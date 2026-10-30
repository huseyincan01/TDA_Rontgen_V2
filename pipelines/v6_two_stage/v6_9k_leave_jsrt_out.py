"""V6.9k — LEAVE-JSRT-OUT (LOSO): JSRT'yi eğitimden TAMAMEN çıkar, %100 yabancı dış test yap.

SORU
----
NODE21'in halka açık eğitim seti dört açık kaynaktan derlenmiştir:

    chestxray14  617 · padchest 314 · jsrt 149 · openi 54   (+ node21_control 3.748)

Omurga `resnet50-res512-all` (pc-nih-rsna-siim-vin) ön-eğitiminde **JSRT'yi hiç görmedi**
(V6.9e'de doğrulandı) — **ama** JSRT hastalarının 107'si NODE21'in kendi **train** bölmesinde
bulunuyordu. Yani şimdiye kadarki JSRT ölçümleri "kaynak-dışı" değil, "aynı kaynaktan başka
hastalar eğitimde vardı" ölçümleridir.

Bu betik, JSRT'yi **eğitimden tamamen çıkarıp** iki aşamayı yeniden eğitir ve JSRT'yi VinDr
gibi %100 yabancı bir dış test seti olarak ölçer. Amaç: **kaynak ezberi var mı?** JSRT'yi
eğitimde görmek, JSRT üzerindeki performansı yapay olarak şişiriyor muydu?

    JSRT dağılımı: train 107 · val 23 · test 19 = 149 hasta (hepsi nodüllü, kontrol YOK)
    LOSO eğitimi: train − {JSRT}  ⇒ 683 nodüllü + 683 kontrol = 1.366 görüntü (1:1)

🔒 İZOLASYON SÖZLEŞMESİ (bu betiğin en önemli kuralı)
-----------------------------------------------------
Üretilen hiçbir şey ana mimariye SIZMAZ. Ana artifact'ler salt-okunur:

    OKUNUR (asla yazılmaz)                     YAZILIR (ayrı ad alanı)
    checkpoints/M5_foundation.pt               checkpoints/M5_nojsrt.pt
    checkpoints/stage2_foundation_tda_hgb.joblib   checkpoints/stage2_nojsrt_tda_hgb.joblib
    artifacts/.../v6_9_test_seal.json          artifacts/.../v6_9k_leave_jsrt_out.json

Ayrıca betik, ana artifact'lerin SHA-256'sını koşunun **başında ve sonunda** hesaplar;
değişmişlerse koşuyu **geçersiz ilan eder**. Bu, sızıntının *kanıtı* değil, *yokluğunun
denetimidir*.

Ayrıca TDA PI aralıkları (`ranges`) da LOSO adaylarından **yeniden türetilir** — ana
bundle'ın aralıkları JSRT görmüş olabilir; LOSO'da göremez. z-referansı val'in
**JSRT-dışı** hastalarından dondurulur.

ÖLÇÜM TASARIMI (JSRT'de kontrol görüntüsü YOK ⇒ AUC için negatif gerekir)
-------------------------------------------------------------------------
1. `recall_at_K`  : lezyon yakalama (K=10/20/50, Kriter-1) — negatif GEREKTİRMEZ.
2. `patient_hit`  : top-50'de ≥1 lezyon yakalayan JSRT hastası oranı — negatif GEREKTİRMEZ.
3. `candidate_auc`: aynı JSRT görüntülerinde TP adaylar vs TP olmayan adaylar ⇒ 2. aşamanın
                    yabancı veride ayrım gücü; dış negatif GEREKTİRMEZ.
4. `mixed_auc`    : JSRT pozitifleri vs NODE21 kontrolleri (yalnız val+test kontrolleri,
                    hiçbiri eğitimde değil) — V6.9d/e konvansiyonuyla aynı, "harici" etiketli.
5. `official`     : JSRT + kontroller üzerinde resmî FROC/rank_metric (kıyas için).

KARŞILAŞTIRMA: ana model (M5_foundation + stage2_foundation_tda_hgb) **aynı** JSRT
hastalarında salt-okunur ölçülür ⇒ Δ = LOSO − ANA. Δ ≈ 0 ise JSRT'yi eğitimde görmenin
JSRT performansına katkısı yoktur (kaynak ezberi yok).

Aşamalar (tek tek de koşulabilir):
    python pipelines/v6_two_stage/v6_9k_leave_jsrt_out.py --stage all
    python pipelines/v6_two_stage/v6_9k_leave_jsrt_out.py --stage train
    python pipelines/v6_two_stage/v6_9k_leave_jsrt_out.py --stage build --skip-train
    python pipelines/v6_two_stage/v6_9k_leave_jsrt_out.py --stage eval  --skip-train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_6_train_foundation as tr  # noqa: E402
from detector_models_foundation import FoundationCenterNet, canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, official_metric, pick_peaks  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
OUT = ART / "v6_9k_leave_jsrt_out.json"

# ---- ayrı ad alanı (ana mimariye DOKUNULMAZ) ------------------------------- #
LOSO_TAG = "M5_nojsrt"
LOSO_BUNDLE = CKPT / "stage2_nojsrt_tda_hgb.joblib"
LOSO_S1 = CKPT / f"{LOSO_TAG}.pt"

# ---- ana mimari (SALT-OKUNUR) ---------------------------------------------- #
MAIN_TAG = "M5_foundation"
MAIN_BUNDLE = CKPT / "stage2_foundation_tda_hgb.joblib"

# ---- kilitli protokol (V6.9 mührüyle birebir) ------------------------------ #
SCALES = (0.9, 1.0, 1.1)
WIN, NMS_PX, TOPK = 3, 8, 50
BOX_HALF, W = 28.0, 0.55
KS = (10, 20, 50)
SRC_JSRT = "jsrt"


# --------------------------------------------------------------------------- #
# yardımcılar
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    if not path.exists():
        return "YOK"
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            d.update(chunk)
    return d.hexdigest()


def guard_snapshot() -> dict:
    """Ana artifact'lerin koşu öncesi parmak izi."""
    return {"M5_foundation.pt": sha256(CKPT / f"{MAIN_TAG}.pt"),
            "stage2_foundation_tda_hgb.joblib": sha256(MAIN_BUNDLE)}


def source_of(pid: str) -> str:
    return str(json.loads(
        (common.PATIENTS_DIR / pid / "patient_info.json").read_text(encoding="utf-8")
    ).get("source_dataset"))


def jsrt_patients() -> dict[str, list[str]]:
    """JSRT hastaları, bölme bazında."""
    sp = common.split_patients()
    return {s: [p for p in pids if source_of(p) == SRC_JSRT] for s, pids in sp.items()}


def _auc(y, s) -> float | None:
    """Eşitlik-duyarlı (Mann–Whitney) AUC. HGB olasılıkları çok eşitli olduğu için
    `argsort` tabanlı AUC yanlı olurdu; ortalama sıra kullanılır."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return None
    from scipy.stats import rankdata

    r = rankdata(s)
    return round(float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N)), 4)


# --------------------------------------------------------------------------- #
# 1) AŞAMA 1 EĞİTİMİ (JSRT hariç)
# --------------------------------------------------------------------------- #
def train_loso(device, args) -> None:
    torch.manual_seed(tr.SEED)
    np.random.seed(tr.SEED)
    js = set(jsrt_patients()["train"])
    pos_all = p2._nodule_pids("train")
    pos = [p for p in pos_all if p not in js]
    neg_all = p2._control_pids("train")
    if args.nodule:
        pos = pos[:args.nodule]
    if args.control:
        neg = neg_all[:args.control]
    else:
        neg = neg_all[:len(pos)]
    print(f"  JSRT train'den çıkarıldı: {len(pos_all)} → {len(pos)} nodüllü "
          f"(−{len(pos_all) - len(pos)})", flush=True)
    print(f"  eğitim: {len(pos)} nodül + {len(neg)} kontrol = {len(pos) + len(neg)}", flush=True)
    assert not (set(pos) & js), "JSRT train'de kaldı — izolasyon ihlali"

    print("\n[1] RAM önbelleği...", flush=True)
    cache = tr.build_cache(list(pos) + list(neg))
    items = [(p, True) for p in pos if p in cache] + [(p, False) for p in neg if p in cache]
    print(f"    kullanılabilir örnek: {len(items)}", flush=True)

    print("\n[2] Model (sıfırdan, CXR vakıf omurgası)...", flush=True)
    model = FoundationCenterNet(weights=args.backbone).to(device)
    print(f"    {model.trainable_summary()}", flush=True)

    epochs, hist = args.epochs, []
    sched = None
    opt = None
    for ep in range(1, epochs + 1):
        if ep == 1:
            model.freeze_backbone()
            opt = torch.optim.Adam(model.param_groups(0.0, tr.LR_HEAD), lr=tr.LR_HEAD,
                                   weight_decay=1e-5)
        elif ep == args.warm + 1:
            model.unfreeze_top()
            print(f"  >>> Faz 2: üst bloklar açıldı ({model.trainable_summary()})", flush=True)
            opt = torch.optim.Adam(model.param_groups(tr.LR_BACKBONE, tr.LR_HEAD),
                                   weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs - args.warm), eta_min=1e-5)

        model.train()
        model.set_bn_eval()
        rng = np.random.RandomState(tr.SEED + ep)
        order = rng.permutation(len(items))
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(order), args.batch):
            sel = [items[i] for i in order[s:s + args.batch]]
            if len(sel) < 2:
                continue
            x, Y, R = tr.make_batch(sel, cache, rng, device)
            loss = tr.loss_fn(model, x, Y, R)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if ep > args.warm and sched is not None:
            sched.step()
        avg = tot / max(1, nb)
        lrs = {g.get("name", "?"): g["lr"] for g in opt.param_groups}
        hist.append({"epoch": ep, "loss": round(avg, 5), "lrs": lrs,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {ep:2d}/{epochs} | loss {avg:.4f} | {time.time() - t0:.0f}s | "
              f"lr_head={lrs.get('head')} " +
              (f"lr_bb={lrs.get('backbone'):.2e}" if "backbone" in lrs else "bb=DONUK"),
              flush=True)
        if ep == args.warm:
            torch.save(model.state_dict(), CKPT / f"{LOSO_TAG}_warm.pt")

    torch.save(model.state_dict(), LOSO_S1)
    (ART / f"{LOSO_TAG}_train_log.json").write_text(json.dumps(
        {"experiment": "V6.9k-leave-jsrt-out", "backbone": args.backbone,
         "excluded_source": SRC_JSRT, "n_excluded_train": len(pos_all) - len(pos),
         "epochs": epochs, "warm": args.warm, "batch": args.batch,
         "n_train": len(items), "trainable": model.trainable_summary(),
         "history": hist}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kaydedildi: {LOSO_S1}", flush=True)


# --------------------------------------------------------------------------- #
# aday üretimi / özellikler
# --------------------------------------------------------------------------- #
def candidates(model, pids, device, tag) -> dict:
    out, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        out[pid] = [(float(cx), float(cy), float(s)) for cx, cy, s, _r in cs]
        if (i + 1) % 200 == 0:
            print(f"    aday {tag} {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    aday {tag}: {sum(len(v) for v in out.values())} kutu "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def features(pids, cands, ranges) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(pid, cands[pid], ranges, False) for pid in pids]
    out, t0 = {}, time.time()
    try:
        with ProcessPoolExecutor(max_workers=8) as pool:
            for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
                out[pid] = res
    except Exception as exc:
        print(f"    (süreç havuzu yok: {type(exc).__name__} → seri)", flush=True)
        for job in jobs:
            pid, res, err = s2._patient_features(job)
            out[pid] = res
    bad = [p for p in pids if out.get(p) is None]
    if bad:
        print(f"    ⚠️ özellik çıkarılamayan {len(bad)} hasta: {bad[:5]}", flush=True)
    print(f"    özellik {len(pids)} hasta ({time.time() - t0:.0f}s)", flush=True)
    return out


def matrix(pids, cands, feats, gts):
    tda, spa, y = [], [], []
    for pid in pids:
        f = feats.get(pid)
        if f is None:
            continue
        labs = s2._labels_for(cands[pid], gts[pid])
        for i in range(len(cands[pid])):
            tda.append(f["tda"][i])
            spa.append(f["spa"][i])
            y.append(1 if labs[i] is not None else 0)
    return (np.asarray(tda, np.float32), np.asarray(spa, np.float32),
            np.asarray(y, np.int64))


def fit_stage2(cands_tr, feats_tr, gts_tr, tr_pids, ranges):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    tda, spa, y = matrix(tr_pids, cands_tr, feats_tr, gts_tr)
    X = np.concatenate([tda, spa], 1)
    sc = StandardScaler().fit(X)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         class_weight="balanced", random_state=42
                                         ).fit(sc.transform(X), y)
    return sc, hgb, X, y


def rescore(bundle, pids, cands, feats, gts):
    """Her hasta için (s1, p_tda) dizileri."""
    sc, hgb = bundle["scaler"], bundle["hgb"]
    out = {}
    for pid in pids:
        cs, f = cands[pid], feats.get(pid)
        if f is None or not cs:
            out[pid] = (np.zeros(0, np.float32), np.zeros(0, np.float32))
            continue
        X = sc.transform(np.concatenate([f["tda"][:len(cs)], f["spa"][:len(cs)]], 1))
        out[pid] = (np.array([c[2] for c in cs], np.float32),
                    hgb.predict_proba(X)[:, 1].astype(np.float32))
    return out


def zref(pids, scored) -> dict:
    s1 = np.concatenate([scored[p][0] for p in pids if len(scored[p][0])])
    td = np.concatenate([scored[p][1] for p in pids if len(scored[p][1])])
    return {"s1_mean": float(s1.mean()), "s1_std": float(s1.std()),
            "tda_mean": float(td.mean()), "tda_std": float(td.std()), "w": W}


def fuse(s1, p_tda, ref, w) -> np.ndarray:
    z1 = (s1 - ref["s1_mean"]) / (ref["s1_std"] + 1e-9)
    zt = (p_tda - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
    return (w * z1 + (1.0 - w) * zt).astype(np.float32)


# --------------------------------------------------------------------------- #
# ölçümler
# --------------------------------------------------------------------------- #
def _recall_at_K(cands, gts, K):
    if not gts:
        return None
    return common.recall_at_K(cands, gts, K, common.hit_criterion1)


def measure(pids_jsrt, cands, scored, ref, gts, ctrl_pids=None, ctrl_cands=None,
            ctrl_scored=None) -> dict:
    """JSRT üzerinde negatif-gerektirmeyen + kontrol-karışık metrikler."""
    n_les = sum(len(gts[p]) for p in pids_jsrt)
    rep: dict = {"n_patients": len(pids_jsrt), "n_lesions": n_les}

    # ---- 1/2: yakalama (negatif gerekmez) --------------------------------- #
    rec = {}
    for K in KS:
        hit = sum(int(round(_recall_at_K(cands[p], gts[p], K) * len(gts[p])))
                  for p in pids_jsrt if gts[p])
        rec[f"K{K}"] = round(hit / max(1, n_les), 4)
    pat_hit = sum(1 for p in pids_jsrt if gts[p]
                  and any(common.hit_criterion1(c[0], c[1], g) for c in cands[p][:TOPK]
                          for g in gts[p]))
    rep["recall_at_K"] = rec
    rep["patient_hit_top50"] = round(pat_hit / max(1, len([p for p in pids_jsrt if gts[p]])), 4)

    # ---- 3: aday-içi AUC (aynı görüntülerde TP vs TP-değil) ---------------- #
    cand_auc = {}
    for key, idx in (("stage1", 0), ("tda", 1)):
        y, s = [], []
        for p in pids_jsrt:
            labs = s2._labels_for(cands[p], gts[p])
            arr = scored[p][idx]
            for i in range(len(cands[p])):
                y.append(1 if labs[i] is not None else 0)
                s.append(float(arr[i]))
        cand_auc[key] = _auc(y, s)
    y, s = [], []
    for p in pids_jsrt:
        labs = s2._labels_for(cands[p], gts[p])
        arr = fuse(scored[p][0], scored[p][1], ref, ref["w"])
        for i in range(len(cands[p])):
            y.append(1 if labs[i] is not None else 0)
            s.append(float(arr[i]))
    cand_auc["fusion"] = _auc(y, s)
    rep["candidate_auc"] = cand_auc

    # ---- resmî metrik (JSRT + kontroller) --------------------------------- #
    if ctrl_pids:
        allp = list(pids_jsrt) + list(ctrl_pids)
        all_c = dict(cands)
        all_c.update(ctrl_cands or {})
        all_s = dict(scored)
        all_s.update(ctrl_scored or {})
        gt_all = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in gts[p]]
                  for p in allp}
        is_nod = {p: bool(gts[p]) for p in allp}
        n_all = sum(len(gt_all[p]) for p in allp)

        def series(p, kind):
            s1, td = all_s.get(p, (np.zeros(0), np.zeros(0)))
            if not len(s1):
                return np.zeros(0)
            if kind == "stage1":
                return s1
            if kind == "tda":
                return td
            return fuse(s1, td, ref, ref["w"])

        flat = {k: [(float(series(p, k)[i]), p, float(cx), float(cy), 0.0)
                    for p in allp for i, (cx, cy, _o) in enumerate(all_c.get(p, []))]
                for k in ("stage1", "tda", "fusion")}
        rep["official_jsrt_plus_controls"] = {
            k: official_metric(v, len(allp), n_all, gt_all, is_nod, allp, box_half=BOX_HALF)
            for k, v in flat.items()}
        rep["official_n_controls"] = len(ctrl_pids)

        # ---- 4: kontrol-karışık hasta düzeyi AUC (top1 / top3 / noisyor) -- #
        def img_score(p, kind, how):
            arr = series(p, kind)
            if not len(arr):
                return 0.0
            if kind == "fusion":
                arr = 1.0 / (1.0 + np.exp(-arr))
            ss = sorted(arr.tolist(), reverse=True)
            if how == "top1":
                return float(ss[0])
            if how == "top3":
                return float(np.mean(ss[:3]))
            return float(1 - np.prod([1 - min(1.0, max(0.0, x)) for x in ss[:5]]))

        mixed = {}
        for kind in ("stage1", "tda", "fusion"):
            mixed[kind] = {}
            for how in ("top1", "top3", "noisyor"):
                yv = [1] * len(pids_jsrt) + [0] * len(ctrl_pids)
                sv = [img_score(p, kind, how) for p in allp]
                mixed[kind][how] = _auc(yv, sv)
        rep["mixed_patient_auc"] = mixed
    return rep


# --------------------------------------------------------------------------- #
# ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "train", "build", "eval"])
    ap.add_argument("--backbone", default="resnet50-res512-all")
    ap.add_argument("--epochs", type=int, default=tr.TOTAL_EPOCHS)
    ap.add_argument("--warm", type=int, default=tr.WARM_EPOCHS)
    ap.add_argument("--batch", type=int, default=tr.BATCH)
    ap.add_argument("--control", type=int, default=0)
    ap.add_argument("--nodule", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--n-controls", type=int, default=250,
                    help="kontrol-karışık AUC için negatif havuz boyu (val kontrolleri)")
    ap.add_argument("--limit", type=int, default=0, help="duman: her grupta N hasta")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9k — LEAVE-JSRT-OUT (LOSO) dış test", flush=True)
    print(f"  device={device} | LOSO_TAG={LOSO_TAG} | ana mimari SALT-OKUNUR", flush=True)

    before = guard_snapshot()
    print(f"  ana artifact parmak izi (öncesi): {json.dumps(before, ensure_ascii=False)[:120]}…",
          flush=True)

    js = jsrt_patients()
    sp = common.split_patients()
    print(f"  JSRT: train {len(js['train'])} · val {len(js['val'])} · "
          f"test {len(js['test'])} = {sum(len(v) for v in js.values())} hasta", flush=True)

    # ---------------- 1) EĞİTİM ---------------- #
    if args.stage in ("all", "train") and not (args.skip_train and args.stage == "all"):
        print("\n" + "=" * 70 + "\n[1] AŞAMA 1 — LOSO eğitimi (JSRT hariç)\n" + "=" * 70, flush=True)
        train_loso(device, args)
    if args.stage == "train":
        return
    if not LOSO_S1.exists():
        raise SystemExit(f"LOSO checkpoint yok: {LOSO_S1} — önce --stage train")

    # ---------------- 2) ADAYLAR + RANGES + HGB ---------------- #
    model = load_foundation(LOSO_TAG, device)
    js_all = js["train"] + js["val"] + js["test"]
    js_set = set(js["train"]) | set(js["val"]) | set(js["test"])
    tr_pids = [p for p in p2._nodule_pids("train") if p not in set(js["train"])]
    ctl_pids = p2._control_pids("train")[:len(tr_pids)]
    va_pids = [p for p in sp["val"] if p not in set(js["val"])]
    # negatif havuzu: val kontrolleri (test bölmesine HİÇ dokunulmaz)
    rng = np.random.RandomState(42)
    _va_ctrl = [p for p in va_pids if not common.load_gt(p)]
    ctrl_eval = list(rng.permutation(_va_ctrl)[:args.n_controls])
    if args.limit:
        tr_pids, ctl_pids = tr_pids[:args.limit], ctl_pids[:args.limit]
        va_pids, js_all = va_pids[:args.limit], js_all[:args.limit]
        ctrl_eval = ctrl_eval[:args.limit]
    fit_pids = tr_pids + ctl_pids
    print(f"\n  eğitim adayı hastası {len(fit_pids)} · val(-JSRT) {len(va_pids)} · "
          f"JSRT {len(js_all)} · negatif havuzu {len(ctrl_eval)} (val kontrolü)", flush=True)

    gts_fit = {p: common.load_gt(p) for p in fit_pids}
    gts_js = {p: common.load_gt(p) for p in js_all}
    gts_ctrl = {p: common.load_gt(p) for p in ctrl_eval}

    print("\n[2] Adaylar (fit)…", flush=True)
    c_fit = candidates(model, fit_pids, device, "fit")
    s2._IDS["s"]["v51"] = object()          # derive_ranges'te kullanılmıyor (import atla)
    sample = [(p, c_fit[p]) for p in tr_pids if gts_fit[p]][:20]
    ranges = s2.derive_ranges(sample)
    print(f"    LOSO PI aralıkları (JSRT görmedi): {ranges}", flush=True)

    print("\n[3] Özellikler (fit)…", flush=True)
    f_fit = features(fit_pids, c_fit, ranges)
    sc_loso, hgb_loso, X_fit, y_fit = fit_stage2(c_fit, f_fit, gts_fit, fit_pids, ranges)
    print(f"    HGB: {len(y_fit)} aday · %{100 * y_fit.mean():.2f} pozitif", flush=True)

    print("\n[4] Val(-JSRT) → z-referansı…", flush=True)
    c_va = candidates(model, va_pids, device, "val")
    f_va = features(va_pids, c_va, ranges)
    gts_va = {p: common.load_gt(p) for p in va_pids}
    sc_va = rescore({"scaler": sc_loso, "hgb": hgb_loso}, va_pids, c_va, f_va, gts_va)
    ref_loso = zref(va_pids, sc_va)
    print(f"    ref: {json.dumps(ref_loso, ensure_ascii=False)}", flush=True)

    bundle_loso = {"hgb": hgb_loso, "scaler": sc_loso, "ranges": ranges,
                   "fusion_ref": ref_loso, "stage1_score_included": False,
                   "meta": {"experiment": "V6.9k-leave-jsrt-out",
                            "excluded_source": SRC_JSRT,
                            "trained_on": f"{len(fit_pids)} hasta (1:1), JSRT train'de YOK",
                            "features": "1600 PI + 4 uzamsal = 1604 (s1 YOK)",
                            "isolation": "ana mimariye SIZMAZ — ayrı ad alanı"}}
    import joblib
    joblib.dump(bundle_loso, LOSO_BUNDLE)
    print(f"    kaydedildi: {LOSO_BUNDLE.name}", flush=True)

    if args.stage == "build":
        print("\n  --stage build bitti (eval atlandı)", flush=True)
        return

    # ---------------- 3) DIŞ TEST: JSRT ---------------- #
    print("\n[5] JSRT dış testi (LOSO)…", flush=True)
    c_js = candidates(model, js_all, device, "jsrt")
    f_js = features(js_all, c_js, ranges)
    sc_js_loso = rescore(bundle_loso, js_all, c_js, f_js, gts_js)

    print("\n[6] Negatif havuzu (val kontrolleri)…", flush=True)
    c_ct = candidates(model, ctrl_eval, device, "ctrl")
    f_ct = features(ctrl_eval, c_ct, ranges)
    sc_ct_loso = rescore(bundle_loso, ctrl_eval, c_ct, f_ct, gts_ctrl)

    gts_all = dict(gts_js)
    gts_all.update(gts_ctrl)
    js_val_eval = [p for p in js["val"] if p in set(js_all)]     # duman modunda alt küme
    loso = measure(js_all, c_js, sc_js_loso, ref_loso, gts_all,
                   ctrl_pids=ctrl_eval, ctrl_cands=c_ct, ctrl_scored=sc_ct_loso)
    loso_val = measure(js_val_eval, c_js, sc_js_loso, ref_loso, gts_all,
                       ctrl_pids=ctrl_eval, ctrl_cands=c_ct, ctrl_scored=sc_ct_loso)

    # ---------------- 4) ANA MODEL (SALT-OKUNUR) aynı JSRT ---------------- #
    print("\n[7] ANA model aynı JSRT üzerinde (salt-okunur)…", flush=True)
    main_model = load_foundation(MAIN_TAG, device)
    main_bundle = joblib.load(MAIN_BUNDLE)
    assert main_bundle.get("stage1_score_included") is False
    c_js_m = candidates(main_model, js_all, device, "jsrt-main")
    f_js_m = features(js_all, c_js_m, main_bundle["ranges"])
    sc_js_main = rescore(main_bundle, js_all, c_js_m, f_js_m, gts_js)
    c_ct_m = candidates(main_model, ctrl_eval, device, "ctrl-main")
    f_ct_m = features(ctrl_eval, c_ct_m, main_bundle["ranges"])
    sc_ct_main = rescore(main_bundle, ctrl_eval, c_ct_m, f_ct_m, gts_ctrl)
    main = measure(js_all, c_js_m, sc_js_main, main_bundle["fusion_ref"], gts_all,
                   ctrl_pids=ctrl_eval, ctrl_cands=c_ct_m, ctrl_scored=sc_ct_main)
    main_val = measure(js_val_eval, c_js_m, sc_js_main, main_bundle["fusion_ref"], gts_all,
                       ctrl_pids=ctrl_eval, ctrl_cands=c_ct_m, ctrl_scored=sc_ct_main)

    # ---- izolasyon kanıtı: LOSO eğitimi ve PI aralıkları JSRT görmedi ----- #
    assert not (set(fit_pids) & js_set), "JSRT adayları eğitim havuzunda — ihlal"
    assert not (set(p for p, _ in sample) & js_set), "JSRT PI aralıklarına sızdı — ihlal"

    # ---------------- 5) Δ ve rapor ---------------- #
    def delta(block_l, block_m, *path):
        a = block_l
        b = block_m
        for k in path:
            a = a.get(k) if isinstance(a, dict) else None
            b = b.get(k) if isinstance(b, dict) else None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(a - b, 4)
        return None

    d = {"recall_at_K": {f"K{k}": delta(loso, main, "recall_at_K", f"K{k}") for k in KS},
         "patient_hit_top50": delta(loso, main, "patient_hit_top50"),
         "candidate_auc": {k: delta(loso, main, "candidate_auc", k)
                           for k in ("stage1", "tda", "fusion")},
         "mixed_patient_auc": {k: {h: delta(loso, main, "mixed_patient_auc", k, h)
                                   for h in ("top1", "top3", "noisyor")}
                               for k in ("stage1", "tda", "fusion")}}

    after = guard_snapshot()
    isolation_ok = before == after
    print(f"\n  🔒 izolasyon denetimi: ana artifact'ler {'DEĞİŞMEDİ ✔' if isolation_ok else 'DEĞİŞTİ ✘'}", flush=True)

    rep = {
        "experiment": "V6.9k-leave-jsrt-out",
        "question": ("JSRT hastalarının 107'si NODE21 train bölmesindeydi. JSRT'yi eğitimden "
                     "tamamen çıkarınca JSRT üzerindeki performans düşüyor mu? "
                     "(kaynak ezberi testi)"),
        "design": {
            "excluded_source": SRC_JSRT,
            "n_jsrt": {k: len(v) for k, v in js.items()},
            "n_jsrt_total": sum(len(v) for v in js.values()),
            "train_excluded": len(js["train"]),
            "loso_train": f"{len(tr_pids)} nodüllü + {len(ctl_pids)} kontrol (1:1, JSRT YOK)",
            "eval": "JSRT'nin 149 hastasının tamamı (hiçbiri LOSO eğitiminde değil)",
        },
        "isolation": {
            "contract": ("LOSO modeli/TDA'sı ana mimariye SIZMAZ: ayrı checkpoint, ayrı "
                         "bundle, ayrı JSON. Ana artifact'ler salt-okunur."),
            "main_artifacts_before": before,
            "main_artifacts_after": after,
            "unchanged": isolation_ok,
            "loso_checkpoint": str(LOSO_S1.relative_to(PROJECT_ROOT)),
            "loso_bundle": str(LOSO_BUNDLE.relative_to(PROJECT_ROOT)),
            "loso_bundle_sha256": sha256(LOSO_BUNDLE),
            "loso_s1_sha256": sha256(LOSO_S1),
        },
        "protocol": {"scales": list(SCALES), "flip_tta": True, "win": WIN,
                     "nms_px": NMS_PX, "top_k": TOPK, "box_px": int(2 * BOX_HALF),
                     "w_fusion": W,
                     "z_ref_loso": ref_loso,
                     "z_ref_main": main_bundle.get("fusion_ref"),
                     "negative_pool": f"val kontrolleri ({len(ctrl_eval)} hasta, seed 42) — "
                                      f"TEST bölmesine dokunulmadı"},
        "loso": {"jsrt_all149": loso, "jsrt_val_only": loso_val},
        "main": {"jsrt_all149": main, "jsrt_val_only": main_val},
        "delta_loso_minus_main": d,
        "n_controls_for_mixed_auc": len(ctrl_eval),
        "smoke_limit": args.limit,
    }
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    print("\n" + "=" * 70, flush=True)
    print(f"  JSRT DIŞ TEST | n={loso['n_patients']} hasta · {loso['n_lesions']} lezyon",
          flush=True)
    for nm, b in (("LOSO", loso), ("ANA ", main)):
        ca = b["candidate_auc"]
        print(f"  [{nm}] Recall@K {b['recall_at_K']} hasta-yakalama {b['patient_hit_top50']}",
              flush=True)
        print(f"  [{nm}] aday-AUC s1={ca['stage1']} tda={ca['tda']} füzyon={ca['fusion']}",
              flush=True)
        if "mixed_patient_auc" in b:
            print(f"  [{nm}] kontrol-karışık AUC füzyon {b['mixed_patient_auc']['fusion']}",
                  flush=True)
    print(f"\n  Δ(LOSO − ANA): {json.dumps(d, ensure_ascii=False)}", flush=True)
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
