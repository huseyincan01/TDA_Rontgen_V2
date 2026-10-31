"""V6.9r — TAM İZOLE SAF JSRT DIŞ TESTİ (149 JSRT Nodüllü + 93 JSRT Sağlıklı = 242 Saf JSRT).

KEŞİF VE MOTİVASYON
-------------------
NODE21 veri setinde JSRT'den gelen 149 nodüllü röntgen `source_dataset = "jsrt"` olarak
etiketlenmiş, ancak JSRT'nin 93 sağlıklı kontrol röntgeni (`JPCNN001..093`) diğer
hastanelerin kontrolleriyle birlikte `node21_control` havuzuna karıştırılmıştı.

`dataset_node21/cxr_images/original_data/images/*.mha` ham başlıkları incelendiğinde JSRT
taramalarının benzersiz fiziksel imzası (`DimSize = 2048 2048`,
`ElementSpacing = 0.17499999999999999 0.17499999999999999`, `ElementType = MET_USHORT`)
sayesinde:
  * 149 pozitif `jsrt` görüntüsünün 149'u (%100),
  * Diğer pozitif kaynakların (`chestxray14`, `padchest`, `openi`) 0'ı (%0),
  * `node21_control` havuzundaki **tam 93 görüntü** (`train: 61, val: 18, test: 14`)
birebir tespit edilmiştir (149 + 93 = 242 JSRT görüntüsü).

Önceki `V6.9k` / `V6.9k4` (Leave-JSRT-Out) deneyinde eğitimden 107 JSRT pozitif hasta
çıkarılmış, ancak `node21_control` içindeki JSRT sağlıklı kontrollerinin bir kısmı eğitimde
ve val z-referansında kalmıştı; ayrıca dış test 149 pozitif + 250 karma `node21_control`
üzerinde yapılmıştı.

BU BETİK (V6.9r — %100 Saf JSRT İzolasyonu)
-------------------------------------------
1. **Eğitimden Tam Çıkarma (0 JSRT Pozitif, 0 JSRT Negatif):**
   `train` bölmesinden hem 107 JSRT nodüllü hasta hem de 61 JSRT sağlıklı kontrol hastası
   tamamen çıkarılır (`683 JSRT-dışı nodüllü + 683 JSRT-dışı sağlıklı = 1.366 hasta`).
2. **Val Kalibrasyonundan Tam Çıkarma:**
   `val` z-referansı (`w = 0.55`) hesaplanırken hem 23 JSRT nodüllü hem de 18 JSRT sağlıklı
   kontrol hastası çıkarılır (`693 JSRT-dışı val hastası`).
3. **TDA PI Aralıkları (`ranges`):**
   Yalnız JSRT-dışı `train` adaylarından türetilir (JSRT görmez).
4. **Saf 1.600 TDA 2. Aşama:**
   `StandardScaler` + `HistGradientBoostingClassifier(max_iter=200, lr=0.08, balanced, rs=42)`
   yalnız 1.600 boyutlu sebat görüntüsü (PI) vektörüyle eğitilir (konum sütunu yok, `s1` yok).
5. **Gerçek Saf JSRT Dış Testi (242 Hasta):**
   Modelin ne 1. aşamada, ne 2. aşamada, ne de kalibrasyonda tek bir örneğini bile görmediği
   **149 JSRT Nodüllü + 93 JSRT Sağlıklı = 242 Saf JSRT (12-bit kayıpsız .mha)** kohortu
   üzerinde sıfır-atış (zero-shot) dış doğrulama yapılır!

🔒 İZOLASYON SÖZLEŞMESİ
-----------------------
Ana mimari ağırlıklarına (`M5_foundation.pt`, `stage2_foundation_tda_hgb_pure1600.joblib`,
`stage2_foundation_tda_hgb.joblib`) ve önceki `M5_nojsrt` ağırlıklarına ASLA dokunulmaz;
SHA-256 özetleri koşu başında ve sonunda doğrulanır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_6_train_foundation as tr  # noqa: E402
import v6_9k_leave_jsrt_out as k  # noqa: E402
from detector_models_foundation import FoundationCenterNet, canon_to_512  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, pick_peaks  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
ORIG_DIR = PROJECT_ROOT / "dataset_node21" / "cxr_images" / "original_data" / "images"

# ---- V6.9r ayrı ad alanı (ana mimariye DOKUNULMAZ) ------------------------ #
TAG_R = "M5_nojsrt_full"
S1_R = CKPT / f"{TAG_R}.pt"
BUNDLE_R = CKPT / "stage2_nojsrt_full_tda_hgb_pure1600.joblib"
CACHE_R = ART / "_v6_9r_cache.npz"
OUT_R = ART / "v6_9r_true_jsrt_external.json"

# ---- Salt-okunur ana ve önceki LOSO artifact'leri ------------------------- #
READONLY_FILES = [
    CKPT / "M5_foundation.pt",
    CKPT / "stage2_foundation_tda_hgb.joblib",
    CKPT / "stage2_foundation_tda_hgb_pure1600.joblib",
    CKPT / "M5_nojsrt.pt",
    CKPT / "stage2_nojsrt_tda_hgb.joblib",
    CKPT / "stage2_nojsrt_tda_hgb_pure1600.joblib",
]

SCALES = k.SCALES
WIN, NMS_PX, TOPK = k.WIN, k.NMS_PX, k.TOPK
BOX_HALF, W = k.BOX_HALF, k.W
SEED = 42


def sha256(path: Path) -> str:
    if not path.exists():
        return "YOK"
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            d.update(chunk)
    return d.hexdigest()


def guard_snapshot() -> dict[str, str]:
    return {p.name: sha256(p) for p in READONLY_FILES}


def find_jsrt_controls() -> list[str]:
    """original_data/images/c*.mha başlıklarından 93 orijinal JSRT sağlıklı kontrolünü bul."""
    out = []
    for p in sorted(ORIG_DIR.glob("c*.mha")):
        hdr = {}
        with open(p, "rb") as f:
            for _ in range(25):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if "=" in line:
                    kh, vh = [x.strip() for x in line.split("=", 1)]
                    hdr[kh] = vh
                if line.startswith("ElementDataFile"):
                    break
        if (hdr.get("DimSize") == "2048 2048"
                and hdr.get("ElementSpacing") == "0.17499999999999999 0.17499999999999999"
                and hdr.get("ElementType") == "MET_USHORT"):
            out.append(p.stem)
    return out


def jsrt_cohorts() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(jsrt_pos_by_split, jsrt_neg_by_split) döndürür."""
    sp = common.split_patients()
    pos_splits = k.jsrt_patients()
    ctrl_set = set(find_jsrt_controls())
    neg_splits = {s: [p for p in pids if p in ctrl_set] for s, pids in sp.items()}
    return pos_splits, neg_splits


def train_stage1_nojsrt_full(device: torch.device, args, tr_pos: list[str], tr_neg: list[str]) -> None:
    torch.manual_seed(tr.SEED)
    np.random.seed(tr.SEED)
    print(f"\n[1] RAM önbelleği ({len(tr_pos)} nodüllü + {len(tr_neg)} kontrol = "
          f"{len(tr_pos) + len(tr_neg)} görüntü)...", flush=True)
    cache = tr.build_cache(list(tr_pos) + list(tr_neg))
    items = [(p, True) for p in tr_pos if p in cache] + [(p, False) for p in tr_neg if p in cache]
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
            torch.save(model.state_dict(), CKPT / f"{TAG_R}_warm.pt")

    torch.save(model.state_dict(), S1_R)
    (ART / f"{TAG_R}_train_log.json").write_text(json.dumps(
        {"experiment": "V6.9r-true-jsrt-external", "backbone": args.backbone,
         "excluded_source": "jsrt_pos_107_and_jsrt_ctrl_61",
         "n_train_pos": len(tr_pos), "n_train_neg": len(tr_neg),
         "epochs": epochs, "warm": args.warm, "batch": args.batch,
         "n_train": len(items), "trainable": model.trainable_summary(),
         "history": hist}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kaydedildi: {S1_R}", flush=True)


def extract_candidates(model, pids: list[str], device: torch.device, tag: str) -> dict[str, list[tuple]]:
    out, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        out[pid] = [(float(cx), float(cy), float(s)) for cx, cy, s, _r in cs]
        if (i + 1) % 200 == 0:
            print(f"    aday {tag} {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    aday {tag}: {len(pids)} hasta, {sum(len(v) for v in out.values())} kutu "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def _tda_worker(args):
    pid, cands, ranges = args
    try:
        img = common.load_image(pid)
        mask_dil = common.load_lung_mask(pid)
        p1, p99 = common.get_lung_p1p99()[pid]
        canon = common.canonicalize(img, mask_dil, float(p1), float(p99))
    except Exception as exc:
        return pid, None, f"{type(exc).__name__}: {exc}"

    ex = ZoneTDAExtractor(s2._settings(ranges))
    n = len(cands)
    tda = np.zeros((n, 2 * 2 * s2.GRID * s2.GRID), np.float32)
    for i, (cx, cy, _s) in enumerate(cands):
        c_in = s2._crop(canon, cx, cy, s2.INNER)
        c_out = s2._crop(canon, cx, cy, s2.OUTER)
        v_in, _ = ex.extract(c_in)
        v_out, _ = ex.extract(c_out)
        tda[i] = np.concatenate([v_in, v_out])
    return pid, tda, None


def extract_pure1600(pids: list[str], cands: dict[str, list[tuple]], ranges: dict, tag: str) -> dict[str, np.ndarray]:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    jobs = [(pid, cands[pid], ranges) for pid in pids]
    out, t0 = {}, time.time()
    try:
        with ProcessPoolExecutor(max_workers=8) as pool:
            for idx, (pid, res, err) in enumerate(pool.map(_tda_worker, jobs, chunksize=8), 1):
                out[pid] = res
                if err:
                    print(f"    ! {pid}: {err}", flush=True)
                if idx % 200 == 0:
                    print(f"    TDA-1600 {tag} {idx}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as exc:
        print(f"    (ProcessPool yok: {type(exc).__name__} → ThreadPool 8 iş parçacığı)", flush=True)
        with ThreadPoolExecutor(max_workers=8) as tpool:
            for idx, (pid, res, err) in enumerate(tpool.map(_tda_worker, jobs), 1):
                out[pid] = res
                if err:
                    print(f"    ! {pid}: {err}", flush=True)
                if idx % 200 == 0:
                    print(f"    TDA-1600 {tag} {idx}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    TDA-1600 {tag}: {len(pids)} hasta tamamlandı ({time.time() - t0:.0f}s)", flush=True)
    return out


def pack_cohort(pids: list[str], cands: dict[str, list[tuple]], feats: dict[str, np.ndarray]) -> dict:
    tda, s1, n, cx, cy = [], [], [], [], []
    for pid in pids:
        cs = cands[pid]
        m = len(cs)
        n.append(m)
        s1.extend([c[2] for c in cs])
        cx.extend([c[0] for c in cs])
        cy.extend([c[1] for c in cs])
        f = feats.get(pid)
        if f is None:
            tda.append(np.zeros((m, 1600), np.float32))
        else:
            tda.append(f[:m].astype(np.float32))
    off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    return {
        "pids": np.asarray(pids),
        "tda": np.concatenate(tda) if tda else np.zeros((0, 1600), np.float32),
        "s1": np.asarray(s1, np.float32),
        "cx": np.asarray(cx, np.float32),
        "cy": np.asarray(cy, np.float32),
        "n": np.asarray(n, np.int64),
        "off": off,
    }


def unpack_cands_scored(d: dict, tag: str, sc, hgb) -> tuple[list[str], dict, dict]:
    pids = [str(p) for p in d[f"{tag}_pids"]]
    off = d[f"{tag}_off"]
    s1 = d[f"{tag}_s1"].astype(np.float32)
    cx = d[f"{tag}_cx"].astype(np.float32)
    cy = d[f"{tag}_cy"].astype(np.float32)
    tda = d[f"{tag}_tda"].astype(np.float32)
    p_tda = hgb.predict_proba(sc.transform(tda))[:, 1].astype(np.float32)
    cands, scored = {}, {}
    for i, p in enumerate(pids):
        lo, hi = int(off[i]), int(off[i + 1])
        cands[p] = [(float(cx[j]), float(cy[j]), float(s1[j])) for j in range(lo, hi)]
        scored[p] = (s1[lo:hi], p_tda[lo:hi])
    return pids, cands, scored


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval"])
    ap.add_argument("--backbone", default="resnet50-res512-all")
    ap.add_argument("--epochs", type=int, default=tr.TOTAL_EPOCHS)
    ap.add_argument("--warm", type=int, default=tr.WARM_EPOCHS)
    ap.add_argument("--batch", type=int, default=tr.BATCH)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    before = guard_snapshot()
    print("=" * 78, flush=True)
    print("V6.9r — TAM İZOLE SAF JSRT DIŞ TESTİ (149 Pozitif + 93 Sağlıklı = 242 Saf JSRT)", flush=True)
    print(f"  device={device} | TAG={TAG_R} | Ana mimari SALT-OKUNUR", flush=True)
    print("=" * 78, flush=True)

    js_pos, js_neg = jsrt_cohorts()
    js_pos_all = js_pos["train"] + js_pos["val"] + js_pos["test"]
    js_neg_all = js_neg["train"] + js_neg["val"] + js_neg["test"]
    js_all_set = set(js_pos_all) | set(js_neg_all)

    assert len(js_pos_all) == 149, f"Beklenen 149 JSRT pozitif, bulunan: {len(js_pos_all)}"
    assert len(js_neg_all) == 93, f"Beklenen 93 JSRT sağlıklı, bulunan: {len(js_neg_all)}"

    sp = common.split_patients()
    tr_pos_all = p2._nodule_pids("train")
    tr_neg_all = p2._control_pids("train")

    tr_pos = [p for p in tr_pos_all if p not in js_all_set]
    tr_neg_clean = [p for p in tr_neg_all if p not in js_all_set]
    tr_neg = tr_neg_clean[:len(tr_pos)]
    fit_pids = tr_pos + tr_neg

    va_clean = [p for p in sp["val"] if p not in js_all_set]

    # Sıfır JSRT sızıntı doğrulaması
    assert not (set(fit_pids) & js_all_set), "İHLAL: Eğitim setinde JSRT hastası kaldı!"
    assert not (set(va_clean) & js_all_set), "İHLAL: Val kalibrasyon setinde JSRT hastası kaldı!"

    print(f"  JSRT Pozitif (149)  : train={len(js_pos['train'])}, val={len(js_pos['val'])}, test={len(js_pos['test'])}", flush=True)
    print(f"  JSRT Sağlıklı (93)  : train={len(js_neg['train'])}, val={len(js_neg['val'])}, test={len(js_neg['test'])}", flush=True)
    print(f"  Eğitim (0 JSRT)     : {len(tr_pos)} nodüllü + {len(tr_neg)} sağlıklı = {len(fit_pids)} hasta (1:1)", flush=True)
    print(f"  Val z-ref (0 JSRT)  : {len(va_clean)} hasta (23 JSRT poz + 18 JSRT sağ çıkarıldı)", flush=True)
    print(f"  Saf JSRT Dış Test   : {len(js_pos_all)} nodüllü + {len(js_neg_all)} sağlıklı = {len(js_pos_all) + len(js_neg_all)} hasta", flush=True)

    # ---------------- 1) 1. AŞAMA EĞİTİMİ ---------------- #
    if args.stage in ("all", "train") and not args.skip_train:
        print("\n" + "=" * 78 + "\n[AŞAMA 1] M5_nojsrt_full Eğitimi (0 JSRT Pozitif, 0 JSRT Negatif)\n" + "=" * 78, flush=True)
        train_stage1_nojsrt_full(device, args, tr_pos, tr_neg)
    if args.stage == "train":
        return
    if not S1_R.exists():
        raise SystemExit(f"Checkpoint bulunamadı: {S1_R}")

    # ---------------- 2) ADAY + SAF 1600 TDA ÖNBELLEĞİ ---------------- #
    if CACHE_R.exists() and not args.refresh_cache:
        print(f"\n[AŞAMA 2] Önbellek yükleniyor: {CACHE_R.name}", flush=True)
        z = np.load(CACHE_R, allow_pickle=True)
        skip_fit_arrays = args.stage == "eval" and BUNDLE_R.exists()
        d = {kk: z[kk] for kk in z.files if not (skip_fit_arrays and kk.startswith("fit_"))}
        ranges = json.loads(str(d["ranges_json"]))
    else:
        print("\n" + "=" * 78 + "\n[AŞAMA 2] Aday Üretimi + PI Aralıkları + Saf 1.600 TDA Çıkarımı\n" + "=" * 78, flush=True)
        model = load_foundation(TAG_R, device)
        c_fit = extract_candidates(model, fit_pids, device, "fit_clean")
        c_va = extract_candidates(model, va_clean, device, "va_clean")
        c_jp = extract_candidates(model, js_pos_all, device, "jsrt_pos_149")
        c_jn = extract_candidates(model, js_neg_all, device, "jsrt_neg_93")

        gts_fit = {p: common.load_gt(p) for p in fit_pids}
        s2._IDS["s"]["v51"] = object()
        sample = [(p, c_fit[p]) for p in tr_pos if gts_fit[p]][:20]
        assert not (set(p for p, _ in sample) & js_all_set), "İHLAL: PI aralıklarına JSRT sızdı!"
        ranges = s2.derive_ranges(sample)
        print(f"    TDA PI aralıkları (0 JSRT): {ranges}", flush=True)

        f_fit = extract_pure1600(fit_pids, c_fit, ranges, "fit_clean")
        f_va = extract_pure1600(va_clean, c_va, ranges, "va_clean")
        f_jp = extract_pure1600(js_pos_all, c_jp, ranges, "jsrt_pos_149")
        f_jn = extract_pure1600(js_neg_all, c_jn, ranges, "jsrt_neg_93")

        fit_y = []
        for p in fit_pids:
            fit_y.extend([1 if lb is not None else 0 for lb in s2._labels_for(c_fit[p], gts_fit[p])])

        d = {"fit_y": np.asarray(fit_y, np.int64), "ranges_json": np.asarray(json.dumps(ranges))}
        for tag, pids, cc, ff in (
            ("fit", fit_pids, c_fit, f_fit),
            ("va", va_clean, c_va, f_va),
            ("jp", js_pos_all, c_jp, f_jp),
            ("jn", js_neg_all, c_jn, f_jn),
        ):
            pk = pack_cohort(pids, cc, ff)
            for kk, vv in pk.items():
                d[f"{tag}_{kk}"] = vv
        np.savez_compressed(CACHE_R, **d)
        print(f"    Önbellek kaydedildi: {CACHE_R.name}", flush=True)

    # ---------------- 3) SAF 1.600 HGB EĞİTİMİ + VAL Z-REF ---------------- #
    if args.stage == "eval" and BUNDLE_R.exists():
        print(f"\n[AŞAMA 3] Mevcut bundle yükleniyor: {BUNDLE_R.name}", flush=True)
        bundle_r = joblib.load(BUNDLE_R)
        sc = bundle_r["scaler"]
        hgb = bundle_r["hgb"]
        ref_r = bundle_r["fusion_ref"]
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        print("\n" + "=" * 78 + "\n[AŞAMA 3] Saf 1.600 TDA HGB Eğitimi ve Val(-JSRT) Kalibrasyonu\n" + "=" * 78, flush=True)
        X_fit = d["fit_tda"].astype(np.float32)
        y_fit = d["fit_y"].astype(np.int64)
        sc = StandardScaler().fit(X_fit)
        hgb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, class_weight="balanced", random_state=SEED
        ).fit(sc.transform(X_fit), y_fit)

        X_va = d["va_tda"].astype(np.float32)
        p_va = hgb.predict_proba(sc.transform(X_va))[:, 1].astype(np.float32)
        s1_va = d["va_s1"].astype(np.float32)
        ref_r = {
            "s1_mean": float(s1_va.mean()),
            "s1_std": float(s1_va.std()),
            "tda_mean": float(p_va.mean()),
            "tda_std": float(p_va.std()),
            "w": W,
        }
        print(f"  HGB fit aday: {len(y_fit)} (%{100 * y_fit.mean():.2f} pozitif)", flush=True)
        print(f"  Val(-JSRT) z-ref: {json.dumps(ref_r, ensure_ascii=False)}", flush=True)

        bundle_r = {
            "hgb": hgb,
            "scaler": sc,
            "ranges": ranges,
            "fusion_ref": ref_r,
            "stage1_score_included": False,
            "meta": {
                "experiment": "V6.9r-true-jsrt-external",
                "excluded_from_train": "107 JSRT pos + 61 JSRT ctrl (0 JSRT in train)",
                "excluded_from_val_zref": "23 JSRT pos + 18 JSRT ctrl (0 JSRT in val)",
                "trained_on": f"{len(fit_pids)} hasta (683 poz + 683 neg, 1:1)",
                "features": "1600 saf PI (konum YOK, s1 YOK)",
            },
        }
        joblib.dump(bundle_r, BUNDLE_R)
        print(f"  Bundle kaydedildi: {BUNDLE_R.name}", flush=True)

    # ---------------- 4) DIŞ TEST ÖLÇÜMLERİ ---------------- #
    print("\n" + "=" * 78 + "\n[AŞAMA 4] Saf JSRT Dış Test Ölçümleri\n" + "=" * 78, flush=True)
    jp_pids, c_jp, s_jp = unpack_cands_scored(d, "jp", sc, hgb)
    jn_pids, c_jn, s_jn = unpack_cands_scored(d, "jn", sc, hgb)
    va_pids, c_va, s_va = unpack_cands_scored(d, "va", sc, hgb)

    all_eval_pids = set(jp_pids) | set(jn_pids) | set(va_pids)
    gts_all = {p: common.load_gt(p) for p in all_eval_pids}

    # A) Birincil Dış Test: 149 JSRT Pozitif + 93 JSRT Sağlıklı = 242 Saf JSRT
    res_242_pure_jsrt = k.measure(
        jp_pids, c_jp, s_jp, ref_r, gts_all,
        ctrl_pids=jn_pids, ctrl_cands=c_jn, ctrl_scored=s_jn
    )

    # B) Val+Test Saf JSRT Alt Kümesi: 42 JSRT Pozitif + 32 JSRT Sağlıklı = 74 Saf JSRT
    jp_valtest_42 = js_pos["val"] + js_pos["test"]
    jn_valtest_32 = js_neg["val"] + js_neg["test"]
    res_74_valtest_jsrt = k.measure(
        jp_valtest_42, c_jp, s_jp, ref_r, gts_all,
        ctrl_pids=jn_valtest_32, ctrl_cands=c_jn, ctrl_scored=s_jn
    )

    # C) Test-Only Saf JSRT Alt Kümesi: 19 JSRT Pozitif + 14 JSRT Sağlıklı = 33 Saf JSRT
    res_33_test_jsrt = k.measure(
        js_pos["test"], c_jp, s_jp, ref_r, gts_all,
        ctrl_pids=js_neg["test"], ctrl_cands=c_jn, ctrl_scored=s_jn
    )

    # D) Kıyas için: 149 JSRT Pozitif + 250 JSRT-dışı Val Kontrol (V6.9k4 protokolüyle birebir)
    rng = np.random.RandomState(SEED)
    va_ctrl_clean = [p for p in va_pids if not gts_all[p]]
    ctrl_250_nonjsrt = list(rng.permutation(va_ctrl_clean)[:250])
    res_149_vs_250_nonjsrt = k.measure(
        jp_pids, c_jp, s_jp, ref_r, gts_all,
        ctrl_pids=ctrl_250_nonjsrt, ctrl_cands=c_va, ctrl_scored=s_va
    )

    # E) Salt-okunur ANA MODEL (M5_foundation + stage2_foundation_tda_hgb_pure1600) aynı 242 ve 74 Saf JSRT üzerinde
    print("\n[AŞAMA 5] Salt-Okunur Ana Model (M5_foundation + pure1600) Kıyas Ölçümü (242 & 74 Saf JSRT)...", flush=True)
    main_b_path = CKPT / "stage2_foundation_tda_hgb_pure1600.joblib"
    main_b = joblib.load(main_b_path)
    if "m_jp_tda" in d and "m_jn_tda" in d and not args.refresh_cache:
        _, cm_jp, sm_jp = unpack_cands_scored(d, "m_jp", main_b["scaler"], main_b["hgb"])
        _, cm_jn, sm_jn = unpack_cands_scored(d, "m_jn", main_b["scaler"], main_b["hgb"])
    else:
        main_model = load_foundation("M5_foundation", device)
        cm_jp_raw = extract_candidates(main_model, jp_pids, device, "main_jsrt_pos_149")
        cm_jn_raw = extract_candidates(main_model, jn_pids, device, "main_jsrt_neg_93")
        fm_jp = extract_pure1600(jp_pids, cm_jp_raw, main_b["ranges"], "main_jsrt_pos_149")
        fm_jn = extract_pure1600(jn_pids, cm_jn_raw, main_b["ranges"], "main_jsrt_neg_93")
        for tag, pids, cc, ff in (("m_jp", jp_pids, cm_jp_raw, fm_jp), ("m_jn", jn_pids, cm_jn_raw, fm_jn)):
            pk = pack_cohort(pids, cc, ff)
            for kk, vv in pk.items():
                d[f"{tag}_{kk}"] = vv
        np.savez_compressed(CACHE_R, **d)
        _, cm_jp, sm_jp = unpack_cands_scored(d, "m_jp", main_b["scaler"], main_b["hgb"])
        _, cm_jn, sm_jn = unpack_cands_scored(d, "m_jn", main_b["scaler"], main_b["hgb"])

    main_242_pure_jsrt = k.measure(
        jp_pids, cm_jp, sm_jp, main_b["fusion_ref"], gts_all,
        ctrl_pids=jn_pids, ctrl_cands=cm_jn, ctrl_scored=sm_jn
    )
    main_74_valtest_jsrt = k.measure(
        jp_valtest_42, cm_jp, sm_jp, main_b["fusion_ref"], gts_all,
        ctrl_pids=jn_valtest_32, ctrl_cands=cm_jn, ctrl_scored=sm_jn
    )
    main_33_test_jsrt = k.measure(
        js_pos["test"], cm_jp, sm_jp, main_b["fusion_ref"], gts_all,
        ctrl_pids=js_neg["test"], ctrl_cands=cm_jn, ctrl_scored=sm_jn
    )

    after = guard_snapshot()
    isolation_ok = before == after
    assert isolation_ok, f"İHLAL: Salt-okunur dosyalar değişti! {before} != {after}"

    rep = {
        "experiment": "V6.9r-true-jsrt-external",
        "discovery": {
            "summary": (
                "dataset_node21/cxr_images/original_data/images/*.mha başlık taramasıyla "
                "DimSize='2048 2048', ElementSpacing='0.175 0.175', ElementType='MET_USHORT' "
                "imzasına sahip 149 JSRT nodüllü ve 93 JSRT sağlıklı kontrol görüntüsü "
                "(toplam 242 saf JSRT röntgeni) eksiksiz tespit edildi."
            ),
            "jsrt_positive_counts": {s: len(v) for s, v in js_pos.items()},
            "jsrt_positive_total": len(jp_pids),
            "jsrt_control_counts": {s: len(v) for s, v in js_neg.items()},
            "jsrt_control_total": len(jn_pids),
            "jsrt_control_ids": jn_pids,
            "pure_jsrt_total": len(jp_pids) + len(jn_pids),
        },
        "design": {
            "stage1_checkpoint": str(S1_R.relative_to(PROJECT_ROOT)),
            "stage2_bundle": str(BUNDLE_R.relative_to(PROJECT_ROOT)),
            "train_pos_non_jsrt": len(tr_pos),
            "train_neg_non_jsrt": len(tr_neg),
            "train_total": len(fit_pids),
            "jsrt_in_train_pos": 0,
            "jsrt_in_train_neg": 0,
            "val_zref_non_jsrt": len(va_clean),
            "jsrt_in_val_zref": 0,
            "stage2_features": "1600 saf TDA PI (konum YOK, s1 YOK)",
            "w_fusion": W,
            "box_px": int(2 * BOX_HALF),
            "z_ref": ref_r,
        },
        "isolation": {
            "readonly_before": before,
            "readonly_after": after,
            "unchanged": isolation_ok,
            "s1_r_sha256": sha256(S1_R),
            "bundle_r_sha256": sha256(BUNDLE_R),
        },
        "results": {
            "pure_jsrt_242_external": res_242_pure_jsrt,
            "pure_jsrt_74_valtest": res_74_valtest_jsrt,
            "pure_jsrt_33_test_only": res_33_test_jsrt,
            "jsrt_149_vs_250_nonjsrt_val_ctrl": res_149_vs_250_nonjsrt,
        },
        "results_main_readonly": {
            "pure_jsrt_242_in_fold_mixed": main_242_pure_jsrt,
            "pure_jsrt_74_valtest": main_74_valtest_jsrt,
            "pure_jsrt_33_test_only": main_33_test_jsrt,
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    OUT_R.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print("ÖZET TABLO — V6.9r TAM İZOLE SAF JSRT DIŞ TESTİ", flush=True)
    print("=" * 78, flush=True)
    for label, m in [
        ("1) ANA DIŞ TEST (V6.9r LOSO): 242 Saf JSRT (149 Nodüllü + 93 Sağlıklı, %100 Sıfır-Eğitim)", res_242_pure_jsrt),
        ("2) ALT KÜME (V6.9r LOSO, val+test): 74 Saf JSRT (42 Nodüllü + 32 Sağlıklı)", res_74_valtest_jsrt),
        ("3) ALT KÜME (V6.9r LOSO, yalnız test): 33 Saf JSRT (19 Nodüllü + 14 Sağlıklı)", res_33_test_jsrt),
        ("4) KIYAS (V6.9k4 protokolü): 149 JSRT Nodüllü + 250 JSRT-dışı Val Kontrol", res_149_vs_250_nonjsrt),
        ("5) ANA MODEL (Salt-Okunur, val+test): 74 Saf JSRT (42 Nodüllü + 32 Sağlıklı)", main_74_valtest_jsrt),
        ("6) ANA MODEL (Salt-Okunur, tüm 242 — 107 poz + 61 neg eğitimdeydi): 242 Saf JSRT", main_242_pure_jsrt),
    ]:
        print(f"\n>>> {label}", flush=True)
        print(f"  Recall@K (Kriter-1) : K10={m['recall_at_K']['K10']:.4f} | "
              f"K20={m['recall_at_K']['K20']:.4f} | K50={m['recall_at_K']['K50']:.4f} | "
              f"Hasta Hit@50={m['patient_hit_top50']:.4f}", flush=True)
        print(f"  JSRT-içi Aday AUC   : Stage1={m['candidate_auc']['stage1']:.4f} | "
              f"TDA={m['candidate_auc']['tda']:.4f} | Fusion={m['candidate_auc']['fusion']:.4f}", flush=True)
        for kind in ("stage1", "tda", "fusion"):
            auc_d = m["mixed_patient_auc"][kind]
            off_d = m["official_jsrt_plus_controls"][kind]
            sens = off_d["sens"]
            print(f"  [{kind.upper():6s}] Hasta AUC (Top-1)={auc_d['top1']:.4f} | "
                  f"Hasta AUC (Noisy-OR)={auc_d['noisyor']:.4f} | "
                  f"Resmî AUC={off_d['auc']:.4f} | "
                  f"S@0.125={sens['0.125']:.4f} | S@0.25={sens['0.25']:.4f} | "
                  f"S@0.5={sens['0.5']:.4f} | rank_metric={off_d['rank_metric']:.4f}", flush=True)
    print(f"\n🔒 İzolasyon: Salt-okunur ana ağırlıklar DEĞİŞMEDİ ✔ | Kayıt: {OUT_R.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
