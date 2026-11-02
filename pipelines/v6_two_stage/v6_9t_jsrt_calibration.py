"""V6.9t — 242 Saf JSRT Saha Kalibrasyonu (5-fold stratified CV, sıfır-eğitim → saha kalibrasyonu simetrisi).

NEDEN: V6.9j2'de düşük kaliteli harici veri (VinDr-CXR, 8-bit JPEG) için hem sıfır-eğitim
(N=0) hem çift-katmanlı saha kalibrasyonu ölçüldü (AUC 0,638 → 0,794). Yüksek kaliteli harici
veride (242 Saf JSRT, 12-bit .mha) ise yalnız sıfır-eğitim (N=0) ölçülmüştü. Bu deney simetriyi
tamamlar: aynı hastane-içi etiketleme bütçesiyle JSRT'ye adapte olunca ne oluyor?

PROTOKOL — V6.9j2 ile BİREBİR aynı (tek fark: kohort + model):
  * Model:  M5_nojsrt_full.pt (JSRT'yi hiç görmemiş 1. aşama) + stage2_nojsrt_full_tda_hgb_pure1600
  * 5-fold STRATIFIED CV: test katmanına asla dokunulmaz; her fold'da model deepcopy ile SIFIRLANIR
    (ağırlıklar dondurulmuş başlangıçtan yüklenir ⇒ fold'lar arası sızıntı yok).
  * N ∈ {0,10,20,40,80,120,160} etiketli görüntü (yarısı pozitif, yarısı sağlıklı).
  * Kol 1 — yalnız 2. katman: saf 1.600 TDA üzerinde LR (görüntü etiketi) + sıfır-eğitim
            öncülüyle shrinkage  p = α·p_LR + (1−α)·p_tda0,  α = N/(N+40).
  * Kol 2 — yalnız 1. katman: omurga DONUK, yalnız neck+head ince ayarı (Gaussian focal +
            sıfır-eğitim öncül çapası); aday KUTULARI sabit (V6.9j2 tasarım kararı) ⇒ yalnız
            sabit merkezlerde yeniden puanlama. Ek olarak 1. katman TEK BAŞINA ölçülür.
  * Kol 3 — çift katmanlı: FT edilmiş 1. katman skoru + kalibre edilmiş TDA (w_tda = 0,45).
  * Füzyon SABİT ve hiçbir veride yeniden öğrenilmez: S = σ(0,55·z(s1) + 0,45·z(p_tda));
    z referansı JSRT-dışı NODE21 val'den dondurulmuş (bundle fusion_ref).

HIZ (V6.9j2'nin önbellek hilesi):
  * 1.600 TDA özellikleri `_v6_9r_cache.npz` içinde HAZIR (GUDHI tekrar çalışmaz).
  * Donuk omurga çıktıları (c2..c5) float16 CPU RAM'de bir kez önbelleğe alınır ⇒ FT ve
    yeniden puanlama omurgayı geçmez, yalnız neck+head çalışır (~10×).
  * Flip geçişi (TTA) önbellekte tutulmaz (RAM sınırı), çalışma anında hesaplanır.

ÇIKTI: artifacts/v6_two_stage/v6_9t_jsrt_calibration.json
Kullanım:
    python pipelines/v6_two_stage/v6_9t_jsrt_calibration.py            # tam koşu
    python pipelines/v6_two_stage/v6_9t_jsrt_calibration.py --smoke    # 2 fold, N={0,10,40}
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import joblib  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import common  # noqa: E402
import v6_6_train_foundation as tr  # noqa: E402
from detector_models import build_targets_heat, center_focal_loss  # noqa: E402
from detector_models_foundation import foundation_input  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import official_metric  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
TAG_R = "M5_nojsrt_full"
CACHE_R = ART / "_v6_9r_cache.npz"
BUNDLE_R = CKPT / "stage2_nojsrt_full_tda_hgb_pure1600.joblib"
OUT = ART / "v6_9t_jsrt_calibration.json"

SEED = 42
N_LIST = (0, 10, 20, 40, 80, 120, 160)
N_FOLDS = 5
BOX_HALF = 28.0            # 56×56 px kutu (1024 uzayı)
HMP = 256                  # ısı haritası ızgarası
FT_EPOCHS = 5              # V6.9j2 ile aynı
FT_LR = 1e-4
FT_ANCHOR = 1e-3           # sıfır-eğitim öncül çapası
FT_BATCH = 4
SHRINK_C = 40.0            # 2. katman shrinkage: α = N/(N+C)
ARMS = ("kol1_stage2", "kol2_stage1", "kol3_both", "stage1_only")


# --------------------------------------------------------------------------- #
# yardımcılar
# --------------------------------------------------------------------------- #
def _fuse(s1: np.ndarray, p_tda: np.ndarray, ref: dict) -> np.ndarray:
    z1 = (s1 - ref["s1_mean"]) / (ref["s1_std"] + 1e-9)
    zt = (p_tda - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
    return 1.0 / (1.0 + np.exp(-(ref["w"] * z1 + (1.0 - ref["w"]) * zt)))


def _noisy_or(scores) -> float:
    """Hasta düzeyi noisy-OR: en iyi 5 kutunun birleşik olasılığı (V6.9j2/V6.9k ile aynı)."""
    ss = np.sort(np.asarray(scores, float))[::-1][:5]
    if len(ss) == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - np.clip(ss, 0, 1)))


def _to_grid(cx1024: float, cy1024: float) -> tuple[int, int]:
    """1024-uzayı merkez → 256×256 ısı ızgarası (pick_peaks'in tersi: c = 4·grid + 2)."""
    gx = int(round((float(cx1024) - 2.0) / 4.0))
    gy = int(round((float(cy1024) - 2.0) / 4.0))
    return min(max(gx, 0), HMP - 1), min(max(gy, 0), HMP - 1)


def load_jsrt_data(z, scaler, hgb) -> list[dict]:
    """jp (149 nodüllü) + jn (93 sağlıklı) → tek 242'lik veri listesi.

    Aday kutuları ve 1.600 TDA özellikleri V6.9r önbelleğinden gelir (GUDHI çalışmaz).
    """
    data: list[dict] = []
    for tag, y in (("jp", 1), ("jn", 0)):
        pids = [str(p) for p in z[f"{tag}_pids"]]
        off = z[f"{tag}_off"]
        tda = z[f"{tag}_tda"].astype(np.float32)
        s1 = z[f"{tag}_s1"].astype(np.float32)
        cx = z[f"{tag}_cx"].astype(np.float32)
        cy = z[f"{tag}_cy"].astype(np.float32)
        Xs = scaler.transform(tda)
        p0 = hgb.predict_proba(Xs)[:, 1].astype(np.float32)
        for i, p in enumerate(pids):
            lo, hi = int(off[i]), int(off[i + 1])
            gts = common.load_gt(p)
            data.append(dict(
                pid=p, y=y, n=hi - lo, gts=gts,
                cands=[(float(cx[j]), float(cy[j]), float(s1[j])) for j in range(lo, hi)],
                Xs=Xs[lo:hi], p_tda0=p0[lo:hi],
                s1_3scale=s1[lo:hi].copy(),
            ))
    return data


@torch.no_grad()
def build_bb_cache(model, data, cache512, device) -> dict:
    """Donuk omurga çıktıları (c2..c5, float16 CPU) — FT ve re-score bunları kullanır."""
    bb, t0 = {}, time.time()
    for i, d in enumerate(data):
        x = foundation_input(cache512[d["pid"]][0], False).unsqueeze(0).to(device)
        bb[d["pid"]] = {"f": [c.detach().cpu().half() for c in model.backbone_features(x)]}
        if (i + 1) % 60 == 0:
            print(f"    omurga önbelleği {i + 1}/{len(data)} ({time.time() - t0:.0f}s)", flush=True)
    return bb


@torch.no_grad()
def _heat_from_feats(model, feats, device) -> np.ndarray:
    c2, c3, c4, c5 = [c.to(device).float() for c in feats]
    p2, _, _, _ = model.fpn(c2, c3, c4, c5)
    f = model.fuse(p2)
    f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.sigmoid(model.heat(f))[0, 0].detach().cpu().numpy()


@torch.no_grad()
def rescore(model, data, bb, cache512, device) -> dict:
    """Sabit aday merkezlerinde 1,0 ölçek + flip TTA ısı haritasından yeniden puanlama."""
    out, t0 = {}, time.time()
    for i, d in enumerate(data):
        pid = d["pid"]
        h1 = _heat_from_feats(model, bb[pid]["f"], device)
        xf = foundation_input(cache512[pid][0], True).unsqueeze(0).to(device)
        feats_f = [c.detach().cpu().half() for c in model.backbone_features(xf)]
        h2 = _heat_from_feats(model, feats_f, device)[:, ::-1]
        heat = (h1 + h2) / 2.0
        s = np.zeros(len(d["cands"]), np.float32)
        for k, (cx, cy, _s) in enumerate(d["cands"]):
            gx, gy = _to_grid(cx, cy)
            s[k] = heat[gy, gx]
        out[pid] = s
        if (i + 1) % 60 == 0:
            print(f"      re-score {i + 1}/{len(data)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def finetune_head(base_model, data, idxs, cache512, bb, device):
    """Omurga DONUK + önbellekten; yalnız neck+head N görüntüyle fine-tune (L2 çapa).

    V6.9j2 hiperparametreleri birebir; tek fark: omurga özellikleri GPU'ya BATCH BAŞINA taşınır
    (N=160'ta tamamını GPU'ya almak 5 GB VRAM istiyordu → 6 GB kartta kırılgan).
    """
    m = copy.deepcopy(base_model)
    m.freeze_backbone()
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=FT_LR, weight_decay=1e-5)
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
    rng = np.random.RandomState(SEED)
    for ep in range(FT_EPOCHS):
        order = rng.permutation(len(idxs))
        tot, nb = 0.0, 0
        for s in range(0, len(order), FT_BATCH):
            sel = [idxs[i] for i in order[s:s + FT_BATCH]]
            Ys, feats = [], []
            for i in sel:
                d = data[i]
                _c512, cen, rad = cache512[d["pid"]]
                Y, _ = build_targets_heat(cen, rad, device)
                Ys.append(Y)
                c2, c3, c4, c5 = [c.to(device).float() for c in bb[d["pid"]]["f"]]
                p2, _, _, _ = m.fpn(c2, c3, c4, c5)
                f = m.fuse(p2)
                feats.append(F.interpolate(f, scale_factor=2, mode="bilinear",
                                           align_corners=False))
            Y = torch.stack(Ys).unsqueeze(1)
            h = m.heat(torch.cat(feats, 0))
            focal = center_focal_loss(h, Y)
            anchor = sum(((p - init[n]).pow(2)).sum() for n, p in m.named_parameters()
                         if p.requires_grad)
            opt.zero_grad()
            (focal + FT_ANCHOR * anchor).backward()
            opt.step()
            tot += float(focal)
            nb += 1
        if ep == 0 or ep == FT_EPOCHS - 1:
            print(f"      FT ep{ep + 1}/{FT_EPOCHS} focal {tot / max(1, nb):.4f}", flush=True)
    m.eval()
    return m


def evaluate(data, idxs, s1_map, p_tda_map, ref) -> tuple[dict, list, np.ndarray, np.ndarray]:
    """Test-fold (veya tüm kohort) için resmî metrik + noisy-OR hasta AUC'si."""
    gts = {data[i]["pid"]: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
                            for g in data[i]["gts"]] for i in idxs}
    is_nod = {data[i]["pid"]: bool(data[i]["gts"]) for i in idxs}
    pids = [data[i]["pid"] for i in idxs]
    n_les = sum(len(gts[p]) for p in pids)
    flat = []
    for i in idxs:
        d = data[i]
        pid = d["pid"]
        pf = _fuse(np.asarray(s1_map[pid]), np.asarray(p_tda_map[pid]), ref)
        for k, (cx, cy, _s) in enumerate(d["cands"]):
            flat.append((float(pf[k]), pid, float(cx), float(cy), 0.0))
    om = official_metric(flat, len(idxs), n_les, gts, is_nod, pids, box_half=BOX_HALF)
    yv = np.array([data[i]["y"] for i in idxs])
    sc = np.array([_noisy_or(_fuse(np.asarray(s1_map[data[i]["pid"]]),
                                   np.asarray(p_tda_map[data[i]["pid"]]), ref))
                   for i in idxs])
    aur = float(roc_auc_score(yv, sc)) if len(set(yv.tolist())) > 1 else float("nan")
    return (dict(auc=om["auc"], s125=om["sens"]["0.125"], s25=om["sens"]["0.25"],
                 s50=om["sens"]["0.5"], rank=om["rank_metric"], auc_nor=aur),
            flat, yv, sc)


def _stat(vals) -> tuple[float, float]:
    a = np.array(vals, float)
    a = a[np.isfinite(a)]
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    sm = args.smoke
    n_list = (0, 10, 40) if sm else N_LIST
    n_folds = 2 if sm else N_FOLDS

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78, flush=True)
    print("V6.9t — 242 SAF JSRT SAHA KALİBRASYONU (5-fold stratified CV, sızıntısız)", flush=True)
    print(f"  device={device} | model={TAG_R} | N={list(n_list)} | folds={n_folds} | smoke={sm}",
          flush=True)
    print("=" * 78, flush=True)

    bundle = joblib.load(BUNDLE_R)
    scaler, hgb, ref = bundle["scaler"], bundle["hgb"], bundle["fusion_ref"]
    print(f"  2. aşama: {BUNDLE_R.name} (saf 1.600, stage1_score_included="
          f"{bundle['stage1_score_included']}, w_s1={ref['w']})", flush=True)

    z = np.load(CACHE_R, allow_pickle=True, mmap_mode="r")
    data = load_jsrt_data(z, scaler, hgb)
    n_pos = sum(d["y"] for d in data)
    print(f"  kohort: {len(data)} hasta ({n_pos} nodüllü + {len(data) - n_pos} sağlıklı), "
          f"{sum(d['n'] for d in data)} aday", flush=True)

    if sm:
        pos = [d for d in data if d["y"] == 1][:40]
        neg = [d for d in data if d["y"] == 0][:24]
        data = pos + neg
        n_pos = sum(d["y"] for d in data)
        print(f"  [smoke] alt küme: {len(data)} hasta ({n_pos} poz + "
              f"{len(data) - n_pos} sağlıklı)", flush=True)

    print("\n[1] Kanonik 512 görüntü önbelleği...", flush=True)
    cache512 = tr.build_cache([d["pid"] for d in data])

    base = load_foundation(TAG_R, device)

    print("\n[2] Donuk omurga özellik önbelleği (c2..c5, float16)...", flush=True)
    bb = build_bb_cache(base, data, cache512, device)
    print(f"    {len(bb)} görüntü", flush=True)

    print("\n[3] Sıfır-eğitim temel çizgisi (donuk model, 1,0 ölçek + flip)...", flush=True)
    s1_base = rescore(base, data, bb, cache512, device)
    # koordinat eşlemesi denetimi: 3-ölçek TTA skoruyla korelasyon yüksek olmalı
    a_all = np.concatenate([d["s1_3scale"] for d in data])
    b_all = np.concatenate([s1_base[d["pid"]] for d in data])
    corr = float(np.corrcoef(a_all, b_all)[0, 1])
    print(f"    koordinat denetimi: korelasyon(3-ölçek TTA, 1,0+flip) = {corr:.3f}", flush=True)
    if corr < 0.5:
        raise SystemExit(f"HATA: ısı haritası eşlemesi şüpheli (korelasyon {corr:.2f})")

    p_tda0 = {d["pid"]: d["p_tda0"] for d in data}

    # ---- sıfır-eğitim referansı: 3 ölçekli TTA skoru (yayımlanmış sayıyı üretmeli)
    print("\n[4] Sıfır-eğitim referansı (3-ölçek TTA s1, yayımlanmış protokol)...", flush=True)
    s1_3s = {d["pid"]: d["s1_3scale"] for d in data}
    ref_om, ref_flat, ref_y, ref_sc = evaluate(data, list(range(len(data))), s1_3s, p_tda0, ref)
    print(f"    N=0 referans: AUC {ref_om['auc']} · nOR {ref_om['auc_nor']:.4f} · "
          f"S@.25 {ref_om['s25']} · rank {ref_om['rank']}", flush=True)

    # ---- 5-fold stratified
    rng = np.random.RandomState(SEED)
    y_img = np.array([d["y"] for d in data])
    folds = np.zeros(len(data), int)
    for lab in (0, 1):
        idx = np.where(y_img == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    print(f"  fold dağılımı: {[int((folds == f).sum()) for f in range(n_folds)]}", flush=True)

    acc = {arm: {N: {"auc": [], "auc_nor": [], "s125": [], "s25": [], "s50": [], "rank": [],
                     "flat": [], "y": [], "sc": []} for N in n_list} for arm in ARMS}

    for N in n_list:
        for f in range(n_folds):
            t0 = time.time()
            tr_idx = [i for i in range(len(data)) if folds[i] != f]
            te_idx = [i for i in range(len(data)) if folds[i] == f]
            pick, n_take = [], (0, 0)
            if N == 0:
                p_tda_map = p_tda0
                s1_ft = s1_base
            else:
                pos = [i for i in tr_idx if y_img[i] == 1]
                neg = [i for i in tr_idx if y_img[i] == 0]
                r2 = np.random.RandomState(SEED + N + f)
                npos = min(N // 2, len(pos))
                nneg = min(N - npos, len(neg))
                cal_pos = list(r2.choice(pos, size=npos, replace=False))
                cal_neg = list(r2.choice(neg, size=nneg, replace=False))
                pick = cal_pos + cal_neg
                n_take = (len(cal_pos), len(cal_neg))
                # --- Kol 1: 2. katman LR (görüntü etiketi, ölçekli 1.600)
                Xc = np.concatenate([data[i]["Xs"] for i in pick])
                yc = np.concatenate([np.full(data[i]["n"], data[i]["y"], np.float32)
                                     for i in pick])
                if len(set(yc.tolist())) < 2:
                    p_tda_map = p_tda0
                else:
                    lr = LogisticRegression(class_weight="balanced", max_iter=500, C=0.1,
                                            random_state=SEED).fit(Xc, yc)
                    alpha = N / (N + SHRINK_C)
                    p_tda_map = {}
                    for d in data:
                        pr = lr.predict_proba(d["Xs"])[:, 1]
                        p_tda_map[d["pid"]] = alpha * pr + (1.0 - alpha) * d["p_tda0"]
                # --- Kol 2: 1. katman neck+head FT + sabit merkez yeniden puanlama
                ft = finetune_head(base, data, pick, cache512, bb, device)
                s1_ft = rescore(ft, data, bb, cache512, device)
                del ft
                torch.cuda.empty_cache()

            zpad = {d["pid"]: np.zeros(d["n"], np.float32) for d in data}
            combos = [
                ("kol1_stage2", s1_base, p_tda_map),
                ("kol2_stage1", s1_ft, p_tda0),
                ("kol3_both", s1_ft, p_tda_map),
                ("stage1_only", s1_ft, zpad),
            ]
            for arm, s1m, ptm in combos:
                om, flat, yv, sc = evaluate(data, te_idx, s1m, ptm, ref)
                a = acc[arm][N]
                for k in ("auc", "auc_nor", "s125", "s25", "s50", "rank"):
                    a[k].append(om[k])
                a["flat"].append(flat)
                a["y"].append(yv)
                a["sc"].append(sc)
            print(f"  N={N:3d} fold {f + 1}/{n_folds} (kalibre {n_take[0]}+{n_take[1]}) "
                  f"{time.time() - t0:.1f}s", flush=True)
        a = acc["kol3_both"][N]
        print(f"  → N={N:3d} kol3 AUC {np.mean(a['auc']):.4f} nOR {np.mean(a['auc_nor']):.4f} "
              f"S@.25 {np.mean(a['s25']):.4f} rank {np.mean(a['rank']):.4f} "
              f"({time.time() - t_start:.0f}s)", flush=True)

    # ---- özet + havuzlanmış (out-of-fold) tahmin
    gts_all = {d["pid"]: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in d["gts"]]
               for d in data}
    is_nod_all = {d["pid"]: bool(d["gts"]) for d in data}
    pids_all = [d["pid"] for d in data]
    n_les_all = sum(len(gts_all[p]) for p in pids_all)

    summary, pooled = {}, {}
    for arm in ARMS:
        summary[arm], pooled[arm] = {}, {}
        for N in n_list:
            a = acc[arm][N]
            m, s = _stat(a["auc"])
            mn, sn = _stat(a["auc_nor"])
            summary[arm][N] = dict(auc_mean=m, auc_std=s, auc_nor_mean=mn, auc_nor_std=sn,
                                   s125=_stat(a["s125"])[0], s125_std=_stat(a["s125"])[1],
                                   s25=_stat(a["s25"])[0], s25_std=_stat(a["s25"])[1],
                                   s50=_stat(a["s50"])[0], s50_std=_stat(a["s50"])[1],
                                   rank=_stat(a["rank"])[0], rank_std=_stat(a["rank"])[1])
            flat = [x for sub in a["flat"] for x in sub]
            po = official_metric(flat, len(data), n_les_all, gts_all, is_nod_all, pids_all,
                                 box_half=BOX_HALF)
            yv = np.concatenate(a["y"])
            sc = np.concatenate(a["sc"])
            pooled[arm][N] = dict(auc=po["auc"], auc_nor=float(roc_auc_score(yv, sc)),
                                  s125=po["sens"]["0.125"], s25=po["sens"]["0.25"],
                                  s50=po["sens"]["0.5"], rank=po["rank_metric"])

    # ---- ekran tablosu
    print("\n" + "=" * 118, flush=True)
    print("V6.9t — 242 SAF JSRT SAHA KALİBRASYONU (5-fold stratified CV, w_tda=0,45 sabit)",
          flush=True)
    print("=" * 118, flush=True)
    hdr = (f"{'N':>4} | {'Kol':12} | {'AUC(top1)':>15} | {'AUC(noisy-OR)':>15} | "
           f"{'S@.125':>6} {'S@.25':>6} {'S@.50':>6} {'rank':>6} | {'OOF AUC':>7} {'OOF rank':>8}")
    print(hdr, flush=True)
    print("-" * 118, flush=True)
    for N in n_list:
        for arm in ARMS:
            sb = summary[arm][N]
            po = pooled[arm][N]
            print(f"{N:4d} | {arm:12} | {sb['auc_mean']:.4f}±{sb['auc_std']:.4f} | "
                  f"{sb['auc_nor_mean']:.4f}±{sb['auc_nor_std']:.4f} | "
                  f"{sb['s125']:.4f} {sb['s25']:.4f} {sb['s50']:.4f} {sb['rank']:.4f} | "
                  f"{po['auc']:.4f}  {po['rank']:.4f}", flush=True)
        print("-" * 118, flush=True)
    print(f"{'ref':>4} | {'N=0 3-TTA':12} | {ref_om['auc']:.4f}{'':9} | "
          f"{ref_om['auc_nor']:.4f}{'':9} | {ref_om['s125']:.4f} {ref_om['s25']:.4f} "
          f"{ref_om['s50']:.4f} {ref_om['rank']:.4f} |", flush=True)

    report = {
        "experiment": "V6.9t-jsrt-site-calibration",
        "smoke": sm,
        "protocol": "V6.9j2 (VinDr) ile birebir: sabit aday havuzu + yeniden puanlama, "
                    "5-fold stratified CV, deepcopy ile sıfırlanan model",
        "cohort": {"n_patients": len(data), "n_positive": int(n_pos),
                   "n_healthy": len(data) - int(n_pos),
                   "n_candidates": int(sum(d["n"] for d in data)),
                   "n_lesions": int(n_les_all), "source": "242 Saf JSRT (149+93)"},
        "model": {"stage1": TAG_R, "stage2": BUNDLE_R.name,
                  "features": "1600 saf PI (V6.9m)", "fusion": ref},
        "N": list(n_list), "folds": n_folds,
        "ft": {"epochs": FT_EPOCHS, "lr": FT_LR, "anchor": FT_ANCHOR, "batch": FT_BATCH,
               "scope": "boyun+CenterNet kafası (omurga donuk ve önbellekten)"},
        "stage2_shrinkage": {"C": SHRINK_C, "alpha": "N/(N+C)"},
        "coord_check_corr": round(corr, 4),
        "ref_zero_shot_3scale": {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in ref_om.items()},
        "summary": {arm: {str(N): summary[arm][N] for N in n_list} for arm in ARMS},
        "pooled_out_of_fold": {arm: {str(N): pooled[arm][N] for N in n_list} for arm in ARMS},
        "elapsed_seconds": round(time.time() - t_start, 1),
        "notes": [
            "S@x = resmî NODE21 IoU>0,2 eşleşmesi, 56×56 sabit kutu; hasta AUC Top-1 = en iyi tek kutu.",
            "noisy-OR = 1-ÇARP(1-p) en iyi 5 kutu (V6.9j2/V6.9k ile aynı tanım).",
            "stage1_only kolu = p_tda sabit (yalnız 1. aşama sıralaması) — 2. katman katkısını izole eder.",
            "Kol 2/3'te aday kutuları DONUK 1. aşamadan sabittir (V6.9j2 tasarım kararı); "
            "güncellenen ısı haritası yalnız sabit merkezlerdeki skoru değiştirir, yeni kutu üretmez.",
            "N=0 referansı 3 ölçekli TTA (yayımlanmış protokol) ile verilir; 1,0+flip taban çizgisi "
            "kol karşılaştırmalarında kullanılır.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSüre: {time.time() - t_start:.0f} sn · Rapor: {args.out}", flush=True)


if __name__ == "__main__":
    main()
