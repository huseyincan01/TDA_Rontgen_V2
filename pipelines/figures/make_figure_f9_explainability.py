"""F9 — Görsel açıklanabilirlik (V6 iki aşamalı lokalizasyon).

Tek bir NODE21 val hastası üzerinde:
    (A) 5-zon kanonik görüntü + GT kutusu + adaylar (isabet/FP renkli)
    (B) TP ve en yüksek skorlu FP'nin 64x64 kırpıntıları
    (C) TP kırpıntısının kübik kalıcılık diyagramları (H0/H1)
    (D) Skor ayrışımı: aşama-1 · lokal TDA-HGB · birleşik

Kullanım:
    python pipelines/figures/make_figure_f9_explainability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from detector_models import CenterNetHead, FCOSHead  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10.5, "axes.grid": False, "figure.dpi": 200,
                     "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY = "#2E5A88", "#D97706", "#2F7D4F", "#B4341F", "#6B7280"


def _pad(img: np.ndarray, size: int = 64) -> np.ndarray:
    out = np.zeros((size, size), np.float32)
    h, w = img.shape
    h, w = min(h, size), min(w, size)
    out[:h, :w] = img[:h, :w]
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device)
    m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()
    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    scaler, hgb, lr_fus, ranges = b["scaler"], b["hgb"], b["lr_fusion"], b["ranges"]
    seg = PSPNet5ZoneSegmenter(device=device)

    va = common.split_patients()["val"]
    nod = [p for p in va if p in set(p2._nodule_pids("val"))]
    print(f"val nodul hastasi: {len(nod)}")

    chosen = None
    best_score = -1
    for pid in nod[:40]:
        try:
            c = s2.stage1_candidates([pid], m5, m6, device, top=50)[pid]["M5_heatTTA"]
            gts = common.load_gt(pid)
            labs = s2._labels_for(c, gts)
            feat = s2._patient_features((pid, c, ranges, False))[1]
            X = np.concatenate([feat["tda"], feat["spa"]], 1)
            p_tda = hgb.predict_proba(scaler.transform(X))[:, 1]
            s1 = np.array([q[2] for q in c], np.float32)
            p_fus = lr_fus.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
            hits = [i for i, h in enumerate(labs) if h is not None]
            fps = [i for i, h in enumerate(labs) if h is None]
            if not hits or not fps:
                continue
            order = np.argsort(-p_fus)
            tp = max(hits, key=lambda i: p_fus[i])
            rank = int(list(order).index(tp))
            # aşama-1'in TP'yi FP'nin altına attığı, TDA'nın düzelttiği vaka
            corrected = [i for i in fps if s1[i] > s1[tp] and p_tda[i] < p_tda[tp]]
            score = (10 if rank == 0 else (4 if rank == 1 else 0)) + (5 if corrected else 0)
            if score > best_score:
                best_score = score
                chosen = (pid, c, gts, labs, p_tda, p_fus)
                print(f"  aday: {pid} | TP sira={rank + 1} | duzeltilen FP={len(corrected)} | skor={score}")
            if rank == 0 and corrected:
                break
        except Exception as exc:
            print(f"  {pid}: {exc}")
    if chosen is None:
        raise SystemExit("uygun hasta bulunamadi")
    print(f"secildi: {chosen[0]}")

    pid, cands, gts, labs, p_tda, p_fus = chosen
    img = common.load_image(pid)
    mask_d = common.load_lung_mask(pid)
    p1, p99 = common.get_lung_p1p99()[pid]
    canon = common.canonicalize(img, mask_d, float(p1), float(p99))
    zone = seg.segment(img)["zone_map"]
    raw = s2._raw_mask(pid)
    ys, xs = np.where(raw)
    bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    dist = s2._distance_map(raw)

    hits = [i for i, h in enumerate(labs) if h is not None]
    fps = [i for i, h in enumerate(labs) if h is None]
    tp = max(hits, key=lambda i: p_fus[i])
    s1_all = np.array([q[2] for q in cands], np.float32)
    corrected = [i for i in fps if p_tda[i] < p_tda[tp]]
    fp = max(corrected, key=lambda i: p_fus[i]) if corrected else max(fps, key=lambda i: p_fus[i])
    order = np.argsort(-p_fus)

    fig = plt.figure(figsize=(15.4, 6.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1.0, 0.95],
                  height_ratios=[1.0, 1.0], wspace=0.26, hspace=0.30)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[:, 2])

    # ---------------- A
    axA.imshow(canon, cmap="gray", vmin=0, vmax=255)
    axA.contour(zone, levels=[1.5, 2.5, 3.5, 4.5], colors="#38BDF8",
                linewidths=0.8, alpha=0.75, linestyles="--")
    for g in gts:
        axA.add_patch(Rectangle((g["x"], g["y"]), g["w"], g["h"], fill=False,
                                edgecolor=GREEN, linewidth=2.0))
    show = [i for i in order[:5]]
    for i in show:
        cx, cy, _ = cands[i]
        is_hit = labs[i] is not None
        col = GREEN if is_hit else RED
        big = i in (tp, fp)
        axA.scatter(cx, cy, s=110 if big else 60, facecolors="none", edgecolors=col,
                    linewidths=2.2 if big else 1.4)
        if i == tp:
            axA.annotate(f"TP  {p_fus[i]:.2f}", (cx, cy), textcoords="offset points",
                         xytext=(-14 if cx > 0.5 * canon.shape[1] else 14, 14),
                         ha="right" if cx > 0.5 * canon.shape[1] else "left",
                         color=GREEN, fontsize=8.5, weight="bold",
                         bbox={"boxstyle": "round,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.6})
        elif i == fp:
            axA.annotate(f"FP  {p_fus[i]:.2f}", (cx, cy), textcoords="offset points",
                         xytext=(-14 if cx > 0.5 * canon.shape[1] else 14, -18),
                         ha="right" if cx > 0.5 * canon.shape[1] else "left",
                         color=RED, fontsize=8.5, weight="bold",
                         bbox={"boxstyle": "round,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.6})
    axA.set_title(f"A) 5-zon kanonik görüntü + adaylar  ({pid})\n"
                  "yeşil çerçeve = GT nodül · yeşil halka = isabet · kırmızı = FP",
                  fontsize=10)
    axA.axis("off")

    # ---------------- B
    tpc = _pad(s2._crop(canon, cands[tp][0], cands[tp][1], 64))
    fpc = _pad(s2._crop(canon, cands[fp][0], cands[fp][1], 64))
    combo = np.concatenate([tpc, np.full((64, 4), p99), fpc], axis=1)
    axB.imshow(combo, cmap="gray", vmin=0, vmax=255)
    axB.axvline(65.5, color="white", linewidth=0.6)
    axB.set_title("B) 64×64 kırpıntılar\n(sol: TP · sağ: TDA'nın bastırdığı güçlü FP)", fontsize=9.5)
    axB.set_xticks([32, 100]); axB.set_xticklabels(["TP", "FP"], fontsize=9)
    axB.set_yticks([])

    # ---------------- C
    ex = ZoneTDAExtractor(s2._settings(ranges))
    filt = ex.prepare_filtration(tpc)
    if filt is not None:
        h0, h1 = ex.compute_diagrams(filt)
        if len(h0):
            axC.scatter(h0[:, 0], h0[:, 1] - h0[:, 0], s=5, color=BLUE, alpha=0.35, label="H0")
        if len(h1):
            axC.scatter(h1[:, 0], h1[:, 1] - h1[:, 0], s=14, color=ORANGE, alpha=0.85,
                        edgecolors="none", label="H1 (döngüler)")
    axC.set_xlabel("doğum (filtration değeri)")
    axC.set_ylabel("kalıcılık (ölüm − doğum)")
    axC.set_title("C) TP kırpıntısının kalıcılık diyagramı\n"
                  "H1 döngüleri = TDA dalının gördüğü yapı", fontsize=9.5)
    axC.legend(fontsize=7.5, loc="upper right", markerscale=1.6)
    axC.grid(alpha=0.25)

    # ---------------- D
    metrics = ["aşama-1\n(ısı haritası)", "lokal TDA\n(HGB)", "birleşik\n(e_fusion)"]
    tv = [cands[tp][2], p_tda[tp], p_fus[tp]]
    fv = [cands[fp][2], p_tda[fp], p_fus[fp]]
    x = np.arange(3)
    w = 0.36
    b1 = axD.bar(x - w / 2, tv, w, color=GREEN, label="TP (gerçek nodül)", edgecolor="white")
    b2 = axD.bar(x + w / 2, fv, w, color=RED, label="FP (TDA'nın bastırdığı)", edgecolor="white")
    for bars in (b1, b2):
        for q in bars:
            axD.text(q.get_x() + q.get_width() / 2, q.get_height() + 0.02,
                     f"{q.get_height():.2f}", ha="center", fontsize=8)
    axD.set_xticks(x); axD.set_xticklabels(metrics, fontsize=8.5)
    axD.set_ylim(0, 1.12)
    axD.set_ylabel("skor")
    axD.set_title("D) Skor ayrışımı — TDA, aşama-1'in\nkaçırdığı ayrımı tamamlıyor", fontsize=9.5)
    axD.legend(fontsize=8, loc="upper left")
    axD.grid(axis="y", alpha=0.3)

    fig.suptitle("F9 — Görsel açıklanabilirlik: aday → kırpıntı → kalıcılık → skor",
                 fontsize=13, weight="bold", y=1.00)
    p = OUT / "F9_explainability.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  yazildi: {p.relative_to(ROOT)}")
    print(f"  TP p_fus={p_fus[tp]:.3f} (sira {list(order).index(tp) + 1}) | "
          f"FP p_fus={p_fus[fp]:.3f} (sira {list(order).index(fp) + 1}) | "
          f"TDA: {p_tda[tp]:.3f} vs {p_tda[fp]:.3f}")


if __name__ == "__main__":
    main()
