"""V6.9p — NİTELİKSEL DOĞRULUK FİGÜRÜ: gerçek test hastalarında TP / FP / kaçırma.

Ne üretir
---------
Kilitli test bölmesinden seçilen 4 hastanın röntgeni üzerinde:
  * **yeşil kalın**  : gerçek tümör kutusu (GT)
  * **gri noktalı**  : 1. aşamanın 50 adayı (top-50)
  * **kırmızı**      : nihai (füzyon) sıralamada FP
  * **yeşil-mavi**   : nihai sıralamada TP (IoU > 0,20)
  * **turuncu kesik** : aynı GT'ye ikinci kez vuran tahmin → ne TP ne FP (yok sayılır)
Her hasta için iki panel: (sol) 1. aşama sıralaması, (sağ) füzyon sıralaması — 2. aşamanın
sıralamayı nasıl değiştirdiği görsel olarak okunur.

Kullanım: python pipelines/v6_two_stage/v6_9p_figure_tp_fp.py
Çıktı   : artifacts/v6_two_stage/v6_9p_tp_fp_figure.png + .json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_9k_leave_jsrt_out as k  # noqa: E402
from v6_6_optimization_audit import iou  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CACHE = ART / "_v6_9m_test_cache.npz"
BUNDLE = ART / "checkpoints" / "stage2_foundation_tda_hgb_pure1600.joblib"
PNG = ART / "v6_9p_tp_fp_figure.png"
JSON = ART / "v6_9p_tp_fp_figure.json"
BOX_HALF = 28.0
TOPK_SHOW = 5


def classify(cands, gts, scores):
    """Resmî kural: skor sırasıyla, IoU>0,2 ve GT ilk kez → TP; aynı GT'ye 2. → ignore."""
    order = np.argsort(-scores)
    matched, out = set(), {}
    for i in order:
        cx, cy = cands[i]
        box = (cx - BOX_HALF, cy - BOX_HALF, cx + BOX_HALF, cy + BOX_HALF)
        best, hit = 0.0, None
        for gi, g in enumerate(gts):
            v = iou(box, g)
            if v > best:
                best, hit = v, gi
        if hit is not None and best > 0.2:
            if hit in matched:
                out[i] = ("ignore", best, hit)
                continue
            matched.add(hit)
            out[i] = ("TP", best, hit)
        else:
            out[i] = ("FP", best, hit)
    return out


def load_patient(pid, z, bundle, imap=None):
    off = z["te_off"]
    if imap is None:
        imap = {str(p): i for i, p in enumerate(z["te_pids"])}
    idx = imap[pid]
    lo, hi = int(off[idx]), int(off[idx + 1])
    cx, cy = z["te_cx"][lo:hi], z["te_cy"][lo:hi]
    s1 = z["te_s1"][lo:hi].astype(np.float32)
    tda = z["te_tda"][lo:hi].astype(np.float32)
    p_tda = bundle["hgb"].predict_proba(bundle["scaler"].transform(tda))[:, 1]
    fus = k.fuse(s1, p_tda, bundle["fusion_ref"], bundle["fusion_ref"]["w"])
    gts = [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in common.load_gt(pid)]
    return list(zip(cx, cy)), s1, p_tda, fus, gts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    bundle = joblib.load(BUNDLE)
    z = np.load(CACHE, allow_pickle=True)
    pids = [str(p) for p in z["te_pids"]]
    print(f"test önbelleği: {len(pids)} hasta", flush=True)

    # ---- adayları tara: dört temsilci vaka bul
    rows = []
    for pid in pids:
        cands, s1, p_tda, fus, gts = load_patient(pid, z, bundle)
        if not cands:
            continue
        c_fus = classify(cands, gts, fus)
        c_s1 = classify(cands, gts, s1)
        top_f = [i for i in np.argsort(-fus)[:TOPK_SHOW]]
        best_iou_f = max((c_fus[i][1] for i in top_f), default=0.0)
        any_iou = max((v[1] for v in c_fus.values()), default=0.0)
        n_tp = sum(1 for i in top_f if c_fus[i][0] == "TP")
        n_fp = sum(1 for i in top_f if c_fus[i][0] == "FP")
        rows.append({"pid": pid, "n_gt": len(gts), "n_cand": len(cands),
                     "best_iou_top5_fus": best_iou_f, "any_iou": any_iou,
                     "tp5": n_tp, "fp5": n_fp,
                     "top1_s1_is_tp": c_s1[int(np.argmax(s1))][0] == "TP",
                     "top1_fus_is_tp": c_fus[int(np.argmax(fus))][0] == "TP"})
    print(f"  taranan: {len(rows)} hasta", flush=True)

    picks: list[tuple[str, str, str]] = []
    # 1) temiz başarı: GT var, füzyon top-5'te TP ve IoU yüksek
    c = [r for r in rows if r["n_gt"] and r["top1_fus_is_tp"] and r["best_iou_top5_fus"] > 0.55
         and r["fp5"] <= 2]
    if c:
        b = max(c, key=lambda r: r["best_iou_top5_fus"])
        picks.append((b["pid"], "Temiz başarı — TP", f"IoU {b['best_iou_top5_fus']:.2f}"))
    # 2) kaçırma: GT var ama hiçbir aday yakın değil
    c = [r for r in rows if r["n_gt"] and r["any_iou"] < 0.20]
    if c:
        b = min(c, key=lambda r: r["any_iou"])
        picks.append((b["pid"], "Kaçırma (FN) — hiçbir aday GT'ye oturmuyor",
                      f"en iyi IoU {b['any_iou']:.2f} < 0,20"))
    # 3) 2. aşamanın sıralamayı düzelttiği vaka
    c = [r for r in rows if r["n_gt"] and not r["top1_s1_is_tp"] and r["top1_fus_is_tp"]]
    if c:
        b = c[0]
        picks.append((b["pid"], "2. aşama sıralamayı düzeltti",
                      "1. aşama top-1 = FP → füzyon top-1 = TP"))
    # 4) temiz kontrol: GT yok, yüksek skorlu FP
    c = [r for r in rows if r["n_gt"] == 0]
    if c:
        b = max(c, key=lambda r: r["fp5"])
        picks.append((b["pid"], "Temiz kontrol — yalnız FP", "GT yok, adaylar yanlış alarm"))

    print(f"  seçilen: {[p[0] for p in picks]}", flush=True)

    # ---- figür
    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(11.5, 5.6 * n), squeeze=False)
    fig.suptitle("NODE21 kilitli test — tahmin doğruluğu: GT vs 1. aşama vs füzyon\n"
                 "saf 1.600 TDA · eşleşme kuralı IoU > 0,20 · sabit 56×56 kutu · eşik YOK, "
                 "çıktı sıralı listedir (top-5 gösteriliyor)", fontsize=12, y=0.996)
    rec = {"experiment": "V6.9p-tp-fp-figure",
           "matching_rule": "IoU > 0.20, aynı hasta; aynı GT'ye 2. vuruş ne TP ne FP",
           "box": "sabit 56×56 px", "patients": []}

    for r, (pid, tag, note) in enumerate(picks):
        cands, s1, p_tda, fus, gts = load_patient(pid, z, bundle)
        info = {}
        for key, scores, col in (("s1", s1, 0), ("fus", fus, 1)):
            ax = axes[r, col]
            img = common.load_image(pid)
            mask = common.load_lung_mask(pid)
            p1, p99 = common.get_lung_p1p99()[pid]
            disp = common.canonicalize(img, mask, float(p1), float(p99))
            ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
            # gösterilecek skor: 1. aşama olasılık (ısı), füzyon ise z-uzayı (sigmoid doyar)
            prob = scores if key == "s1" else scores
            cls = classify(cands, gts, scores)
            # tüm adaylar (soluk)
            for (cx, cy) in cands:
                ax.add_patch(Rectangle((cx - BOX_HALF, cy - BOX_HALF), 2 * BOX_HALF, 2 * BOX_HALF,
                                       fill=False, ec="white", lw=0.4, alpha=0.3, ls=":"))
            # GT
            for gi, g in enumerate(gts):
                ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1],
                                       fill=False, ec="lime", lw=2.6))
                if gi == 0:
                    ax.text(g[0] + 2, g[1] - 8, "GT", color="lime", fontsize=10,
                            weight="bold", va="bottom",
                            bbox=dict(fc="black", alpha=0.6, pad=1.2, ec="none"))
            # top-5: kutu + içine SIRA NUMARASI
            order = np.argsort(-scores)[:TOPK_SHOW]
            for rank, i in enumerate(order):
                cx, cy = cands[i]
                kind, biou, _gi = cls[i]
                ec = {"TP": "deepskyblue", "FP": "red", "ignore": "orange"}[kind]
                ls = "--" if kind == "ignore" else "-"
                ax.add_patch(Rectangle((cx - BOX_HALF, cy - BOX_HALF), 2 * BOX_HALF,
                                       2 * BOX_HALF, fill=False, ec=ec, lw=2.2, ls=ls))
                ax.text(cx, cy, f"{rank + 1}", color=ec, fontsize=14, weight="bold",
                        ha="center", va="center",
                        bbox=dict(fc="black", alpha=0.5, pad=1.4, ec="none"))
            # köşe tablosu — kesilme/üst üste binme yok
            head = f"{'#':<2}{'sınıf':<7}{'skor':>6}{'IoU':>7}" if key == "s1" else \
                   f"{'#':<2}{'sınıf':<7}{'z':>6}{'IoU':>7}"
            lines = [head]
            for rank, i in enumerate(order):
                kind, biou, _gi = cls[i]
                sc = f"{prob[i]:>6.2f}" if key == "s1" else f"{prob[i]:>+6.2f}"
                lines.append(f"{rank + 1:<2}{kind:<7}{sc}{biou:>7.2f}")
            ntp, nfp = sum(1 for i in order if cls[i][0] == "TP"), \
                sum(1 for i in order if cls[i][0] == "FP")
            ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                    fontsize=7.2, family="monospace", va="top", ha="left", color="white",
                    bbox=dict(fc="black", alpha=0.62, pad=3.0, ec="none"))
            ax.text(0.985, 0.985, f"top-5\nTP {ntp} · FP {nfp}", transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="right", color="white", weight="bold",
                    bbox=dict(fc="black", alpha=0.62, pad=3.0, ec="none"))
            lab = ("1. AŞAMA sıralaması — ısı haritası skoru" if key == "s1"
                   else "FÜZYON sıralaması — 0,55·s1 + 0,45·p_TDA")
            ax.set_title(f"{pid} · {lab}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            top = [cls[i][0] for i in order]
            info[key] = {"top5": top, "n_tp_top5": top.count("TP"),
                         "n_fp_top5": top.count("FP"),
                         "top1_score": round(float(prob[np.argmax(scores)]), 4),
                         "best_iou_top5": round(float(max(cls[i][1] for i in order)), 4)}
        axes[r, 0].set_ylabel(f"{tag}\n{note}", fontsize=10, weight="bold")
        rec["patients"].append({"pid": pid, "tag": tag, "note": note,
                                "n_gt": len(gts), "n_candidates": len(cands), **info})

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, ec="lime", lw=2.4, label="GT (gerçek tümör)"),
        Rectangle((0, 0), 1, 1, fill=False, ec="deepskyblue", lw=2.0, label="TP  (IoU > 0,20)"),
        Rectangle((0, 0), 1, 1, fill=False, ec="red", lw=2.0, label="FP  (yanlış alarm)"),
        Rectangle((0, 0), 1, 1, fill=False, ec="orange", lw=2.0, ls="--",
                  label="yok sayılan (aynı GT'ye 2. vuruş)"),
        Rectangle((0, 0), 1, 1, fill=False, ec="white", lw=0.6, ls=":",
                  label="1. aşamanın 50 adayı"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    fig.savefig(PNG, dpi=145)
    print(f"  PNG: {PNG.relative_to(PROJECT_ROOT)}", flush=True)

    for p in rec["patients"]:
        print(f"  {p['pid']:8s} {p['tag'][:34]:34s} GT={p['n_gt']} | "
              f"s1 top5 TP/FP={p['s1']['n_tp_top5']}/{p['s1']['n_fp_top5']} · "
              f"füzyon top5 TP/FP={p['fus']['n_tp_top5']}/{p['fus']['n_fp_top5']}", flush=True)
    JSON.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON: {JSON.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
