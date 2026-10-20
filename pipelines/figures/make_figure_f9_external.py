"""F9 — Harici doğrulama (VinDr-CXR) figürü.

Kaynak: `data/external_vindr/vindr_report_{linear,matched,matched_scalenorm}.json`
        + `vindr_summary_report.json`  (V4 koşusu, sıfır eğitim)

Paneller:
    (A) Kol başına TDA / ResNet18 / Füzyon(a=0.45) ROC-AUC, %95 bootstrap CI ile
    (B) Füzyon a-taraması (VinDr'de optimal a = 1,00 ⇒ TDA katkısı ~0)

Kullanım:
    python pipelines/figures/make_figure_f9_external.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = PROJECT_ROOT / "data/external_vindr"
OUT = PROJECT_ROOT / "artifacts/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY = "#2E5A88", "#D97706", "#2F7D4F", "#B4341F", "#6B7280"

ARMS = ["linear", "matched", "matched_scalenorm"]
ARM_LABELS = {
    "linear": "linear\n(8-bit × 16)",
    "matched": "matched\n(200-hasta havuzu)",
    "matched_scalenorm": "matched + ölçek\n(akciğer yüksekliği)",
}


def load(arm: str) -> dict:
    return json.loads((EXT_DIR / f"vindr_report_{arm}.json").read_text(encoding="utf-8"))


def main() -> None:
    reports = {a: load(a) for a in ARMS}
    summary = json.loads((EXT_DIR / "vindr_summary_report.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # ---------------- Panel A: kol başına AUC + CI ----------------
    x = np.arange(len(ARMS))
    w = 0.26
    series = [("tda", "TDA ensemble", ORANGE), ("resnet", "ResNet18", GREEN),
              ("fusion", "Füzyon (a=0.45)", BLUE)]
    for k, (key, label, color) in enumerate(series):
        vals, lo, hi = [], [], []
        for a in ARMS:
            r = reports[a]
            vals.append(r["auc"][key])
            l, h = r["ci95"][key]
            lo.append(r["auc"][key] - l)
            hi.append(h - r["auc"][key])
        bars = ax1.bar(x + (k - 1) * w, vals, w, yerr=[lo, hi], capsize=4, label=label,
                       color=color, edgecolor="white", linewidth=0.8,
                       error_kw={"elinewidth": 1.0, "ecolor": GREY})
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=7.5)

    ax1.axhline(0.5, color=RED, linestyle="--", linewidth=1.2)
    ax1.text(-0.48, 0.505, "şans (0.5)", color=RED, fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ARM_LABELS[a] for a in ARMS], fontsize=9)
    ax1.set_ylim(0.40, 0.78)
    ax1.set_ylabel("ROC-AUC (hasta/görüntü düzeyi)")
    ax1.set_title("A) VinDr-CXR (400 görüntü) — sıfır eğitim", fontsize=11)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.text(2.45, 0.752, "TDA ≈ şans · füzyon, ResNet'in altında",
             ha="right", fontsize=8.5, color=GREY)

    # ---------------- Panel B: a-taraması ----------------
    for a, color in zip(ARMS, [GREY, ORANGE, GREEN]):
        sw = reports[a]["a_sweep"]
        xs = np.array([float(k) for k in sw]) if list(sw)[0] in ("0.00", "0.0") else \
            np.array(sorted(float(k) for k in sw))
        ys = np.array([sw[f"{v:.2f}"] for v in xs])
        ax2.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, color=color,
                 label=ARM_LABELS[a].replace("\n", " "))
    ax2.axvline(0.45, color=BLUE, linestyle="--", linewidth=1.3)
    ax2.text(0.455, 0.585, "E0.6 kilitli a=0.45", color=BLUE, fontsize=8.5)
    ax2.scatter([1.0], [reports["matched"]["auc"]["resnet"]], marker="*", s=150,
                color=GREEN, zorder=5, edgecolor="black", linewidth=0.6)
    ax2.annotate("VinDr optimum a=1.00\n(= saf görüntü modeli)",
                 xy=(1.0, reports["matched"]["auc"]["resnet"]), xytext=(0.62, 0.66),
                 fontsize=8.5, arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.0})
    ax2.axhline(0.5, color=RED, linestyle="--", linewidth=1.0)
    ax2.set_xlabel("füzyon ağırlığı a  (hasta skoru = a·görüntü + (1−a)·TDA)")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.48, 0.70)
    ax2.set_title("B) Füzyon ağırlığı taraması — dış veride TDA katkısı yok", fontsize=11)
    ax2.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("F9 — Harici doğrulama: alan kayması altında TDA taşınmıyor",
                 fontsize=12.5, weight="bold", y=1.02)
    p = OUT / "F9_external_validation.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  yazıldı: {p.relative_to(PROJECT_ROOT)}")
    print(f"  özet  : n_images={summary['cohort']['n_images']} "
          f"resnet={reports['matched']['auc']['resnet']:.4f} "
          f"matched_tda={reports['matched']['auc']['tda']:.4f}")


if __name__ == "__main__":
    main()
