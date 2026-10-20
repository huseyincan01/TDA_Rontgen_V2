"""F10 — FROC eğrileri ve VinDr-CXR transfer merdiveni.

Paneller:
    (A) NODE21 FROC: val aşama-1 / TDA-HGB / e_fusion / hiyerarşik + TEST mührü (e_fusion)
    (B) VinDr-CXR merdiveni: V4 → V5.3 → V6.4b → V6.5 (zero-shot → few-shot)
    (C) VinDr FROC@2FP vs etiketli hedef görüntü sayısı N (V6.5 / V6.5b / V6.5c)

Tüm sayılar kayıtlı raporlardan okunur (elle girilmez).

Kullanım:
    python pipelines/figures/make_figure_f10_froc_ladder.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "v6_two_stage"
EXT = ROOT / "data" / "external_vindr"
OUT = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2E5A88", "#D97706", "#2F7D4F", "#B4341F", "#6B7280", "#6B4C9A"

FP = np.array([0.5, 1.0, 2.0, 4.0])


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def froc_xy(froc: dict) -> tuple[np.ndarray, np.ndarray]:
    return FP, np.array([froc[str(int(f)) if f != 0.5 else "0.5"] for f in FP], float)


def main() -> None:
    seal = load(ART / "v6_4_domain_and_test_seal.json")["step3_test_seal"]
    v62 = load(ART / "v6_2_froc_and_transfer.json")["node21_val"]
    v64b = load(ART / "v6_4b_vindr_adabn_tda.json")
    v65 = load(ART / "v6_5_vindr_transfer_boost.json")
    v65b = load(ART / "v6_5b_fewshot_compact.json")
    v65c = load(ART / "v6_5c_decoupled_calibration.json")
    v53 = load(EXT / "v5_3_corrected_tda_vindr_report.json")["results"]
    v4 = load(EXT / "vindr_report_matched.json")["auc"]

    fig = plt.figure(figsize=(15.2, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.18, 1.0], wspace=0.30)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---------------- (A) NODE21 FROC ----------------
    curves = [
        ("M5_heatTTA|K30|a_stage1", "Aşama-1 dedektör", GREY, ":"),
        ("M5_heatTTA|K30|b_tda_hgb", "Lokal TDA-HGB", ORANGE, "--"),
        ("M5_heatTTA|K30|e_fusion_tda", "e_fusion (V6 nihai)", BLUE, "-"),
        ("M5_heatTTA|K30|f_hierarchical", "Hiyerarşik (FROC-opt.)", GREEN, "-"),
    ]
    for key, label, color, ls in curves:
        x, y = froc_xy(v62[key]["froc"])
        ax1.plot(x, y, ls, color=color, marker="o", markersize=3.5, linewidth=1.7, label=label)
    xt, yt = froc_xy(seal["v6"]["froc"])
    ax1.plot(xt, yt, color=BLUE, linestyle=(0, (4, 2)), marker="s", markersize=4.5,
             linewidth=2.0, label="TEST mührü (e_fusion)")
    for xx, yy in zip(xt, yt):
        ax1.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points", xytext=(0, -12),
                     ha="center", fontsize=7.5, color=BLUE)
    ax1.set_xscale("log")
    ax1.set_xticks(FP)
    ax1.set_xticklabels(["0,5", "1", "2", "4"])
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlim(0.42, 4.6)
    ax1.set_xlabel("yanlış pozitif / görüntü (FP)")
    ax1.set_ylabel("duyarlılık (lezyon yakalama)")
    ax1.set_ylim(0.20, 0.92)
    ax1.set_title("A) NODE21 FROC — 229 lezyon\n(val eğrileri + tek seferlik TEST mührü)", fontsize=10.5)
    ax1.legend(fontsize=7.6, loc="upper left")

    # ---------------- (B) VinDr merdiveni ----------------
    rungs = [
        ("V4 füzyon (a=0,45)", v4["fusion"], GREY),
        ("V5.3 düzeltilmiş TDA", v53["fusion_a045"], GREY),
        ("V6.4b AdaBN-M5 + lokal", v64b["adabn_m5_e_fusion_tda_auc"], BLUE),
        ("V6.4b V5.3 + lokal", v64b["combined_v53_adabn_m5_tda_auc"], BLUE),
        ("V6.5 zero-shot (0 etiket)", v65["kol1_zero_shot"]["denoise1.0_scale_coral"]["auc"], ORANGE),
        ("V6.5 few-shot N=80", v65["kol2_few_shot"]["curve"]["80"]["stage2_only_mean"], GREEN),
        ("V6.5 few-shot N=160", v65["kol2_few_shot"]["curve"]["160"]["stage2_only_mean"], GREEN),
    ]
    names = [r[0] for r in rungs][::-1]
    vals = [r[1] for r in rungs][::-1]
    cols = [r[2] for r in rungs][::-1]
    yy = np.arange(len(names))
    bars = ax2.barh(yy, vals, color=cols, height=0.62, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax2.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", fontsize=8.2)
    ax2.axvline(0.5, color=RED, linestyle="--", linewidth=1.1)
    ax2.text(0.503, -0.45, "şans", color=RED, fontsize=8)
    ax2.text(0.985, 0.03, "NODE21 test: 0,988 (referans)", transform=ax2.transAxes,
             ha="right", fontsize=8, color=PURPLE)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(names, fontsize=8.3)
    ax2.set_xlim(0.50, 0.77)
    ax2.set_xlabel("VinDr-CXR görüntü AUC (400 görüntü)")
    ax2.set_title("B) Transfer merdiveni — domain farkı kapanmıyor,\nyönetilebiliyor (0,544 → 0,724)", fontsize=10.5)
    ax2.grid(axis="y", visible=False)

    # ---------------- (C) FROC@2 vs N ----------------
    Ns = [0, 10, 20, 40, 80, 160]
    series = [
        ("V6.5 görüntü etiketi", [v65["kol2_few_shot"]["curve"][str(n)]["froc_mean"]["2"] for n in Ns], BLUE, "-"),
        ("V6.5b bbox etiketi", [v65b["curve"][str(n)]["froc_mean"]["2"] for n in Ns], ORANGE, "--"),
        ("V6.5c çift-kafa", [v65c["curve"][str(n)]["froc_g1.0"]["2"] for n in Ns], RED, "-"),
    ]
    for label, ys, color, ls in series:
        ax3.plot(Ns, ys, ls, color=color, marker="o", markersize=4.5, linewidth=1.8, label=label)
    ax3.axhline(0.22, color=GREY, linestyle=":", linewidth=1.2)
    ax3.text(2, 0.224, "hedef ≥ 0,22", color=GREY, fontsize=8)
    ax3.set_xscale("symlog", linthresh=10)
    ax3.set_xticks(Ns)
    ax3.set_xticklabels([str(n) for n in Ns])
    ax3.set_xlabel("etiketli hedef görüntü sayısı N")
    ax3.set_ylabel("FROC @2 FP")
    ax3.set_ylim(0.04, 0.245)
    ax3.set_title("C) AUC–FROC ödünleşimi\n(az örnekte kalibrasyon sıralamayı bozuyor)", fontsize=10.5)
    ax3.legend(fontsize=8)

    fig.suptitle("F10 — FROC ve VinDr-CXR transfer merdiveni",
                 fontsize=13, weight="bold", y=1.03)
    p = OUT / "F10_froc_vindr_ladder.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  yazildi: {p.relative_to(ROOT)}")
    print(f"  test FROC@2={seal['v6']['froc']['2']:.4f} | VinDr zero-shot="
          f"{v65['kol1_zero_shot']['denoise1.0_scale_coral']['auc']:.4f} | N160="
          f"{v65['kol2_few_shot']['curve']['160']['stage2_only_mean']:.4f}")


if __name__ == "__main__":
    main()
