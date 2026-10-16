"""E5.2-fig — Makale figürleri F1–F8.

Her figür, [ablation_matrix.md](../../ablation_matrix.md) §11 planındaki bir maddeyi üretir.
Sayılar doğrulanmış deney raporlarından alınmıştır; F3 için val olasılıkları yeniden hesaplanır.

Çıktı: artifacts/figures/F1..F8 (PNG, 200 dpi)

Kullanım:
    python pipelines/figures/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

OUT = PROJECT_ROOT / "artifacts/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

BLUE, ORANGE, GREEN, RED, GREY = "#2E5A88", "#D97706", "#2F7D4F", "#B4341F", "#6B7280"


def save(fig, name: str) -> None:
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  yazildi: {p.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------- F1
def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.7); ax.axis("off")

    def box(x, y, w, h, text, color, fc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.6, edgecolor=color, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color=GREY, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.5, color=color,
                                     connectionstyle=f"arc3,rad={rad}"))

    # Ust siralama
    box(0.2, 5.15, 1.9, 0.9, "Göğüs röntgeni\n(1024×1024, 16-bit)", BLUE, "#EAF1F8")
    box(2.45, 5.15, 1.9, 0.9, "PSPNet akciğer\nbölütlemesi", BLUE, "#EAF1F8")
    box(4.7, 5.15, 1.7, 0.9, "5 anatomik\nzon kırpıntısı", BLUE, "#EAF1F8")
    arrow(2.12, 5.6, 2.43, 5.6); arrow(4.37, 5.6, 4.68, 5.6)

    # Iki dal
    box(0.3, 3.05, 4.0, 1.35,
        "TDA DALI\nkübik kalıcı homoloji (H0/H1)\n→ Kalıcılık İmgesi (2×20×20)\n"
        "→ küçük CNN + RBF-SVM", ORANGE, "#FDF3E7")
    box(5.4, 3.05, 4.3, 1.35,
        "GÖRÜNTÜ DALI\nResNet18 (son blok ince ayarlı)\n→ 512-boyut özellik\n"
        "→ lojistik regresyon", GREEN, "#EAF4EE")
    arrow(5.5, 5.13, 2.4, 4.45, ORANGE, rad=-0.12)
    arrow(5.9, 5.13, 7.4, 4.45, GREEN, rad=0.12)

    # Hasta duzeyi max-pooling
    box(0.3, 1.75, 4.0, 0.8, "hasta skoru = max(5 zon)", ORANGE, "#FDF3E7")
    box(5.4, 1.75, 4.3, 0.8, "hasta skoru = max(5 zon)", GREEN, "#EAF4EE")
    arrow(2.3, 3.03, 2.3, 2.57, ORANGE); arrow(7.55, 3.03, 7.55, 2.57, GREEN)

    # Fuzyon
    box(3.3, 0.45, 3.4, 0.8, "füzyon:  a·görüntü + (1−a)·TDA", RED, "#FBEBE8", fs=9.5)
    arrow(4.25, 1.73, 4.6, 1.27, RED, rad=-0.15)
    arrow(5.6, 1.73, 5.4, 1.27, RED, rad=0.15)

    ax.text(5, 0.12, "karar eşiği validation'da seçilir (Youden)", ha="center",
            fontsize=8.5, color=GREY)
    ax.text(5, 6.45, "F1 — Boru hattı", ha="center", fontsize=12, weight="bold")
    save(fig, "F1_pipeline")


# ---------------------------------------------------------------- F2
def fig2_leaderboard() -> None:
    models = ["Füzyon (ince ayarlı + TDA)", "İnce ayarlı ResNet18", "Füzyon (donmuş + TDA)",
              "TDA ensemble (CNN+SVM)", "TDA PI küçük CNN", "ResNet18 (donmuş)",
              "RBF-SVM", "k-NN", "Ham görüntü CNN", "PL küçük CNN", "PL RBF-SVM"]
    aucs = [0.9516, 0.9284, 0.9190, 0.8776, 0.8631, 0.8349, 0.8289, 0.7162, 0.7043, 0.5450, 0.5217]
    cols = [RED, GREEN, BLUE, ORANGE, ORANGE, GREEN, GREY, GREY, GREY, GREY, GREY]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(models))[::-1]
    ax.barh(y, aucs, color=cols, height=0.68)
    for yy, a in zip(y, aucs):
        ax.text(a + 0.006, yy, f"{a:.4f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim(0.45, 1.0); ax.set_xlabel("Test ROC-AUC (hasta düzeyi, 734 hasta)")
    ax.axvline(0.877, color=GREY, ls="--", lw=1.2)
    ax.text(0.879, 0.3, "NODE21 leaderboard\nensemble 0.877", fontsize=8, color=GREY)
    ax.set_title("F2 — Model liderlik tablosu", fontsize=12, weight="bold")
    save(fig, "F2_leaderboard")


# ---------------------------------------------------------------- F3 (gercek veri)
def fig3_complementarity() -> None:
    zs = pd.read_csv(PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv"
                     ).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    X_res = np.load(PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    meta = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    y = meta["patient_label"].values

    # TDA dali: kayitli bundle'lar, hasta duzeyi max-pool, sonra fuzyon (Varyant A)
    cb = torch.load(PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt",
                    map_location="cpu", weights_only=False)
    sb = joblib.load(PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    drop = float(cb["config"].get("dropout", 0.3))
    cnn_rows, svm_rows = [], []
    for z in range(1, 6):
        m = ((zs["split"] == "val") & (zs["zone_id"] == z)).values
        Xz = X_tda[zs[m]["vector_index"].values]
        sc = cb["scalers"][z]
        net = SmallZoneCNN(in_channels=2, dropout=drop)
        net.load_state_dict(cb["models_state_dict"][z]); net.eval()
        with torch.no_grad():
            pc = torch.sigmoid(net(torch.tensor(vector_to_2channel_image(sc.transform(Xz)),
                                                 dtype=torch.float32))).numpy()
        ps = sb.predict_zone_proba(z, Xz)
        pid = zs.loc[m, "patient_id"].values
        cnn_rows.append(pd.DataFrame({"patient_id": pid, "p": pc}))
        svm_rows.append(pd.DataFrame({"patient_id": pid, "p": ps}))
    p_cnn = pd.concat(cnn_rows).groupby("patient_id")["p"].max()
    p_svm = pd.concat(svm_rows).groupby("patient_id")["p"].max()
    p_tda = (0.55 * p_cnn + 0.45 * p_svm).reindex(meta.index).values

    # Goruntu dali: donmus ResNet18 + lojistik regresyon (zon basina)
    p_res = np.zeros(len(meta))
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    acc = np.zeros(len(meta)); cnt = np.zeros(len(meta))
    idx_of = {pid: i for i, pid in enumerate(meta.index)}
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & trm
        scaler = StandardScaler().fit(X_res[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000, random_state=42).fit(
            scaler.transform(X_res[tr]), zs["zone_label"].values[tr])
        for split in ("val",):
            m = zm & (zs["split"] == split).values
            pr = clf.predict_proba(scaler.transform(X_res[m]))[:, 1]
            for pid, p in zip(zs.loc[m, "patient_id"].values, pr):
                i = idx_of[pid]
                acc[i] = max(acc[i], p); cnt[i] += 1
    p_res = acc

    r = float(np.corrcoef(p_res, p_tda)[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, lab, col in ((axes[0], "Sağlıklı (0)", BLUE), (axes[1], "Nodül-pozitif (1)", RED)):
        m = y == (0 if "Sağlıklı" in lab else 1)
        ax.scatter(p_res[m], p_tda[m], s=22, alpha=0.55, color=col, edgecolors="none")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Görüntü dalı hasta olasılığı")
        ax.set_ylabel("TDA dalı hasta olasılığı")
        ax.set_title(lab, fontsize=10)
    fig.suptitle(f"F3 — Tamamlayıcılık: iki temsil neredeyse dik (r = {r:.2f})",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "F3_complementarity")
    print(f"    (hesaplanan korelasyon r = {r:.4f})")


# ---------------------------------------------------------------- F4
def fig4_pl_envelope() -> None:
    lam = [1, 2, 3, 4, 5]
    h1_env = [109.89, 102.61, 97.56, 93.60, 89.61]
    h0_env = [173.87, 173.42, 172.67, 171.79, 170.49]
    h1_sig = [1.62, 1.90, 1.54, 0.37, 1.72]
    h0_sig = [0.02, 0.21, 0.39, 0.78, 0.89]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(lam, h1_env, "o-", color=ORANGE, label="H1 zarf tepesi")
    axes[0].plot(lam, h0_env, "s-", color=BLUE, label="H0 zarf tepesi")
    axes[0].set_xlabel("Zarf seviyesi λ"); axes[0].set_ylabel("Ortalama zarf tepesi")
    axes[0].set_title("Zarf tepeleri λ1→λ5 boyunca yüksek ve birbirine yakın", fontsize=10)
    axes[0].legend(fontsize=9)
    w = 0.38
    axes[1].bar(np.array(lam) - w / 2, h1_sig, w, color=ORANGE, label="H1")
    axes[1].bar(np.array(lam) + w / 2, h0_sig, w, color=BLUE, label="H0")
    axes[1].set_xlabel("Zarf seviyesi λ"); axes[1].set_ylabel("Sınıf sinyali / zarf (%)")
    axes[1].set_title("Ama sınıf sinyali zarfın yalnızca %0–2'si", fontsize=10)
    axes[1].legend(fontsize=9)
    fig.suptitle("F4 — PL neden çöker: üst-zarf arka planla doymuş", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "F4_pl_envelope")


# ---------------------------------------------------------------- F5
def fig5_decisive() -> None:
    names = ["Donmuş\nResNet18", "TDA\nensemble", "İnce ayarlı\nResNet18",
             "Füzyon\n(donmuş)", "FÜZYON\n(ince ayarlı)"]
    vals = [0.8349, 0.8776, 0.9284, 0.9190, 0.9516]
    cols = [GREEN, ORANGE, GREEN, BLUE, RED]
    errs = [0, 0, 0, 0.0056, 0]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(names))
    ax.bar(x, vals, color=cols, width=0.62, yerr=errs, capsize=5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.006, f"{v:.4f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.78, 0.99); ax.set_ylabel("Test ROC-AUC")
    ax.annotate("", xy=(4, 0.9516), xytext=(4, 0.9284),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(4.32, 0.940, "TDA katkısı\n+0.0232\n[0.0132, 0.0338]\np=0.0000",
            fontsize=9, color=RED, va="center")
    ax.set_title("F5 — Belirleyici test: TDA güçlü baseline'a da katkı sağlıyor",
                 fontsize=12, weight="bold")
    save(fig, "F5_decisive_test")


# ---------------------------------------------------------------- F6
def fig6_robustness() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    sig = [0.5, 1.0, 1.5, 2.0]; sigv = [0.8460, 0.8697, 0.8734, 0.8712]
    grid = [15, 20, 25]; gridv = [0.8545, 0.8697, 0.8469]
    wl = ["p", "arctan", "√p", "sabit", "p²"]
    wv = [0.8736, 0.8640, 0.8624, 0.8591, 0.8547]
    seeds = [42, 43, 44, 45, 46]; seedv = [0.8922, 0.8546, 0.8395, 0.8369, 0.8701]

    ax = axes[0, 0]; ax.plot(sig, sigv, "o-", color=BLUE)
    ax.axhline(0.8697, ls="--", color=GREY, lw=1)
    ax.set_xlabel("σ (Gaussian)"); ax.set_ylabel("Val ROC-AUC")
    ax.set_title("σ duyarlılığı (E2.8) · maks sapma 0.0236", fontsize=10)

    ax = axes[0, 1]; ax.plot(grid, gridv, "s-", color=ORANGE)
    ax.axhline(0.8697, ls="--", color=GREY, lw=1)
    ax.set_xlabel("PI ızgara"); ax.set_ylabel("Val ROC-AUC")
    ax.set_title("Izgara duyarlılığı (E2.8) · maks sapma 0.0227", fontsize=10)

    ax = axes[1, 0]; ax.bar(wl, wv, color=[GREEN] + [GREY] * 4)
    ax.set_ylabel("Val ROC-AUC"); ax.set_ylim(0.84, 0.88)
    ax.set_title("PI ağırlık fonksiyonu (E2.7) · mevcut seçim optimal", fontsize=10)

    ax = axes[1, 1]; ax.plot(seeds, seedv, "o-", color=RED)
    ax.set_xlabel("Seed"); ax.set_ylabel("Val ROC-AUC")
    ax.set_title("CNN seed varyansı ±0.02–0.03 (E4.6) → Kural 10", fontsize=10)

    fig.suptitle("F6 — Sağlamlık panelleri", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "F6_robustness")


# ---------------------------------------------------------------- F7
def fig7_cross_source() -> None:
    src = ["chestxray14", "padchest", "jsrt"]
    nat = [0.7821, 0.8499, 0.9085]
    eq = [0.7821, 0.7625, 0.8847]
    x = np.arange(len(src)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - w / 2, nat, w, color=GREY, label="Doğal eğitim boyutu (confound'lu)")
    ax.bar(x + w / 2, eq, w, color=BLUE, label="Eşit-boyut (geçerli)")
    for xi, (a, b) in enumerate(zip(nat, eq)):
        ax.text(xi - w / 2, a + 0.006, f"{a:.4f}", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, b + 0.006, f"{b:.4f}", ha="center", fontsize=8.5, weight="bold")
    ax.axhline(0.9190, ls="--", color=RED, lw=1.3)
    ax.text(2.05, 0.925, "rastgele split 0.9190", fontsize=8, color=RED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(src)
    ax.set_ylabel("Füzyon test ROC-AUC"); ax.set_ylim(0.70, 0.97)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("F7 — Kaynak-dışı genelleme (leave-one-source-out)", fontsize=12, weight="bold")
    save(fig, "F7_cross_source")


# ---------------------------------------------------------------- F8
def fig8_limitation() -> None:
    models = ["TDA CNN", "TDA ensemble", "ResNet18", "Füzyon"]
    zone_auc = [0.4837, 0.5037, 0.5258, 0.5156]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(models, zone_auc, color=[ORANGE, ORANGE, GREEN, RED], width=0.6)
    axes[0].axhline(0.5, ls="--", color=GREY, lw=1.4)
    axes[0].text(3.4, 0.503, "şans", fontsize=9, color=GREY)
    for i, v in enumerate(zone_auc):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    axes[0].set_ylim(0.44, 0.56); axes[0].set_ylabel("Zon düzeyi AUC")
    axes[0].set_title("Model lokalize ETMİYOR (E4.7): zon AUC ≈ 0.5", fontsize=10)
    axes[0].tick_params(axis="x", labelsize=9)

    axes[1].bar(["Oracle yama\n(ground-truth bbox)", "Eşleştirilmemiş\n(confound'lu)"],
                [0.6866, 0.9940], color=[BLUE, GREY], width=0.5)
    for i, v in enumerate([0.6866, 0.9940]):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10, weight="bold")
    axes[1].axhline(0.5, ls="--", color=GREY, lw=1.2)
    axes[1].set_ylim(0.4, 1.08); axes[1].set_ylabel("Yama düzeyi AUC")
    axes[1].set_title("Nodülün tam merkezinde bile yerel TDA zayıf (E3.6)", fontsize=10)
    fig.suptitle("F8 — Doğrulanmış sınırlama: ayırt edici bilgi global/arka plan topolojisinde",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "F8_limitation")


def main() -> None:
    print("=" * 70)
    print("E5.2-fig — Makale figurleri F1-F8")
    print("=" * 70)
    fig1_pipeline()
    fig2_leaderboard()
    fig3_complementarity()
    fig4_pl_envelope()
    fig5_decisive()
    fig6_robustness()
    fig7_cross_source()
    fig8_limitation()
    print(f"\nToplam: {len(list(OUT.glob('F*.png')))} figur -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
