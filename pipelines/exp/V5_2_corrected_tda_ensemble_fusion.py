"""V5.2 — Düzeltilmiş TDA (V5.1 `a_density`) ile CNN ensemble + füzyon (val-only).

NEDEN
    V5.1'de kök neden bulundu: `intensity_max=65535` sabiti 12-bit veri için 16× fazla
    olduğundan filtrasyon ~16 seviyeye (fiilen ~4-bit) kuantalanıyordu. Doku p1–p99
    kanonikleştirmesi bunu düzeltir: H1 nokta sayısı 610k → 1,63M, PI doluluğu %38 → %81,
    val label AUC +0,088, LOO +0,061.

    Bu deney, düzeltilmiş temsilin **projenin gerçek modeli** (5-zon küçük CNN + RBF-SVM
    ensemble ve ince ayarlı ResNet18 ile geç füzyon) altında da kazandığını doğrular.

KURALLAR
    * **Test bütçesi 6/6 tükendi** ⇒ bu deney **yalnızca validation**'da karar verir.
    * Tek değişken: temsil (v2 dondurulmuş `features_v2` ↔ V5.1 `a_density`).
      Model tarifi, hiperparametre ızgarası, eşik yöntemi ve füzyon ızgarası SABİT.
    * Yanlılık kontrolü: her zon için CNN tohumu iki kolda **aynı** (`SEED + z`).
    * Ön-kayıtlı kural: düzeltilmiş kol, (i) TDA ensemble val AUC ve (ii) füzyon val AUC
      bakımından **her ikisinde de** dondurulmuş v2'yi geçerse KEEP.

Kullanım:
    python pipelines/exp/V5_2_corrected_tda_ensemble_fusion.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "V5.2"
SLUG = "corrected_tda_ensemble_fusion"
SEED = 42
TDA_CWT = 0.55                    # E3.4 ensemble ağırlığı (CNN)
# --- v2.1 model tarifi (05_train_cnn.py / configs/models.yaml ile birebir) ---
CNN_LR, CNN_WD, CNN_BATCH, CNN_EPOCHS, DROPOUT = 1e-3, 1e-4, 32, 40, 0.3
SVM_C_GRID = [0.1, 1.0, 10.0, 50.0, 100.0]
SVM_G_GRID = ["scale", "auto", 0.01, 0.005, 0.001]
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
E0_6_A = 0.45

SPLITS = PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv"
FROZEN = PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy"
CANON = PROJECT_ROOT / "data/processed/features_v5_1/X_a_density.npy"
RESNET_CACHE = PROJECT_ROOT / "data/processed/features_resnet18/trainval_gray224.npy"
RESNET_CKPT = PROJECT_ROOT / "artifacts/models/zone_resnet18/resnet18_e0_6.pt"
REPORT_JSON = PROJECT_ROOT / "artifacts/runs/V5.2_corrected_tda_ensemble_fusion/report.json"
MODEL_DIR = PROJECT_ROOT / "artifacts/models"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ CNN (v2.1 protokolü)
def train_zone_cnn(X_tr, y_tr, X_val, y_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallZoneCNN(in_channels=2, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=CNN_WD)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(vector_to_2channel_image(X_tr), dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.float32)),
                        batch_size=CNN_BATCH, shuffle=True)
    xval = torch.tensor(vector_to_2channel_image(X_val), dtype=torch.float32)
    best_auc, best_w, best_ep = -1.0, copy.deepcopy(model.state_dict()), 1
    for ep in range(1, CNN_EPOCHS + 1):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(xval)).numpy()
        auc = roc_auc_score(y_val, p) if len(set(y_val)) > 1 else -1.0
        if auc > best_auc:
            best_auc, best_w, best_ep = auc, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_w)
    model.eval()
    return model, float(best_auc), best_ep


@torch.no_grad()
def cnn_probs(model, X):
    model.eval()
    return torch.sigmoid(model(torch.tensor(vector_to_2channel_image(X), dtype=torch.float32))).numpy().flatten()


# ------------------------------------------------------------------ SVM (v2.1 protokolü)
def train_zone_svm(X_tr, y_tr, X_val, y_val, seed=SEED):
    """AUC için karar fonksiyonu (monoton) ile tune, sonra kazananı olasılıklı refit."""
    best = {"auc": -1.0, "C": None, "gamma": None}
    for c in SVM_C_GRID:
        for g in SVM_G_GRID:
            clf = SVC(kernel="rbf", C=c, gamma=g, probability=False,
                      random_state=seed, cache_size=1000)
            clf.fit(X_tr, y_tr)
            auc = roc_auc_score(y_val, clf.decision_function(X_val))
            if auc > best["auc"]:
                best = {"auc": float(auc), "C": c, "gamma": g}
    final = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], probability=True,
                random_state=seed, cache_size=1000)
    final.fit(X_tr, y_tr)
    return final, best


def zone_proba(clf, X):
    return clf.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ ResNet18 (E0.6 checkpoint)
def raw_batch(gray):
    return torch.from_numpy(gray).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)


def normalize_img(x):
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def resnet_patient_probs(sub: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """`mask`, ResNet önbelleğiyle hizalı `sub` (E0.6 alt kümesi) üzerindedir."""
    from torchvision.models import resnet18

    gray = np.load(RESNET_CACHE)
    if len(gray) != len(sub):
        raise ValueError(f"ResNet önbelleği uyuşmuyor: {len(gray)} vs {len(sub)}")
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, 1))
    ckpt = torch.load(RESNET_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    idx = np.where(mask)[0]
    out = []
    for i in range(0, len(idx), 128):
        b = idx[i:i + 128]
        x = normalize_img(raw_batch(gray[b]))
        out.append(torch.sigmoid(model(x).squeeze(-1)).numpy())
    p = np.concatenate(out)
    t = sub.loc[mask, ["patient_id", "patient_label"]].copy()
    t["p"] = p
    return t.groupby(["patient_id", "patient_label"])["p"].max().reset_index()


# ------------------------------------------------------------------ Ana akış
def evaluate_arm(name: str, X: np.ndarray, splits: pd.DataFrame,
                 img_val: pd.DataFrame, save_bundles: bool) -> dict:
    y = splits["zone_label"].values.astype(float)
    cnn_rows, svm_rows = [], []
    cnn_bundle, svm_bundle = {}, {}
    t0 = time.time()

    for z in range(1, 6):
        zm = (splits["zone_id"] == z).values
        tr = zm & splits["is_train_balanced_sample"].values.astype(bool)
        va_m = zm & (splits["split"] == "val").values & splits["is_zone_metric_sample"].values.astype(bool)
        va_all = zm & (splits["split"] == "val").values
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva_m, Xva_all = sc.transform(X[tr]), sc.transform(X[va_m]), sc.transform(X[va_all])

        model, cnn_val_auc, ep = train_zone_cnn(Xtr, y[tr], Xva_m, y[va_m], SEED + z)
        clf, sbest = train_zone_svm(Xtr, y[tr], Xva_m, y[va_m])

        p_cnn_all, p_svm_all = cnn_probs(model, Xva_all), zone_proba(clf, Xva_all)
        p_cnn_m, p_svm_m = cnn_probs(model, Xva_m), zone_proba(clf, Xva_m)

        cnn_rows.append(pd.DataFrame({"patient_id": splits.loc[va_all, "patient_id"].values,
                                      "patient_label": splits.loc[va_all, "patient_label"].values,
                                      "p": p_cnn_all}))
        svm_rows.append(pd.DataFrame({"patient_id": splits.loc[va_all, "patient_id"].values,
                                      "patient_label": splits.loc[va_all, "patient_label"].values,
                                      "p": p_svm_all}))
        print(f"    [zon {z}] CNN val(zon) AUC={cnn_val_auc:.4f} (ep {ep}) | "
              f"SVM C={sbest['C']} γ={sbest['gamma']} (zon AUC={sbest['auc']:.4f})", flush=True)

        if save_bundles:
            cnn_bundle[z] = {"state_dict": model.state_dict(), "scaler": sc}
            svm_bundle[z] = {"model": clf, "scaler": sc}

    def pat(df):
        return df.groupby(["patient_id", "patient_label"])["p"].max().reset_index()

    cnn_p, svm_p = pat(pd.concat(cnn_rows)), pat(pd.concat(svm_rows))
    ens = cnn_p.copy()
    ens["p"] = TDA_CWT * cnn_p["p"].values + (1 - TDA_CWT) * svm_p["p"].values
    yv = ens["patient_label"].values

    iv = img_val.reindex(ens["patient_id"]).values
    tv = ens["p"].values
    if not np.isfinite(iv).all():
        miss = sorted(set(ens["patient_id"]) - set(img_val.index))
        raise ValueError(f"ResNet val tahminleri eksik hasta: {len(miss)} (ör. {miss[:5]})")
    best = {"a": None, "auc": -1.0}
    for a in A_GRID:
        s = roc_auc_score(yv, a * iv + (1 - a) * tv)
        if s > best["auc"]:
            best = {"a": float(a), "auc": float(s)}
    fv = best["a"] * iv + (1 - best["a"]) * tv
    thr, _ = find_optimal_threshold(yv, fv, method="youden")

    res = {
        "val_tda_cnn": float(roc_auc_score(yv, cnn_p["p"].values)),
        "val_tda_svm": float(roc_auc_score(yv, svm_p["p"].values)),
        "val_tda_ensemble": float(roc_auc_score(yv, tv)),
        "val_image_resnet_e0.6": float(roc_auc_score(yv, iv)),
        "val_fusion_best_a": best["auc"],
        "fusion_best_a": best["a"],
        f"val_fusion_a{E0_6_A}": float(roc_auc_score(yv, E0_6_A * iv + (1 - E0_6_A) * tv)),
        "fusion_metrics_best_a": compute_binary_metrics(yv, fv, threshold=thr),
        "n_val_patients": int(len(ens)),
        "n_pos": int((yv == 1).sum()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if save_bundles:
        cnn_path = MODEL_DIR / "zone_cnn_v5_1/zone_cnn_v5_1_bundle.pt"
        svm_path = MODEL_DIR / "zone_svm_v5_1/zone_svm_v5_1_bundle.joblib"
        cnn_path.parent.mkdir(parents=True, exist_ok=True)
        svm_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models_state_dict": {z: b["state_dict"] for z, b in cnn_bundle.items()},
                    "scalers": {z: b["scaler"] for z, b in cnn_bundle.items()},
                    "config": {"in_channels": 2, "dropout": DROPOUT,
                               "patient_aggregation": "max", "tda_cwt": TDA_CWT}},
                   cnn_path)
        import joblib

        joblib.dump({z: svm_bundle[z]["model"] for z in svm_bundle}, svm_path)
        res["bundles"] = {"cnn": str(cnn_path), "svm": str(svm_path)}
    return res


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 82)
    print("V5.2 — DÜZELTİLMİŞ TDA (65535 niceleme hatası giderilmiş) · CNN ensemble + füzyon")
    print("=" * 82)
    print("⚠️  Test bütçesi 6/6 tükendi → karar YALNIZCA validation'da.")

    splits = pd.read_csv(SPLITS).sort_values("vector_index").reset_index(drop=True)
    # ResNet önbelleği E0.6 alt kümesiyle (train_balanced | val | test) hizalıdır.
    need = (splits["is_train_balanced_sample"].values.astype(bool)
            | (splits["split"] == "val").values | (splits["split"] == "test").values)
    sub = splits[need].reset_index(drop=True)
    img_val_df = resnet_patient_probs(sub, (sub["split"] == "val").values)
    img_val = img_val_df.set_index("patient_id")["p"]
    print(f"ResNet18 (E0.6 checkpoint) val hasta tahminleri hazır: {len(img_val)} hasta", flush=True)

    X_frozen = np.load(FROZEN, mmap_mode="r")
    X_canon = np.load(CANON)
    print(f"Temsiller: frozen {X_frozen.shape} | canonical {X_canon.shape}\n", flush=True)

    report = {"experiment": EXPERIMENT_ID, "scope": "validation-only (test bütçesi 6/6 kapalı)",
              "config": {"tda_cwt": TDA_CWT, "cnn": {"lr": CNN_LR, "wd": CNN_WD, "batch": CNN_BATCH,
                                                     "epochs": CNN_EPOCHS, "dropout": DROPOUT},
                         "svm_grid": {"C": SVM_C_GRID, "gamma": SVM_G_GRID},
                         "fusion_e0_6_a": E0_6_A}, "arms": {}}

    for name, X, save in (("v2_frozen", X_frozen, False), ("v5_1_canonical", X_canon, True)):
        print(f"[kol] {name}", flush=True)
        report["arms"][name] = evaluate_arm(name, X, splits, img_val, save)

    f, c = report["arms"]["v2_frozen"], report["arms"]["v5_1_canonical"]
    keep = (c["val_tda_ensemble"] > f["val_tda_ensemble"] and c["val_fusion_best_a"] > f["val_fusion_best_a"])
    report["comparison"] = {
        "delta_val_tda_cnn": c["val_tda_cnn"] - f["val_tda_cnn"],
        "delta_val_tda_svm": c["val_tda_svm"] - f["val_tda_svm"],
        "delta_val_tda_ensemble": c["val_tda_ensemble"] - f["val_tda_ensemble"],
        "delta_val_fusion_best_a": c["val_fusion_best_a"] - f["val_fusion_best_a"],
        "delta_val_fusion_a045": c[f"val_fusion_a{E0_6_A}"] - f[f"val_fusion_a{E0_6_A}"],
        "decision": "KEEP" if keep else "KILL",
        "rule": "TDA ensemble val ↑ VE füzyon val ↑",
    }

    print("\n" + "=" * 82)
    for key, label in (("val_tda_cnn", "TDA CNN (tek)"), ("val_tda_svm", "TDA SVM (tek)"),
                       ("val_tda_ensemble", "TDA ensemble"), ("val_image_resnet_e0.6", "ResNet18 (E0.6)"),
                       ("val_fusion_best_a", "FÜZYON (val'de en iyi a)")):
        delta = c[key] - f[key]
        print(f"  {label:28s} | v2 {f[key]:.4f}  →  düzeltilmiş {c[key]:.4f}  ({delta:+.4f})")
    print(f"\n  val füzyon doğruluk (en iyi a={c['fusion_best_a']:.2f}): "
          f"%{c['fusion_metrics_best_a']['accuracy'] * 100:.1f} "
          f"(v2: %{f['fusion_metrics_best_a']['accuracy'] * 100:.1f})")
    print(f"  KARAR: {report['comparison']['decision']} "
          f"(Δ ensemble {report['comparison']['delta_val_tda_ensemble']:+.4f}, "
          f"Δ füzyon {report['comparison']['delta_val_fusion_best_a']:+.4f})")
    print("=" * 82)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    write_manifest(run_dir, EXPERIMENT_ID, config=report["config"],
                   inputs=[SPLITS, FROZEN, CANON, RESNET_CACHE, RESNET_CKPT], seed=SEED,
                   metrics={f"{a}_{k}": report["arms"][a][k]
                            for a in report["arms"] for k in ("val_tda_cnn", "val_tda_svm",
                                                              "val_tda_ensemble", "val_fusion_best_a")},
                   decision=report["comparison"]["decision"],
                   notes="Düzeltilmiş TDA (V5.1 a_density) — val-only; test bütçesi kapalı.")
    print(f"Rapor: {REPORT_JSON}")
    print(f"Manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
