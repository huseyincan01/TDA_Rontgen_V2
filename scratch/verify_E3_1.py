"""E3.1 fuzyon sonucunun bagimsiz dogrulamasi (sizinti/bug kontrolu)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

SEED = 42
splits = pd.read_csv(PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv").sort_values("vector_index").reset_index(drop=True)
X_tda = np.load(PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
X_res = np.load(PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")

y_all = splits["zone_label"].values
train_m = splits["is_train_balanced_sample"].values.astype(bool)


def patient_meta(split_name):
    m = splits[splits["split"] == split_name].drop_duplicates("patient_id").set_index("patient_id")
    return m["patient_label"]


def resnet_probs():
    out = {"val": {}, "test": {}}
    for z in range(1, 6):
        zm = (splits["zone_id"] == z).values
        tr, va, te = zm & train_m, zm & (splits["split"] == "val").values, zm & (splits["split"] == "test").values
        sc = StandardScaler().fit(X_res[tr])
        Xtr, Xva, Xte = sc.transform(X_res[tr]), sc.transform(X_res[va]), sc.transform(X_res[te])
        best = (-1, None)
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xtr, y_all[tr])
            a = roc_auc_score(y_all[va], clf.predict_proba(Xva)[:, 1])
            if a > best[0]:
                best = (a, clf)
        clf = best[1]
        for name, mask, X in (("val", va, Xva), ("test", te, Xte)):
            for pid, p in zip(splits.loc[mask, "patient_id"], clf.predict_proba(X)[:, 1]):
                out[name][pid] = max(out[name].get(pid, 0.0), p)
    return out


def tda_probs():
    cb = torch.load(PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt", map_location="cpu", weights_only=False)
    sb = joblib.load(PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    drop = float(cb["config"].get("dropout", 0.3))
    cnn_p, svm_p = {"val": {}, "test": {}}, {"val": {}, "test": {}}
    for z in range(1, 6):
        for name in ("val", "test"):
            mask = ((splits["split"] == name) & (splits["zone_id"] == z)).values
            idx = splits[mask]["vector_index"].values
            Xz = X_tda[idx]
            Xs = cb["scalers"][z].transform(Xz)
            m = SmallZoneCNN(in_channels=2, dropout=drop).eval()
            m.load_state_dict(cb["models_state_dict"][z])
            with torch.no_grad():
                pc = torch.sigmoid(m(torch.tensor(vector_to_2channel_image(Xs), dtype=torch.float32))).numpy()
            ps = sb.predict_zone_proba(z, Xz)
            for pid, a, b in zip(splits.loc[mask, "patient_id"], pc, ps):
                cnn_p[name][pid] = max(cnn_p[name].get(pid, 0.0), float(a))
                svm_p[name][pid] = max(svm_p[name].get(pid, 0.0), float(b))
    return {name: {pid: 0.55 * cnn_p[name][pid] + 0.45 * svm_p[name][pid] for pid in cnn_p[name]}
            for name in ("val", "test")}


print("Bagimsiz yeniden hesaplama...")
R = resnet_probs()
T = tda_probs()

for name in ("val", "test"):
    yy = patient_meta(name)
    r = pd.Series(R[name]).reindex(yy.index)
    t = pd.Series(T[name]).reindex(yy.index)
    print(f"\n=== {name.upper()} (n={len(yy)}) ===")
    print("  NaN  resnet:", int(r.isna().sum()), "| tda:", int(t.isna().sum()))
    print(f"  resnet AUC = {roc_auc_score(yy.values, r.values):.4f}")
    print(f"  tda    AUC = {roc_auc_score(yy.values, t.values):.4f}")
    print(f"  korelasyon(resnet, tda) = {np.corrcoef(r.values, t.values)[0,1]:.4f}")
    for a in (0.5, 0.55, 0.6):
        f = a * r.values + (1 - a) * t.values
        print(f"  fuzyon a={a:.2f} -> AUC = {roc_auc_score(yy.values, f):.4f}")
    # en iyi a taramasi (val'de)
    if name == "val":
        grid = np.arange(0, 1.0001, 0.05)
        aucs = [roc_auc_score(yy.values, a * r.values + (1 - a) * t.values) for a in grid]
        bi = int(np.argmax(aucs))
        print(f"  val en iyi a={grid[bi]:.2f} (AUC={aucs[bi]:.4f})")

# Sizinti kontrolu: etiketleri karistirip ayni islem
print("\n=== SIZINTI KONTROLU (etiket permutasyonu) ===")
rng = np.random.RandomState(0)
yy = patient_meta("test")
r = pd.Series(R["test"]).reindex(yy.index).values
t = pd.Series(T["test"]).reindex(yy.index).values
perm = rng.permutation(yy.values)
print(f"  gercek etiketle fuzyon AUC = {roc_auc_score(yy.values, 0.55*r+0.45*t):.4f}")
print(f"  permute etiketle fuzyon AUC = {roc_auc_score(perm, 0.55*r+0.45*t):.4f}  (~0.5 beklenir)")
