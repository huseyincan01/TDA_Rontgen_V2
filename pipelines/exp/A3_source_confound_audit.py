"""A3 — Kaynak/etiket confound denetimi.

Soru: NODE21 test setinde "kaynak-içi AUC" hesaplanabilir mi? Model ayrımı
kaynaklar arasında tutarlı mı, yoksa kaynak kimliğinden mi besleniyor?

Bulgı (bu betik üretir):
  * NODE21'de tüm pozitifler dört dış kaynaktan (chestxray14, padchest, jsrt, openi),
    tüm negatifler `node21_control` havuzundan gelir ⇒ **her kaynak tek sınıflıdır**
    ve klasik "kaynak-içi AUC" tanımsızdır.
  * Yerine geçen doğru metrik: **kaynak pozitifleri vs ortak kontrol havuzu** AUC'si
    (kaynak başına) + bunların **makro ortalaması** (kaynak prevalansından bağımsız).

Girdi: artifacts/runs/test_patient_predictions*.csv (v3 tek modelleri) +
       data/processed/zones/patients/*/patient_info.json (`source_dataset`).
Not: v3 füzyonu, V5.2 ve V6 için hasta-düzeyi tahmin CSV'si kaydedilmemiştir;
bu betik yalnızca kayıtlı tahminleri denetler.

Kullanım:
    python pipelines/exp/A3_source_confound_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"
OUT = RUNS / "A3_source_confound_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
B = 1000

LABELS = {
    "test_patient_predictions_cnn.csv": "TDA PI CNN (başlık tek model)",
    "test_patient_predictions.csv": "TDA PI RBF-SVM",
    "test_patient_predictions_knn.csv": "TDA PI k-NN",
    "test_patient_predictions_raw_cnn.csv": "Ham görüntü CNN",
    "test_patient_predictions_landscape_cnn.csv": "PL CNN",
    "test_patient_predictions_landscape_svm.csv": "PL SVM",
}


def auc_ci(y: np.ndarray, p: np.ndarray, rng: np.random.RandomState) -> tuple[float, float]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        try:
            vals.append(roc_auc_score(y[i], p[i]))
        except ValueError:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    rng = np.random.RandomState(SEED)
    splits = pd.read_csv(ROOT / "data/processed/splits_v2/patient_splits.csv")
    src = {}
    for pid in splits["patient_id"]:
        try:
            info = json.loads((ROOT / "data/processed/zones/patients" / pid /
                               "patient_info.json").read_text(encoding="utf-8"))
            src[pid] = info["source_dataset"]
        except Exception:
            src[pid] = "unknown"
    splits["source"] = splits["patient_id"].map(src)

    cont = (splits.groupby(["split", "source"])
            .agg(n=("patient_id", "size"), n_pos=("patient_label", "sum"))
            .reset_index())
    cont["n_neg"] = cont["n"] - cont["n_pos"]
    contingency = {}
    for split, g in cont.groupby("split"):
        contingency[split] = {
            r.source: {"n": int(r.n), "n_pos": int(r.n_pos), "n_neg": int(r.n_neg)}
            for r in g.itertuples()
        }

    test = splits[splits["split"] == "test"]
    meta = test.set_index("patient_id")
    sources = sorted(s for s, g in test.groupby("source") if g["patient_label"].max() == 1)

    print(f"Test: {len(test)} hasta | pozitif kaynaklar: {sources}")
    print("Test dağılımı:")
    for s in sources + ["node21_control"]:
        row = contingency["test"].get(s)
        if row:
            print(f"  {s:16s} n={row['n']:3d}  poz={row['n_pos']:3d}  neg={row['n_neg']:3d}")

    report = {
        "experiment": "A3",
        "question": "Model ayrımı kaynak kimliğinden mi besleniyor? Kaynak-içi AUC hesaplanabilir mi?",
        "protocol": {"bootstrap": B, "seed": SEED,
                     "per_source_auc": "kaynak pozitifleri vs TÜM node21_control negatifleri",
                     "macro_auc": "4 pozitif kaynağın AUC'lerinin ağırlıksız ortalaması"},
        "contingency": contingency,
        "key_finding": "Her pozitif kaynak tek sınıflı (tüm negatifler node21_control) ⇒ "
                       "klasik kaynak-içi AUC TANIMSIZDIR. Etiket ile kaynak kimliği bu kohortta "
                       "tamamen iç içedir; bu yüzden kaynak-başına AUC + makro ortalama raporlanır.",
        "models": {},
    }

    for fname, label in LABELS.items():
        path = RUNS / fname
        if not path.exists():
            continue
        d = pd.read_csv(path).set_index("patient_id")
        d = d.drop(columns=[c for c in ("patient_label",) if c in d.columns]).join(meta[["source", "patient_label"]])
        y = d["patient_label"].values.astype(int)
        p = d["predicted_patient_prob"].values.astype(float)
        ctrl = d[d["source"] == "node21_control"]
        entry = {
            "overall_auc": round(float(roc_auc_score(y, p)), 4),
            "controls": {"n": int(len(ctrl)),
                         "mean_prob": round(float(ctrl["predicted_patient_prob"].mean()), 4),
                         "specificity_at_0.5": round(float((ctrl["predicted_patient_prob"] < 0.5).mean()), 4)},
            "sources": {},
        }
        aucs = []
        for s in sources:
            pos = d[d["source"] == s]
            if len(pos) == 0:
                continue
            yy = np.concatenate([np.ones(len(pos), int), np.zeros(len(ctrl), int)])
            pp = np.concatenate([pos["predicted_patient_prob"].values, ctrl["predicted_patient_prob"].values])
            a = float(roc_auc_score(yy, pp))
            lo, hi = auc_ci(yy, pp, rng)
            aucs.append(a)
            entry["sources"][s] = {
                "n_pos": int(len(pos)),
                "auc_vs_controls": round(a, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "mean_prob": round(float(pos["predicted_patient_prob"].mean()), 4),
                "sensitivity_at_0.5": round(float((pos["predicted_patient_prob"] >= 0.5).mean()), 4),
            }
        entry["macro_auc"] = round(float(np.mean(aucs)), 4) if aucs else None
        entry["overall_minus_macro"] = (round(entry["overall_auc"] - entry["macro_auc"], 4)
                                        if entry["macro_auc"] is not None else None)
        report["models"][label] = entry

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nModel                         | genel AUC | makro AUC | fark")
    print("-" * 66)
    for label, e in report["models"].items():
        print(f"{label:29s} |   {e['overall_auc']:.4f}  |   {e['macro_auc']:.4f}  | {e['overall_minus_macro']:+.4f}")

    print("\nKaynak bazında (TDA PI CNN):")
    e = report["models"].get("TDA PI CNN (başlık tek model)")
    if e:
        for s, r in e["sources"].items():
            print(f"  {s:14s} n={r['n_pos']:3d} AUC={r['auc_vs_controls']:.3f} "
                  f"[{r['ci95'][0]:.3f}; {r['ci95'][1]:.3f}] ort.p={r['mean_prob']:.3f}")
        print(f"  {'control':14s} n={e['controls']['n']:3d} ort.p={e['controls']['mean_prob']:.3f} "
              f"özgüllük={e['controls']['specificity_at_0.5']:.3f}")

    print(f"\nRapor: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
