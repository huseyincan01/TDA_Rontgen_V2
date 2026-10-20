"""A2 — Kaynak (cihaz/kurum) ayrıştırılabilirliği: temsillerimiz fizik mi kodluyor?

Neden bu deney:
    Alan kaymasının kaynağını, VinDr'e dokunmadan **NODE21'in kendi içindeki 4
    farklı kaynağı** ile ölçebiliriz (chestxray14 0.143 mm/px · padchest 0.275 ·
    jsrt 0.309 · openi 0.331). NODE21 etiketleri bu kaynaklarla **ilişkilidir**
    (tüm kontroller `node21_control`), bu yüzden kaynağı doğrudan tahmin etmek
    etiket sızıntısı gibi görünür. Bunu kırmak için test **yalnızca nodül-pozitif
    hastalar arasında** yapılır: burada kaynak, etiketten bağımsız bir "cihaz/kurum"
    değişkenidir. Yüksek doğruluk ⇒ temsil **fiziksel/kurumsal imza** taşıyor ⇒
    alan kaymasına açık.

Ön-kayıtlı beklenti (falsifiye edilebilir):
    H0: temsiller kaynağı şans düzeyinde ayırt eder (fizik kodlamıyor).
    H1: temsiller kaynağı belirgin biçimde ayırt eder (AUC > 0.70) ⇒ alan kayması riski.
    Ek karşılaştırma: yalnızca spacing/boyut (2-6 sayısal meta-veri) ne kadar ayırt ediyor?

Protokol:
    * Hasta düzeyi; pozitif hastalar; 5 zon vektörünün ortalaması.
    * Bölme: `patient_splits` train → val (hasta-düzeyi ayrık). **Test hastaları kullanılmaz.**
    * Sınıflandırıcılar: Çok-sınıflı Lojistik Regresyon + HistGradientBoosting.
    * Metrikler: doğruluk, makro one-vs-rest ROC-AUC, çoğunluk-sınıfı referansı.

Çıktı:
    data/external_vindr/a2_source_predictability_report.json
    artifacts/runs/A2_source_predictability/manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TDA_FEATURES = PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy"
RESNET_FEATURES = PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy"
FEATURES_META = PROJECT_ROOT / "data/processed/features_v2/features_metadata.csv"
PATIENT_SPLITS = PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv"
INVENTORY = PROJECT_ROOT / "data/processed/metadata_inventory_v2.csv"
REPORT_JSON = PROJECT_ROOT / "data/external_vindr/a2_source_predictability_report.json"
SEED = 42


def patient_level(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Zon vektörlerini hasta düzeyinde ortalar."""
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["patient_id"] = meta["patient_id"].values
    return df.groupby("patient_id").mean()


def evaluate(model, X_tr, y_tr, X_va, y_va, classes) -> dict:
    from sklearn.pipeline import Pipeline

    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)
    pred = model.predict(X_va)
    aucs = {}
    for i, c in enumerate(classes):
        if (y_va == c).any() and (y_va != c).any():
            aucs[str(c)] = float(roc_auc_score((y_va == c).astype(int), proba[:, i]))
    acc = float(accuracy_score(y_va, pred))
    baseline = float(pd.Series(y_va).value_counts(normalize=True).max())
    return {
        "accuracy": acc,
        "majority_baseline_accuracy": baseline,
        "macro_ovr_roc_auc": float(np.mean(list(aucs.values()))) if aucs else float("nan"),
        "per_class_ovr_auc": aucs,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "class_support_val": {str(c): int((y_va == c).sum()) for c in classes},
        "pipeline": type(model).__name__ if not isinstance(model, Pipeline) else
        " -> ".join(type(s).__name__ for _, s in model.steps),
    }


def lr_pipeline():
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=SEED, class_weight="balanced")),
    ])


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    meta = pd.read_csv(FEATURES_META)
    inv = pd.read_csv(INVENTORY)[["patient_id", "source_dataset", "label", "spacing_y",
                                  "spacing_x", "orig_h", "orig_w"]]
    splits = pd.read_csv(PATIENT_SPLITS)[["patient_id", "split"]]

    info = inv.merge(splits, on="patient_id", how="inner")
    pos = info[info["label"] == 1].copy()
    pos = pos[~pos["source_dataset"].isin(["node21_control", "unknown"])]
    classes = sorted(pos["source_dataset"].unique())
    print(f"Pozitif hastalar: {len(pos)} | sınıflar: {classes}")
    print(pos.groupby(["split", "source_dataset"]).size().unstack(fill_value=0).to_string())

    tr_ids = set(pos.loc[pos["split"] == "train", "patient_id"])
    va_ids = set(pos.loc[pos["split"] == "val", "patient_id"])
    print(f"Train pozitif: {len(tr_ids)} | Val pozitif: {len(va_ids)} (test hastaları KULLANILMADI)")

    tables = {}
    for name, path in (("tda_pi_800", TDA_FEATURES), ("resnet18_512", RESNET_FEATURES)):
        X = np.load(path)
        agg = patient_level(X, meta)
        agg = agg.loc[agg.index.intersection(pos["patient_id"].values)]
        sub = pos.set_index("patient_id").loc[agg.index]
        tr_m = agg.index.isin(tr_ids)
        va_m = agg.index.isin(va_ids)
        Xtr, ytr = agg[tr_m].values, sub.loc[tr_m, "source_dataset"].values
        Xva, yva = agg[va_m].values, sub.loc[va_m, "source_dataset"].values
        tables[name] = {
            "logreg": evaluate(lr_pipeline(), Xtr, ytr, Xva, yva, classes),
            "hgb": evaluate(HistGradientBoostingClassifier(random_state=SEED),
                            Xtr, ytr, Xva, yva, classes),
        }
        print(f"[{name}] LR makro AUC={tables[name]['logreg']['macro_ovr_roc_auc']:.4f} "
              f"acc={tables[name]['logreg']['accuracy']:.4f} | "
              f"HGB AUC={tables[name]['hgb']['macro_ovr_roc_auc']:.4f} "
              f"acc={tables[name]['hgb']['accuracy']:.4f}")

    # Referans: yalnızca meta-veri (spacing/boyut)
    mcols = ["spacing_y", "spacing_x", "orig_h", "orig_w"]
    m = pos.set_index("patient_id")
    tr_m = m.index.isin(tr_ids)
    va_m = m.index.isin(va_ids)
    meta_res = evaluate(lr_pipeline(), m.loc[tr_m, mcols].values, m.loc[tr_m, "source_dataset"].values,
                        m.loc[va_m, mcols].values, m.loc[va_m, "source_dataset"].values, classes)
    print(f"[meta-only spacing/size] makro AUC={meta_res['macro_ovr_roc_auc']:.4f} "
          f"acc={meta_res['accuracy']:.4f}")

    report = {
        "experiment": "A2 — kaynak (cihaz/kurum) ayrıştırılabilirliği",
        "question": "Temsillerimiz etiketten bağımsız olarak hangi cihazdan geldiğini kodluyor mu?",
        "design": {
            "scope": "yalnızca nodül-pozitif hastalar (etiketten bağımsız cihaz değişkeni)",
            "level": "hasta düzeyi (5 zon ortalaması)",
            "split": "patient_splits train → val; TEST HASTALARI KULLANILMADI",
            "classes": classes,
        },
        "chance_reference": tables["tda_pi_800"]["logreg"]["majority_baseline_accuracy"],
        "representations": tables,
        "metadata_only_baseline": meta_res,
        "interpretation": (
            "Yüksek AUC, temsilin fiziksel/kurumsal imza taşıdığını gösterir; "
            "bu imza iç testte etiketle ilişkili olduğu için başarımı şişirir, "
            "dış merkezde ise çöker. Hedef: label AUC'sini korurken bu AUC'yi düşürmek."
        ),
        "caveat": (
            "Kaynaklar nodül boyutu/görünüş dağılımı bakımından da farklı olabilir; "
            "ayrıştırılabilirliğin bir kısmı fizik değil anatomi/hastalık spektrumu olabilir. "
            "Bu yüzden 'spacing-only' referansı ve boyut-eşleştirilmiş kontroller birlikte okunmalı."
        ),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {REPORT_JSON}")

    try:
        from node21_tda.experiment import new_run_dir, write_manifest

        run_dir = new_run_dir("A2", "source_predictability")
        write_manifest(
            run_dir, "A2",
            config={"classes": classes, "level": "patient", "scope": "positives_only"},
            inputs=[TDA_FEATURES, RESNET_FEATURES, FEATURES_META, PATIENT_SPLITS, INVENTORY],
            seed=SEED,
            metrics={
                "tda_logreg_macro_auc": tables["tda_pi_800"]["logreg"]["macro_ovr_roc_auc"],
                "resnet_logreg_macro_auc": tables["resnet18_512"]["logreg"]["macro_ovr_roc_auc"],
                "metadata_only_macro_auc": meta_res["macro_ovr_roc_auc"],
            },
            decision="DIAGNOSTIC",
            notes="Kaynak ayrıştırılabilirliği (fiziksel imza) teşhisi; test hastaları kullanılmadı.",
        )
        print(f"Manifest: {run_dir / 'manifest.json'}")
    except Exception as exc:
        print(f"Manifest yazılamadı: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
