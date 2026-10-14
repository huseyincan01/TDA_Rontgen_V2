"""Train Persistence Diyagramlarının Gerçek Doğum (Birth) ve Persistence Dağılım Analizi.

Hızlı ve istatistiksel olarak güvenilir analiz: 40 temsili train hastası (20 nodüllü, 20 kontrol, 200 zon).
Her adımda flush=True ile anlık loglama yapar.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path("c:/Users/Huseyincan/codes/rontgen_V2")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.features.extractor import ZoneTDAExtractor

ZONES_DIR = PROJECT_ROOT / "data" / "processed" / "zones" / "patients"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "patient_splits.csv"

def analyze_distribution(data: list[float], name: str) -> dict:
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "name": name,
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50_median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }

def main():
    print("Train hasta listesi okunuyor...", flush=True)
    patients_df = pd.read_csv(SPLITS_PATH)
    train_df = patients_df[patients_df["split"] == "train"]

    # 20 nodüllü + 20 sağlıklı train hastası (toplam 200 zon)
    train_pos = train_df[train_df["patient_label"] == 1].sample(n=20, random_state=42)
    train_neg = train_df[train_df["patient_label"] == 0].sample(n=20, random_state=42)
    selected_patients = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=42)

    print(f"Toplam seçilen train hastası: {len(selected_patients)} (20 Pozitif, 20 Kontrol = 200 zon)", flush=True)

    extractor = ZoneTDAExtractor()

    h0_births = []
    h0_persistences = []
    h1_births = []
    h1_persistences = []

    zone_names = [
        "zone_1_sag_ust",
        "zone_2_sag_orta",
        "zone_3_sag_alt",
        "zone_4_sol_ust",
        "zone_5_sol_alt",
    ]

    total_zones = 0
    success_zones = 0

    print("Zon MHA kesitleri okunuyor ve persistence diyagramları hesaplanıyor...", flush=True)
    for idx, (_, row) in enumerate(selected_patients.iterrows()):
        pid = row["patient_id"]
        p_dir = ZONES_DIR / pid

        for zid, zname in enumerate(zone_names, start=1):
            mha_file = p_dir / f"{zname}.mha"
            if not mha_file.exists():
                continue

            total_zones += 1
            itk_img = sitk.ReadImage(str(mha_file))
            arr = sitk.GetArrayFromImage(itk_img).squeeze()

            filtration = extractor.prepare_filtration(arr)
            if filtration is None:
                continue

            h0, h1 = extractor.compute_diagrams(filtration)
            success_zones += 1

            if len(h0) > 0:
                h0_births.extend(h0[:, 0].tolist())
                h0_persistences.extend((h0[:, 1] - h0[:, 0]).tolist())

            if len(h1) > 0:
                h1_births.extend(h1[:, 0].tolist())
                h1_persistences.extend((h1[:, 1] - h1[:, 0]).tolist())

        if (idx + 1) % 5 == 0:
            print(f"  [{idx + 1}/{len(selected_patients)}] hasta tamamlandı (Toplam H0: {len(h0_births):,}, H1: {len(h1_births):,})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("TRAIN PERSISTENCE DİYAGRAMLARI DAĞILIM İSTATİSTİKLERİ", flush=True)
    print("=" * 80, flush=True)

    h0_b_stat = analyze_distribution(h0_births, "H0 Doğum (Birth)")
    h0_p_stat = analyze_distribution(h0_persistences, "H0 Persistence (Death - Birth)")
    h1_b_stat = analyze_distribution(h1_births, "H1 Doğum (Birth)")
    h1_p_stat = analyze_distribution(h1_persistences, "H1 Persistence (Death - Birth)")

    for stat in [h0_b_stat, h0_p_stat, h1_b_stat, h1_p_stat]:
        print(f"\n{stat['name']} (Toplam Nokta Sayısı: {stat['count']:,}):", flush=True)
        print(f"  Min: {stat['min']:.1f} | Max: {stat['max']:.1f} | Ortalama: {stat['mean']:.1f}", flush=True)
        print(f"  %1:  {stat['p01']:.1f} | %5:  {stat['p05']:.1f} | %10: {stat['p10']:.1f}", flush=True)
        print(f"  %25: {stat['p25']:.1f} | %50 (Medyan): {stat['p50_median']:.1f} | %75: {stat['p75']:.1f}", flush=True)
        print(f"  %90: {stat['p90']:.1f} | %95: {stat['p95']:.1f} | %99: {stat['p99']:.1f}", flush=True)

    out_report = {
        "n_patients": len(selected_patients),
        "n_zones": success_zones,
        "h0_birth": h0_b_stat,
        "h0_persistence": h0_p_stat,
        "h1_birth": h1_b_stat,
        "h1_persistence": h1_p_stat,
    }

    report_path = PROJECT_ROOT / "data" / "processed" / "zones" / "reports" / "train_diagram_distribution_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)
    print(f"\nRapor kaydedildi: {report_path}", flush=True)

if __name__ == "__main__":
    main()
