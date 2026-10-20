"""V4 harici doğrulama (VinDr-CXR) raporlarının şema ve tutarlılık testleri.

Bu testler, daha önce belgelerle çelişen iddiaların (mm cinsinden alt gruplar,
farklı füzyon ağırlığı, kayıtsız/denetimsiz koşular) tekrar ortaya çıkmasını
engeller. Raporlar yoksa testler atlanır (harici koşuya bağlı).

Kullanım:
    python tests/test_v4_external_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = PROJECT_ROOT / "data" / "external_vindr"
ARMS = ["linear", "matched", "matched_scalenorm"]
EXPECTED_FUSION_A = 0.45
E0_6_THRESHOLD = 0.7172159718111142


class Skip(Exception):
    pass


def load(name: str):
    p = EXT_DIR / name
    if not p.exists():
        raise Skip(f"{name} bulunamadı (V4 koşusu çalıştırılmamış).")
    return json.loads(p.read_text(encoding="utf-8"))


def test_summary_declares_image_level_and_no_vindr_training():
    summary = load("vindr_summary_report.json")
    assert summary["cohort"]["level"].startswith("image")
    assert summary["cohort"]["n_images"] == 400
    proto = summary["protocol"]
    assert proto["training_on_vindr"] is False
    assert proto["fusion_a_locked"] == EXPECTED_FUSION_A
    assert "a*resnet" in proto["fusion_definition"]
    assert proto["resnet_val_consistent_with_e0_6"] is True
    assert proto["resnet_val_auc_delta_vs_e0_6"] < 5e-3
    assert "rel_size" in proto["subgroup_definition"]
    print("  [PASS] test_summary_declares_image_level_and_no_vindr_training")


def test_arm_reports_schema_and_consistency():
    for arm in ARMS:
        rep = load(f"vindr_report_{arm}.json")
        assert rep["arm"] == arm
        assert rep["a_locked"] == EXPECTED_FUSION_A

        for key in ("tda", "resnet", "fusion"):
            auc = rep["auc"][key]
            assert 0.0 <= auc <= 1.0, (arm, key, auc)
            lo, hi = rep["ci95"][key]
            assert lo <= auc <= hi, (arm, key, auc, lo, hi)

        assert len(rep["a_sweep"]) == 21
        assert "0.45" in rep["a_sweep"]
        assert abs(rep["a_sweep"]["1.00"] - rep["auc"]["resnet"]) < 1e-6

        subs = rep["subgroups_scale_free"]
        assert set(subs) == {"kucuk_tercile", "orta_tercile", "buyuk_tercile"}
        for name, sg in subs.items():
            assert "mm" not in json.dumps(sg, ensure_ascii=False).lower()
            assert "tercile" in name or "tercile" in sg["definition"]

        keys = list(rep["thresholds"])
        assert any(f"{E0_6_THRESHOLD:.4f}" in k for k in keys), keys
    print("  [PASS] test_arm_reports_schema_and_consistency")


def test_metadata_shortcut_report_is_bounded():
    rep = load("a0_a1_metadata_report.json")
    assert rep["n_patients"] == 4882
    auc = rep["metrics"]["max_val_roc_auc"]
    assert 0.5 < auc < 1.0
    assert rep["decision"] in {"TEMIZ", "KISMI_KISAYOL", "CIDDI_KISAYOL"}
    print("  [PASS] test_metadata_shortcut_report_is_bounded")


def main() -> int:
    print("=" * 60)
    print("V4 HARİCİ DOĞRULAMA RAPOR TESTLERİ")
    print("=" * 60)
    tests = [
        test_summary_declares_image_level_and_no_vindr_training,
        test_arm_reports_schema_and_consistency,
        test_metadata_shortcut_report_is_bounded,
    ]
    for t in tests:
        try:
            t()
        except Skip as exc:
            print(f"  [SKIP] {t.__name__}: {exc}")
    print("=" * 60)
    print("TAMAMLANDI")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
