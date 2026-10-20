"""V5.1b raporuna yorum bloğu ekler (kural çıktısı korunur, okunuşu netleşir)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/external_vindr/v5_1b_fixed_reference_report.json"

INTERP = {
    "summary": (
        "Kırpıntı-başına uyarlanabilir pencere, füzyon val kazancının BİR KISMINI açıklıyor "
        "(-0.0078); nicelleme düzeltmesinin kendisi baskın ve gerçek."
    ),
    "adaptive_window_contribution": {
        "fusion_val_auc": -0.0078,
        "note": "0.0127'lik kazancın ~%60'ı; topolojik değil, histogram/pencere kaynaklı "
                "(kırpıntının kendi p99'u lezyondan hafifçe etkilenir). Test zamanında "
                "hesaplanabilir olduğu için hile değildir; ancak 'topolojik parmak izi' "
                "iddiası için ayrıştırılmalıdır.",
    },
    "quantization_fix_survives": {
        "tda_ensemble_v2": 0.8811,
        "tda_ensemble_c_lungref": 0.9335,
        "delta": 0.0524,
        "share_of_a_density_gain": 0.74,
        "note": "+0.0711'lik TDA-ensemble kazancının %74'ü pencere olmadan da var.",
    },
    "recommended_reporting": {
        "conservative_fusion_val_auc": 0.9778,
        "per_crop_fusion_val_auc": 0.9856,
        "text": "Makalede muhafazakâr sayı c_lungref (0.9778) olarak raporlanmalı; "
                "0.9856 pencere katkısını içerir ve bu ayrım açıkça yazılmalıdır.",
    },
    "decision_final": "KEEP korunur (çiftleşmesiz varyantta da TDA ensemble +0.0524, LOO 0.7897).",
}


def main() -> None:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    rep["interpretation"] = INTERP
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("interpretation eklendi.")
    print("  kural çıktısı :", rep.get("decision"))
    print("  nihai         :", INTERP["decision_final"])


if __name__ == "__main__":
    main()
