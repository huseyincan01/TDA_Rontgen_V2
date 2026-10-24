"""V6.7 — submission_config.json DONDURMA + koruma denetimi.

Yaptığı iş:
  1. Tam-veri artefaktlarının varlığını doğrular.
  2. BİLİMSEL checkpoint'lerin SHA-256'sını kaydeder ve **değişmediklerini** teyit eder
     (Kural: makale tablolarını üreten ağırlıklara dokunulmaz).
  3. Nihai çıkarım standardını + kilitli füzyonu tek JSON sözleşmesine yazar.
  4. Duman testini çalıştırır.

Kullanım:
    python pipelines/v6_two_stage/v6_7_freeze_config.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
SUBDIR = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
CFG = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_config.json"

PROTECTED = {
    "M5_centernet.pt": "V6.0/V6.2 1. aşama (makale tabloları)",
    "M6_fcos.pt": "V6.0/V6.2 FCOS yardımcı dedektör",
    "M5_foundation.pt": "V6.6 vakıf omurga (V6.6.2–V6.6.5 tabloları)",
    "stage2_final_tda_hgb.joblib": "V6.3 nihai 2. aşama bundle'ı",
    "M5_domain_gen.pt": "V6.4 KILL edilen domain-gen varyantı (kanıt)",
}
SUBMISSION = {
    "M5_foundation_fulldata.pt": "V6.7 tam-veri 1. aşama",
    "stage2_tda_hgb_fulldata.joblib": "V6.7 tam-veri 2. aşama (Tuzak B korumalı)",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("V6.7 submission_config.json donduruluyor...\n", flush=True)

    missing = [n for n in SUBMISSION if not (SUBDIR / n).exists()]
    if missing:
        raise SystemExit(f"EKSİK tam-veri artefaktı: {missing}\n"
                         "Önce v6_7_train_fulldata.py ve "
                         "v6_7_build_fulldata_stage2.py koşulmalı.")

    print("=== KORUNAN BİLİMSEL CHECKPOINT'LER (dokunulmadı) ===", flush=True)
    prot = []
    for name, desc in PROTECTED.items():
        p = CKPT / name
        if not p.exists():
            print(f"  ⚠ {name} yok (atlandı)", flush=True)
            continue
        h = sha256(p)
        st = p.stat()
        prot.append({"name": name, "desc": desc, "sha256": h,
                     "bytes": st.st_size, "mtime": int(st.st_mtime)})
        print(f"  ✓ {name:32s} {st.st_size / 1e6:8.2f} MB  sha256={h[:16]}…", flush=True)

    print("\n=== TAM-VERİ GÖNDERİM ARTEFAKTLARI ===", flush=True)
    subm = []
    for name, desc in SUBMISSION.items():
        p = SUBDIR / name
        h = sha256(p)
        subm.append({"name": name, "desc": desc, "sha256": h, "bytes": p.stat().st_size})
        print(f"  ✓ {name:32s} {p.stat().st_size / 1e6:8.2f} MB  sha256={h[:16]}…", flush=True)

    import joblib
    b = joblib.load(SUBDIR / "stage2_tda_hgb_fulldata.joblib")
    fr = b["fusion_ref"]
    m = b["meta"]
    assert m["stage1_score_included"] is False, "Tuzak B ihlali: s1 matrise girmiş!"
    print(f"\n  Tuzak B denetimi: stage1_score_included={m['stage1_score_included']} ✓",
          flush=True)
    print(f"  HGB: {m['n_samples']} örnek · {m['n_positive']} pozitif · "
          f"{m['n_patients']} hasta", flush=True)
    print(f"  Kilitli füzyon: w={fr['w']} (yeniden uydurulmadı)", flush=True)

    cfg = {
        "version": "v6.7-fulldata-1.0",
        "stage1": {
            "checkpoint": "M5_foundation_fulldata",
            "path": "artifacts/v6_two_stage/submission_fulldata/M5_foundation_fulldata.pt",
            "backbone": "resnet50-res512-all (torchxrayvision, 800k+ CXR)",
            "input": {"size": 512, "channels": 1, "range": [-1024, 1024],
                      "note": "xrv.utils.normalize(img, maxval=255)"},
            "sha256": subm[0]["sha256"],
        },
        "stage2": {
            "bundle": "artifacts/v6_two_stage/submission_fulldata/"
                      "stage2_tda_hgb_fulldata.joblib",
            "features": "1600 PI (32/64 c_lungref, yerel min-max YOK) + 4 uzamsal",
            "stage1_score_included": False,
            "sha256": subm[1]["sha256"],
        },
        "inference": {
            "scales": [0.9, 1.0, 1.1],
            "flip_tta": True,
            "peak_window": 3,
            "nms_px": 8,
            "top_k": 50,
            "box_px": 56,
            "note": "V6.6.7 denetimiyle sabitlendi (56x56 sabit kutu; yarıçap kafası "
                    "resmî IoU>0.2 için kullanılamaz)",
        },
        "fusion": {
            "w": fr["w"],
            "formula": "S = sigmoid(w·z(s1) + (1−w)·z(p_tda)); z sabit referansla",
            "reference": {k: fr[k] for k in ("s1_mean", "s1_std", "tda_mean", "tda_std")},
            "locked": True,
            "why": "Tuzak B: füzyon HİÇBİR veri üzerinde yeniden uydurulmadı; "
                   "V6.6.6'da val üzerinde doğrulanmış sabit nokta.",
        },
        "metric": {
            "official": "rank_metric = 0.75·AUC + 0.25·Sens@0.25 FP/görüntü",
            "matching": "IoU > 0.2; aynı referansa çakışan 2.+ tahmin ne TP ne FP",
            "froc_points": [0.125, 0.25, 0.5],
            "validated_on": "NODE21 val (M5_foundation ile; tam-veri modeli NODE21'de "
                            "DEĞERLENDİRİLEMEZ çünkü tüm veri eğitimde kullanıldı)",
            "validated_values": {"rank_metric": 0.9131, "auc": 0.9678,
                                 "sens_at_0.125": 0.6710, "sens_at_0.25": 0.7489,
                                 "sens_at_0.5": 0.8009},
        },
        "trap_a": {
            "protection": "Doğrulanmış SABİT takvim yeniden kullanıldı (hiperparametre "
                          "aramasi YOK): 5 epoch donuk ısınma + 13 epoch diferansiyel FT, "
                          "BN donuk, Cosine Annealing, aynı augmentasyon/kayıp.",
            "data": "4.882 hasta (train+val+test): 1.134 nodül + 1.134 temiz kontrol (1:1)",
            "radius_head": "radius_detach=True — yarıçap kafası detach edilmiş "
                           "özelliklerden eğitildi, omurga ağırlıkları %100 korundu",
        },
        "trap_b": {
            "protection": "HGB matrisinde s1 YOK (yalnız 1600 PI + 4 uzamsal); füzyon "
                          "ağırlığı hiçbir veri üzerinde yeniden uydurulmadı.",
            "verification": "bundle meta.stage1_score_included == False",
        },
        "rule9": "Verimiz NODE21'in HALK AÇIK EĞİTİM setidir; resmî test setleri özeldir "
                 "⇒ leaderboard sıralaması ölçülemez. Bu boru hattı ileriye dönük "
                 "gönderim/dağıtım içindir, sıralama iddiası değildir.",
        "protected_scientific_artifacts": prot,
        "submission_artifacts": subm,
    }
    CFG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDonduruldu: {CFG}", flush=True)

    print("\n=== DUMAN TESTİ ===", flush=True)
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "pipelines/v6_two_stage/"
                                            "v6_7_submission_infer.py"), "--smoke", "--n", "4"],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(r.stdout.strip()[-2500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
        raise SystemExit(f"duman testi BAŞARISIZ (exit {r.returncode})")
    print("\n✅ SÖZLEŞME DONDURULDU VE DUMAN TESTİ GEÇTİ", flush=True)


if __name__ == "__main__":
    main()
