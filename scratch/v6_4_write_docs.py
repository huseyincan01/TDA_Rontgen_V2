"""V6.4 bölümünü experiments.md ve ablation_matrix.md'ye ekler (idempotent)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXP = r"""
### V6.4 — Domain stabilizasyonu + NODE21 TEST mühürü (tek sefer)

**Adım 1 — VinDr test-time stabilizasyon (eğitimsiz):** Aşama-1 görüntü AUC (top-1): baseline **0,5443** · SNR-diff 0,5440 · SNR-z 0,5453 ⇒ **SNR etkisiz**. **AdaBN 0,6332 (+0,089)**, recall@50 0,907 ⇒ M5 için gerçek kazanç; ancak M6 AdaBN'ı bozduğu için M5+M6 füzyonu 0,512 → 0,441 ⇒ **AdaBN yalnız M5'e uygulanmalı**.
*(Not: V6.3'teki 0,4618 noisy-OR toplamasıydı; burada top-1 — fark bundan.)*

**Adım 1b — M6'sız AdaBN'li M5 + lokal TDA (VinDr):** AdaBN-M5 Stage-1 **0,6346** · lokal TDA-HGB 0,6082 · **e_fusion_tda 0,6471** · V5.3 global 0,6283 · **V5.3 + birleşik 0,6494** (r = 0,460). ⇒ **M6 çıkarılınca AdaBN V5.3'ü geçiyor** (+0,019); birleşim en iyi (+0,021). AdaBN yalnız M5'e uygulanmalı.
*(Not: V6.3'teki 0,4618 noisy-OR toplamasıydı; burada top-1 — fark bundan.)*

**Adım 2 — Fiziksel domain augmentation ile M5_domain_gen:** NODE21 val `M5_heatTTA@50` 0,978 → **0,919**, `M5M6TTA@50` 0,983 → **0,971**; VinDr AUC 0,544 → **0,516**. ⇒ **KILL**: in-domain kayıp var, transfer kazancı yok (gamma/CLAHE/jitter doku istatistiğini bozuyor).

**Adım 3 — NODE21 TEST mühürü (734 hasta / 229 lezyon; tek sefer, sıfır ayar):**
| Model | Test hasta AUC | Test FROC @0.5/1/2/4 FP |
| :--- | :---: | :---: |
| V6 lokal (`e_fusion_tda`) | 0,9464 (Noisy-OR) | **0,480 / 0,555 / 0,642 / 0,738** |
| V5.2 global (c_lungref, 5-zon) | 0,9612 | — |
| **Global + Lokal birleşik** | **0,9876** (r = 0,376) | — |

⇒ **Test setindeki en yüksek hasta AUC'si**: v3 headline 0,9516 ve V5.2 0,9612'nin üzerinde. Küresel ve lokal temsiller r=0,38 ile yeterince dik ⇒ birleşim **gerçek tamamlayıcılık** sağlıyor. Test seti bu mühürle **bir kez** kullanıldı; eşik/ağırlık seçimi yapılmadı.
* **Kayıt:** `artifacts/v6_two_stage/v6_4_domain_and_test_seal.json` · `checkpoints/M5_domain_gen.pt`
"""

ABL = r"""
### V6.4 — Domain stabilizasyonu + test mühürü
| Adım | Sonuç | Karar |
| :--- | :--- | :---: |
| SNR (diff/z) | AUC 0,544 → 0,544/0,545 (etkisiz) | KILL |
| **AdaBN (M5)** | **0,544 → 0,633** (+0,089); recall@50 0,907 | KEEP (yalnız M5) |
| AdaBN (M6) | M6 recall 0,893 → 0,740; füzyon 0,512 → 0,441 | KILL |
| **AdaBN-M5 + lokal TDA (M6'sız)** | Stage-1 0,6346 · TDA 0,6082 · **e_fusion 0,6471** · **V5.3+birleşik 0,6494** (r=0,46) | **V5.3'ü geçti** |
| Domain-gen M5 (aug) | NODE21 val @50: 0,978 → **0,919**; VinDr 0,544 → 0,516 | **KILL** |
| **TEST mühürü** | V6 lokal **0,9464** · V5.2 global **0,9612** · **birleşik 0,9876** (r=0,376) | **KAYIT** |
"""


def append(path: Path, text: str, marker: str) -> None:
    s = path.read_text(encoding="utf-8")
    if marker in s:
        print(f"{path.name}: '{marker}' zaten var — atlandı")
        return
    path.write_text(s.rstrip() + "\n" + text, encoding="utf-8")
    print(f"{path.name}: eklendi")


if __name__ == "__main__":
    append(ROOT / "experiments.md", EXP, "### V6.4")
    append(ROOT / "ablation_matrix.md", ABL, "### V6.4")
