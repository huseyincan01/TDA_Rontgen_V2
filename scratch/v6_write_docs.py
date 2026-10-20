"""V6 nihai bölümünü experiments.md ve ablation_matrix.md'ye ekler (mevcut içeriğe dokunmadan)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V6_EXPERIMENTS = r"""
---

## 9. V6 — İki Aşamalı Lokalizasyon (Aday Üretici + Lokal TDA FP Eleme)

> **Kapsam:** v3/v5 programından **bağımsız** yeni program. Disk kuralı: görüntü/kırpıntı **yazılmaz**; yalnız hafif JSON + model checkpoint. Kod `pipelines/v6_two_stage/`, kayıtlar `artifacts/v6_two_stage/`.
> **Hedef:** 2. aşamadaki lokal TDA sınıflandırıcıya geçmeden, K=10–50 aday içinde **%98–99** nodül yakalama.

### V6.0 — 1. Aşama aday üretici (oracle Recall@K, val 231 lezyon)

| Yöntem | K10 | K20 | K30 | K50 | ms/görüntü |
| :--- | :---: | :---: | :---: | :---: | :---: |
| M1 LoG+CLAHE (eğitimsiz) | 0,069 | 0,143 | 0,173 | 0,229 | 1929 |
| M2 DoH+damar bastırma | 0,052 | 0,117 | 0,229 | 0,325 | 697 |
| M3 radyal simetri | 0,056 | 0,108 | 0,143 | 0,216 | 286 |
| M4 bilateral asimetri | 0,004 | 0,009 | 0,013 | 0,069 | 1347 |
| **M5 CenterNet (ResNet18-stride2)** | **0,866** | **0,961** | **0,974** | **0,978** | **26** |
| M6 FCOS-lite (focal objectness) | 0,481 | 0,576 | 0,654 | 0,758 | 54 |
| **M5 + heatTTA** | **0,909** | **0,965** | 0,970 | 0,978 | ~60 |
| **M5+M6 (RRF)** | 0,818 | 0,918 | 0,974 | **0,983** | ~150 |
| M7 hibrit (M5 %70 + M2 %30) | 0,784 | 0,939 | 0,970 | 0,978 | 686 |

**Faz 1 dersleri:** (i) eğitimsiz filtreler K50'de bile %7–33'te kalıyor (kaburga/hilus parlaklığı sıralamayı domine ediyor); (ii) **pozitif-only eğitim tuzağı doğrulandı** — 1:1 temiz kontrol (boş hedef) olmadan model her görüntüde tepe üretiyor (0,30 → kontrollerle 0,0 → 15 epoch'ta 0,97); (iii) M6'nın ilk hali dengesiz BCE yüzünden 0,017'de çöküyordu, **focal objectness** ile 0,76'ya çıktı; (iv) **stride-2 (256×256 çıktı = 1024'e stride-4)** ve heatmap eşiğinin 0,05→0,01'e indirilmesi M5'i 0,853→0,974 taşıdı.

### V6.1/V6.2 — 2. Aşama: lokal TDA + uzamsal öncül + FP eleme (val, 734 hasta)

Temsil: aday merkezli **32×32 (çekirdek) + 64×64 (çevre)** H0/H1 PI (1600 boyut, **c_lungref ölçeği korunur**; yerel min-max yok) + 4 uzamsal özellik (rel_x, rel_y, hilus, plevra) + 64×64 yama.

| Ayar | Model | Hasta AUC (Top-1 / Top-3 / Noisy-OR) | FROC @0.5/1/2/4 FP |
| :--- | :--- | :---: | :---: |
| M5M6TTA·K50 | (a) yalnız Aşama-1 | 0,688 / 0,709 / 0,711 | 0,281 / 0,403 / 0,520 / 0,623 |
| M5M6TTA·K50 | (b) **lokal TDA-HGB** | **0,882 / 0,922 / 0,926** | 0,312 / 0,411 / 0,515 / 0,615 |
| M5M6TTA·K50 | (c) yama CNN | 0,650 / 0,658 / 0,652 | 0,052 / 0,117 / 0,186 / 0,255 |
| M5M6TTA·K50 | (d) tam füzyon | 0,906 / 0,942 / 0,951 | 0,511 / 0,610 / 0,680 / 0,766 |
| **M5M6TTA·K50** | **(e) e_fusion_tda** | **0,911 / 0,945 / 0,955** | **0,528 / 0,615 / 0,688 / 0,771** |
| **M5_heatTTA·K30** | **(f) f_hierarchical** | **0,923 / 0,933 / 0,936** | **0,589 / 0,658 / 0,753 / 0,857** |

* **Yama CNN mimariden ÇIKARILDI** (AUC 0,646; onsuz füzyon eşit/daha iyi ve daha ucuz) ⇒ *"lokal ölçekte ayırt edici bilgi piksel dokusunda değil, topolojide"* bir kez daha doğrulandı.
* **Doğrusal olmayan topluluk (SVM+MLP rank-ortalaması) reddedildi**: HGB tek başına 0,919 iken topluluk 0,656 (zayıf üyeler güçlü HGB'yi aşağı çekti).
* **FROC protokolü düzeltildi:** post-fusion NMS + duplikasyon **IGNORE** (LUNA16/NODE21 stili) ⇒ hiyerarşik kalibrasyon FROC'u tüm eşiklerde yukarı taşıdı (0,857 @4 FP).
* **Hiyerarşik kalibrasyon:** `S_final = p_aday × P_NoisyOR(hasta)`.

### V6.3 — VinDr-CXR sıfır-eğitim bileşen otopsisi (400 görüntü)

| Bileşen | VinDr AUC | NODE21 val AUC |
| :--- | :---: | :---: |
| (a) Aşama-1 dedektör | **0,4618** | 0,7111 |
| (b) lokal TDA-HGB | **0,5901** | 0,9261 |
| (c) e_fusion_tda | **0,5911** | **0,9546** |
| V5.3 global 5-zon (0,45·ResNet+0,55·TDA) | **0,6283** | 0,9821 |
| r(V5.3, V6.2) | 0,359 | 0,385 |
| Birleşik (eşit rank-ort.) | 0,6214 | **0,9921** |

**VinDr bbox recall@K (3× indirgeme doğrulandı, fit=1,00; 150 pozitif):** M5_heatTTA **K30=0,880 · K50=0,900** · M5M6TTA K30=0,873 · K50=0,893.

**Kritik teşhis:** VinDr'de **aday üretici nodülleri buluyor** (K50 recall 0,90) ama **görüntü düzeyi güven skoru şans altı** (AUC 0,46) ⇒ darboğaz **lokalizasyon değil, skor kalibrasyonu** (normal görüntülerde de yüksek güvenli adaylar var). Lokal TDA (0,590) en taşınabilir V6 bileşeni; ancak V5.3 global (0,6283) hâlâ en iyi ve birleşim onu geçmiyor (0,6214).

### V6 nihai modeller (donduruldu)
* **Maks AUC:** `M5M6TTA · K50 · e_fusion_tda` — hasta AUC **0,955** (Noisy-OR), FROC 0,528/0,615/0,688/0,771.
* **FROC-optimal:** `M5_heatTTA · K30 · f_hierarchical` — FROC **0,589/0,658/0,753/0,857**.
* Checkpoint'ler: `artifacts/v6_two_stage/checkpoints/{M5_centernet.pt, M6_fcos.pt, stage2_final_tda_hgb.joblib}` (yama CNN yok).
* Raporlar: `v6_0_candidate_benchmark.json` · `v6_0_miss_autopsy.json` · `v6_2_froc_and_transfer.json` · `v6_3_final_report.json` · `v6_3_vindr_recall.json`.
"""

V6_ABLATION = r"""
---

## 13. V6 — İki Aşamalı Lokalizasyon (özet)

**Kod:** `pipelines/v6_two_stage/` · **Kayıt:** `artifacts/v6_two_stage/` · ayrıntı: [experiments.md §9](experiments.md) · **disk kuralı:** görüntü/kırpıntı yazımı yok.

### 1. Aşama (aday üretici, oracle Recall@K, val)
| Model | K20 | K30 | K50 | Not |
| :--- | :---: | :---: | :---: | :--- |
| Eğitimsiz en iyi (M2 DoH) | 0,117 | 0,229 | 0,325 | yetersiz |
| **M5 CenterNet** | **0,961** | **0,974** | **0,978** | 26 ms; stride-4/1024 |
| M6 FCOS-lite | 0,576 | 0,654 | 0,758 | focal objectness şart |
| M5+heatTTA | 0,965 | 0,970 | 0,978 | K≤30 için en iyi |
| **M5+M6 (RRF)** | 0,918 | 0,974 | **0,983** | K≥30 için en iyi |

### 2. Aşama (lokal TDA + FP eleme, val 734 hasta)
| Model | Hasta AUC (Noisy-OR) | FROC @4 FP |
| :--- | :---: | :---: |
| yalnız Aşama-1 | 0,711 | 0,623 |
| lokal TDA-HGB | 0,926 | 0,615 |
| yama CNN | 0,652 | 0,255 |
| **e_fusion_tda** | **0,955** | 0,771 |
| **f_hierarchical** | 0,936 | **0,857** |

* **Yama CNN çıkarıldı** (0,646 AUC, gereksiz) · SVM+MLP topluluğu reddedildi (0,656 vs HGB 0,919).
* FROC protokolü: post-fusion NMS + **duplikasyon IGNORE**.

### VinDr sıfır-eğitim (400 görüntü)
| Bileşen | AUC |
| :--- | :---: |
| Aşama-1 dedektör | **0,4618** |
| lokal TDA-HGB | **0,5901** |
| e_fusion_tda | 0,5911 |
| **V5.3 global 5-zon** | **0,6283** |
| Birleşik (V5.3+V6) | 0,6214 |
| bbox recall@K50 (M5) | **0,900** |

**Sonuç:** V6 aday üretici **lokalizasyonu taşıyor** (VinDr bbox recall@K50=0,90) ama **görüntü düzeyi güveni taşımıyor** (0,46). Kurum-içi hasta AUC 0,955; dış merkezde V5.3 global temsili hâlâ daha iyi ⇒ **lokal TDA + global temsil** birlikte kullanılmalı (NODE21 val'de birleşik 0,992).
"""


def append(path: Path, text: str, marker: str) -> None:
    s = path.read_text(encoding="utf-8")
    if marker in s:
        print(f"{path.name}: V6 bölümü zaten var, atlandı")
        return
    path.write_text(s.rstrip() + "\n" + text, encoding="utf-8")
    print(f"{path.name}: V6 bölümü eklendi")


def main() -> None:
    append(ROOT / "experiments.md", V6_EXPERIMENTS, "## 9. V6")
    append(ROOT / "ablation_matrix.md", V6_ABLATION, "## 13. V6")


if __name__ == "__main__":
    main()
