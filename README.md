# node21_tda

NODE21 verilerini hasta bazında hazırlama, 5 anatomik bölgeden TDA (Topological Data Analysis) özellikleri çıkarma ve makine/derin öğrenme modelleri ile nodül tespiti projesi.

- Mimari ve çalışma günlüğü: [progress.md](progress.md)
- v3 araştırma programı ve deney yönetimi: [roadmap.md](roadmap.md)
- v4 araştırma programı (aktif dal `exp/v4-directional-radial`): [v4_roadmap.md](v4_roadmap.md)
- **Tüm sonuçlar tek yerde (makale tabloları): [ablation_matrix.md](ablation_matrix.md)**
- **Makale taslağı: [paper_draft.md](paper_draft.md)**
- Literatür taraması ve konumlandırma: [literature.md](literature.md)
- Model deneyleri, doğruluk oranları ve liderlik tablosu: [experiments.md](experiments.md)



## Dallar ve belge rehberi (yeni bir agent/okuyucu için)

| Dal / etiket | Anlamı |
| :--- | :--- |
| `main` | **v3 — kapandı, yayına hazır.** Güncel gerçek kaynak. Etiket: `v3-publication-ready` (yalnızca belge düzeltmeleri sonradan eklendi) |
| `exp/v4-directional-radial` | **v4 — aktif.** Yalnızca [v4_roadmap.md](v4_roadmap.md) ve `pipelines/exp/V*.py` ekler; v3 belgelerine dokunmaz. Şu ana dek tüm adaylar KILL |
| `exp/*` | Deneysel programlar; kendi roadmap belgeleri vardır |

**Okuma sırası:** [progress.md](progress.md) (mimari, v1→v3 özeti) → [roadmap.md](roadmap.md) (protokol kuralları 1–10, karar günlüğü, §13 kapanış) → [ablation_matrix.md](ablation_matrix.md) (nihai sayılar) → [experiments.md](experiments.md) (deney ayrıntıları).

**Kurallar:** Test seti bütçesi **6/6 tükendi** (kapalı). Bu veri NODE21'in halka açık eğitim setidir; **leaderboard ile karşılaştırma yapılmaz** (Kural 9). Sayılarda çelişki görürseniz `ablation_matrix.md` ve `roadmap.md §13` esas alınır. v3 belgeleri yalnızca **hata düzeltmesi** için `main`'de değiştirilir; yeni program sonuçları kendi dalındaki belgeye yazılır.