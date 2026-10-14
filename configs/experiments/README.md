# Deney Konfigürasyonları

Bu klasör, [roadmap.md](../../roadmap.md) backlog'undaki her deneyin konfigürasyonunu barındırır.

## Dosya adı sözleşmesi

```
configs/experiments/<ID>_<slug>.yaml     # ör. E1_1_bone_suppression.yaml
```

`<ID>`, roadmap.md'deki backlog ID'sidir (E0.0, E1.1, E2.3 …). `<slug>` kısa ve ASCII'dir.

## Çalıştırma çıktısı sözleşmesi

```
artifacts/runs/<ID>_<slug>/
    ├── manifest.json       # git commit + config hash + girdi hash'leri + seed + metrikler
    ├── val_report.json
    └── test_report.json    # YALNIZCA faz sonunda (test bütçesi, roadmap §0.3)
```

Manifest yazmak için `src/node21_tda/experiment.py` kullanılır:

```python
from node21_tda.experiment import new_run_dir, write_manifest

run_dir = new_run_dir("E1.1", "bone_suppression")
write_manifest(
    run_dir, "E1.1",
    config=cfg, config_path=config_path,
    inputs=[features_path, splits_path],
    seed=42, metrics={"val_roc_auc": 0.8712},
)
```

> **Not:** `artifacts/` ve `data/processed/` git tarafından yok sayılır. Kalıcı kanıt kaydı
> [experiments.md](../../experiments.md) ve [roadmap.md](../../roadmap.md) durum tablosudur;
> manifest'ler yalnızca yerel tekrar üretilebilirlik içindir.

## Kurallar (roadmap §0)

1. Her deney **tek bir faktörü** değiştirir.
2. Başarı eşiği deney başlamadan yazılır, sonradan değiştirilmez (ön-kayıt).
3. Başarı sayılması için val ROC-AUC farkı ≥ **+0.015** ve bootstrap %95 CI kesişmemesi gerekir.
4. Test setine yalnızca faz sonunda dokunulur.
5. Öldürme kriteri tetiklenirse deney **KILL** işaretlenir ve zaman harcanmaz.
