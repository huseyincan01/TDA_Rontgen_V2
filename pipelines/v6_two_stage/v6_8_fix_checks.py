"""V6.8 — Hata düzeltme denetimleri: (B) ısınma/ince ayar ablasyonu · (C) yarıçap kafası onarımı.

NEDEN VAR
---------
V6.6'da `M5_foundation_warm.pt` (epoch 5, donuk ısınma) **teşhis için** kaydedilmiş,
sonra "temizlik" sırasında **değerlendirilmeden silinmişti** ⇒ "iki fazlı eğitimin
2. fazı gerçekten kazandırıyor mu?" sorusunun kanıtı kayboldu. Bu betik o kanıtı
yeniden üretir. Ayrıca V6.6.7'de bulunan **çökmüş yarıçap kafası** (R ≈ 0,01 vs GT
35–119 px; kayıp ağırlığı 0,1 iken focal kaybı 1–13 mertebesinde) onarılır ve resmî
NODE21 metriğinde sabit 56×56 px kutuya karşı ölçülür.

GÖREVLER
--------
  --task warm : 5 epoch DONUK ısınma (v6_6_train_foundation ile birebir aynı tohum,
                takvim, augmentasyon, kayıp) → checkpoint kaydedilir → val'de
                **final (18 epoch) M5_foundation** ile kıyaslanır.
                ⇒ Δ(final − warm) = **2. fazın (diferansiyel ince ayar) katkısı**.
  --task bn   : Isınma checkpoint'inden devam edip 2. fazı **BN istatistikleri
                AÇIK** (set_bn_eval çağrılmaz) koşar → "unutma-önleme" mekanizmasının
                (BN dondurma) payı ölçülür. Aynı 18 epoch takvimi, tek fark BN.
  --task radius: Omurga DONUK; yalnız 2 katmanlı yeni bir yarıçap kafası
                (64 kanal detach özellik üzerinde) **kayıp ağırlığı 1,0** ile eğitilir.
                Resmî metrik (IoU>0,2): **öğrenilmiş yarıçap** vs **sabit 48/56/64 px**.
                İKİ DENETİM KOLU (`--radius-mask`):
                  * `gauss`  = kayıp, ısı haritasının pozitif bölgesinin TAMAMI üzerinde
                               ortalanır (V6.6'nın yaptığı) → R=0 pikselleri seyreltir.
                  * `center` = kayıp YALNIZ R>0 (merkez) piksellerinde (CenterNet standardı).
                Ölçüm: ağırlık 1,0 tek başına yetmiyor; asıl darboğaz **maske/normalizasyon**.

PROTOKOL NOTU (dürüstlük)
-------------------------
* `warm` kolu "2. faz katkısını" ölçer; bunu tek başına "unutma-önleme katkısı" diye
  adlandırmak fazla iddia olurdu ⇒ onun için ayrı `bn` kolu vardır.
* Eğitim **yalnız train split**'inde, değerlendirme **yalnız val**'de yapılır ⇒
  tam-veri (V6.7) gönderim modeline sızıntı yok.
* Bilimsel checkpoint'lere DOKUNULMAZ; teşhis checkpoint'leri
  `artifacts/v6_two_stage/submission_fulldata/` altına yazılır (takip edilmez).

Kullanım:
    python pipelines/v6_two_stage/v6_8_fix_checks.py --task warm
    python pipelines/v6_two_stage/v6_8_fix_checks.py --task bn
    python pipelines/v6_two_stage/v6_8_fix_checks.py --task radius
    ... --limit 20 --train-limit 40 --epochs 1     (duman testi: yalnız hız kontrolü)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as v62  # noqa: E402
from detector_models_foundation import (FoundationCenterNet, canon_to_512,  # noqa: E402
                                        predict_tta)
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import (dist_nms, heat_maps, official_metric,  # noqa: E402
                                     pick_peaks)
from v6_7_train_fulldata import _aug_batch, _prep512, loss_fn  # noqa: E402

OUT_DEFAULT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_8_fix_checks.json"
WORK = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
BACKBONE = "resnet50-res512-all"
SEED, BATCH = 42, 4
WARM_EPOCHS, TOTAL_EPOCHS = 5, 18
LR_HEAD, LR_BACKBONE = 1e-3, 1e-4
SCALES, WIN, NMS_PX, TOPK = (0.9, 1.0, 1.1), 3, 8, 50
BOX_HALVES = (24, 28, 32)          # 48 / 56 / 64 px (V6.6.7: optimum 56)


# --------------------------------------------------------------------------- #
# rapor I/O
# --------------------------------------------------------------------------- #
def _load_report(out: Path) -> dict:
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    return {"experiment": "V6.8-fix-checks",
            "purpose": "V6.6'da silinen ısınma kanıtını yeniden üret + yarıçap kafasını onar"}


def _save(out: Path, rep: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")


# --------------------------------------------------------------------------- #
# ortak veri / değerlendirme
# --------------------------------------------------------------------------- #
def _build_cache(pids, workers: int = 8):
    """RAM önbelleği — İŞ PARÇACIKLI (v6_7'deki ProcessPool sürümüyle aynı çıktı).

    Neden iş parçacığı: kısıtlı sandbox'ta süreçler arası `Pipe` açılamıyor
    (WinError 5); ölçüm: 1580 hasta ≈ 30 s (8 iş parçacığı) vs ≈ 3,3 dk (tek çekirdek).
    Girdi/çıktı ağırlıklı iş (SimpleITK okuma) olduğu için iş parçacığı yeterli.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (pid, c512, cen, rad) in enumerate(pool.map(_prep512, pids, chunksize=8)):
            cache[pid] = (c512, cen, rad)
            if (i + 1) % 400 == 0:
                print(f"    önbellek {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)",
                      flush=True)
    print(f"    önbellek hazır: {len(cache)} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def _train_items(limit: int = 0):
    pos = p2._nodule_pids("train")
    neg = p2._control_pids("train")[:len(pos)]        # 1:1 denge (V6.6 ile aynı)
    if limit:
        pos, neg = pos[:limit], neg[:limit]
    return pos, neg


def _val_pids(limit: int = 0):
    val = common.split_patients()["val"]
    if not limit:
        return val
    nod = [p for p in val if common.load_gt(p)]       # AUC/FROC için iki sınıf da şart
    ctl = [p for p in val if not common.load_gt(p)]
    h = max(1, limit // 2)
    return nod[:h] + ctl[:limit - h]


def _detect_metrics(model, val_pids, device, tag):
    """Aşama-1 protokolü (predict_tta, NMS 20px) → Recall@K + FROC + hasta AUC."""
    gts_map = {p: common.load_gt(p) for p in val_pids}
    is_nod = {p: len(gts_map[p]) > 0 for p in val_pids}
    n_les = sum(len(g) for g in gts_map.values())
    hits = {str(k): 0 for k in (10, 20, 30, 50)}
    recs, t0 = [], time.time()
    for i, pid in enumerate(val_pids):
        canon, _, _ = p2._prep(pid)
        cs = [tuple(c) for c in predict_tta(model, canon, device)]
        gts = gts_map[pid]
        if gts:
            for k in hits:
                hits[k] += int(round(common.recall_at_K(cs, gts, int(k),
                                                        common.hit_criterion1) * len(gts)))
        labs = s2._labels_for(cs, gts)
        recs.append({"pid": pid, "is_nodule": is_nod[pid],
                     "cands": [(cs[j][0], cs[j][1], float(cs[j][2]), labs[j])
                               for j in range(len(cs))]})
        if (i + 1) % 100 == 0:
            print(f"      {tag}: {i + 1}/{len(val_pids)} ({time.time() - t0:.0f}s)",
                  flush=True)
    fr, au = v62.froc_fixed(recs, len(recs), n_les)
    return {"recall_at_K": {k: round(v / max(1, n_les), 4) for k, v in hits.items()},
            "froc": fr, "patient_auc": au, "n_patients": len(val_pids),
            "n_lesions": int(n_les)}


def _print_metrics(name, r):
    rk = r["recall_at_K"]
    print(f"  {name:10s} K10={rk['10']:.4f} K20={rk['20']:.4f} K30={rk['30']:.4f} "
          f"K50={rk['50']:.4f} | FROC@0.5={r['froc']['0.5']:.4f} "
          f"@1={r['froc']['1']:.4f} @4={r['froc']['4']:.4f} | "
          f"AUC nOR={r['patient_auc']['noisyor']} top1={r['patient_auc']['top1']}",
          flush=True)


# --------------------------------------------------------------------------- #
# eğitim (v6_6_train_foundation ile birebir takvim)
# --------------------------------------------------------------------------- #
def _train(device, items, cache, epochs, warm, bn_frozen, tag, seed=SEED,
           warm_probe: Path | None = None, resume: Path | None = None):
    """Aynı tohum/takvim/augmentasyon/kayıp. Döner: (model, history, warm_probe_path).

    bn_frozen=False ⇒ 2. fazda BN istatistikleri GÜNCELLENİR (unutma-önleme kontrolü).
    resume verilirse 5 epoch'luk ısınma yeniden koşulmaz, checkpoint'ten devam edilir.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FoundationCenterNet(weights=BACKBONE).to(device)
    model.radius_detach = True                 # gradyan paylaşılan gövdeye akmaz
    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"    ısınma checkpoint'inden devam: {resume}", flush=True)
    hist, sched, start = [], None, 1
    if resume is not None:
        start = warm + 1
        model.unfreeze_top()
        opt = torch.optim.Adam(model.param_groups(LR_BACKBONE, LR_HEAD),
                               weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs - warm), eta_min=1e-5)
        # resume edilen koşuda faz-2 LR takvimi baştan başlar (aşağıda ep>warm için aynı)
    for ep in range(start, epochs + 1):
        if ep == 1:
            model.freeze_backbone()
            opt = torch.optim.Adam(model.param_groups(0.0, LR_HEAD), lr=LR_HEAD,
                                   weight_decay=1e-5)
        elif ep == warm + 1 and resume is None:
            model.unfreeze_top()
            print(f"    >>> Faz 2: üst bloklar açıldı ({model.trainable_summary()})",
                  flush=True)
            opt = torch.optim.Adam(model.param_groups(LR_BACKBONE, LR_HEAD),
                                   weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs - warm), eta_min=1e-5)
        model.train()
        if bn_frozen or ep <= warm:
            model.set_bn_eval()                # 800k CXR istatistiği korunur
        rng = np.random.RandomState(seed + ep)
        order = rng.permutation(len(items))
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(order), BATCH):
            sel = [items[i] for i in order[s:s + BATCH]]
            if len(sel) < 2:
                continue
            x, Y, R = _aug_batch(sel, cache, rng, device)
            loss = loss_fn(model, x, Y, R)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if sched is not None:
            sched.step()
        avg = tot / max(1, nb)
        hist.append({"epoch": ep, "loss": round(avg, 5),
                     "seconds": round(time.time() - t0, 1),
                     "bn": "train" if (not bn_frozen and ep > warm) else "frozen"})
        print(f"    epoch {ep:2d}/{epochs} | loss {avg:.4f} | {time.time() - t0:.0f}s | "
              f"BN={hist[-1]['bn']}", flush=True)
        if ep == min(warm, epochs) and warm_probe is not None:
            warm_probe.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), warm_probe)
            print(f"    ısınma checkpoint kaydedildi: {warm_probe}", flush=True)
    return model, hist, warm_probe


def _cache(device, pos, neg):
    print(f"  train: {len(pos)} nodül + {len(neg)} kontrol = {len(pos) + len(neg)}",
          flush=True)
    print("\n[1] RAM önbelleği...", flush=True)
    cache = _build_cache(list(pos) + list(neg))
    return cache, [(p, True) for p in pos if p in cache] + \
                  [(p, False) for p in neg if p in cache]


# --------------------------------------------------------------------------- #
# B) ısınma / ince ayar ablasyonu
# --------------------------------------------------------------------------- #
def task_warm(device, out: Path, limit: int = 0, train_limit: int = 0, epochs: int = 0):
    print("=" * 70)
    print("B) ISINMA ABLASYONU — 5 epoch donuk ısınma vs final (18 epoch)")
    print("=" * 70, flush=True)
    pos, neg = _train_items(train_limit)
    cache, items = _cache(device, pos, neg)
    ep_total = epochs or WARM_EPOCHS
    probe = WORK / "M5_foundation_warm_probe.pt"

    print(f"\n[2] Isınma fazı yeniden üretiliyor ({ep_total} epoch, v6_6 takvimi)...",
          flush=True)
    model, hist, _ = _train(device, items, cache, ep_total, WARM_EPOCHS, True,
                            "warm", warm_probe=probe)
    del model
    torch.cuda.empty_cache()

    val_pids = _val_pids(limit)
    print(f"\n[3] Val değerlendirmesi ({len(val_pids)} hasta): ısınma vs final...",
          flush=True)
    warm_m = FoundationCenterNet(weights=BACKBONE).to(device)
    warm_m.load_state_dict(torch.load(probe, map_location=device))
    warm_m.eval()
    warm = _detect_metrics(warm_m, val_pids, device, "warm")
    del warm_m
    torch.cuda.empty_cache()
    final = _detect_metrics(load_foundation("M5_foundation", device), val_pids, device,
                            "final")

    _print_metrics("warm", warm)
    _print_metrics("final", final)
    delta = {
        "recall_K20": round(final["recall_at_K"]["20"] - warm["recall_at_K"]["20"], 4),
        "recall_K50": round(final["recall_at_K"]["50"] - warm["recall_at_K"]["50"], 4),
        "froc_0.5": round(final["froc"]["0.5"] - warm["froc"]["0.5"], 4),
        "froc_4": round(final["froc"]["4"] - warm["froc"]["4"], 4),
        "auc_noisyor": round((final["patient_auc"]["noisyor"] or 0)
                             - (warm["patient_auc"]["noisyor"] or 0), 4),
        "auc_top1": round((final["patient_auc"]["top1"] or 0)
                          - (warm["patient_auc"]["top1"] or 0), 4),
    }
    rep = _load_report(out)
    rep["warm_ablation"] = {
        "question": "İki fazlı eğitimin 2. fazı (diferansiyel ince ayar) gerçekten kazandırıyor mu?",
        "restored_evidence": ("V6.6'da M5_foundation_warm.pt değerlendirilmeden silinmişti; "
                              "bu koşu onu aynı tohum/takvimle yeniden üretir"),
        "protocol": (f"train split 1:1 ({len(items)} örnek) · ısınma {ep_total} epoch donuk "
                     f"omurga · aug/kayıp/tohum v6_6_train_foundation ile birebir · "
                     f"val {len(val_pids)} hasta"),
        "smoke": bool(limit or train_limit or epochs),
        "warm_history": hist,
        "warm_only": warm, "final_18ep": final, "delta_final_minus_warm": delta,
        "warm_ckpt": "submission_fulldata/M5_foundation_warm_probe.pt (teşhis, bilimsel değil)",
        "caveat": ("Δ(final−warm) faz-2'nin TOPLAM katkısıdır (13 ek epoch + üst blokların "
                   "açılması). Bunu tek başına 'unutma-önleme katkısı' saymak fazla iddia "
                   "olurdu → --task bn kolu BN dondurmanın payını ayrı ölçer."),
    }
    _save(out, rep)
    print(f"\n  Δ(final−warm): {delta}", flush=True)
    print(f"  kayıt: {out}", flush=True)


def task_bn(device, out: Path, limit: int = 0, train_limit: int = 0, epochs: int = 0):
    """2. fazı BN istatistikleri AÇIK koşar ⇒ BN dondurmanın (unutma-önleme) payı."""
    print("=" * 70)
    print("B2) UNUTMA-ÖNLEME KOLU — 2. faz BN AÇIK vs BN DONUK (final)")
    print("=" * 70, flush=True)
    pos, neg = _train_items(train_limit)
    cache, items = _cache(device, pos, neg)
    ep_total = epochs or TOTAL_EPOCHS
    probe = WORK / "M5_foundation_warm_probe.pt"
    if not probe.exists():
        print("\n[2a] Isınma checkpoint'i yok → önce 5 epoch ısınma...", flush=True)
        _train(device, items, cache, WARM_EPOCHS, WARM_EPOCHS, True, "warm",
               warm_probe=probe)
        torch.cuda.empty_cache()

    print(f"\n[2b] 2. faz BN AÇIK ({WARM_EPOCHS + 1}..{ep_total})...", flush=True)
    model, hist, _ = _train(device, items, cache, ep_total, WARM_EPOCHS, False,
                            "bn-unfrozen", warm_probe=None, resume=probe)
    bn_path = WORK / "M5_foundation_bn_unfrozen_probe.pt"
    torch.save(model.state_dict(), bn_path)
    print(f"    teşhis checkpoint: {bn_path}", flush=True)
    del model
    torch.cuda.empty_cache()

    val_pids = _val_pids(limit)
    print(f"\n[3] Val değerlendirmesi ({len(val_pids)} hasta)...", flush=True)
    bn_m = FoundationCenterNet(weights=BACKBONE).to(device)
    bn_m.load_state_dict(torch.load(bn_path, map_location=device))
    bn_m.eval()
    bn = _detect_metrics(bn_m, val_pids, device, "bn-open")
    del bn_m
    torch.cuda.empty_cache()
    final = _detect_metrics(load_foundation("M5_foundation", device), val_pids, device,
                            "final")

    _print_metrics("bn-open", bn)
    _print_metrics("final(BN donuk)", final)
    delta = {
        "recall_K20": round(bn["recall_at_K"]["20"] - final["recall_at_K"]["20"], 4),
        "recall_K50": round(bn["recall_at_K"]["50"] - final["recall_at_K"]["50"], 4),
        "froc_0.5": round(bn["froc"]["0.5"] - final["froc"]["0.5"], 4),
        "froc_4": round(bn["froc"]["4"] - final["froc"]["4"], 4),
        "auc_noisyor": round((bn["patient_auc"]["noisyor"] or 0)
                             - (final["patient_auc"]["noisyor"] or 0), 4),
    }
    rep = _load_report(out)
    rep["bn_unfrozen_ablation"] = {
        "question": "BN dondurma (unutma-önleme) gerçekten koruyor mu?",
        "protocol": (f"aynı 18 epoch takvimi; tek fark 2. fazda BN running istatistikleri "
                     f"güncellenir (batch={BATCH}) · val {len(val_pids)} hasta"),
        "smoke": bool(limit or train_limit or epochs),
        "phase2_history": hist,
        "bn_open": bn, "final_bn_frozen": final, "delta_bn_open_minus_final": delta,
        "caveat": ("Δ<0 ⇒ BN dondurma koruyor (unutma-önleme kanıtı). Δ≈0 ⇒ etkisiz. "
                   "Δ>0 ⇒ dondurma zararlı. Tek fark BN'dir; veri/takvim/tohum aynıdır."),
    }
    _save(out, rep)
    print(f"\n  Δ(BN açık − BN donuk): {delta}", flush=True)
    print(f"  kayıt: {out}", flush=True)


# --------------------------------------------------------------------------- #
# C) yarıçap kafası onarımı
# --------------------------------------------------------------------------- #
class RadiusHead(nn.Module):
    """Donuk 64 kanal özellik üzerinde 2 katmanlı yarıçap kafası (teşhis probu)."""

    def __init__(self, in_ch=64, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))

    def forward(self, f):
        return self.net(f)


def _feats(model, x):
    """forward()'daki `f` ile BİREBİR aynı özellik (fuse → ×2 interpolate, 256x256)."""
    c2, c3, c4, c5 = model.backbone_features(x)
    p2v, _, _, _ = model.fpn(c2, c3, c4, c5)
    f = model.fuse(p2v)
    return F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)


def task_radius(device, out: Path, limit: int = 0, train_limit: int = 0,
                epochs: int = 0, mask_mode: str = "center"):
    print("=" * 70)
    print(f"C) YARIÇAP KAFASI ONARIMI — donuk özellik, kayıp ağırlığı 1,0, "
          f"maske={mask_mode}")
    print("=" * 70, flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ep_total = epochs or 6

    src = load_foundation("M5_foundation", device)
    for p in src.parameters():
        p.requires_grad = False
    src.radius_detach = True

    # referans: mevcut (çökmüş) kafa tahminleri
    pos, neg = _train_items(train_limit)
    cache, items = _cache(device, pos, neg)

    head = RadiusHead(64, 32).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    print(f"\n[2] Yarıçap kafası eğitimi (ağırlık 1,0, {ep_total} epoch, "
          f"maske={mask_mode})...", flush=True)
    hist = []
    for ep in range(1, ep_total + 1):
        head.train()
        rng = np.random.RandomState(SEED + ep)
        order = rng.permutation(len(items))
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(order), BATCH):
            sel = [items[i] for i in order[s:s + BATCH]]
            if len(sel) < 2:
                continue
            x, Y, R = _aug_batch(sel, cache, rng, device)
            with torch.no_grad():
                f = _feats(src, x)
            pred = head(f)
            # `gauss`: ısı haritasının tüm pozitif bölgesi (R=0 pikselleri seyreltir)
            # `center`: yalnız R>0 pikselleri (CenterNet standardı)
            pos_mask = (R > 0).float() if mask_mode == "center" else (Y > 0).float()
            loss = (F.smooth_l1_loss(pred, R, reduction="none") * pos_mask).sum() \
                / pos_mask.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        hist.append({"epoch": ep, "loss": round(tot / max(1, nb), 5),
                     "seconds": round(time.time() - t0, 1)})
        print(f"    epoch {ep}/{ep_total} | L1(radius) {hist[-1]['loss']:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)

    label = {"gauss": "weight1_gauss_posmask",
             "center": "weight1_center_only"}[mask_mode]
    # KANIT KORUMASI: eğitilen kafa DİSKE yazılır. V6.6'da ısınma checkpoint'i
    # değerlendirilmeden silinmişti; aynı hatayı tekrarlamıyoruz.
    head_path = WORK / f"radius_head_{label}.pt"
    WORK.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), head_path)
    print(f"    yarıçap kafası kaydedildi: {head_path}", flush=True)

    old_radius = src.radius
    src.radius = head                            # ısı dalı DEĞİŞMEZ (fuse donuk)
    src.eval()

    val_pids = _val_pids(limit)
    gt_boxes = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
                    for g in common.load_gt(p)] for p in val_pids}
    is_nod = {p: len(gt_boxes[p]) > 0 for p in val_pids}
    n_les = sum(len(g) for g in gt_boxes.values())

    print(f"\n[3] Isı + yarıçap haritaları ({len(val_pids)} hasta)...", flush=True)
    flat, t0 = [], time.time()
    for i, pid in enumerate(val_pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(src, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        flat.extend((s, pid, cx, cy, r) for cx, cy, s, r in cs)
        if (i + 1) % 100 == 0:
            print(f"      {i + 1}/{len(val_pids)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[4] Resmî NODE21 metriği (IoU>0,2) — sabit kutu vs öğrenilmiş yarıçap...",
          flush=True)
    res = {}
    for bh in BOX_HALVES:
        r = official_metric(flat, len(val_pids), n_les, gt_boxes, is_nod, val_pids,
                            box_half=float(bh))
        res[f"fixed_{2 * bh}px"] = r
        print(f"  sabit {2 * bh:3d}px  rank={r['rank_metric']} AUC={r['auc']} "
              f"S@.125={r['sens']['0.125']} S@.25={r['sens']['0.25']} "
              f"S@.5={r['sens']['0.5']}", flush=True)
    rr = official_metric(flat, len(val_pids), n_les, gt_boxes, is_nod, val_pids,
                         use_radius=True)
    res["learned_radius"] = rr
    print(f"  ÖĞRENİLMİŞ     rank={rr['rank_metric']} AUC={rr['auc']} "
          f"S@.125={rr['sens']['0.125']} S@.25={rr['sens']['0.25']} "
          f"S@.5={rr['sens']['0.5']}", flush=True)

    rs = np.array([f[4] for f in flat], np.float32)
    pct = {q: round(float(np.percentile(rs, q)), 2) for q in (10, 50, 90)}
    print(f"  tahmin edilen r1024: medyan {pct[50]} px · p10 {pct[10]} · p90 {pct[90]}"
          f"  (GT 17,5–59,5 px ⇒ çökme {'DÜZELDİ' if pct[50] > 5 else 'SÜRÜYOR'})",
          flush=True)

    best_fixed = max((f"fixed_{2 * bh}px" for bh in BOX_HALVES),
                     key=lambda k: res[k]["rank_metric"] or 0)
    verdict = ("öğrenilmiş yarıçap üstün" if (rr.get("rank_metric") or 0)
               > (res[best_fixed]["rank_metric"] or 0) else "sabit kutu hâlâ daha iyi")
    rep = _load_report(out)
    rhf = rep.get("radius_head_fix") or {}
    if "results" in rhf:                       # eski düz biçim → etiketli yapıya taşı
        rhf = {"weight1_gauss_posmask": {
            **rhf, "supervision": "kayıp ağırlığı 1,0 · maske=Y>0 (tüm Gaussian bölgesi)"}}
    rhf[label] = {
        "question": "Yarıçap kafası onarılırsa (ağırlık 0,1 → 1,0) sabit 56 px kutuyu geçer mi?",
        "supervision": (f"kayıp ağırlığı 1,0 · maske="
                        f"{'R>0 (yalnız merkez pikseli)' if mask_mode == 'center' else 'Y>0 (tüm Gaussian bölgesi)'}"),
        "protocol": (f"omurga DONUK, yalnız 2 katmanlı yarıçap kafası (fuse özelliği, "
                     f"256x256), smooth-L1 ağırlık 1,0, {ep_total} epoch · "
                     f"değerlendirme: çok ölçekli TTA + NMS 8px, resmî IoU>0,2 · "
                     f"val {len(val_pids)} hasta"),
        "smoke": bool(limit or train_limit or epochs),
        "history": hist,
        "head_ckpt": f"submission_fulldata/{head_path.name} (teşhis; git dışı)",
        "results": res,
        "best_fixed": best_fixed,
        "predicted_r1024": pct,
        "old_radius_head_control": ("V6.6.7: R ≈ 0,01–0,04 → Sens@0,25 = 0,000 (kutu ~0,1 px); "
                                    "v6_6_optimization_audit.json → radius_head_control"),
        "verdict": verdict,
        "caveat": ("Yarıçap kafası omurgayı eğitmez (donuk özellik probu) ⇒ kazanç yalnız kutu "
                   "geometrisinden gelir; ısı haritası/sıralama değişmez (AUC sabit kalır)."),
    }
    rep["radius_head_fix"] = rhf
    _save(out, rep)
    src.radius = old_radius
    print(f"\n  KARAR: {verdict}  (en iyi sabit: {best_fixed})", flush=True)
    print(f"  kayıt: {out}", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["warm", "bn", "radius"])
    ap.add_argument("--limit", type=int, default=0, help="val hasta sınırı (yalnız duman)")
    ap.add_argument("--train-limit", type=int, default=0, help="sınıf başına örnek (duman)")
    ap.add_argument("--epochs", type=int, default=0, help="takvimi kısalt (yalnız duman)")
    ap.add_argument("--radius-mask", choices=["center", "gauss"], default="center",
                    help="yarıçap kaybının süpervizyon maskesi (bkz. docstring)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    print(f"V6.8 hata düzeltme denetimi | task={args.task} | device={device}", flush=True)
    if args.task == "warm":
        task_warm(device, out, args.limit, args.train_limit, args.epochs)
    elif args.task == "bn":
        task_bn(device, out, args.limit, args.train_limit, args.epochs)
    else:
        task_radius(device, out, args.limit, args.train_limit, args.epochs,
                    args.radius_mask)


if __name__ == "__main__":
    main()
