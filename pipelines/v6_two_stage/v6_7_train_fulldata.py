"""V6.7 — %100 Tam Veri (4.882 hasta) NİHAİ GÖNDERİM EĞİTİMİ: M5_fulldata.

KURAL — Bilimsel checkpoint'ler korunur
---------------------------------------
Makale tablolarını üreten `M5_centernet.pt`, `M5_foundation.pt`,
`stage2_final_tda_hgb.joblib` **ASLA** değiştirilmez. Bu betik her şeyi
`checkpoints/submission_fulldata/` altına yazar.

TUZAK A KORUMASI
----------------
Doğrulanmış SABİT öğrenme takvimi aynen kullanılır (hiçbir yeni hiperparametre
aranmaz): 5 epoch donuk ısınma + 13 epoch diferansiyel fine-tune, BN donuk,
Cosine Annealing, aynı augmentasyon, aynı kayıp. Veri: 4.882 hastanın tamamı
(train + val + test), 1.134 nodüllü + 1:1 dengeli temiz kontrol.

Yarıçap kafası `radius_detach=True` ile **detach edilmiş özellikler** üzerinde
eğitilir ⇒ gradyanı paylaşılan gövdeye/ısı haritasına geri akmaz ve omurga
ağırlıkları %100 korunur (V6.6.7 Bulgu 1: aksi hâlde kafa çöküyor).

Kullanım:
    python pipelines/v6_two_stage/v6_7_train_fulldata.py
    python pipelines/v6_two_stage/v6_7_train_fulldata.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
from detector_models import build_targets_heat, center_focal_loss  # noqa: E402
from detector_models_foundation import (INPUT, FoundationCenterNet,  # noqa: E402
                                        canon_to_512, foundation_input,
                                        foundation_input_aug, scale_centers)

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
SEED = 42
WARM_EPOCHS, TOTAL_EPOCHS = 5, 18
LR_HEAD, LR_BACKBONE, BATCH = 1e-3, 1e-4, 4


def _prep512(pid: str):
    canon, centers, radii = p2._prep(pid)
    return pid, canon_to_512(canon), centers, radii


def build_cache(pids, workers=10):
    cache, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (pid, c512, cen, rad) in enumerate(pool.map(_prep512, pids, chunksize=8)):
            cache[pid] = (c512, cen, rad)
            if (i + 1) % 400 == 0:
                print(f"    önbellek {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    önbellek hazır: {len(cache)} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def _aug_batch(items, cache, rng, device):
    """Augmentasyonlu batch (gama + ölçek + flip), ölçek merkezlere de uygulanır."""
    xs, ys, rs = [], [], []
    for pid, _ in items:
        c512, centers, radii = cache[pid]
        gamma = float(rng.uniform(0.8, 1.25))
        scale = float(rng.uniform(0.8, 1.2))
        flip = bool(rng.rand() < 0.5)
        img = foundation_input_aug(c512, gamma=gamma, scale=scale)
        cen, rad = scale_centers(centers, radii, scale)
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
            cen = [(INPUT - cx, cy) for cx, cy in cen]
        Y, R = build_targets_heat(cen, rad, device)
        xs.append(foundation_input(img, flip=False))
        ys.append(Y)
        rs.append(R)
    return (torch.stack(xs).to(device), torch.stack(ys).unsqueeze(1),
            torch.stack(rs).unsqueeze(1))


def loss_fn(model, x, Y, R):
    h, r = model(x)
    L = center_focal_loss(h, Y)
    L = L + 0.1 * F.smooth_l1_loss(r, R, reduction="none").mul(
        (Y > 0).float()).sum() / (Y > 0).sum().clamp_min(1)
    return L


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50-res512-all")
    ap.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    ap.add_argument("--warm", type=int, default=WARM_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--tag", default="M5_foundation_fulldata")
    ap.add_argument("--no-radius-detach", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    print(f"V6.7 TAM VERİ eğitimi | omurga={args.backbone} | device={device}", flush=True)
    print(f"  çıktı dizini: {CKPT}", flush=True)

    # ---- TUZAK A: doğrulanmış sabit takvim; veri = TÜM splitler
    splits = common.split_patients()
    all_pids = list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    pos = [p for p in all_pids if common.load_gt(p)]
    neg_all = [p for p in all_pids if not common.load_gt(p)]
    rng0 = np.random.RandomState(SEED)
    neg = list(rng0.permutation(neg_all)[:len(pos)])           # 1:1 denge
    if args.smoke:
        pos, neg = pos[:20], neg[:20]
    print(f"  TÜM VERİ: {len(all_pids)} hasta | {len(pos)} nodüllü + {len(neg)} temiz "
          f"kontrol = {len(pos) + len(neg)}", flush=True)

    print("\n[1] RAM önbelleği...", flush=True)
    cache = build_cache(list(pos) + list(neg))
    items = [(p, True) for p in pos if p in cache] + [(p, False) for p in neg if p in cache]
    print(f"    kullanılabilir: {len(items)}", flush=True)

    print("\n[2] Model (radius_detach={})...".format(not args.no_radius_detach), flush=True)
    model = FoundationCenterNet(weights=args.backbone).to(device)
    model.radius_detach = not args.no_radius_detach
    print(f"    {model.trainable_summary()}", flush=True)

    hist = []
    sched = None
    for ep in range(1, args.epochs + 1):
        if ep == 1:
            model.freeze_backbone()
            opt = torch.optim.Adam(model.param_groups(0.0, LR_HEAD), lr=LR_HEAD,
                                   weight_decay=1e-5)
        elif ep == args.warm + 1:
            model.unfreeze_top()
            print(f"  >>> Faz 2: üst bloklar açıldı ({model.trainable_summary()})", flush=True)
            opt = torch.optim.Adam(model.param_groups(LR_BACKBONE, LR_HEAD),
                                   weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.epochs - args.warm), eta_min=1e-5)

        model.train()
        model.set_bn_eval()
        rng = np.random.RandomState(SEED + ep)
        order = rng.permutation(len(items))
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(order), args.batch):
            sel = [items[i] for i in order[s:s + args.batch]]
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
        lrs = {g.get("name", "?"): g["lr"] for g in opt.param_groups}
        hist.append({"epoch": ep, "loss": round(avg, 5), "lrs": lrs,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {ep:2d}/{args.epochs} | loss {avg:.4f} | {time.time() - t0:.0f}s | "
              + (f"lr_head={lrs.get('head'):.2e} lr_bb={lrs.get('backbone', 0):.2e}"
                 if "backbone" in lrs else "bb=DONUK"), flush=True)

    torch.save(model.state_dict(), CKPT / f"{args.tag}.pt")
    (CKPT / f"{args.tag}_train_log.json").write_text(json.dumps({
        "experiment": "V6.7-fulldata", "backbone": args.backbone,
        "epochs": args.epochs, "warm": args.warm, "batch": args.batch,
        "radius_detach": model.radius_detach, "n_train": len(items),
        "n_nodule": len(pos), "n_control": len(neg),
        "data": "TÜM splitler (train+val+test)", "trainable": model.trainable_summary(),
        "history": hist}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {CKPT / (args.tag + '.pt')}", flush=True)
    print("Bilimsel checkpoint'lere DOKUNULMADI.", flush=True)


if __name__ == "__main__":
    main()
