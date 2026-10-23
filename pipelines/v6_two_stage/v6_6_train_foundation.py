"""V6.6 — Foundation X-Ray Backbone Stage-1 EĞİTİMİ (iki fazlı diferansiyel).

Faz planı
---------
Epoch 1-5   (Isınma)          : CXR vakıf omurgası DONUK (requires_grad=False) ve
                                BN .eval() (800k röntgenlik running_mean/var
                                bozulmasın); yalnız FPN+ASPP boynu ve CenterNet
                                kafası lr=1e-3 ile eğitilir.
Epoch 6-18  (Diferansiyel FT) : Omurganın üst blokları (layer3/layer4) açılır;
                                lr_backbone=1e-4, lr_head=1e-3, Cosine Annealing.
                                BN yine .eval() (batch küçük ⇒ running istatistik
                                güncellemesi zararlı olurdu).

Kayıp: Gaussian Focal (alpha=2, beta=4) + 0.1 * L1 (radius, yalnız pozitifler)
Augmentasyon: yatay çevirme + gama gamma in [0.8, 1.25] + ölçek +-%20
Kayıt: checkpoints/M5_foundation.pt (final) ve M5_foundation_warm.pt (epoch 5)

Kullanım:
    python pipelines/v6_two_stage/v6_6_train_foundation.py
    python pipelines/v6_two_stage/v6_6_train_foundation.py --backbone densenet121-res224-all
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
from detector_models import build_targets_heat, center_focal_loss  # noqa: E402
from detector_models_foundation import (INPUT, FoundationCenterNet,  # noqa: E402
                                        canon_to_512, foundation_input,
                                        foundation_input_aug, scale_centers)

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
SEED = 42
WARM_EPOCHS = 5
TOTAL_EPOCHS = 18
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
BATCH = 4


def _prep512(pid: str):
    """NODE21 train örneği: 512x512 uint8 kanonik + 512 uzayında merkez/yarıçap."""
    canon, centers, radii = p2._prep(pid)
    return pid, canon_to_512(canon), centers, radii


def build_cache(pids, workers: int = 10):
    """512 kanonikleri RAM'de önbellekle (1580 x 256 KB ≈ 0.4 GB) — I/O'yu keser."""
    cache = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (pid, c512, cen, rad) in enumerate(pool.map(_prep512, pids, chunksize=8)):
            cache[pid] = (c512, cen, rad)
            if (i + 1) % 200 == 0:
                print(f"    önbellek {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    önbellek hazır: {len(cache)} örnek ({time.time() - t0:.0f}s)", flush=True)
    return cache


def make_batch(items, cache, rng, device):
    """Augmentasyonlu batch: flip + gama + ölçek (girdi [-1024, +1024])."""
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
    return (torch.stack(xs).to(device),
            torch.stack(ys).unsqueeze(1),
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
    ap.add_argument("--nodule", type=int, default=0, help="0 = tümü")
    ap.add_argument("--control", type=int, default=0, help="0 = nodül sayısı kadar (1:1)")
    ap.add_argument("--tag", default="M5_foundation")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.6 eğitim | omurga={args.backbone} | device={device}", flush=True)

    pos = p2._nodule_pids("train")
    neg_all = p2._control_pids("train")
    if args.smoke:
        pos, neg_all = pos[:20], neg_all[:20]
    if args.control:
        neg = neg_all[:args.control]
    else:
        neg = neg_all[:len(pos)]                       # 1:1 denge
    if args.nodule:
        pos = pos[:args.nodule]
    print(f"  train: {len(pos)} nodül + {len(neg)} temiz kontrol = {len(pos) + len(neg)}", flush=True)

    print("\n[1] RAM önbelleği...", flush=True)
    cache = build_cache(list(pos) + list(neg))
    items = [(p, True) for p in pos if p in cache] + [(p, False) for p in neg if p in cache]
    print(f"    kullanılabilir örnek: {len(items)}", flush=True)

    print("\n[2] Model...", flush=True)
    model = FoundationCenterNet(weights=args.backbone).to(device)
    print(f"    {model.trainable_summary()}", flush=True)

    epochs = args.epochs
    hist = []
    for ep in range(1, epochs + 1):
        # ---- faz geçişi
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
                opt, T_max=max(1, epochs - args.warm), eta_min=1e-5)

        model.train()
        model.set_bn_eval()                            # BN istatistikleri DONUK
        rng = np.random.RandomState(SEED + ep)
        order = rng.permutation(len(items))
        tot, nb = 0.0, 0
        t0 = time.time()
        for s in range(0, len(order), args.batch):
            sel = [items[i] for i in order[s:s + args.batch]]
            if len(sel) < 2:                           # BN/istatistik için en az 2
                continue
            x, Y, R = make_batch(sel, cache, rng, device)
            loss = loss_fn(model, x, Y, R)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if ep > args.warm:
            sched.step()
        avg = tot / max(1, nb)
        lrs = {g.get("name", "?"): g["lr"] for g in opt.param_groups}
        hist.append({"epoch": ep, "loss": round(avg, 5), "lrs": lrs,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {ep:2d}/{epochs} | loss {avg:.4f} | {time.time() - t0:.0f}s | "
              f"lr_head={lrs.get('head')} " +
              (f"lr_bb={lrs.get('backbone'):.2e}" if "backbone" in lrs else "bb=DONUK"),
              flush=True)
        if ep == args.warm:
            torch.save(model.state_dict(), CKPT / f"{args.tag}_warm.pt")
            print(f"    ısınma checkpoint: {args.tag}_warm.pt", flush=True)

    torch.save(model.state_dict(), CKPT / f"{args.tag}.pt")
    (CKPT.parent / f"{args.tag}_train_log.json").write_text(
        json.dumps({"backbone": args.backbone, "epochs": epochs, "warm": args.warm,
                    "batch": args.batch, "n_train": len(items),
                    "trainable": model.trainable_summary(), "history": hist},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {CKPT / (args.tag + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
