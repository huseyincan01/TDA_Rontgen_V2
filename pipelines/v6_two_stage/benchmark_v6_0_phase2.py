"""V6.0 Faz 2 — M5/M6 eğitimi + M5/M6/M7 oracle recall benchmark (val + train).

Disk kuralı: eğitim on-the-fly; diske yalnızca checkpoint (2 adet) + JSON rapor.
Kullanım:
    python pipelines/v6_two_stage/benchmark_v6_0_phase2.py --epochs 10 --limit 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
from detector_models import (HMP, STRIDE, INPUT, CenterNetHead, FCOSHead,
                             build_targets_heat, _build_box_targets, center_focal_loss)  # noqa: E402
import candidates_classical as cc  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
BENCH_JSON = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_0_candidate_benchmark.json"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _nodule_pids(split: str) -> list[str]:
    pids = common.split_patients()[split]
    out = []
    for p in pids:
        gts = common.load_gt(p)
        if gts:
            out.append(p)
    return out


def _control_pids(split: str, seed: int = 42) -> list[str]:
    """Temiz kontrol (nodül içermeyen) hastalar."""
    pids = common.split_patients()[split]
    out = []
    for p in pids:
        gts = common.load_gt(p)
        if not gts:
            out.append(p)
    rng = np.random.RandomState(seed)
    return list(rng.permutation(out))


def _prep(pid: str) -> tuple[np.ndarray, list]:
    """kanonik 512x512 (3ch tensor girdi) + bbox (512 uzayında cx,cy,r listesi)."""
    img = common.load_image(pid)
    mask = common.load_lung_mask(pid)
    p1p99 = common.get_lung_p1p99().get(pid)
    p1, p99 = float(p1p99[0]), float(p1p99[1])
    canon = common.canonicalize(img, mask, p1, p99)
    gts = common.load_gt(pid)
    centers, radii = [], []
    for g in gts:
        cx = (g["x"] + g["w"] / 2) * 0.5
        cy = (g["y"] + g["h"] / 2) * 0.5
        r = max(g["w"], g["h"]) * 0.5 * 0.5
        centers.append((cx, cy))
        radii.append(max(r, 2.0))
    return canon, centers, radii


def _to_input(canon: np.ndarray, flip: bool) -> torch.Tensor:
    """V6.0: mono 3 kanal. V6.0b (V6_BIPOLAR=1): [kanonik, ters, kanonik]."""
    import os

    im = Image.fromarray(np.clip(canon, 0, 255).astype(np.uint8)).resize(
        (INPUT, INPUT), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    if flip:
        a = np.ascontiguousarray(a[:, ::-1])
    t = torch.from_numpy(a)
    if os.environ.get("V6_BIPOLAR", "0") == "1":
        x = torch.stack([t, 1.0 - t, t], dim=0)
    else:
        x = t.unsqueeze(0).repeat(3, 1, 1)
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (x - m) / s


def _flip_centers(centers, radii, flip: bool):
    if not flip:
        return centers, radii
    return [(INPUT - cx, cy) for cx, cy in centers], radii


# ------------------------------------------------------------------ eğitim
def train_model(kind: str, pos_pids: list[str], neg_pids: list[str], epochs: int, device):
    torch.manual_seed(42)
    np.random.seed(42)
    model = CenterNetHead() if kind == "M5" else FCOSHead()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    B = 4
    # 1:1 pozitif + temiz kontrol (boş hedef) — normal anatomi bastırma
    items = [(p, True) for p in pos_pids] + [(p, False) for p in neg_pids]
    for ep in range(1, epochs + 1):
        model.train()
        rng = np.random.RandomState(42 + ep)
        order = rng.permutation(len(items))
        tot_loss, nb = 0.0, 0
        xs, ts = [], []
        for i in order:
            pid, _ = items[i]
            try:
                canon, centers, radii = _prep(pid)
            except Exception:
                continue
            flip = bool(rng.rand() < 0.5)
            x = _to_input(canon, flip)
            centers, radii = _flip_centers(centers, radii, flip)
            if kind == "M5":
                Y, R = build_targets_heat(centers, radii, device)
                t = (Y, R)
            else:
                obj, box = _build_box_targets(centers, radii)
                t = (torch.tensor(obj, device=device), torch.tensor(box, device=device))
            xs.append(x)
            ts.append(t)
            if len(xs) == B:
                loss = _batch_loss(model, xs, ts, kind)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += loss.item()
                nb += 1
                xs, ts = [], []
        if xs:
            loss = _batch_loss(model, xs, ts, kind)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
        print(f"  {kind} epoch {ep}/{epochs} | loss {tot_loss / max(1, nb):.4f}", flush=True)
    return model


def _batch_loss(model, xs, ts, kind):
    x = torch.stack(xs).to(next(model.parameters()).device)
    if kind == "M5":
        h, r = model(x)
        Y = torch.stack([t[0] for t in ts]).unsqueeze(1)
        R = torch.stack([t[1] for t in ts]).unsqueeze(1)
        L = center_focal_loss(h, Y)
        L = L + 0.1 * F.smooth_l1_loss(r, R, reduction="none").mul((Y > 0).float()).sum() / (Y > 0).sum().clamp_min(1)
        return L
    else:
        o, b = model(x)
        Y = torch.stack([t[0] for t in ts])
        Bx = torch.stack([t[1] for t in ts])
        pos = Y > 0.5
        # dengesizlik: 65k negatif hücreye karşı ~169 pozitif → focal zorunlu
        from torchvision.ops import sigmoid_focal_loss
        L_obj = sigmoid_focal_loss(o, Y.unsqueeze(1).expand_as(o), alpha=0.25, gamma=2.0,
                                   reduction="mean")
        if pos.any():
            L_box = F.smooth_l1_loss(b.permute(0, 2, 3, 1)[pos], Bx[pos], reduction="mean")
        else:
            L_box = torch.tensor(0.0, device=x.device)
        return L_obj + L_box


# ------------------------------------------------------------------ çıkarım + decode
@torch.no_grad()
def predict(model, canon: np.ndarray, device, kind: str, top: int = 1000):
    model.eval()
    x = _to_input(canon, False).unsqueeze(0).to(device)
    if kind == "M5":
        h, _ = model(x)
        heat = torch.sigmoid(h)[0, 0].cpu().numpy()
        maxf = ndi.maximum_filter(heat, size=3)
        ys, xs = np.where((heat >= maxf) & (heat > 0.01))
        scores = heat[ys, xs]
    else:
        o, b = model(x)
        obj = torch.sigmoid(o)[0, 0].cpu().numpy()
        box = b[0].cpu().numpy()  # (4,128,128)
        ys, xs = np.where(obj > 0.02)
        scores = obj[ys, xs]
        # kutu merkezi = hücre merkezi + box merkez kayması
        l, t, r, bb = box[0], box[1], box[2], box[3]
        cx_off = (r[ys, xs] - l[ys, xs]) / 2.0 * STRIDE
        cy_off = (bb[ys, xs] - t[ys, xs]) / 2.0 * STRIDE
        xs = (xs * STRIDE + STRIDE / 2 + cx_off) / 0.5
        ys = (ys * STRIDE + STRIDE / 2 + cy_off) / 0.5
        order = np.argsort(-scores)
        cands = [(float(xs[k]), float(ys[k]), float(scores[k])) for k in order]
        return cands[:top]
    order = np.argsort(-scores)
    cands = [(float(xs[k] * STRIDE + STRIDE / 2) / 0.5,
              float(ys[k] * STRIDE + STRIDE / 2) / 0.5,
              float(scores[k])) for k in order]
    return cc._nms(cands, min_dist=20.0, top=top)


# ------------------------------------------------------------------ hibrit M7
def predict_m7(model_best, model_classical, canon, mask, device, kind, top=1000):
    a = predict(model_best, canon, device, kind, top=top)
    b = cc.GENERATORS["M2_doh_vesselsuppress"](canon, mask)
    n_b = max(1, int(round(len(a) * 0.3)))
    merged = a[: max(1, int(len(a) * 0.7))] + b[:n_b]
    merged.sort(key=lambda c: -c[2])
    return cc._nms(merged, min_dist=20.0, top=top)


# ------------------------------------------------------------------ değerlendirme (tek geçiş)
def _empty():
    return {"n_lesions": 0, "hits_k1": {str(k): 0 for k in (10, 20, 30, 50)},
            "hits_k2": {str(k): 0 for k in (10, 20, 30, 50)}, "total_ms": 0.0}


def _accumulate(a, cands, gts, ms):
    a["total_ms"] += ms
    a["n_lesions"] += len(gts)
    for k in (10, 20, 30, 50):
        a["hits_k1"][str(k)] += int(round(common.recall_at_K(cands, gts, k, common.hit_criterion1) * len(gts)))
        a["hits_k2"][str(k)] += int(round(common.recall_at_K(cands, gts, k, common.hit_criterion2) * len(gts)))


def _m7_orthogonal(trained: list, m2: list, n_anchor: int = 35, frac_t: float = 0.7):
    """Sıralı/ortogonal kota (danışman önerisi):
    1) eğitimliden NMS'li ilk `n_anchor` (=0.7*50) adayı sabitle,
    2) M2'den bunlara dist>20px olan en yüksek skorluları al,
    3) %70/%30 deseninde iç-içe geçir (kota NMS'te kaybolmaz)."""
    t = cc._nms(trained, min_dist=20, top=4000)
    anchors = t[:n_anchor]
    m = [c for c in m2 if all((c[0] - a) ** 2 + (c[1] - b) ** 2 >= 400 for a, b, _ in anchors)]
    m = cc._nms(m, min_dist=20, top=4000)
    out, i, j = [], 0, 0
    while i < len(t) or j < len(m):
        for _ in range(7):
            if i < len(t):
                out.append(t[i]); i += 1
        for _ in range(3):
            if j < len(m):
                out.append(m[j]); j += 1
    return out


def eval_one_pass(models: dict, pids: list, device) -> dict:
    """Her hasta için görüntü/maske/GT bir kez yüklenir; M2 bir kez hesaplanır."""
    keys = list(models.keys()) + ["M7a_M5M2", "M7b_M6M2"]
    agg = {k: _empty() for k in keys}
    for i, pid in enumerate(pids):
        canon, _, _ = _prep(pid)
        mask = common.load_lung_mask(pid)
        gts = common.load_gt(pid)

        cands = {}
        for kind, m in models.items():
            t0 = time.perf_counter()
            c = predict(m, canon, device, kind)
            ms = (time.perf_counter() - t0) * 1000
            cands[kind] = c
            _accumulate(agg[kind], c, gts, ms)

        t0 = time.perf_counter()
        m2 = cc.GENERATORS["M2_doh_vesselsuppress"](canon, mask)
        ms2 = (time.perf_counter() - t0) * 1000
        _accumulate(agg["M7a_M5M2"], _m7_orthogonal(cands["M5"], m2), gts, ms2)
        _accumulate(agg["M7b_M6M2"], _m7_orthogonal(cands["M6"], m2), gts, ms2)

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(pids)} hasta", flush=True)
    return agg


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Faz 2 | device={device} | epochs={args.epochs}")

    pos_pids = _nodule_pids("train")
    neg_pids = _control_pids("train")
    if args.limit:
        pos_pids = pos_pids[: args.limit]
        neg_pids = neg_pids[: args.limit]
    neg_pids = neg_pids[: len(pos_pids)]   # en az 1:1 temiz kontrol
    print(f"Eğitim: {len(pos_pids)} nodül + {len(neg_pids)} kontrol (1:1)")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for kind in ("M5", "M6"):
        ckpt = CKPT_DIR / f"{kind}_centernet.pt" if kind == "M5" else CKPT_DIR / f"{kind}_fcos.pt"
        if args.skip_train and ckpt.exists():
            m = (CenterNetHead() if kind == "M5" else FCOSHead()).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            print(f"{kind} checkpoint yüklendi.")
        else:
            m = train_model(kind, pos_pids, neg_pids, args.epochs, device)
            torch.save(m.state_dict(), ckpt)
            print(f"{kind} checkpoint kaydedildi: {ckpt.name}")
        models[kind] = m

    # --- değerlendirme (val + train referans, tek geçiş) ---
    report = {"phase2": {}}
    for split in ("val", "train"):
        pids = _nodule_pids(split)
        print(f"\n=== {split.upper()} ({len(pids)} nodül hastası) ===")
        agg = eval_one_pass(models, pids, device)

        def k50(a):
            return a["hits_k1"]["50"] / max(1, a["n_lesions"])

        best_trained = max(models, key=lambda k: k50(agg[k]))
        m7_key = "M7a_M5M2" if k50(agg["M7a_M5M2"]) >= k50(agg["M7b_M6M2"]) else "M7b_M6M2"

        summary = {}
        for k, v in agg.items():
            summary[k] = {
                "recall_k1_froc": {str(kk): round(v["hits_k1"][str(kk)] / max(1, v["n_lesions"]), 4) for kk in (10, 20, 30, 50)},
                "recall_k2_containment": {str(kk): round(v["hits_k2"][str(kk)] / max(1, v["n_lesions"]), 4) for kk in (10, 20, 30, 50)},
                "mean_ms_per_patient": round(v["total_ms"] / max(1, len(pids)), 1),
                "n_lesions": v["n_lesions"],
            }
            print(f"  {k:12s} " + "  ".join(f"K{kk}={summary[k]['recall_k1_froc'][str(kk)]}" for kk in (10, 20, 30, 50))
                  + f"   ({summary[k]['mean_ms_per_patient']:.0f} ms/hasta)")

        report["phase2"][split] = {
            "methods": summary,
            "best_trained_for_M7": best_trained,
            "m7_selected": m7_key,
        }

    # merge into existing benchmark json
    if BENCH_JSON.exists():
        base = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    else:
        base = {"experiment": "V6.0", "stage": "1. asama aday uretici benchmark", "config": {}}
    base["phase2"] = report["phase2"]
    base["phase2_config"] = {"epochs": args.epochs, "input": INPUT, "stride": STRIDE,
                             "output_map": f"{HMP}x{HMP} (orijinal 1024'e stride=4)",
                             "backbone": "ResNet18 stride-2 (maxpool yok)",
                             "loss": "M5: Gaussian focal(a2,b4); M6: BCE+smoothL1",
                             "training": "1:1 nodul + temiz kontrol (bos hedef)",
                             "m7": "ortogonal kota: eğitimli %70 + M2 %30 (dist>20px), iç-içe"}
    BENCH_JSON.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor güncellendi: {BENCH_JSON}")


if __name__ == "__main__":
    main()
