"""V6.9c — VinDr kontaminasyonunun ÖLÇÜLMESİ (temiz omurga deneyi).

SORUN
-----
V6.6.3 / V6.6.5'te "VinDr zero-shot" diye raporlanan sayılar `resnet50-res512-all`
ile üretildi. Bu omurga torchxrayvision'da *"trained on pc-nih-rsna-siim-**vin**"*
olarak tanımlı ve değerlendirme görüntülerimizin **400/400'ü (%100)** onun ön-eğitim
listesinde (bkz. LEAK_AUDIT.md §3). Yani sayılar sıfır-atış değil.

DENEY TASARIMI (tek değişken: ön-eğitim korpusu)
-----------------------------------------------
Aynı tarif, üç omurga — **yalnız omurganın gördüğü veri değişir**, mimari/kafa/veri/
tohum/takvim aynıdır:

  A) resnet50-res512-all      → VinDr **VAR**  (mevcut omurga; kontamine referans)
  B) densenet121-res224-all   → VinDr **VAR**  (farklı mimari, aynı kontaminasyon)
  C) densenet121-res224-nih   → VinDr **YOK**  (NIH-14; **temiz**)

Mantık: B vs C farkı **kontaminasyonun büyüklüğüdür** (mimari sabit). A satırı mevcut
sonucun aynı tarifteki karşılığıdır. NODE21 val ölçümü **kontrol koludur**: orada üç
omurga da VinDr görmemiş sayılır, farklar mimari/kapasite kaynaklıdır.

Protokol: omurga **DONUK** (V6.8 ısınma tarifi: 5 epoch, lr_head 1e-3, BN donuk,
aynı augmentasyon/kayıp/tohum). VinDr değerlendirmesi **strict** (AdaBN yok, tekil
görüntü) — V6.6.3 ile aynı ölçüm.

Kullanım:
    python pipelines/v6_two_stage/v6_9c_contamination.py            # 3 satır (~30 dk)
    python pipelines/v6_two_stage/v6_9c_contamination.py --rows C   # yalnız temiz omurga
    ... --train-limit 40 --vindr-limit 40                          # duman
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as v62  # noqa: E402
import v6_5d_strict_no_adabn as v65d  # noqa: E402
from detector_models_foundation import (FoundationCenterNet, canon_to_512,  # noqa: E402
                                        predict_tta)
from v6_7_train_fulldata import _aug_batch, _prep512, loss_fn  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9c_contamination.json"
WORK = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
# xrv ağırlıkları çalışma alanındaki önbellekten okunur: kısıtlı ortamda
# ~/.torchxrayvision yazılamıyor (PermissionError) ve indirme tekrarlanmasın.
CACHE = PROJECT_ROOT / "artifacts" / "xrv_weights"
SEED, BATCH, EPOCHS, LR_HEAD = 42, 4, 5, 1e-3

ROWS = {
    "A_resnet50_512_all_VINDR": ("resnet50-res512-all", True,
                                 "VinDr VAR (mevcut omurga)"),
    "B_densenet121_224_all_VINDR": ("densenet121-res224-all", True,
                                    "VinDr VAR (mimari kontrolü)"),
    "C_densenet121_224_nih_CLEAN": ("densenet121-res224-nih", False,
                                    "VinDr YOK (temiz omurga)"),
}


def sha_note() -> str:
    return "omurga DONUK · 5 epoch · lr_head 1e-3 · BN donuk · seed 42 · train 1:1"


# --------------------------------------------------------------------------- #
# eğitim (V6.8 ısınma tarifi, tek fark: --weights)
# --------------------------------------------------------------------------- #
def train_probe(weights: str, device, items, cache, epochs: int):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = FoundationCenterNet(weights=weights, cache_dir=str(CACHE)).to(device)
    model.radius_detach = True
    model.freeze_backbone()
    opt = torch.optim.Adam(model.param_groups(0.0, LR_HEAD), lr=LR_HEAD,
                           weight_decay=1e-5)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        model.set_bn_eval()
        rng = np.random.RandomState(SEED + ep)
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
        hist.append({"epoch": ep, "loss": round(tot / max(1, nb), 4),
                     "seconds": round(time.time() - t0, 1)})
        print(f"      epoch {ep}/{epochs} loss {hist[-1]['loss']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model, hist


# --------------------------------------------------------------------------- #
# NODE21 val (kontrol kolu)
# --------------------------------------------------------------------------- #
def eval_node21(model, device, limit: int = 0):
    pids = common.split_patients()["val"]
    if limit:
        pids = pids[:limit]
    gts = {p: common.load_gt(p) for p in pids}
    n_les = sum(len(g) for g in gts.values())
    hits = {k: 0 for k in (10, 20, 50)}
    recs = []
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        cs = [tuple(c) for c in predict_tta(model, canon, device)]
        if gts[pid]:
            for k in hits:
                hits[k] += int(round(common.recall_at_K(cs, gts[pid], k,
                                                        common.hit_criterion1)
                                     * len(gts[pid])))
        labs = s2._labels_for(cs, gts[pid])
        recs.append({"pid": pid, "is_nodule": bool(gts[pid]),
                     "cands": [(cs[j][0], cs[j][1], float(cs[j][2]), labs[j])
                               for j in range(len(cs))]})
        if (i + 1) % 200 == 0:
            print(f"      val {i + 1}/{len(pids)}", flush=True)
    fr, au = v62.froc_fixed(recs, len(recs), n_les)
    return {"n_patients": len(pids), "n_lesions": int(n_les),
            "recall_at_K": {str(k): round(v / max(1, n_les), 4) for k, v in hits.items()},
            "froc": fr, "patient_auc": au}


# --------------------------------------------------------------------------- #
# VinDr-CXR strict (hedef kol)
# --------------------------------------------------------------------------- #
def eval_vindr(model, device, data):
    from sklearn.metrics import roc_auc_score

    y = np.array([d["y"] for d in data])
    cands = {}
    t0 = time.time()
    for i, d in enumerate(data):
        cands[d["pid"]] = predict_tta(model, d["canon"].astype(np.float32), device,
                                      shape=d["canon"].shape)
        if (i + 1) % 100 == 0:
            print(f"      vindr {i + 1}/{len(data)} ({time.time() - t0:.0f}s)", flush=True)

    auc = {}
    for agg in ("top1", "top3", "noisyor"):
        sc = []
        for d in data:
            ss = sorted([c[2] for c in cands[d["pid"]]], reverse=True)
            if not ss:
                sc.append(0.0)
            elif agg == "top1":
                sc.append(float(ss[0]))
            elif agg == "top3":
                sc.append(float(np.mean(ss[:3])))
            else:
                sc.append(float(1 - np.prod([1 - min(1.0, max(0.0, s)) for s in ss[:5]])))
        auc[agg] = round(float(roc_auc_score(y, sc)), 4)

    pos = [d for d in data if d["gts"]]
    n_les = sum(len(d["gts"]) for d in pos)
    rec = {}
    for k in (30, 50):
        hit = sum(int(round(common.recall_at_K(cands[d["pid"]], d["gts"], k,
                                               common.hit_criterion1) * len(d["gts"])))
                  for d in pos)
        rec[f"K{k}"] = round(hit / max(1, n_les), 4)
    return {"n_images": len(data), "n_positive": int(y.sum()),
            "patient_auc": auc, "bbox_recall": rec}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="A,B,C", help="hangi omurgalar (A,B,C)")
    ap.add_argument("--train-limit", type=int, default=0, help="sınıf başına (duman)")
    ap.add_argument("--vindr-limit", type=int, default=0, help="VinDr görüntü (duman)")
    ap.add_argument("--val-limit", type=int, default=0, help="val hasta (duman)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    want = [r.strip().upper() for r in args.rows.split(",") if r.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.9c — VinDr kontaminasyon deneyi | device={device} | satırlar={want}",
          flush=True)

    # ---- NODE21 eğitim verisi (1:1, V6.6/V6.8 ile aynı)
    pos = p2._nodule_pids("train")
    neg = p2._control_pids("train")[:len(pos)]
    if args.train_limit:
        pos, neg = pos[:args.train_limit], neg[:args.train_limit]
    print(f"  train: {len(pos)} nodül + {len(neg)} kontrol", flush=True)
    cache, t0 = {}, time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pid, c512, cen, rad in pool.map(_prep512, list(pos) + list(neg), chunksize=8):
            cache[pid] = (c512, cen, rad)
    items = [(p, True) for p in pos if p in cache] + [(p, False) for p in neg if p in cache]
    print(f"  önbellek hazır ({len(cache)}, {time.time() - t0:.0f}s)", flush=True)

    # ---- VinDr kohortu (bir kez kur; 400 görüntü RAM'de)
    print("\nVinDr kohortu hazırlanıyor...", flush=True)
    import pandas as pd
    import torchxrayvision as xrv
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter

    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(v65d.EXT / "cohort_with_diameters.csv")
    if args.vindr_limit:
        cohort = cohort.head(args.vindr_limit)
    seg = PSPNet5ZoneSegmenter(device=device)
    data = v65d.build_vindr(seg, cohort, nod)
    print(f"  VinDr: {len(data)} görüntü, {sum(1 for d in data if d['y'])} pozitif",
          flush=True)

    rep_path = Path(args.out)
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {
        "experiment": "V6.9c-vindr-contamination",
        "question": ("VinDr 'zero-shot' sayıları ön-eğitim kontaminasyonundan ne kadar "
                     "şişmiş? Aynı tarif, yalnız omurganın gördüğü veri değişiyor."),
        "protocol": sha_note(),
        "rows": {}}

    for key in want:
        tag = next((k for k in ROWS if k.startswith(key + "_")), None)
        if tag is None:
            raise SystemExit(f"bilinmeyen satır: {key} (A/B/C)")
        weights, vindr_seen, desc = ROWS[tag]
        print(f"\n{'=' * 70}\n[{key}] {weights} — {desc}\n{'=' * 70}", flush=True)
        t_row = time.time()
        model, hist = train_probe(weights, device, items, cache, args.epochs)
        ck = WORK / f"contam_probe_{tag}.pt"
        WORK.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ck)
        print(f"    checkpoint: {ck.name}", flush=True)

        print("    NODE21 val (kontrol)...", flush=True)
        val = eval_node21(model, device, args.val_limit)
        print(f"      val K20={val['recall_at_K']['20']} K50={val['recall_at_K']['50']} "
              f"AUC nOR={val['patient_auc']['noisyor']}", flush=True)

        print("    VinDr strict (hedef)...", flush=True)
        vin = eval_vindr(model, device, data)
        print(f"      VinDr AUC top1={vin['patient_auc']['top1']} "
              f"nOR={vin['patient_auc']['noisyor']} | bbox {vin['bbox_recall']}", flush=True)

        rep["rows"][tag] = {"weights": weights, "vindr_in_pretraining": vindr_seen,
                            "description": desc, "history": hist,
                            "node21_val": val, "vindr_strict": vin,
                            "checkpoint": f"submission_fulldata/{ck.name} (teşhis; git dışı)",
                            "seconds": round(time.time() - t_row, 1),
                            "smoke": bool(args.train_limit or args.vindr_limit
                                          or args.val_limit or args.epochs != EPOCHS)}
        rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
        del model
        torch.cuda.empty_cache()

    # ---- hüküm: B vs C
    r = rep["rows"]
    b, c = r.get("B_densenet121_224_all_VINDR"), r.get("C_densenet121_224_nih_CLEAN")
    if b and c:
        for k in ("top1", "noisyor"):
            dv = (b["vindr_strict"]["patient_auc"][k] - c["vindr_strict"]["patient_auc"][k])
            dn = (b["node21_val"]["patient_auc"]["noisyor"]
                  - c["node21_val"]["patient_auc"]["noisyor"])
            rep.setdefault("verdict", {})[f"delta_vindr_{k}_B_minus_C"] = round(dv, 4)
            rep["verdict"][f"delta_node21_nor_B_minus_C"] = round(dn, 4)
        print(f"\n  HÜKÜM: VinDr AUC farkı (B−C) = "
              f"{rep['verdict']['delta_vindr_top1_B_minus_C']:+.4f} (top1) · "
              f"NODE21 farkı = {rep['verdict']['delta_node21_nor_B_minus_C']:+.4f}",
              flush=True)
        print("  Okuma: fark yalnız VinDr'de büyük, NODE21'de küçükse ⇒ kontaminasyon "
              "kanıtlanmış olur.", flush=True)
        rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\n  kayıt: {rep_path}", flush=True)


if __name__ == "__main__":
    main()
