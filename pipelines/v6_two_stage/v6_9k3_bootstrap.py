"""V6.9k3 — LOSO farkının EŞLEŞTİRİLMİŞ bootstrap güven aralığı.

NEDEN
-----
V6.9k, JSRT'yi eğitimden çıkarınca JSRT metriklerinin düştüğünü gösterdi. Ama iki model
**aynı hastalar** üzerinde ölçüldüğü için bağımsız iki AUC farkı değil, **eşleştirilmiş**
bir farktır; ve JSRT yalnız 149 hasta. Bu betik, aynı hastaları **birlikte** yeniden
örnekleyerek Δ'nın (LOSO − ANA) bootstrap dağılımını üretir.

Ayrıca **fark-farkı (DiD)** hesaplanır:

    DiD = Δ(JSRT)  −  Δ(val, JSRT-dışı)

Δ(val) genel "daha az veri" maliyetidir (V6.9k2). DiD ≈ 0 ⇒ fark veri azalmasıyla açıklanır;
DiD < 0 ⇒ JSRT'ye özgü ek kayıp var (kaynak ezberi lehine kanıt).

Ölçülenler
----------
JSRT (149 pozitif): Recall@K10/20/50 · hasta AUC (1. aşama s1) · hasta AUC (füzyon) ·
                    aday düzeyi AUC (füzyon)
val (149 pozitif, JSRT-dışı): Recall@K10/20/50 · hasta AUC (1. aşama s1) — yalnız aday

İzolasyon: hiçbir artifact yazmaz (npz önbelleği hariç); ana mimari salt-okunur.

Kullanım: python pipelines/v6_two_stage/v6_9k3_bootstrap.py --reps 2000
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
import v6_9k_leave_jsrt_out as k  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, pick_peaks  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
OUT = ART / "v6_9k3_bootstrap.json"
CACHE = ART / "_v6_9k3_cache.npz"


def _cands(model, pids, device, tag) -> dict:
    out, t0 = {}, time.time()
    for pid in pids:
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, k.SCALES)
        cs = dist_nms(pick_peaks(heat, rad, k.WIN), k.NMS_PX, top=k.TOPK)
        out[pid] = [(float(cx), float(cy), float(s)) for cx, cy, s, _r in cs]
    print(f"    [{tag}] aday {len(pids)} hasta ({time.time() - t0:.0f}s)", flush=True)
    return out


def collect(pids_js, ctrl, pids_val, device) -> dict:
    """İki model için JSRT(val+test) tam skorlar + val adayları; npz'e önbellekler."""
    import joblib

    js_pos = [p for p in pids_js]
    gts = {p: common.load_gt(p) for p in js_pos + ctrl + pids_val}
    data = {"js_pids": np.array(js_pos), "ctrl_pids": np.array(ctrl),
            "val_pids": np.array(pids_val)}
    for tag, bundle_path in ((k.LOSO_TAG, k.LOSO_BUNDLE), (k.MAIN_TAG, k.MAIN_BUNDLE)):
        model = load_foundation(tag, device)
        bundle = joblib.load(bundle_path)
        ranges = bundle["ranges"]
        ref = bundle["fusion_ref"]

        c_js = _cands(model, js_pos, device, tag + "/jsrt")
        f_js = k.features(js_pos, c_js, ranges)
        sc_js = k.rescore(bundle, js_pos, c_js, f_js, gts)

        c_ct = _cands(model, ctrl, device, tag + "/ctrl")
        f_ct = k.features(ctrl, c_ct, ranges)
        sc_ct = k.rescore(bundle, ctrl, c_ct, f_ct, gts)

        c_va = _cands(model, pids_val, device, tag + "/val")

        data[f"{tag}_js_s1"] = np.array([sc_js[p][0] for p in js_pos], dtype=object)
        data[f"{tag}_js_tda"] = np.array([sc_js[p][1] for p in js_pos], dtype=object)
        data[f"{tag}_js_n"] = np.array([len(c_js[p]) for p in js_pos])
        data[f"{tag}_js_lab"] = np.array(
            [np.array([1 if lb is not None else 0
                       for lb in k.s2._labels_for(c_js[p], gts[p])], np.int64)
             for p in js_pos], dtype=object)
        data[f"{tag}_ct_s1"] = np.array([sc_ct[p][0] for p in ctrl], dtype=object)
        data[f"{tag}_ct_tda"] = np.array([sc_ct[p][1] for p in ctrl], dtype=object)
        data[f"{tag}_ref"] = np.array([ref["s1_mean"], ref["s1_std"],
                                       ref["tda_mean"], ref["tda_std"], ref["w"]])
        # val (yalnız 1. aşama)
        data[f"{tag}_va_top1"] = np.array([max([c[2] for c in c_va[p]]) if c_va[p] else 0.0
                                           for p in pids_val])
        data[f"{tag}_va_nles"] = np.array([len(gts[p]) for p in pids_val])
        data[f"{tag}_va_hit"] = np.array([
            [int(round(common.recall_at_K(c_va[p], gts[p], K, common.hit_criterion1)
                       * len(gts[p]))) if gts[p] else 0 for K in k.KS] for p in pids_val])
        print(f"    [{tag}] toplandı", flush=True)
    return data


def _recall_from_hits(hit_rows, n_les) -> dict:
    return {f"K{K}": round(float(hit_rows[:, i].sum() / max(1, n_les)), 4)
            for i, K in enumerate(k.KS)}


def _patient_auc(js_scores, ctrl_scores) -> float | None:
    y = [1] * len(js_scores) + [0] * len(ctrl_scores)
    return k._auc(y, list(js_scores) + list(ctrl_scores))


def _flat_arrays(d, tag):
    return d[f"{tag}_js_s1"], d[f"{tag}_js_tda"], d[f"{tag}_js_lab"]


def stats(d, idx_js, idx_ct, tag) -> dict:
    """Yeniden örneklenmiş hasta indeksleriyle metrikler."""
    s1, tda, lab = _flat_arrays(d, tag)
    ref = d[f"{tag}_ref"]
    n = d[f"{tag}_js_n"]
    counts = n[idx_js]
    total = int(counts.sum())
    if total == 0:
        return {}

    def gather(arr):
        return np.concatenate([np.asarray(arr[i], float) for i in idx_js]) \
            if len(idx_js) else np.zeros(0)

    g_s1, g_tda, g_lab = gather(s1), gather(tda), gather(lab)
    ref_l, ref_w = ref[:4], float(ref[4])
    refd = {"s1_mean": ref_l[0], "s1_std": ref_l[1],
            "tda_mean": ref_l[2], "tda_std": ref_l[3]}
    g_fus = k.fuse(g_s1, g_tda, refd, ref_w)

    def ctrl_scores(kind):
        out = []
        for i in idx_ct:
            s1c = np.asarray(d[f"{tag}_ct_s1"][i], float)
            if not len(s1c):
                out.append(0.0)
                continue
            if kind == "stage1":
                out.append(float(s1c.max()))
                continue
            tdc = np.asarray(d[f"{tag}_ct_tda"][i], float)
            z = k.fuse(s1c, tdc, refd, ref_w)
            out.append(float((1.0 / (1.0 + np.exp(-z))).max()))
        return np.array(out)

    def img_top1(arr_g, counts_):
        out, o = [], 0
        for c in counts_:
            seg = arr_g[o:o + c]
            out.append(float(seg.max()) if len(seg) else 0.0)
            o += c
        return np.array(out)

    js_s1_img = img_top1(g_s1, counts)
    js_fus_img = img_top1(1.0 / (1.0 + np.exp(-g_fus)), counts)
    out = {
        "label_auc_stage1": k._auc(g_lab, g_s1),
        "label_auc_fusion": k._auc(g_lab, g_fus),
        "patient_auc_stage1": _patient_auc(js_s1_img, ctrl_scores("stage1")),
        "patient_auc_fusion": _patient_auc(js_fus_img, ctrl_scores("fusion")),
    }
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.9k3 — eşleştirilmiş bootstrap | device={device}", flush=True)

    js = k.jsrt_patients()
    js_set = set(js["train"]) | set(js["val"]) | set(js["test"])
    val = [p for p in common.split_patients()["val"] if p not in js_set]
    gts_val = {p: common.load_gt(p) for p in val}
    val_pos = [p for p in val if gts_val[p]]
    val_ctrl = list(np.random.RandomState(42).permutation(
        [p for p in val if not gts_val[p]])[:250])
    js_pos = js["val"] + js["test"]          # dış test: hiçbiri eğitimde değil
    ctrl = list(np.random.RandomState(42).permutation(
        [p for p in val if not gts_val[p]])[:250])
    print(f"  JSRT dış (val+test): {len(js_pos)} hasta · kontrol {len(ctrl)} · "
          f"val(−JSRT) pozitif {len(val_pos)}", flush=True)

    if CACHE.exists() and not args.refresh:
        print(f"  önbellek: {CACHE.name}", flush=True)
        z = np.load(CACHE, allow_pickle=True)
        d = {key: z[key] for key in z.files}
    else:
        d = collect(js_pos, ctrl, val_pos + val_ctrl, device)
        np.savez_compressed(CACHE, **d)
        print(f"  önbellek yazıldı: {CACHE.name}", flush=True)

    n_js = len(js_pos)
    n_va_pos = len(val_pos)
    idx_va_ctrl = np.arange(n_va_pos, n_va_pos + len(val_ctrl))

    # ---- nokta tahminleri ---------------------------------------------- #
    full_js, full_ct = np.arange(n_js), np.arange(len(ctrl))
    full_va, full_vc = np.arange(n_va_pos), idx_va_ctrl
    pt = {}
    for tag in (k.LOSO_TAG, k.MAIN_TAG):
        m = stats(d, full_js, full_ct, tag)
        hit = d[f"{tag}_va_hit"]
        nles = int(d[f"{tag}_va_nles"][full_va].sum())
        va_rec = _recall_from_hits(hit[full_va], nles)
        va_auc = _patient_auc(d[f"{tag}_va_top1"][full_va], d[f"{tag}_va_top1"][full_vc])
        pt[tag] = {**m, "val_recall_at_K": va_rec, "val_patient_auc_stage1": va_auc}
    print("\n  nokta tahminleri:", flush=True)
    for tag in (k.LOSO_TAG, k.MAIN_TAG):
        print(f"    {tag:14s} {json.dumps(pt[tag], ensure_ascii=False)}", flush=True)

    def _num(v):
        return isinstance(v, (int, float)) and v is not None

    delta_pt = {key: (round(pt[k.LOSO_TAG][key] - pt[k.MAIN_TAG][key], 4)
                      if _num(pt[k.LOSO_TAG][key]) and _num(pt[k.MAIN_TAG][key]) else None)
                for key in pt[k.LOSO_TAG]}
    delta_pt["val_recall_at_K"] = {kk: round(pt[k.LOSO_TAG]["val_recall_at_K"][kk]
                                             - pt[k.MAIN_TAG]["val_recall_at_K"][kk], 4)
                                   for kk in pt[k.LOSO_TAG]["val_recall_at_K"]}

    # ---- eşleştirilmiş bootstrap --------------------------------------- #
    rng = np.random.RandomState(args.seed)
    keys_js = ("label_auc_stage1", "label_auc_fusion",
               "patient_auc_stage1", "patient_auc_fusion")
    boot_js = {key: [] for key in keys_js}
    boot_did = []
    for _ in range(args.reps):
        bj = rng.randint(0, n_js, n_js)
        bc = rng.randint(0, len(ctrl), len(ctrl))
        bl, bm = stats(d, bj, bc, k.LOSO_TAG), stats(d, bj, bc, k.MAIN_TAG)
        for key in keys_js:
            if bl.get(key) is not None and bm.get(key) is not None:
                boot_js[key].append(bl[key] - bm[key])
        # DiD — yalnız 1. aşama hasta AUC'si (val tarafı aday tabanlı)
        bv = rng.randint(0, n_va_pos, n_va_pos)
        bvc = rng.randint(0, len(val_ctrl), len(val_ctrl))
        dva = {}
        for tag in (k.LOSO_TAG, k.MAIN_TAG):
            dva[tag] = _patient_auc(d[f"{tag}_va_top1"][bv], d[f"{tag}_va_top1"][bvc])
        if (bl.get("patient_auc_stage1") is not None
                and bm.get("patient_auc_stage1") is not None
                and dva[k.LOSO_TAG] is not None and dva[k.MAIN_TAG] is not None):
            boot_did.append((bl["patient_auc_stage1"] - bm["patient_auc_stage1"])
                            - (dva[k.LOSO_TAG] - dva[k.MAIN_TAG]))

    def ci(vals):
        a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
        if a.size < 10:
            return {"note": "yetersiz örnek"}
        return {"mean": round(float(a.mean()), 4),
                "lo95": round(float(np.percentile(a, 2.5)), 4),
                "hi95": round(float(np.percentile(a, 97.5)), 4),
                "share_negative": round(float((a < 0).mean()), 3)}

    rep = {"experiment": "V6.9k3-bootstrap",
           "question": "LOSO − ANA farkı eşleştirilmiş bootstrap ile anlamlı mı?",
           "design": {"jsrt_eval": len(js_pos), "controls": len(ctrl),
                      "val_pos": len(val_pos), "reps": args.reps, "seed": args.seed},
           "point_estimates": pt,
           "delta_point": delta_pt,
           "paired_bootstrap_jsrt": {key: ci(vals) for key, vals in boot_js.items()},
           "did_delta_minus_val_delta_stage1_auc": ci(boot_did),
           "note": ("JSRT dış testi = NODE21 val+test içindeki JSRT hastaları (42 hasta); "
                    "bunların hiçbiri LOSO eğitiminde yok. Kontroller val'den (250, seed 42). "
                    "DiD: Δ(JSRT) − Δ(val, JSRT-dışı) — 1. aşama hasta AUC'si.")}

    print("\n  Δ(LOSO − ANA) nokta:", json.dumps(delta_pt, ensure_ascii=False), flush=True)
    for key, v in rep["paired_bootstrap_jsrt"].items():
        print(f"    {key:22s} {v}", flush=True)
    print(f"    DiD (1. aşama AUC)     "
          f"{rep['did_delta_minus_val_delta_stage1_auc']}", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
