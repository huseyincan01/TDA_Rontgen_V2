"""V6.0 — 1. Aşama Aday Üretici Benchmark (oracle Recall @ K). FAZ 1: M1–M4 (eğitimsiz).

Disk kuralı: görüntü/kırpıntı YAZILMAZ; yalnızca JSON rapor (artifacts/v6_two_stage/).
Kullanım:
    python pipelines/v6_two_stage/benchmark_v6_0.py --phase 1 --split val --workers 10
    python pipelines/v6_two_stage/benchmark_v6_0.py --phase 1 --split val --limit 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
from candidates_classical import GENERATORS  # noqa: E402

K_VALUES = [10, 20, 30, 50]
OUT_JSON = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_0_candidate_benchmark.json"


def _empty_method():
    return {
        "n_lesions": 0,
        "hits_k1": {str(k): 0 for k in K_VALUES},
        "hits_k2": {str(k): 0 for k in K_VALUES},
        "sub_k1_K20": {"kucuk": [0, 0], "orta": [0, 0], "buyuk": [0, 0]},
        "sub_k2_K20": {"kucuk": [0, 0], "orta": [0, 0], "buyuk": [0, 0]},
        "total_ms": 0.0,
    }


def _process_patient(args):
    pid, methods = args
    out = {}
    try:
        img = common.load_image(pid)
        mask = common.load_lung_mask(pid)
        p1p99 = common.get_lung_p1p99().get(pid)
        if p1p99 and np.isfinite(p1p99[0]) and np.isfinite(p1p99[1]):
            p1, p99 = float(p1p99[0]), float(p1p99[1])
        else:
            p1, p99 = np.percentile(img[mask], [1, 99])
        canon = common.canonicalize(img, mask, p1, p99)
        gts = common.load_gt(pid)
    except Exception as exc:
        return pid, {"__error__": f"{type(exc).__name__}: {exc}"}

    for mname in methods:
        res = _empty_method()
        gen = GENERATORS[mname]
        t0 = time.perf_counter()
        try:
            cands = gen(canon, mask)
        except Exception as exc:
            res["__error__"] = f"{type(exc).__name__}: {exc}"
            out[mname] = res
            continue
        ms = (time.perf_counter() - t0) * 1000.0
        res["total_ms"] = ms
        res["n_lesions"] = len(gts)

        for k in K_VALUES:
            res["hits_k1"][str(k)] = int(round(common.recall_at_K(cands, gts, k, common.hit_criterion1) * len(gts)))
            res["hits_k2"][str(k)] = int(round(common.recall_at_K(cands, gts, k, common.hit_criterion2) * len(gts)))

        for gt in gts:
            sg = common.subgroup_of(gt["size_mm"])
            hit1 = any(common.hit_criterion1(cx, cy, gt) for cx, cy, _ in cands[:20])
            hit2 = any(common.hit_criterion2(cx, cy, gt) for cx, cy, _ in cands[:20])
            res["sub_k1_K20"][sg][1] += 1
            res["sub_k2_K20"][sg][1] += 1
            res["sub_k1_K20"][sg][0] += int(hit1)
            res["sub_k2_K20"][sg][0] += int(hit2)
        out[mname] = res
    return pid, out


def _aggregate(rows: list) -> dict:
    agg = {}
    for _, per_method in rows:
        for mname, r in per_method.items():
            if "__error__" in r:
                continue
            a = agg.setdefault(mname, _empty_method())
            a["n_lesions"] += r["n_lesions"]
            a["total_ms"] += r["total_ms"]
            for k in K_VALUES:
                a["hits_k1"][str(k)] += r["hits_k1"][str(k)]
                a["hits_k2"][str(k)] += r["hits_k2"][str(k)]
            for sg in ("kucuk", "orta", "buyuk"):
                a["sub_k1_K20"][sg][0] += r["sub_k1_K20"][sg][0]
                a["sub_k1_K20"][sg][1] += r["sub_k1_K20"][sg][1]
                a["sub_k2_K20"][sg][0] += r["sub_k2_K20"][sg][0]
                a["sub_k2_K20"][sg][1] += r["sub_k2_K20"][sg][1]
    return agg


def _summarize(agg: dict, n_patients: int) -> dict:
    out = {}
    for mname, a in agg.items():
        n = a["n_lesions"]
        out[mname] = {
            "recall_k1_froc": {str(k): round(a["hits_k1"][str(k)] / n, 4) if n else None for k in K_VALUES},
            "recall_k2_containment": {str(k): round(a["hits_k2"][str(k)] / n, 4) if n else None for k in K_VALUES},
            "subgroup_k1_K20": {sg: (round(v[0] / v[1], 4) if v[1] else None) for sg, v in a["sub_k1_K20"].items()},
            "subgroup_k2_K20": {sg: (round(v[0] / v[1], 4) if v[1] else None) for sg, v in a["sub_k2_K20"].items()},
            "mean_ms_per_patient": round(a["total_ms"] / n_patients, 1) if n_patients else None,
            "n_lesions": n,
        }
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="V6.0 aday üretici benchmark (Faz 1)")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
    ap.add_argument("--split", nargs="+", default=["val"])
    ap.add_argument("--methods", nargs="+", default=list(GENERATORS.keys()))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    report = {
        "experiment": "V6.0",
        "stage": "1. asama aday uretici benchmark (oracle recall @ K)",
        "phase": args.phase,
        "config": {
            "dilation_px": common.DILATION_PX, "candidate_box": "64x64",
            "canonicalization": "c_lungref (tum-akciger p1-p99)",
            "hit_criteria": {"kriter1": "FROC: merkez bbox icinde VEYA d<=max(16,0.5*R_bbox)",
                             "kriter2": "kapsama: IoBB>=0.50 VEYA IoU>=0.15"},
            "subgroup": "kucuk<15mm, orta 15-30, buyuk>30 (bbox maks kenar x spacing)",
            "K": K_VALUES,
            "methods_phase1": args.methods,
        },
        "splits": {},
    }

    for split in args.split:
        pids = common.split_patients()[split]
        if args.limit:
            pids = pids[: args.limit]
        print(f"Faz {args.phase} | split={split} | {len(pids)} hasta | methods={args.methods}")

        t0 = time.time()
        tasks = [(pid, args.methods) for pid in pids]
        if args.workers <= 1:
            rows = [_process_patient(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(_process_patient, tasks, chunksize=4))

        errors = {pid: r for pid, r in rows if "__error__" in r}
        ok_rows = [(pid, r) for pid, r in rows if "__error__" not in r]
        if errors:
            print(f"  ! {len(errors)} hasta hatayla atlandı: {list(errors.values())[:3]}")

        agg = _aggregate(ok_rows)
        summary = _summarize(agg, len(ok_rows))
        elapsed = time.time() - t0

        report["splits"][split] = {
            "n_patients_processed": len(ok_rows),
            "n_patients_error": len(errors),
            "methods": summary,
            "elapsed_sec": round(elapsed, 1),
        }

        print(f"\n=== {split.upper()} — ORACLE RECALL (Kriter 1 / FROC) @K ===")
        for m in args.methods:
            if m not in summary:
                continue
            r = summary[m]["recall_k1_froc"]
            print(f"  {m:28s} " + "  ".join(f"K{k}={r[str(k)]}" for k in K_VALUES)
                  + f"   ({summary[m]['mean_ms_per_patient']:.0f} ms/hasta)")

    report["disk_writes"] = "yalnizca bu JSON"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {out}")


if __name__ == "__main__":
    main()
