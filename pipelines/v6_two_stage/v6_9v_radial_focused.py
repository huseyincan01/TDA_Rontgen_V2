"""V6.9v — Radyal Kübik TDA odaklı doğrulama (M2/M3 elendi, yalnız M1 + Pure1600).

NEDEN: V6.9u'da M2 (EDT) ve M3 (VR-Graf) çöktü; ama **M1 (kutu-merkezli radyal)** iki kısıtla
çalışmıştı:
  (i)  PI eksen aralıkları Pure1600'deki gibi **eğitimden türetilmemiş**, sabit ve aşırı geniş
       `b_range=(0,255)`, `p_range=(0,255)` idi ⇒ H1 ömürleri (0..~33) ızgaranın ilk satırına
       sıkışıyordu (grid collapse).
  (ii) Yalnız **sert** `alpha = 60.0` denendi; JSRT'de kutuların %47'si "göbeğin" dışında olduğu
       için alpha=60 aşırı ceza kesiyor olabilir.
Ayrıca Arm_5 **NODE21 Test (734)** üzerinde hiç ölçülmemişti.

BU BETİK (V6.9v):
  * 3 alpha {60, 40, 25}; her biri için eksen aralıkları **o kulvarın eğitim diyagramlarından**
    p1–p99 ile türetilir (Pure1600'ün `derive_ranges` mantığıyla birebir).
  * 4 varyant: `a60_fixed` (V6.9u'yu birebir üretmeli — harness kontrolü), `a60_cal`, `a40_cal`,
    `a25_cal`.
  * Kontrol kolu **Arm_0 = Pure1600**, aynı betikte aynı protokolle yeniden eğitilir.
  * Kohortlar: NODE21 Val 734 · NODE21 Test 734 (kilitli — 6. kullanım) · 242 Saf JSRT (kalibresiz).
  * Her kol kendi HGB'sini sıfırdan eğitir (yeni scaler; rs=42, max_iter=200, lr=0.08, balanced).
  * Füzyon kilitli `w_s1=0.55 / w_tda=0.45`; z-ref yalnız kalibrasyon kohortundan.

Çıktı: artifacts/v6_two_stage/v6_9v_radial_focused.json
    python pipelines/v6_two_stage/v6_9v_radial_focused.py            # tam koşu
    python pipelines/v6_two_stage/v6_9v_radial_focused.py --limit 30 # duman
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from v6_9t_geometric_graph_tda import (  # noqa: E402  metrik hattı V6.9u ile BİREBİR
    _cubical_h0_h1, _diag_stats6, _pi, _r_norm_grid, _gts, _labels, _slice_cohort,
    evaluate_predictions, merge_cohorts,
)

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
CACHE_L, CACHE_M, CACHE_R = ART / "_v6_9l_cache.npz", ART / "_v6_9m_test_cache.npz", ART / "_v6_9r_cache.npz"
CACHE_GEO = ART / "_v6_9t_geo_cache.npz"
OUT = ART / "v6_9v_radial_focused.json"

BOX_HALF, W_TDA, W_S1, SEED = 28.0, 0.45, 0.55, 42
ALPHAS = (60.0, 40.0, 25.0)
GRID = 10
VARIANTS = (("a60_fixed", 60.0, "fixed"), ("a60_cal", 60.0, "cal"),
            ("a40_cal", 40.0, "cal"), ("a25_cal", 25.0, "cal"))
VNAMES = tuple(v[0] for v in VARIANTS)
FIXED = {"h0_birth": [0.0, 255.0], "h0_pers": [0.0, 255.0],
         "h1_birth": [0.0, 255.0], "h1_pers": [0.0, 255.0]}
READONLY = [CKPT / "M5_foundation.pt", CKPT / "stage2_foundation_tda_hgb_pure1600.joblib",
            CKPT / "stage2_foundation_tda_hgb.joblib", CKPT / "M5_nojsrt_full.pt",
            CKPT / "stage2_nojsrt_full_tda_hgb_pure1600.joblib", CACHE_L, CACHE_M, CACHE_R]


def sha256(p: Path) -> str:
    if not p.exists():
        return "YOK"
    d = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            d.update(chunk)
    return d.hexdigest()


# --------------------------------------------------------------------------- #
# M1 (radyal) — parametrik eksen aralıklı sürüm
# --------------------------------------------------------------------------- #
def _radial_field(patch: np.ndarray, alpha: float) -> np.ndarray | None:
    valid = np.isfinite(patch) & (patch > 0)
    if np.count_nonzero(valid) < 4:
        return None
    rn = _r_norm_grid(patch.shape[0])
    f = np.full(patch.shape, 255.0, dtype=np.float32)
    f[valid] = np.clip((255.0 - patch[valid]) + alpha * rn[valid], 0.0, 255.0)
    return f


def _canon(pid):
    img = common.load_image(pid)
    mask = common.load_lung_mask(pid)
    p1, p99 = common.get_lung_p1p99()[pid]
    return common.canonicalize(img, mask, float(p1), float(p99))


def _worker(job):
    """Bir hasta: 3 alpha × 2 yama → 4 varyantın 424 boyutlu vektörleri.

    Diyagram (H0/H1) her (alpha, yama) için **BİR KEZ** hesaplanır; alpha=60 iki varyantta
    (fixed + cal) ortak kullanılır ⇒ %25 daha az cubical PH.
    """
    pid, cands, ranges = job
    canon = _canon(pid)
    out = {v: np.zeros((len(cands), 424), np.float32) for v in VNAMES}
    for k, (cx, cy, _s) in enumerate(cands):
        for off, size in ((0, s2.INNER), (212, s2.OUTER)):
            c = s2._crop(canon, cx, cy, size)
            for alpha in ALPHAS:
                f = _radial_field(c, alpha)
                if f is None:
                    continue
                h0, h1 = _cubical_h0_h1(f)
                names = [f"a{int(alpha)}_cal"]
                if alpha == 60.0:
                    names.insert(0, "a60_fixed")
                for name in names:
                    rng = FIXED if name == "a60_fixed" else ranges[alpha]
                    out[name][k, off:off + 212] = _pack(h0, h1, rng)
    return pid, out


def _pack(h0: np.ndarray, h1: np.ndarray, rng: dict) -> np.ndarray:
    """2×PI(10×10) + 2×6 özet = 212 boyut (verilen eksen aralıklarıyla)."""
    pi0 = _pi(h0, GRID, tuple(rng["h0_birth"]), tuple(rng["h0_pers"]), 1.0)
    pi1 = _pi(h1, GRID, tuple(rng["h1_birth"]), tuple(rng["h1_pers"]), 1.0)
    return np.concatenate([pi0, pi1, _diag_stats6(h0), _diag_stats6(h1)]).astype(np.float32)


def extract(cands_map, ranges, tag, pool, workers=10):
    pids = list(cands_map)
    jobs = [(p, cands_map[p], ranges) for p in pids]
    res, t0 = {}, time.time()
    for i, (pid, arrs) in enumerate(pool.map(_worker, jobs, chunksize=2), 1):
        res[pid] = arrs
        if i % 200 == 0:
            print(f"    radyal {tag} {i}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    radyal {tag}: {len(pids)} hasta ({time.time() - t0:.0f}s)", flush=True)
    return res


def cands_of(c):
    return {p: [(float(a), float(b), float(s)) for a, b, s in
                zip(c["cx"][int(c["off"][i]):int(c["off"][i + 1])],
                    c["cy"][int(c["off"][i]):int(c["off"][i + 1])],
                    c["s1"][int(c["off"][i]):int(c["off"][i + 1])])]
            for i, p in enumerate(c["pids"])}


# --------------------------------------------------------------------------- #
# Eksen aralıkları (v6_1.derive_ranges mantığı, radyal alan için)
# --------------------------------------------------------------------------- #
def derive_ranges(pids_pos, cands_map, n_pat=40, n_cand=10):
    acc = {a: {"h0b": [], "h0p": [], "h1b": [], "h1p": []} for a in ALPHAS}
    for pid in list(pids_pos)[:n_pat]:
        try:
            canon = _canon(pid)
        except Exception:  # noqa: BLE001
            continue
        for cx, cy, _s in cands_map[pid][:n_cand]:
            for size in (s2.INNER, s2.OUTER):
                c = s2._crop(canon, cx, cy, size)
                for a in ALPHAS:
                    f = _radial_field(c, a)
                    if f is None:
                        continue
                    h0, h1 = _cubical_h0_h1(f)
                    if len(h0):
                        acc[a]["h0b"].extend(h0[:, 0].tolist())
                        acc[a]["h0p"].extend((h0[:, 1] - h0[:, 0]).tolist())
                    if len(h1):
                        acc[a]["h1b"].extend(h1[:, 0].tolist())
                        acc[a]["h1p"].extend((h1[:, 1] - h1[:, 0]).tolist())

    def bnd(v, birth):
        v = np.asarray(v, float)
        if not v.size:
            return [0.0, 10.0]
        lo = float(np.floor(np.percentile(v, 1))) if birth else 0.0
        hi = float(np.ceil(np.percentile(v, 99)))
        return [lo, hi if hi > lo else lo + 10.0]

    return {a: {"h0_birth": bnd(acc[a]["h0b"], True), "h0_pers": bnd(acc[a]["h0p"], False),
                "h1_birth": bnd(acc[a]["h1b"], True), "h1_pers": bnd(acc[a]["h1p"], False)}
            for a in ALPHAS}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def fit_hgb(X, y, balanced=True):
    sc = StandardScaler().fit(X)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                       class_weight="balanced" if balanced else None,
                                       random_state=SEED)
    return sc, m.fit(sc.transform(X), y)


def predict(model, X):
    sc, m = model
    return m.predict_proba(sc.transform(X))[:, 1].astype(np.float32)


def _ref(s1_cal, p_cal):
    return {"s1_mean": float(s1_cal.mean()), "s1_std": float(s1_cal.std()),
            "tda_mean": float(p_cal.mean()), "tda_std": float(p_cal.std())}


def design(c, name, radial):
    if name == "Arm_0_Pure1600":
        return c["tda"]
    extra = np.concatenate([radial[p][name] for p in c["pids"]], axis=0)
    m = int(c["off"][-1])
    return np.concatenate([c["tda"][:m], extra[:m]], axis=1)



def _safe_eval(cohort, y, p, p0, ref):
    """Tek sınıflı kohortta official_metric AUC=None döndürür; koşuyu düşürmemek için sar."""
    try:
        return evaluate_predictions(cohort, y, p, p0, ref)
    except (TypeError, ValueError):
        return {"candidate_auc": None, "corr_with_s1": None, "corr_with_pure1600": None,
                "tda_only": {"patient_auc": None, "S@0.125": None, "S@0.25": None, "S@0.5": None,
                             "rank_metric": None, "top1_box_hit_rate": None},
                "fusion_w045": {"patient_auc": None, "S@0.125": None, "S@0.25": None, "S@0.5": None,
                                "rank_metric": None, "top1_box_hit_rate": None}}


def run_lane(lane, fit, cal, evals, radial, balanced=True):
    print(f"\n=== {lane} · fit {len(fit['y'])} aday · cal {len(cal['s1'])} · "
          f"eval {[(n, len(c['s1'])) for n, c in evals]} ===", flush=True)
    res, preds = {}, {}
    for name in ("Arm_0_Pure1600",) + VNAMES:
        t0 = time.time()
        mdl = fit_hgb(design(fit, name, radial), fit["y"], balanced=balanced)
        p_cal = predict(mdl, design(cal, name, radial))
        preds[name] = (p_cal, {n: (p_cal if c is cal else predict(mdl, design(c, name, radial)))
                               for n, c in evals})
        res[name] = {"n_features": int(design(cal, name, radial).shape[1])}
        print(f"  [{name}] eğitildi ({time.time() - t0:.0f}s · "
              f"{res[name]['n_features']} özellik)", flush=True)

    p0 = preds["Arm_0_Pure1600"][1]
    for name, (p_cal, per_eval) in preds.items():
        ref = _ref(cal["s1"], p_cal)
        res[name]["ref"] = {k: round(v, 4) for k, v in ref.items()}
        for ename, cohort in evals:
            res[name][ename] = _safe_eval(cohort, cohort["y"], per_eval[ename],
                                          p0[ename], ref)
    return res


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    lim = args.limit or None
    balanced = lim is None   # duman: class_weight kapali (hiz) · tam kosu: balanced ACIK

    t_start = time.time()
    print("=" * 104, flush=True)
    print("V6.9v — RADYAL KÜBİK TDA ODAKLI DOĞRULAMA (M1 + Pure1600 · 3 alpha · kalibre aralıklar)",
          flush=True)
    print(f"  limit={args.limit} | alpha={list(ALPHAS)} | w_tda={W_TDA} | "
          f"kohortlar: Val · Test · JSRT-242", flush=True)
    print("=" * 104, flush=True)
    before = {p.name: sha256(p) for p in READONLY}
    pool = ProcessPoolExecutor(max_workers=10)

    zl = np.load(CACHE_L, allow_pickle=True, mmap_mode="r")
    zr = np.load(CACHE_R, allow_pickle=True, mmap_mode="r")
    zm = np.load(CACHE_M, allow_pickle=True, mmap_mode="r")
    co = {"tr": _slice_cohort(zl, "tr", lim), "va": _slice_cohort(zl, "va", lim),
          "rfit": _slice_cohort(zr, "fit", lim), "rva": _slice_cohort(zr, "va", lim),
          "jp": _slice_cohort(zr, "jp", lim), "jn": _slice_cohort(zr, "jn", lim),
          "te": _slice_cohort(zm, "te", lim)}
    co["tr"]["y"] = zl["tr_y"][:int(co["tr"]["off"][-1])].astype(np.int64)
    co["va"]["y"] = zl["va_y"][:int(co["va"]["off"][-1])].astype(np.int64)
    co["rfit"]["y"] = zr["fit_y"][:int(co["rfit"]["off"][-1])].astype(np.int64)
    co["te"]["y"] = zm["te_y"][:int(co["te"]["off"][-1])].astype(np.int64)
    for k in ("rva", "jp", "jn"):
        co[k]["y"] = _labels(co[k])
    for k in ("va", "rva", "jp", "jn", "te"):
        co[k]["gts"], co[k]["is_nod"] = _gts(co[k]["pids"])

    print("\n[1] Radyal PI eksen aralıkları (p1–p99, eğitim diyagramlarından)...", flush=True)
    c_tr, c_rfit = cands_of(co["tr"]), cands_of(co["rfit"])
    pos_tr = [p for p in co["tr"]["pids"] if common.load_gt(p)]
    pos_rfit = [p for p in co["rfit"]["pids"] if common.load_gt(p)]
    rng_L2 = derive_ranges(pos_tr, c_tr, n_pat=10 if lim else 40)
    rng_L1 = derive_ranges(pos_rfit, c_rfit, n_pat=10 if lim else 40)
    for tag, rr in (("L2 (NODE21 train)", rng_L2), ("L1 (JSRT-dışı fit)", rng_L1)):
        print(f"    {tag}:", flush=True)
        for a in ALPHAS:
            print(f"      alpha={a:4.0f} · H0 doğum {rr[a]['h0_birth']} ömür {rr[a]['h0_pers']}"
                  f" · H1 doğum {rr[a]['h1_birth']} ömür {rr[a]['h1_pers']}", flush=True)

    print("\n[2] Preflight: a60_fixed ≡ V6.9t M1 (ilk 424 sütun) birebir mi?", flush=True)
    if CACHE_GEO.exists():
        cz = np.load(CACHE_GEO, allow_pickle=True)
        sample = list(cands_of(co["jp"]).items())[:2]
        if sample:
            got = extract(dict(sample), rng_L1, "preflight", pool)
            for p, _ in sample:
                i = co["jp"]["pids"].index(p)
                lo, hi = int(co["jp"]["off"][i]), int(co["jp"]["off"][i + 1])
                ref = cz["jp_geo"][lo:hi, :424]
                diff = float(np.abs(got[p]["a60_fixed"][:len(ref)] - ref).max())
                print(f"    {p}: max|Δ| {diff:.2e} · {'AYNI ✓' if diff < 1e-3 else 'FARKLI ✗'}",
                      flush=True)
        else:
            print("    (atlandı)", flush=True)
    else:
        print("    (atlandı: geo önbelleği yok)", flush=True)

    print("\n[3] Kulvar 2 (NODE21): radyal özellikler...", flush=True)
    rad_L2 = {}
    for tag, c in (("tr", co["tr"]), ("va", co["va"]), ("te", co["te"])):
        rad_L2.update(extract(cands_of(c), rng_L2, tag, pool))
    res_V = run_lane("Kulvar 2 — NODE21 Val 734", co["tr"], co["va"],
                     [("NODE21_Val", co["va"]), ("NODE21_Test", co["te"])], rad_L2,
                     balanced=balanced)
    del rad_L2

    print("\n[4] Kulvar 1 (kalibresiz 242 Saf JSRT): radyal özellikler...", flush=True)
    rad_L1 = {}
    for tag, c in (("rfit", co["rfit"]), ("rva", co["rva"]), ("jp", co["jp"]), ("jn", co["jn"])):
        rad_L1.update(extract(cands_of(c), rng_L1, tag, pool))
    ev1 = merge_cohorts(co["jp"], co["jn"])
    ev1["y"] = np.concatenate([co["jp"]["y"], co["jn"]["y"]])
    ev1["gts"] = {**co["jp"]["gts"], **co["jn"]["gts"]}
    ev1["is_nod"] = {**co["jp"]["is_nod"], **co["jn"]["is_nod"]}
    res_J = run_lane("Kulvar 1 — kalibresiz 242 Saf JSRT", co["rfit"], co["rva"],
                     [("JSRT_242", ev1)], rad_L1, balanced=balanced)

    pool.shutdown(wait=True)
    after = {p.name: sha256(p) for p in READONLY}
    report = {"experiment": "V6.9v-radial-focused", "limit": args.limit,
              "alphas": list(ALPHAS), "variants": list(VNAMES),
              "fusion": {"w_s1": W_S1, "w_tda": W_TDA},
              "ranges": {"L2_node21_train": rng_L2, "L1_jsrt_free_fit": rng_L1},
              "track_2_node21": res_V, "track_1_jsrt242": res_J,
              "sha256_unchanged": before == after,
              "test_budget_note": "NODE21 Test 6. kullanım; karar Val + JSRT ile verilmeli",
              "elapsed_s": round(time.time() - t_start, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def g(v):
        return ("%.4f" % v).replace(".", ",")

    print("\n" + "=" * 112, flush=True)
    print("V6.9v — RADYAL (M1) + PURE1600 · 3 ALPHA · KALİBRE EKSEN ARALIKLARI", flush=True)
    print("=" * 112, flush=True)
    for key, title in (("track_2_node21", "NODE21 — Val 734 (seçim) + Test 734 (6. kullanım)"),
                       ("track_1_jsrt242", "242 SAF JSRT — kalibresiz dış test")):
        arms = report[key]
        evals = [k for k in arms["Arm_0_Pure1600"] if k.startswith(("NODE21", "JSRT"))]
        print(f"\n### {title}", flush=True)
        head = f"{'Kol':22}{'boyut':>7}" + "".join(f"{e + ' rank/S.25':>22}" for e in evals)
        print(head, flush=True)
        for arm, r in arms.items():
            line = f"{arm:22}{r['n_features']:>7}"
            for e in evals:
                fo = (r.get(e) or {}).get("fusion_w045") or {}
                if fo.get("rank_metric") is None:
                    line += f"{'—':>22}"
                    continue
                line += f"{g(fo['rank_metric'])} / {g(fo['S@0.25']):>6}".rjust(22)
            print(line, flush=True)
        print("  ayrıntı — füzyon AUC / S@0.125 / S@0.25 / S@0.50 / rank / top1-hit ‖ TDA-tek AUC",
              flush=True)
        for arm, r in arms.items():
            for e in evals:
                fo = (r.get(e) or {}).get("fusion_w045") or {}
                t = (r.get(e) or {}).get("tda_only") or {}
                if fo.get("rank_metric") is None:
                    print(f"    {arm:22}{e:14} —", flush=True)
                    continue
                print(f"    {arm:22}{e:14} {g(fo['patient_auc'])} {g(fo['S@0.125'])} "
                      f"{g(fo['S@0.25'])} {g(fo['S@0.5'])} {g(fo['rank_metric'])} "
                      f"{g(fo['top1_box_hit_rate'])} ‖ {g(t['patient_auc'])}", flush=True)
        try:
            print("  aday düzeyi: " + " · ".join(
                f"{a}: AUC {g(arms[a][evals[0]]['candidate_auc'])} "
                f"r(s1) {g(arms[a][evals[0]]['corr_with_s1'])}" for a in arms), flush=True)
        except (TypeError, KeyError):
            print("  aday düzeyi: —", flush=True)

    print(f"\nyazıldı: {args.out}", flush=True)
    print(f"SHA-256 değişmedi: {report['sha256_unchanged']} · süre {report['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
