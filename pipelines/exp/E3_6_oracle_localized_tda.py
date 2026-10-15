"""E3.6 — Oracle-lokalize TDA testi: lokal topolojik sinyal HİÇ var mı?

Neden bu deney:
    E4.7 modellerin nodülü lokalize ETMEDİĞİNİ, E1.2 ise arka plan topolojisinin
    aslında FAYDALI olduğunu gösterdi. Bu, "lokal nodül sinyali zayıf" tezini
    destekliyor. Ama **doğrudan test edilmedi**: nodülün tam üzerinde hesaplanan
    TDA özellikleri, nodülsüz kontrol yamalarından ayırt edilebiliyor mu?

    Ground-truth bbox kullanarak ("oracle") nodül merkezine oturan bir yama
    çıkarıp TDA hesaplıyoruz. Eğer bu bile ayırt edemiyorsa, **lokal sinyal yoktur**
    ve aday-bazlı mimari (E3.3) de işe yaramaz.

Ön-kayıtlı hipotez:
    Nodül merkezine oturan yamaların TDA özellikleri, eşleştirilmiş kontrol
    yamalarından **val ROC-AUC ≥ 0.70** ile ayırt edilebilir.

Tasarım:
    Tek değişken: yama içeriği (nodül merkezi vs kontrol).
    Özellik: 800 boyutlu PI (aynı extractor). Sınıflandırıcı: lojistik regresyon
    (deterministik, seed varyansı yok). Hasta split'i sabit.

Kapsam (test bütçesi korunur):
    Yamalar hasta splitine göre train/val'e ayrılır; **test hastaları kullanılmaz**.

Kullanım:
    python pipelines/exp/E3_6_oracle_localized_tda.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402

EXPERIMENT_ID = "E3.6"
SLUG = "oracle_localized_tda"
SEED = 42
PATCH_SCALE = 2.5          # yama kenari = PATCH_SCALE * max(bbox_w, bbox_h)
MIN_PATCH = 48
NEG_PER_POS = 1
SUCCESS_AUC = 0.70
FAIL_AUC = 0.60
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3.6 oracle lokalize TDA testi")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--patients-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def clamp_patch(cy: int, cx: int, h: int, w: int, half: int) -> Tuple[int, int, int, int]:
    y0 = max(0, cy - half); y1 = min(h, cy + half)
    x0 = max(0, cx - half); x1 = min(w, cx + half)
    return y0, y1, x0, x1


_W: Dict[str, Any] = {}


def _init_worker() -> None:
    _W["ex"] = ZoneTDAExtractor(TDASettings())


def process_zone(item: Tuple[str, int, str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Bir zon icin tum yamalari cikarir."""
    pid, zid, path_str, specs = item
    ex: ZoneTDAExtractor = _W["ex"]
    out: List[Dict[str, Any]] = []
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if img.ndim == 3:
            img = img[0]
        h, w = img.shape
        for sp in specs:
            y0, y1, x0, x1 = clamp_patch(sp["cy"], sp["cx"], h, w, sp["half"])
            patch = img[y0:y1, x0:x1]
            if patch.size == 0 or (patch > 0).sum() < 200:
                continue
            vec, status = ex.extract(patch)
            if status != "ok":
                continue
            out.append({"patient_id": pid, "zone_id": zid, "kind": sp["kind"],
                        "split": sp["split"], "vector": vec})
    except Exception:
        return out
    return out


def build_specs(args) -> Tuple[List[Tuple[str, int, str, List[Dict[str, Any]]]], Dict[str, int]]:
    """Pozitif yamalar + BOYUT-ESLESTIRILMIS negatif yamalar.

    KRITIK: her negatif yama, eslestigi pozitif yamayla AYNI `half` degerini kullanir.
    Aksi halde siniflandirici nodul yerine yama BOYUTUNU ogrenir (confound).
    """
    splits = pd.read_csv(args.splits_path)
    p2s = splits.drop_duplicates("patient_id").set_index("patient_id")["split"].to_dict()
    zones = splits.set_index(["patient_id", "zone_id"])

    rng = np.random.RandomState(SEED)
    tasks: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    stats = {"pos": 0, "neg": 0, "skipped": 0}
    positives: List[Dict[str, Any]] = []
    neg_pool: Dict[str, List[Tuple[str, int, int, int]]] = {"train": [], "val": []}

    for pdir in sorted(args.patients_dir.iterdir()):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        split = p2s.get(pid)
        if split not in ("train", "val"):
            continue
        info_path = pdir / "patient_info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        zinfo = {int(k): v for k, v in info.get("zones", {}).items()}

        # temiz-negatif zon havuzu (boyut eslestirmesi icin)
        for zid in range(1, 6):
            try:
                row = zones.loc[(pid, zid)]
            except KeyError:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if row["label_state"] == "negative" and zid in zinfo:
                shp = zinfo[zid].get("crop_shape")
                if shp:
                    neg_pool[split].append((pid, zid, int(shp[0]), int(shp[1])))

        # POZITIF yamalar
        for nod in info.get("nodules", []):
            zid = int(nod.get("assigned_zone_id") or 0)
            if zid not in ZONE_FILES or zid not in zinfo:
                stats["skipped"] += 1
                continue
            cb = zinfo[zid].get("crop_bbox")
            if not cb:
                stats["skipped"] += 1
                continue
            cy = int(round(nod["center_y"] - cb["min_y"]))
            cx = int(round(nod["center_x"] - cb["min_x"]))
            size = max(int(round(PATCH_SCALE * max(nod["bbox_w"], nod["bbox_h"]))), MIN_PATCH)
            half = size // 2
            tasks.setdefault((pid, zid), []).append(
                {"kind": "positive", "split": split, "cy": cy, "cx": cx, "half": half})
            positives.append({"patient_id": pid, "zone_id": zid, "half": half, "split": split})
            stats["pos"] += 1

    # BOYUT-ESLESTIRILMIS negatifler: her pozitif icin ayni `half` ile bir negatif
    for pos in positives:
        pool = neg_pool[pos["split"]]
        if not pool:
            continue
        npid, nzid, h, w = pool[rng.randint(0, len(pool))]
        half = pos["half"]
        if h <= 2 * half or w <= 2 * half:
            half = max(MIN_PATCH // 2, min(half, min(h, w) // 2 - 1))
        if half < 8:
            continue
        cy = int(rng.randint(half, max(half + 1, h - half)))
        cx = int(rng.randint(half, max(half + 1, w - half)))
        tasks.setdefault((npid, nzid), []).append(
            {"kind": "negative", "split": pos["split"], "cy": cy, "cx": cx, "half": half})
        stats["neg"] += 1

    items = [(pid, zid, str(args.patients_dir / pid / ZONE_FILES[zid]), specs)
             for (pid, zid), specs in tasks.items()]
    return items, stats


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("E3.6 — ORACLE-LOKALIZE TDA: LOKAL TOPOLOJIK SINYAL VAR MI?")
    print("=" * 78)
    items, stats = build_specs(args)
    print(f"Zon gorevi: {len(items)} | pozitif yama: {stats['pos']} | negatif yama: {stats['neg']} "
          f"| atlanan: {stats['skipped']}")
    print("TEST HASTALARI KULLANILMIYOR.")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        done = 0
        for res in pool.map(process_zone, items, chunksize=4):
            rows.extend(res)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(items)} zon | {len(rows)} yama | {time.time()-t0:.0f} s", flush=True)
    print(f"  Cikarim tamamlandi: {len(rows)} yama, {time.time()-t0:.0f} s")

    df = pd.DataFrame(rows)
    X = np.stack(df["vector"].values)
    y = (df["kind"] == "positive").astype(int).values
    tr = (df["split"] == "train").values
    va = (df["split"] == "val").values
    print(f"\n  train yama: {int(tr.sum())} (poz {int(y[tr].sum())}) | val yama: {int(va.sum())} (poz {int(y[va].sum())})")

    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=0.01, max_iter=3000, random_state=SEED).fit(scaler.transform(X[tr]), y[tr])
    val_auc = float(roc_auc_score(y[va], clf.predict_proba(scaler.transform(X[va]))[:, 1]))
    tr_auc = float(roc_auc_score(y[tr], clf.predict_proba(scaler.transform(X[tr]))[:, 1]))

    print("\n--- Yama duzeyi ROC-AUC (nodul merkezi vs kontrol) ---")
    print(f"  train AUC: {tr_auc:.4f}")
    print(f"  VAL   AUC: {val_auc:.4f}")

    if val_auc >= SUCCESS_AUC:
        decision, msg = "KEEP", "Lokal sinyal VAR -> aday-bazli mimari (E3.3) umut verici."
    elif val_auc < FAIL_AUC:
        decision, msg = "KILL", "Lokal sinyal YOK -> aday-bazli mimari (E3.3) de ise yaramaz."
    else:
        decision, msg = "KILL", "Lokal sinyal ZAYIF -> aday-bazli mimari riskli."
    print(f"\n  Esikler: >={SUCCESS_AUC} KEEP | <{FAIL_AUC} KILL (yok)")
    print(f"  KARAR: {decision} — {msg}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Nodul merkezine oturan yamalarin TDA ozellikleri kontrol yamalarindan val AUC >= 0.70 ile ayrilir.",
        "scope": "train+val yamalari; TEST hastalari kullanilmadi",
        "patch_scale": PATCH_SCALE, "min_patch": MIN_PATCH,
        "n_positive_patches": int(stats["pos"]), "n_negative_patches": int(stats["neg"]),
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "train_auc": tr_auc, "val_auc": val_auc,
        "thresholds": {"success": SUCCESS_AUC, "fail": FAIL_AUC},
        "interpretation": msg,
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=decision,
        metrics={"val_patch_auc": val_auc, "train_patch_auc": tr_auc},
        notes=f"Oracle-lokalize yama TDA: val AUC={val_auc:.4f}; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
