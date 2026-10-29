"""V6.9d — NODE21 kaynak örtüşmesinin ölçümü (kaynak-ayrıştırmalı değerlendirme).

SORUN
-----
NODE21'in **halka açık eğitim seti** yeni hastane verisi değil, dört açık veri kümesinden
gelen nodüllü röntgenlerin + bir kontrol havuzunun yeniden etiketlenmesidir. Verinin kendi
`patient_info.json` alanı bunu söyler (4.882 hasta):

    node21_control  3.748  (hepsi label 0)
    chestxray14       617  (hepsi label 1)   = NIH ChestX-ray14
    padchest          314  (hepsi label 1)
    jsrt              149  (hepsi label 1)
    openi              54  (hepsi label 1)

Kullandığımız omurga `resnet50-res512-all`, torchxrayvision'da **"pc-nih-rsna-siim-vin"**
ile ön-eğitilmiştir ⇒ `pc` (PadChest) ve `nih` (ChestX-ray14) **görülmüştür**; JSRT ve
OpenI **görülmemiştir**. Yani pozitiflerin **%82,9'u** omuranın gördüğü korpuslardan.

Ölçmek istediğimiz: bu örtüşme skoru **şişiriyor mu?**
Omurga bu görüntüleri **kaba rapor etiketiyle** (nodule=1) gördü, **kutu koordinatı
görmedi** ⇒ lokalizasyon metrikleri (IoU>0,2) yapısal olarak sızıntıya kapalı olmalı.
Şüpheli olan **hasta düzeyi AUC**dir. Test: aynı kilitli boru hattının performansı
**görülmüş** (chestxray14, padchest) vs **görülmemiş** (jsrt, openi) kaynaklarda ayrışıyor mu?

YÖNTEM (eğitim YOK, yeni seçim YOK)
-----------------------------------
Kilitli protokol (M5_foundation + 2. aşama, 3 ölçek + flip, NMS 8 px, 56 px kutu, w=0,55)
tek kez koşulur; sonra aynı aday listesi kaynak gruplarına göre ayrıştırılır:
  * AUC: grup pozitifleri vs kontroller (hasta düzeyi, maksimum aday skoru)
  * Sens@t: yalnız o grubun lezyonları TP sayılır, FP ekseni **tüm** görüntü kümesi üzerinden
Doğrulama: tam küme metrikleri mühür kaydıyla birebir uyuşmazsa betik durur.

Kullanım: python pipelines/v6_two_stage/v6_9d_source_overlap.py [--split test|val|both]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_9_test_seal as seal  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import iou  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9d_source_overlap.json"
SEALED = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9_test_seal.json"

# Omurganın (pc-nih-rsna-siim-vin) ön-eğitiminde GÖRÜLEN kaynaklar
SEEN = {"chestxray14", "padchest"}
UNSEEN = {"jsrt", "openi"}
FROC = (0.125, 0.25, 0.5)


def _auc(y, s) -> float | None:
    y, s = np.asarray(y, float), np.asarray(s, float)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return None
    o = np.argsort(-s)
    yy = y[o]
    tpr = np.concatenate([[0], np.cumsum(yy) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - yy) / N])
    return round(float(np.trapezoid(tpr, fpr)), 4)


def _stratified(recs, pids, gts, ref, w):
    """Kaynak grubuna göre AUC + Sens@t (yalnız o grubun lezyonları TP sayılır)."""
    src = {p: str(json.loads(
        (common.PATIENTS_DIR / p / "patient_info.json").read_text(encoding="utf-8")
    ).get("source_dataset")) for p in pids}

    # hasta düzeyi skor (füzyon, sigmoid) — official_metric ile aynı yol
    pscore, flat = {}, []
    for pid in pids:
        r = recs[pid]
        sc = seal._fuse(r, ref, w)
        sc = 1.0 / (1.0 + np.exp(-sc)) if len(sc) else sc
        pscore[pid] = float(sc.max()) if len(sc) else 0.0
        for i, (cx, cy, _o) in enumerate(r["cs"]):
            flat.append((float(sc[i]), pid, float(cx), float(cy)))
    flat.sort(key=lambda t: -t[0])

    ctrl = [p for p in pids if not recs[p]["is_nodule"]]
    pos_by_src = {}
    for p in pids:
        if recs[p]["is_nodule"]:
            pos_by_src.setdefault(src[p], []).append(p)

    # ölçüm varlıkları: her kaynak + iki birleşim
    entities = {}
    for s in sorted(pos_by_src):
        entities[s] = {"pids": pos_by_src[s], "kind": "src"}
    entities["SEEN_union"] = {"pids": [p for s in SEEN for p in pos_by_src.get(s, [])],
                              "kind": "seen"}
    entities["UNSEEN_union"] = {"pids": [p for s in UNSEEN for p in pos_by_src.get(s, [])],
                                "kind": "unseen"}
    for e in entities.values():
        e["n_lesions"] = sum(len(gts[p]) for p in e["pids"])
        e["label"] = set(e["pids"])
        e["auc"] = _auc([1] * len(e["pids"]) + [0] * len(ctrl),
                        [pscore[p] for p in e["pids"]] + [pscore[p] for p in ctrl]) \
            if e["pids"] else None
    # birleşimlerin TP'si kaynak TP'lerinin toplamıdır (aynı lezyon iki kaynakta olamaz)
    member = {}
    for e_name, e in entities.items():
        member[e_name] = set(e["pids"])

    matched = {p: set() for p in pids}
    fp, tp = 0, {k: 0 for k in entities}
    curves = {k: [] for k in entities}
    for sc, pid, cx, cy in flat:
        box = (cx - seal.BOX_HALF, cy - seal.BOX_HALF, cx + seal.BOX_HALF, cy + seal.BOX_HALF)
        best, hit = 0.0, None
        for gi, g in enumerate(gts[pid]):
            v = iou(box, g)
            if v > best:
                best, hit = v, gi
        if hit is not None and best > 0.2:
            if hit in matched[pid]:
                continue
            matched[pid].add(hit)
            for k in entities:
                if pid in member[k]:
                    tp[k] += 1
        else:
            fp += 1
        for k in entities:
            n_les = entities[k]["n_lesions"]
            if n_les:
                curves[k].append((fp / len(pids), tp[k] / n_les))

    out = {}
    for k, e in entities.items():
        sens = {}
        for t in FROC:
            cand = [sn for f, sn in curves[k] if f <= t + 1e-9]
            sens[f"{t:g}"] = round(max(cand) if cand else 0.0, 4)
        out[k] = {"n_patients": len(e["pids"]), "n_lesions": int(e["n_lesions"]),
                  "auc_vs_control": e["auc"], "sens": sens,
                  "group": ("SEEN" if k in ("SEEN_union",) or k in SEEN
                            else ("UNSEEN" if k in ("UNSEEN_union",) or k in UNSEEN
                                  else "?"))}
    out["__node21_control__"] = {"n_patients": len(ctrl), "n_lesions": 0}
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["test", "val", "both"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    ref = sealed["frozen_z_reference"]
    w = sealed["locked"]["w_fusion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.9d — kaynak örtüşmesi | device={device} | w={w}", flush=True)

    import joblib
    bundle = joblib.load(seal.CKPT / seal.STAGE2_FILE)
    model = load_foundation(seal.STAGE1_TAG, device)
    sp = common.split_patients()
    rep = {"experiment": "V6.9d-source-overlap",
           "question": ("Omurganın ön-eğitiminde gördüğü kaynaklar (chestxray14, padchest) "
                        "ile görmediği kaynaklar (jsrt, openi) arasında performans ayrışıyor mu?"),
           "seen": sorted(SEEN), "unseen": sorted(UNSEEN),
           "backbone_corpora": "pc-nih-rsna-siim-vin", "w_fusion": w,
           "note": ("Eğitim/seçim YOK. Kilitli protokol bir kez koşulur; aynı aday listesi "
                    "kaynaklara göre ayrıştırılır."),
           "splits": {}}

    for split in (["test", "val"] if args.split == "both" else [args.split]):
        pids = list(sp[split])
        recs = seal._pass(pids, model, device, bundle["ranges"], bundle, split.upper())
        gts = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
                   for g in recs[p]["gts"]] for p in pids}

        # mühür doğrulaması (yalnız test)
        if split == "test":
            a1 = seal._official(recs, pids, "s1")
            a2 = seal._official(recs, pids, "fus", ref)
            for nm, got, want in (("A1", a1, sealed["test"]["A1_stage1_only"]),
                                  ("A2", a2, sealed["test"]["A2a_fusion_tda"])):
                for k in ("rank_metric", "auc"):
                    if abs((got[k] or 0) - (want[k] or 0)) > 1e-6:
                        raise SystemExit(f"MÜHÜR UYUŞMAZLIĞI {nm}.{k}: {got[k]} vs {want[k]}")
            print(f"  ✔ {split}: mühürle birebir aynı (rank {a2['rank_metric']}, "
                  f"auc {a2['auc']})", flush=True)

        rep["splits"][split] = _stratified(recs, pids, gts, ref, w)
        b = rep["splits"][split]
        print(f"\n  [{split}]", flush=True)
        for g in ("chestxray14", "padchest", "jsrt", "openi",
                  "SEEN_union", "UNSEEN_union"):
            if g in b:
                d = b[g]
                print(f"    {g:13s} n={d['n_patients']:3d} lez={d['n_lesions']:3d} "
                      f"AUC={d['auc_vs_control']} S@.125={d['sens']['0.125']} "
                      f"S@.25={d['sens']['0.25']} S@.5={d['sens']['0.5']}", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
