"""V6.9e — Omurga × kaynak ETKİLEŞİM testi (örtüşme avantaj sağlıyor mu?).

SORU
----
V6.9d, kilitli boru hattının **görülen** kaynaklarda (chestxray14, padchest) **görülmeyenlere**
(jsrt, openi) göre daha iyi olduğunu gösterdi (AUC 0,9698 vs 0,9332). Ama bu fark iki şeyle
karışık: (i) kaynakların cihaz/geometri farkı, (ii) modelin *kendisi*.

Bu betik, iki modeli **aynı protokolde** kıyaslayarak karıştırıcıyı azaltır:

    M5  = ImageNet ResNet18 (hiç CXR görmemiş)          → "temiz" omurga
    FND = xrv resnet50-res512-all (pc-nih-rsna-siim-vin) → NODE21 pozitiflerinin %82,9'unu görmüş

Her ikisi de **heat-map TTA** yoluyla aday üretir. Ölçülen: her kaynak grubunda hasta düzeyi
AUC, ve **Δ = FND − M5**.

    ETKİLEŞİM = Δ(görülen) − Δ(görülmeyen)

Mantık: FND'nin avantajı **her yerde aynıysa** (etkileşim ≈ 0) ⇒ avantaj "genel CXR
ön-eğitimi"dir, örtüşme kaynaklı değildir. Avantaj **yalnız görülen kaynaklarda** varsa
(etkileşim > 0) ⇒ örtüşme katkısı vardır.

MÜHÜR KURALI
------------
Yalnız **val** bölmesi kullanılır. Test bölmesi V6.4 + V6.9 mühürleriyle kapalıdır
(üçüncü kullanım mühür sayılmaz). Val yeterlidir: bu bir **seçim** değil, iki model
arasındaki **etkileşim** kıyasıdır; FND'nin val'de seçilmiş olması *genel* avantajını
şişirir ama *etkileşim* kontrastını doğrudan şişirmez.

Kullanım: python pipelines/v6_two_stage/v6_9e_backbone_interaction.py --reps 1000
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
import eval_variants as ev  # noqa: E402
from detector_models import CenterNetHead  # noqa: E402
from detector_models_foundation import predict_tta  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9e_backbone_interaction.json"
SEEN = {"chestxray14", "padchest"}
UNSEEN = {"jsrt", "openi"}
KS = (10, 20, 50)


def _auc(y, s) -> float | None:
    y, s = np.asarray(y, float), np.asarray(s, float)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return None
    o = np.argsort(-s)
    yy = y[o]
    tpr = np.concatenate([[0], np.cumsum(yy) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - yy) / N])
    return float(np.trapezoid(tpr, fpr))


def _img_scores(cands: dict, pids, agg: str) -> np.ndarray:
    out = []
    for p in pids:
        ss = sorted([c[2] for c in cands[p]], reverse=True)
        if not ss:
            out.append(0.0)
        elif agg == "top1":
            out.append(float(ss[0]))
        elif agg == "top3":
            out.append(float(np.mean(ss[:3])))
        else:  # noisy-OR (ilk 5)
            out.append(float(1 - np.prod([1 - min(1.0, max(0.0, x)) for x in ss[:5]])))
    return np.array(out)


def _delta(cands_a, cands_b, pos, ctrl, agg):
    """Δ = AUC(b) − AUC(a) için (grup pozitifleri + kontroller)."""
    pids = list(pos) + list(ctrl)
    y = [1] * len(pos) + [0] * len(ctrl)
    a = _auc(y, list(_img_scores(cands_a, pids, agg)))
    b = _auc(y, list(_img_scores(cands_b, pids, agg)))
    if a is None or b is None:
        return None, a, b
    return b - a, a, b


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.9e — omurga × kaynak etkileşimi | device={device}", flush=True)

    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m5.eval()
    fnd = load_foundation("M5_foundation", device)

    pids = list(common.split_patients()["val"])
    gts = {p: common.load_gt(p) for p in pids}
    src = {p: str(json.loads(
        (common.PATIENTS_DIR / p / "patient_info.json").read_text(encoding="utf-8")
    ).get("source_dataset")) for p in pids}

    print(f"  val: {len(pids)} hasta · aday üretimi (M5_heatTTA ve FND_heatTTA)...",
          flush=True)
    c_m5, c_fnd, t0 = {}, {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        c_m5[pid] = ev._predict_heat_tta(m5, canon, device, "M5")
        c_fnd[pid] = predict_tta(fnd, canon, device)
        if (i + 1) % 150 == 0:
            print(f"    {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    bitti ({time.time() - t0:.0f}s)", flush=True)

    # ---- üretim hattı doğrulaması (yayımlanmış V6.6.2 recall@K ile)
    n_les = sum(len(g) for g in gts.values())
    checks = {}
    for name, cands in (("M5_heatTTA", c_m5), ("FND_heatTTA", c_fnd)):
        hits = {k: 0 for k in KS}
        for pid in pids:
            if gts[pid]:
                for k in KS:
                    hits[k] += int(round(common.recall_at_K(cands[pid], gts[pid], k,
                                                            common.hit_criterion1)
                                         * len(gts[pid])))
        checks[name] = {f"K{k}": round(hits[k] / max(1, n_les), 4) for k in KS}
        print(f"  {name} recall: {checks[name]}", flush=True)

    pos_seen = [p for p in pids if gts[p] and src[p] in SEEN]
    pos_unseen = [p for p in pids if gts[p] and src[p] in UNSEEN]
    ctrl = [p for p in pids if not gts[p]]
    print(f"  görülen pozitif {len(pos_seen)} · görülmeyen pozitif {len(pos_unseen)} · "
          f"kontrol {len(ctrl)}", flush=True)

    rep = {"experiment": "V6.9e-backbone-source-interaction",
           "question": ("FND (CXR ön-eğitimli) ile M5 (ImageNet) arasındaki fark, omurganın "
                        "gördüğü kaynaklarda daha mı büyük? (etkileşim testi)"),
           "split": "val (test mühürlü — üçüncü kullanım yapılmaz)",
           "models": {"M5": "ImageNet ResNet18 CenterNet (CXR görmemiş)",
                      "FND": "xrv resnet50-res512-all (pc-nih-rsna-siim-vin)"},
           "protocol": "heat-map TTA (çok ölçek + yatay flip) — iki model için de aynı yol",
           "pipeline_check_recall_at_K": checks,
           "counts": {"n_val": len(pids), "seen_pos": len(pos_seen),
                      "unseen_pos": len(pos_unseen), "controls": len(ctrl),
                      "seen_lesions": sum(len(gts[p]) for p in pos_seen),
                      "unseen_lesions": sum(len(gts[p]) for p in pos_unseen)},
           "groups": {}, "interaction": {}}

    for agg in ("top1", "top3", "noisyor"):
        d_seen, a_seen, b_seen = _delta(c_m5, c_fnd, pos_seen, ctrl, agg)
        d_uns, a_uns, b_uns = _delta(c_m5, c_fnd, pos_unseen, ctrl, agg)
        rep["groups"][agg] = {
            "seen": {"auc_M5": round(a_seen, 4), "auc_FND": round(b_seen, 4),
                     "delta_FND_minus_M5": round(d_seen, 4)},
            "unseen": {"auc_M5": round(a_uns, 4), "auc_FND": round(b_uns, 4),
                       "delta_FND_minus_M5": round(d_uns, 4)}}
        rep["interaction"][agg] = round(d_seen - d_uns, 4)
        print(f"\n  [{agg}] görülen: M5 {a_seen:.4f} → FND {b_seen:.4f} (Δ {d_seen:+.4f}) | "
              f"görülmeyen: M5 {a_uns:.4f} → FND {b_uns:.4f} (Δ {d_uns:+.4f}) | "
              f"ETKİLEŞİM {d_seen - d_uns:+.4f}", flush=True)

    # ---- bootstrap (hasta düzeyi, grup içinde yerine koyarak)
    rng = np.random.RandomState(args.seed)
    boot = {agg: [] for agg in ("top1", "noisyor")}
    for _ in range(args.reps):
        bs = [pos_seen[i] for i in rng.randint(0, len(pos_seen), len(pos_seen))]
        bu = [pos_unseen[i] for i in rng.randint(0, len(pos_unseen), len(pos_unseen))]
        bc = [ctrl[i] for i in rng.randint(0, len(ctrl), len(ctrl))]
        for agg in boot:
            ds, _, _ = _delta(c_m5, c_fnd, bs, bc, agg)
            du, _, _ = _delta(c_m5, c_fnd, bu, bc, agg)
            if ds is not None and du is not None:
                boot[agg].append(ds - du)
    ci = {}
    for agg, vals in boot.items():
        a = np.array(vals)
        ci[agg] = {"mean": round(float(a.mean()), 4),
                   "lo95": round(float(np.percentile(a, 2.5)), 4),
                   "hi95": round(float(np.percentile(a, 97.5)), 4),
                   "share_positive": round(float((a > 0).mean()), 3)}
        print(f"  ETKİLEŞİM [{agg}] 95% GA: {ci[agg]['lo95']:+.4f} … {ci[agg]['hi95']:+.4f} "
              f"| P(etkileşim>0) = {ci[agg]['share_positive']}", flush=True)
    rep["interaction_ci95"] = ci
    key = "top1"
    lo, hi = ci[key]["lo95"], ci[key]["hi95"]
    if lo <= 0 <= hi:
        conclusion = ("Etkileşim %95 GA'sı 0'ı İÇİYOR ⇒ FND'nin avantajı kaynaklara göre "
                      "ayrışmıyor; örtüşmenin avantaj sağladığına dair KANIT YOK.")
    elif lo > 0:
        conclusion = ("Etkileşim tamamen POZİTİF ⇒ FND'nin avantajı görülen kaynaklarda "
                      "belirgin biçimde daha büyük; örtüşme lehine kanıt VAR.")
    else:
        conclusion = ("Etkileşim tamamen NEGATİF ⇒ FND görülmeyen kaynaklarda daha avantajlı; "
                      "örtüşme lehine kanıt yok, ters yönde işaret.")
    rep["verdict"] = {
        "interaction_top1": rep["interaction"][key],
        "ci95": ci[key],
        "conclusion": conclusion,
        "note": ("Val, FND'nin seçildiği bölmedir; bu FND'nin *genel* avantajını şişirir ama "
                 "etkileşim kontrastını doğrudan şişirmez."),
        "confound": ("M5 ResNet18 (11M) ile FND ResNet50 (23M) kapasite farkı ve TTA yolu "
                     "aynı olsa da mimari farklıdır; etkileşim yorumu bu sınırla okunmalıdır.")}
    print(f"\n  HÜKÜM: {conclusion}", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
