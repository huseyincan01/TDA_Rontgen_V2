"""V6.7 — NİHAİ GÖNDERİM ÇIKARIMI (kilitli standart).

Borunun tamamı tek yerde: kanonikleştirme → M5_fulldata → çok ölçekli flip TTA →
3x3 tepe → 8 px NMS → 56x56 sabit kutu → TDA-HGB_fulldata → KİLİTLİ füzyon (w=0,55).

`submission_config.json` bu betiğin okuduğu dondurulmuş sözleşmedir.

Kullanım:
    python pipelines/v6_two_stage/v6_7_submission_infer.py --smoke
    python pipelines/v6_two_stage/v6_7_submission_infer.py --image <yol>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, pick_peaks  # noqa: E402

CFG_PATH = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_config.json"
SUBDIR = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"


class SubmissionPipeline:
    """Dondurulmuş gönderim boru hattı."""

    def __init__(self, cfg_path: Path = CFG_PATH, device=None, cache_dir: str = None):
        self.cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c = self.cfg
        self.model = load_foundation(c["stage1"]["checkpoint"], self.device, subdir=True,
                                     cache_dir=cache_dir)
        b = joblib.load(PROJECT_ROOT / c["stage2"]["bundle"])
        self.sc = b["scaler"]
        self.hgb = b["hgb"]
        self.ranges = b["ranges"]
        self.fusion = b["fusion_ref"]
        self.scales = tuple(c["inference"]["scales"])
        self.win = int(c["inference"]["peak_window"])
        self.nms_px = int(c["inference"]["nms_px"])
        self.top_k = int(c["inference"]["top_k"])
        self.box_px = int(c["inference"]["box_px"])
        self._ex = ZoneTDAExtractor(self._settings())

    def _settings(self):
        r = self.ranges
        return TDASettings(grid_size=20, intensity_max=255.0,
                           h0_birth_range=tuple(r["h0_birth"]),
                           h0_persistence_range=tuple(r["h0_pers"]),
                           h1_birth_range=tuple(r["h1_birth"]),
                           h1_persistence_range=tuple(r["h1_pers"]),
                           pi_sigma_pixels=1.0)

    # ------------------------------------------------------------------ adaylar
    def stage1(self, canon: np.ndarray):
        """Kilitli standart: çok ölçekli flip TTA → tepe → NMS.

        `pick_peaks` koordinatları 1024-referanslı uzayda üretir; görüntü 1024x1024
        değilse gerçek (H, W) uzayına ölçeklenir (NODE21'de ölçek = 1).
        """
        H, W = canon.shape
        heat, rad = heat_maps(self.model, canon_to_512(canon), self.device, self.scales)
        pk = pick_peaks(heat, rad, self.win)
        cs = dist_nms(pk, self.nms_px, top=self.top_k)            # (cx, cy, skor, r)
        if (H, W) != (1024, 1024):
            sx, sy = W / 1024.0, H / 1024.0
            cs = [(cx * sx, cy * sy, s, r * (sx + sy) / 2.0) for cx, cy, s, r in cs]
        return cs

    # ------------------------------------------------------------------ aşama-2
    def stage2(self, canon: np.ndarray, raw: np.ndarray, cands):
        if not cands:
            return np.zeros(0, np.float32)
        dist = ndi.distance_transform_edt(raw).astype(np.float32)
        ys, xs = np.where(raw)
        bbox = ((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
                if ys.size else (0, 0, 0, 0))
        tda = np.zeros((len(cands), 2 * 2 * 20 * 20), np.float32)
        spa = np.zeros((len(cands), 4), np.float32)
        for i, (cx, cy, _, _) in enumerate(cands):
            v_in, _ = self._ex.extract(s2._crop(canon, cx, cy, s2.INNER))
            v_out, _ = self._ex.extract(s2._crop(canon, cx, cy, s2.OUTER))
            tda[i] = np.concatenate([v_in, v_out])
            spa[i] = s2._spatial(cx, cy, raw, dist, bbox)
        X = self.sc.transform(np.concatenate([tda, spa], 1))
        return self.hgb.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ füzyon
    def fuse(self, s1: np.ndarray, p_tda: np.ndarray) -> np.ndarray:
        f = self.fusion
        z1 = (s1 - f["s1_mean"]) / f["s1_std"]
        zt = (p_tda - f["tda_mean"]) / f["tda_std"]
        return 1.0 / (1.0 + np.exp(-(f["w"] * z1 + (1.0 - f["w"]) * zt)))

    # ------------------------------------------------------------------ boru hattı
    @torch.no_grad()
    def predict_from_arrays(self, img: np.ndarray, raw_mask: np.ndarray):
        """img: ham görüntü; raw_mask: akciğer (dilate EDİLMEMİŞ) bool maskesi."""
        p1, p99 = np.percentile(img[raw_mask], [1, 99]) if raw_mask.any() else (0.0, 1.0)
        mask = ndi.binary_dilation(raw_mask, structure=ndi.generate_binary_structure(2, 1),
                                   iterations=common.DILATION_PX)
        canon = common.canonicalize(img.astype(np.float32), mask, float(p1), float(p99))
        cands = self.stage1(canon)
        if not cands:
            return []
        p_tda = self.stage2(canon, raw_mask, cands)
        s1 = np.array([c[2] for c in cands], np.float32)
        s = self.fuse(s1, p_tda)
        h = self.box_px / 2.0
        out = []
        for i, (cx, cy, _, _) in enumerate(cands):
            out.append({"x1": float(cx - h), "y1": float(cy - h),
                        "x2": float(cx + h), "y2": float(cy + h),
                        "score": float(s[i])})
        out.sort(key=lambda d: -d["score"])
        return out

    def predict_image(self, img: np.ndarray, seg=None):
        """Ham görüntüden tahmin. `seg` verilmezse PSPNet akciğer segmentasyonu kurulur."""
        if seg is None:
            from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter
            seg = PSPNet5ZoneSegmenter(device=self.device)
        raw = seg.segment(img)["zone_map"] > 0
        return self.predict_from_arrays(img, raw)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--image", default="")
    args = ap.parse_args()

    print("V6.7 NİHAİ GÖNDERİM ÇIKARIMI", flush=True)
    if not CFG_PATH.exists():
        raise SystemExit(f"submission_config.json yok: {CFG_PATH}\n"
                         "Önce v6_7_build_fulldata_stage2.py koşulmalı.")
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    print(f"  sözleşme: {cfg['version']} | w={cfg['fusion']['w']} (kilitli)", flush=True)

    t0 = time.time()
    pipe = SubmissionPipeline(CFG_PATH)
    print(f"  model + bundle yüklendi ({time.time() - t0:.1f}s)", flush=True)
    print(f"  aşama-1: {cfg['stage1']['checkpoint']} | aşama-2: "
          f"{Path(cfg['stage2']['bundle']).name}", flush=True)
    print(f"  standart: ölçek{tuple(cfg['inference']['scales'])} + flip · tepe "
          f"{cfg['inference']['peak_window']}x{cfg['inference']['peak_window']} · NMS "
          f"{cfg['inference']['nms_px']}px · kutu {cfg['inference']['box_px']}px", flush=True)

    if args.image:
        img = np.array(Image.open(args.image).convert("L"), dtype=np.float32)
        out = pipe.predict_image(img)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if not args.smoke:
        raise SystemExit("--smoke veya --image gerekli")

    print(f"\nDUMAN TESTİ ({args.n} görüntü, val'den)...", flush=True)
    val = common.split_patients()["val"]
    nod = [p for p in val if common.load_gt(p)][:max(1, args.n - 1)]
    ctl = [p for p in val if not common.load_gt(p)][:max(1, args.n - len(nod))]
    ok = True
    for pid in (nod + ctl):
        t0 = time.time()
        img = common.load_image(pid)
        raw = s2._raw_mask(pid)
        out = pipe.predict_from_arrays(img, raw)
        boxes = [d for d in out if d["score"] > 0]
        scores = [d["score"] for d in out]
        n_gt = len(common.load_gt(pid))
        assert all(0.0 <= s <= 1.0 for s in scores), "skor [0,1] dışında!"
        assert all(d["x2"] - d["x1"] == cfg["inference"]["box_px"] for d in out), \
            "kutu boyutu yanlış!"
        assert all(d["x2"] > d["x1"] and d["y2"] > d["y1"] for d in out), "geçersiz kutu!"
        assert scores == sorted(scores, reverse=True), "skorlar sıralı değil!"
        print(f"  {pid}: {len(out)} aday (GT {n_gt}) | ilk skor "
              f"{scores[0]:.4f} | {time.time() - t0:.1f}s/kare", flush=True)
        ok = ok and len(out) > 0 and len(out) <= cfg["inference"]["top_k"]
    print(f"\n{'✅ DUMAN TESTİ GEÇTİ' if ok else '❌ BAŞARISIZ'} — "
          f"çıktı: her aday için (x1,y1,x2,y2,score), skorlar azalan, "
          f"kutu sabit {cfg['inference']['box_px']}px", flush=True)


if __name__ == "__main__":
    main()
