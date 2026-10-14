"""MHA zon kesitlerinden cubical persistent homology özellikleri çıkarır."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gudhi
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class TDASettings:
    """Persistence Image üretiminde kullanılan ayarlar."""

    grid_size: int = 20
    birth_range: Tuple[float, float] = (0.0, 255.0)
    persistence_range: Tuple[float, float] = (0.0, 255.0)
    intensity_max: float = 65535.0
    h0_birth_range: Tuple[float, float] | None = None
    h0_persistence_range: Tuple[float, float] | None = None
    h1_birth_range: Tuple[float, float] | None = None
    h1_persistence_range: Tuple[float, float] | None = None
    pi_sigma_pixels: float = 1.0

    @property
    def vector_size(self) -> int:
        return 2 * self.grid_size * self.grid_size


class ZoneTDAExtractor:
    """Tek bir maskelenmiş zon görüntüsünü 800 boyutlu TDA vektörüne çevirir.

    Zon kesitlerinin dışı ``0`` olduğundan, doku maskesi ``image > 0`` ile
    elde edilir. Görüntü önce 0--255 aralığına dönüştürülür; yalnızca doku
    pikselleri ters çevrilir ve arka plan tekrar 0 bırakılır.
    """

    def __init__(self, settings: TDASettings | None = None) -> None:
        self.settings = settings or TDASettings()
        if self.settings.grid_size <= 0:
            raise ValueError("grid_size pozitif olmalıdır.")
        if self.settings.intensity_max <= 0:
            raise ValueError("intensity_max pozitif olmalıdır.")

        h0_br = self.settings.h0_birth_range or self.settings.birth_range
        h0_pr = self.settings.h0_persistence_range or self.settings.persistence_range
        h1_br = self.settings.h1_birth_range or self.settings.birth_range
        h1_pr = self.settings.h1_persistence_range or self.settings.persistence_range

        self.h0_ranges = (h0_br, h0_pr)
        self.h1_ranges = (h1_br, h1_pr)
        if self.settings.pi_sigma_pixels <= 0:
            raise ValueError("pi_sigma_pixels pozitif olmalıdır.")

    def prepare_filtration(self, image: np.ndarray) -> np.ndarray | None:
        """MHA piksel dizisini maskeli, ters çevrilmiş uint8 filtrasyona hazırlar."""
        if image.ndim != 2:
            raise ValueError(f"2B zon bekleniyordu, gelen şekil: {image.shape}")

        image = np.asarray(image)
        valid_mask = np.isfinite(image) & (image > 0)
        if not np.any(valid_mask):
            return None

        y_coords, x_coords = np.where(valid_mask)
        y0, y1 = y_coords.min(), y_coords.max() + 1
        x0, x1 = x_coords.min(), x_coords.max() + 1
        tissue = image[y0:y1, x0:x1]
        tissue_mask = valid_mask[y0:y1, x0:x1]

        scaled = np.clip(
            np.rint(tissue.astype(np.float32) * 255.0 / self.settings.intensity_max),
            0,
            255,
        ).astype(np.uint8)
        filtration = np.zeros_like(scaled, dtype=np.uint8)
        filtration[tissue_mask] = 255 - scaled[tissue_mask]
        return filtration

    @staticmethod
    def _finite_intervals(intervals: np.ndarray) -> np.ndarray:
        if len(intervals) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return intervals[np.isfinite(intervals[:, 1])]

    def compute_diagrams(self, filtration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kübik kompleks üzerinde sonlu H0 ve H1 persistence diyagramlarını döndürür."""
        complex_ = gudhi.CubicalComplex(
            dimensions=filtration.shape,
            top_dimensional_cells=filtration.astype(np.float64, copy=False).ravel(),
        )
        complex_.compute_persistence()
        h0 = self._finite_intervals(complex_.persistence_intervals_in_dimension(0))
        h1 = self._finite_intervals(complex_.persistence_intervals_in_dimension(1))
        return h0, h1

    def vectorize(self, h0: np.ndarray, h1: np.ndarray) -> np.ndarray:
        """H0 ve H1 diyagramlarını sabit çözünürlüklü Persistence Image'a dönüştürür.

        Diyagramlar kendi train-türetilmiş aralıklarında 20x20 hücreye binlenir.
        Gaussian genişliği piksel birimindedir; böylece H0 ve H1'in ham yoğunluk
        aralıkları farklı olsa bile ikisi de aynı mekânsal yumuşatmayı alır.
        """
        image_h0 = self._persistence_image(h0, *self.h0_ranges)
        image_h1 = self._persistence_image(h1, *self.h1_ranges)

        h0_vector = np.asarray(image_h0, dtype=np.float32).ravel()
        h1_vector = np.asarray(image_h1, dtype=np.float32).ravel()
        expected = self.settings.grid_size * self.settings.grid_size
        if h0_vector.size != expected or h1_vector.size != expected:
            raise ValueError(
                "Persistence Image boyutu beklenen ızgara boyutuyla eşleşmedi: "
                f"H0={h0_vector.size}, H1={h1_vector.size}, beklenen={expected}."
            )
        return np.concatenate((h0_vector, h1_vector)).astype(np.float32, copy=False)

    def _persistence_image(
        self,
        diagram: np.ndarray,
        birth_range: Tuple[float, float],
        persistence_range: Tuple[float, float],
    ) -> np.ndarray:
        """Diyagramı desteklenen NumPy/SciPy işlemleriyle sabit ızgaraya rasterleştirir."""
        size = self.settings.grid_size
        image = np.zeros((size, size), dtype=np.float64)
        if len(diagram) == 0:
            return image

        diagram = np.asarray(diagram, dtype=np.float64)
        births = diagram[:, 0]
        persistences = diagram[:, 1] - diagram[:, 0]
        valid = np.isfinite(births) & np.isfinite(persistences) & (persistences > 0)
        if not np.any(valid):
            return image

        births = np.clip(births[valid], *birth_range)
        persistences = np.clip(persistences[valid], *persistence_range)
        b_width = birth_range[1] - birth_range[0]
        p_width = persistence_range[1] - persistence_range[0]
        if b_width <= 0 or p_width <= 0:
            raise ValueError("Persistence Image aralıkları pozitif genişlikte olmalıdır.")

        cols = np.minimum(((births - birth_range[0]) / b_width * size).astype(int), size - 1)
        rows = np.minimum(((persistences - persistence_range[0]) / p_width * size).astype(int), size - 1)
        np.add.at(image, (rows, cols), persistences)
        return gaussian_filter(image, sigma=self.settings.pi_sigma_pixels, mode="constant", truncate=3.0)

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Bir zon için ``(800-boyutlu vektör, durum)`` döndürür."""
        filtration = self.prepare_filtration(image)
        if filtration is None:
            return np.zeros(self.settings.vector_size, dtype=np.float32), "empty_zone"
        h0, h1 = self.compute_diagrams(filtration)
        return self.vectorize(h0, h1), "ok"
