"""5-Zon modellerini paketleyen ve hasta düzeyinde tahmin üreten modül."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np


class ZoneModelBundle:
    """5 anatomik bölge için eğitilmiş ölçekleyici (scaler) ve modelleri yönetir."""

    def __init__(
        self,
        models: Optional[Dict[int, Any]] = None,
        scalers: Optional[Dict[int, Any]] = None,
        thresholds: Optional[Dict[int, float]] = None,
        best_params: Optional[Dict[int, Dict[str, Any]]] = None,
        patient_threshold: float = 0.5,
        aggregation: Literal["max", "noisy_or"] = "max",
    ) -> None:
        self.models = models or {}
        self.scalers = scalers or {}
        self.thresholds = thresholds or {z: 0.5 for z in range(1, 6)}
        self.best_params = best_params or {}
        self.patient_threshold = patient_threshold
        self.aggregation = aggregation

    def set_zone_model(
        self,
        zone_id: int,
        model: Any,
        scaler: Any,
        threshold: float = 0.5,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Belirtilen zona ait eğitilmiş modeli ve ölçekleyiciyi kaydeder."""
        self.models[zone_id] = model
        self.scalers[zone_id] = scaler
        self.thresholds[zone_id] = float(threshold)
        if params:
            self.best_params[zone_id] = params

    def predict_zone_proba(self, zone_id: int, X: np.ndarray) -> np.ndarray:
        """Belirtilen zon için girdi özellik vektörlerinin pozitif sınıf (1) olasılıklarını döner."""
        if zone_id not in self.models:
            raise ValueError(f"Zon {zone_id} için model eğitilmemiş!")

        scaler = self.scalers[zone_id]
        model = self.models[zone_id]

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        X_scaled = scaler.transform(X_arr)
        # binary classification: second column is positive probability
        proba = model.predict_proba(X_scaled)[:, 1]
        return proba

    def predict_zone(self, zone_id: int, X: np.ndarray) -> np.ndarray:
        """Zon karar eşiğine göre ikili tahmin (0 veya 1) döner."""
        proba = self.predict_zone_proba(zone_id, X)
        thresh = self.thresholds.get(zone_id, 0.5)
        return (proba >= thresh).astype(int)

    def aggregate_patient_proba(
        self,
        zone_probs: Dict[int, float],
        method: Optional[Literal["max", "noisy_or"]] = None,
    ) -> float:
        """5 zonun pozitiflik olasılıklarından tek bir hasta pozitiflik olasılığı üretir.
        
        - 'max': En şüpheli zonun olasılığını alır: max(p_1, ..., p_5)
        - 'noisy_or': Herhangi bir zonun pozitif olma olasılığı: 1 - prod(1 - p_z)
        """
        agg_method = method or self.aggregation
        probs = list(zone_probs.values())
        if not probs:
            return 0.0

        if agg_method == "max":
            return float(max(probs))
        elif agg_method == "noisy_or":
            comp = np.prod([1.0 - np.clip(p, 0.0, 1.0) for p in probs])
            return float(1.0 - comp)
        else:
            raise ValueError(f"Bilinmeyen birleştirme yöntemi: {agg_method}")

    def save(self, filepath: str | Path) -> None:
        """Model paketini diske kaydeder."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "ZoneModelBundle":
        """Diskteki model paketini yükler."""
        return joblib.load(filepath)
