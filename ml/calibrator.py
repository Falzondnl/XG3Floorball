"""
Isotonic regression calibrator — framework-independent dict schema.
Follows the XG3 pattern used in American Football MS.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.isotonic import IsotonicRegression


class FloorballCalibrator:
    """
    Wraps sklearn IsotonicRegression for post-hoc probability calibration.
    Serialises to / deserialises from a plain dict (schema_version field).
    """

    SCHEMA_VERSION = "floorball_calibrator_v1"

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "FloorballCalibrator":
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw_probs, labels)
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return np.clip(self._iso.predict(raw_probs), 0.02, 0.98)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        if self._iso is None:
            raise RuntimeError("Calibrator not fitted.")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "X_thresholds_": self._iso.X_thresholds_.tolist(),
            "y_thresholds_": self._iso.y_thresholds_.tolist(),
            "out_of_bounds": self._iso.out_of_bounds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FloorballCalibrator":
        if d.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible schema: {d.get('schema_version')} != {cls.SCHEMA_VERSION}"
            )
        obj = cls()
        iso = IsotonicRegression(out_of_bounds=d.get("out_of_bounds", "clip"))
        # Reconstruct internal state
        x = np.array(d["X_thresholds_"])
        y = np.array(d["y_thresholds_"])
        iso.fit(x, y)  # re-fit on the stored thresholds (idempotent for step function)
        obj._iso = iso
        return obj

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=5)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FloorballCalibrator":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if isinstance(d, dict):
            return cls.from_dict(d)
        # Legacy: direct IsotonicRegression object (should not happen in new builds)
        obj = cls()
        obj._iso = d
        return obj
