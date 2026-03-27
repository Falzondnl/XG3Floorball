"""
Floorball ML Trainer
====================
Trains a 3-model stacking ensemble (CatBoost + LightGBM + XGBoost).
Falls back to a Logistic Regression blend if the dataset is too small.

Anti-bias rule:
  50% of training rows are randomly home/away swapped so the model cannot
  learn a P1-always-wins bias from training data ordering.

Temporal split:
  - Train  : matches before 2023-01-01
  - Val    : 2023-01-01 – 2024-01-01
  - Test   : 2024-01-01 onwards

Outputs saved to models/{regime}/:
  ensemble.pkl, calibrator.pkl, extractor.pkl
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import MIN_ROWS_FOR_ENSEMBLE, MODELS_DIR, RANDOM_SEED
from ml.calibrator import FloorballCalibrator
from ml.features import FEATURE_COLUMNS, FloorballFeatureExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    from config import DATA_SOURCES
    path = DATA_SOURCES["sofascore_events"]
    if not path.exists():
        raise FileNotFoundError(f"Primary data not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Anti-bias swap
# ---------------------------------------------------------------------------

def _apply_home_away_swap(
    X: pd.DataFrame, y: pd.Series, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    """
    For ~50% of rows, swap home ↔ away features and flip the label.
    This prevents the model from learning home=P1 bias.
    """
    X = X.copy()
    y = y.copy()

    swap_mask = rng.random(len(X)) < 0.5
    idx = X.index[swap_mask]

    # Pairs to swap
    swap_pairs = [
        ("elo_home_pre", "elo_away_pre"),
        ("form_pts_home_last5", "form_pts_away_last5"),
        ("form_gd_home_last5", "form_gd_away_last5"),
        ("goals_scored_home_avg5", "goals_scored_away_avg5"),
    ]
    for a, b in swap_pairs:
        if a in X.columns and b in X.columns:
            X.loc[idx, [a, b]] = X.loc[idx, [b, a]].values

    # Flip elo_diff
    if "elo_diff" in X.columns:
        X.loc[idx, "elo_diff"] = -X.loc[idx, "elo_diff"]

    # Flip h2h (home win rate becomes away win rate after swap)
    if "h2h_home_win_rate" in X.columns:
        X.loc[idx, "h2h_home_win_rate"] = 1.0 - X.loc[idx, "h2h_home_win_rate"]

    # Flip labels
    y.loc[idx] = 1 - y.loc[idx]

    return X, y


# ---------------------------------------------------------------------------
# Ensemble components
# ---------------------------------------------------------------------------

def _train_catboost(X_tr, y_tr, X_val, y_val):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=False,
        early_stopping_rounds=30,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
    )
    return model


def _train_lightgbm(X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return model


def _train_xgboost(X_tr, y_tr, X_val, y_val):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _logistic_blend(X_tr, y_tr):
    """Fallback when data < MIN_ROWS_FOR_ENSEMBLE."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_scaled, y_tr)
    return {"type": "logistic", "scaler": scaler, "lr": lr}


# ---------------------------------------------------------------------------
# Ensemble container
# ---------------------------------------------------------------------------

class FloorballEnsemble:
    """
    Wraps the 3-model blend or logistic fallback.
    All predict calls return P(home_win).
    """

    def __init__(
        self,
        models: list,
        weights: list[float],
        mode: Literal["ensemble", "logistic"] = "ensemble",
        logistic_bundle: dict | None = None,
    ):
        self.models = models
        self.weights = weights
        self.mode = mode
        self.logistic_bundle = logistic_bundle

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.mode == "logistic":
            scaler = self.logistic_bundle["scaler"]
            lr = self.logistic_bundle["lr"]
            X_s = scaler.transform(X[FEATURE_COLUMNS])
            return lr.predict_proba(X_s)[:, 1]
        # Ensemble average
        preds = []
        for model, w in zip(self.models, self.weights):
            preds.append(w * model.predict_proba(X[FEATURE_COLUMNS])[:, 1])
        return np.clip(np.sum(preds, axis=0), 0.02, 0.98)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=5)

    @classmethod
    def load(cls, path: Path) -> "FloorballEnsemble":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_regime(
    regime: str,
    X: pd.DataFrame,
    y: pd.Series,
    enriched_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Train ensemble for a single regime and save artefacts.
    Returns evaluation metrics.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    out_dir = MODELS_DIR / regime
    out_dir.mkdir(parents=True, exist_ok=True)

    # Temporal split — work entirely in positional space after reset_index.
    # X, y, enriched_df all come from fit_transform which produces them
    # aligned on the same integer RangeIndex. We subset first, then split.
    enriched_df = enriched_df.copy()
    enriched_df["dt"] = pd.to_datetime(
        enriched_df["start_timestamp"], unit="s", errors="coerce"
    )
    # Keep only binary target rows (consistent with what fit_transform returns)
    binary_mask = enriched_df["winner_code"].isin([1, 2])
    enriched_df = enriched_df[binary_mask].copy()
    X = X[binary_mask].copy()
    y = y[binary_mask].copy()

    # Reset all to a clean 0-based RangeIndex to avoid misalignment
    enriched_df = enriched_df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Sort by time (already sorted in fit_transform, but confirm)
    order = enriched_df["dt"].argsort()
    enriched_df = enriched_df.iloc[order].reset_index(drop=True)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    n = len(enriched_df)
    train_end_idx = int(n * 0.60)
    val_end_idx = int(n * 0.80)

    t_split = enriched_df.iloc[train_end_idx]["dt"] if train_end_idx < n else None
    v_split = enriched_df.iloc[val_end_idx]["dt"] if val_end_idx < n else None
    logger.info("[%s] Temporal split — train ends: %s, val ends: %s", regime, t_split, v_split)

    X_tr_raw = X.iloc[:train_end_idx]
    y_tr_raw = y.iloc[:train_end_idx]
    X_val = X.iloc[train_end_idx:val_end_idx]
    y_val = y.iloc[train_end_idx:val_end_idx]
    X_te = X.iloc[val_end_idx:]
    y_te = y.iloc[val_end_idx:]

    logger.info(
        "[%s] Train=%d Val=%d Test=%d", regime, len(X_tr_raw), len(X_val), len(X_te)
    )

    if len(X_tr_raw) < 50:
        logger.warning(
            "[%s] Insufficient training data (%d rows). Skipping.", regime, len(X_tr_raw)
        )
        return {}

    # Anti-bias swap on training data
    X_tr, y_tr = _apply_home_away_swap(X_tr_raw, y_tr_raw, rng)
    logger.info(
        "[%s] After swap — target balance: %.1f%% home wins",
        regime, 100.0 * y_tr.mean()
    )

    use_ensemble = len(X_tr) >= MIN_ROWS_FOR_ENSEMBLE

    if use_ensemble:
        logger.info("[%s] Training CatBoost...", regime)
        cb = _train_catboost(
            X_tr[FEATURE_COLUMNS], y_tr,
            X_val[FEATURE_COLUMNS] if len(X_val) > 0 else X_tr[FEATURE_COLUMNS],
            y_val if len(y_val) > 0 else y_tr,
        )
        logger.info("[%s] Training LightGBM...", regime)
        lgb_m = _train_lightgbm(
            X_tr[FEATURE_COLUMNS], y_tr,
            X_val[FEATURE_COLUMNS] if len(X_val) > 0 else X_tr[FEATURE_COLUMNS],
            y_val if len(y_val) > 0 else y_tr,
        )
        logger.info("[%s] Training XGBoost...", regime)
        xgb_m = _train_xgboost(
            X_tr[FEATURE_COLUMNS], y_tr,
            X_val[FEATURE_COLUMNS] if len(X_val) > 0 else X_tr[FEATURE_COLUMNS],
            y_val if len(y_val) > 0 else y_tr,
        )
        ensemble = FloorballEnsemble(
            models=[cb, lgb_m, xgb_m],
            weights=[1 / 3, 1 / 3, 1 / 3],
            mode="ensemble",
        )
    else:
        logger.info(
            "[%s] Dataset small (%d rows) — using Logistic Regression.", regime, len(X_tr)
        )
        lb = _logistic_blend(X_tr[FEATURE_COLUMNS], y_tr)
        ensemble = FloorballEnsemble(
            models=[],
            weights=[],
            mode="logistic",
            logistic_bundle=lb,
        )

    # Evaluate
    metrics: dict[str, float] = {}

    def _eval(split_name: str, Xs: pd.DataFrame, ys: pd.Series) -> None:
        if len(Xs) == 0:
            return
        preds = ensemble.predict_proba(Xs)
        auc = roc_auc_score(ys, preds)
        brier = brier_score_loss(ys, preds)
        logger.info("[%s] %s AUC=%.4f Brier=%.4f", regime, split_name, auc, brier)
        metrics[f"{split_name}_auc"] = round(auc, 4)
        metrics[f"{split_name}_brier"] = round(brier, 4)

    _eval("train", X_tr[FEATURE_COLUMNS], y_tr)
    if len(X_val) > 0:
        _eval("val", X_val, y_val)
    if len(X_te) > 0:
        _eval("test", X_te, y_te)

    # Calibrate on val (or test if no val)
    cal_X = X_val if len(X_val) >= 30 else (X_te if len(X_te) >= 30 else X_tr)
    cal_y = y_val if len(y_val) >= 30 else (y_te if len(y_te) >= 30 else y_tr)
    raw_cal = ensemble.predict_proba(cal_X)
    calibrator = FloorballCalibrator()
    calibrator.fit(raw_cal, cal_y.values)

    # Save artefacts
    ensemble_path = out_dir / "ensemble.pkl"
    cal_path = out_dir / "calibrator.pkl"
    ext_path = out_dir / "extractor.pkl"

    ensemble.save(ensemble_path)
    calibrator.save(cal_path)

    extractor = FloorballFeatureExtractor()
    with open(ext_path, "wb") as fh:
        pickle.dump(extractor, fh, protocol=5)

    logger.info(
        "[%s] Saved: %s (%d bytes), %s, %s",
        regime,
        ensemble_path,
        ensemble_path.stat().st_size,
        cal_path,
        ext_path,
    )
    metrics["n_train"] = len(X_tr)
    metrics["n_val"] = len(X_val)
    metrics["n_test"] = len(X_te)
    metrics["mode"] = "ensemble" if use_ensemble else "logistic"
    return metrics


def train_all() -> dict[str, dict]:
    """Entry point — train R0 (men) and R1 (women) regimes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Loading raw data...")
    df_raw = load_raw_data()

    extractor = FloorballFeatureExtractor()
    X_all, y_all, enriched_all = extractor.fit_transform(df_raw)

    results: dict[str, dict] = {}

    # --- R0: Men ---
    is_women = enriched_all["tournament_name"].str.contains(
        "Women|women", na=False
    )
    X_men = X_all[~is_women.values]
    y_men = y_all[~is_women.values]
    enr_men = enriched_all[~is_women.values]
    logger.info("Training R0 (men): %d rows", len(X_men))
    results["r0"] = train_regime("r0", X_men, y_men, enr_men)

    # --- R1: Women ---
    X_women = X_all[is_women.values]
    y_women = y_all[is_women.values]
    enr_women = enriched_all[is_women.values]
    logger.info("Training R1 (women): %d rows", len(X_women))
    results["r1"] = train_regime("r1", X_women, y_women, enr_women)

    logger.info("Training complete. Summary:")
    for r, m in results.items():
        logger.info("  %s: %s", r, m)

    return results


if __name__ == "__main__":
    train_all()
