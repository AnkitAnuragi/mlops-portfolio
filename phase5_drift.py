"""
Feature drift detection using the Population Stability Index (PSI) — a
standard, interpretable metric for comparing a feature's distribution in
a reference dataset (e.g. training data) against a new batch.

PSI rule of thumb:
    < 0.1   no significant shift
    0.1-0.25  moderate shift, worth investigating
    > 0.25  significant shift, likely needs retraining/investigation
"""
import numpy as np
import pandas as pd


def _psi_for_bins(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Computes PSI for a single numeric feature between two samples."""
    combined_min = min(reference.min(), current.min())
    combined_max = max(reference.max(), current.max())
    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    return _psi_for_bins(reference.values, current.values, bins)


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Computes PSI for each feature and flags whether it exceeds the threshold."""
    rows = []
    for col in feature_cols:
        psi = compute_psi(reference_df[col], current_df[col])
        rows.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "flagged": psi > threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
