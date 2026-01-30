from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _safe_normalize(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def psi_numeric(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> float:
    """
    Population Stability Index for numeric features.
    Uses quantile bins from reference distribution.
    """
    ref = ref.dropna().astype(float)
    cur = cur.dropna().astype(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    # Quantile-based bins from reference to stabilize
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(ref, quantiles))
    if len(bins) < 3:
        return 0.0  # nearly constant feature

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = _safe_normalize(ref_counts)
    cur_pct = _safe_normalize(cur_counts)

    # Avoid division by zero
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """
    PSI for categorical features based on category frequencies.
    """
    ref = ref.dropna().astype(str)
    cur = cur.dropna().astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    cats = sorted(set(ref_counts.index).union(set(cur_counts.index)))
    ref_pct = np.array([ref_counts.get(c, 0) for c in cats], dtype=float)
    cur_pct = np.array([cur_counts.get(c, 0) for c in cats], dtype=float)

    ref_pct = _safe_normalize(ref_pct)
    cur_pct = _safe_normalize(cur_pct)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
