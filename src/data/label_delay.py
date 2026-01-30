from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelDelayConfig:
    delay_windows: int


def apply_label_delay(
    windows: Dict[str, pd.DataFrame],
    cfg: LabelDelayConfig,
    label_col: str = "y",
) -> Dict[str, pd.DataFrame]:
    """
    Simulates delayed label availability:
    - For the most recent cfg.delay_windows windows, labels are hidden (set to NaN).
    - Older windows keep labels.

    Returns new dict with modified frames.
    """
    keys = sorted(windows.keys(), key=lambda x: int(x[1:]))  # T0, T1, ...
    n = len(keys)
    cutoff = max(0, n - cfg.delay_windows)

    out: Dict[str, pd.DataFrame] = {}
    for i, k in enumerate(keys):
        df = windows[k].copy()
        if label_col not in df.columns:
            raise ValueError(f"Expected label column '{label_col}' in window {k}")

        if i >= cutoff:
            df[label_col] = np.nan  # hidden labels
        out[k] = df

    return out
