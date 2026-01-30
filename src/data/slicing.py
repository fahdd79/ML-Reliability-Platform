from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class SliceConfig:
    window_days: int


def slice_by_time(df: pd.DataFrame, cfg: SliceConfig) -> Dict[str, pd.DataFrame]:
    """
    Slices data into consecutive time windows of length cfg.window_days.
    Returns a dict: window_name -> dataframe
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column")

    df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("D")
    min_date = df["date"].min()
    max_date = df["date"].max()

    windows: Dict[str, pd.DataFrame] = {}
    start = min_date
    idx = 0

    while start <= max_date:
        end = start + pd.Timedelta(days=cfg.window_days)
        mask = (df["date"] >= start) & (df["date"] < end)
        chunk = df.loc[mask].drop(columns=["date"]).reset_index(drop=True)

        window_name = f"T{idx}"
        windows[window_name] = chunk

        start = end
        idx += 1

    return windows
