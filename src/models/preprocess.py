from __future__ import annotations

from typing import List, Tuple
import pandas as pd


NUMERIC = ["amount", "hour", "velocity_1h", "is_night", "is_new_account"]
CATEGORICAL = ["merchant_cat", "country", "device"]
LABEL = "y"


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if LABEL not in df.columns:
        raise ValueError("Label column 'y' not found")
    x = df[NUMERIC + CATEGORICAL].copy()
    y = df[LABEL].astype(int)
    return x, y
