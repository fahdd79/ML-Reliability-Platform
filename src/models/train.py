from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.preprocess import NUMERIC, CATEGORICAL, split_xy


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    metrics: Dict[str, float]
    n_train: int
    fraud_rate: float


def train_logreg(train_df: pd.DataFrame) -> TrainResult:
    x_train, y_train = split_xy(train_df)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    model.fit(x_train, y_train)

    # training metrics (rough sanity only)
    probs = model.predict_proba(x_train)[:, 1]
    metrics = {
        "roc_auc_train": float(roc_auc_score(y_train, probs)) if len(np.unique(y_train)) > 1 else float("nan"),
        "pr_auc_train": float(average_precision_score(y_train, probs)) if len(np.unique(y_train)) > 1 else float("nan"),
    }

    return TrainResult(
        model=model,
        metrics=metrics,
        n_train=int(len(train_df)),
        fraud_rate=float(y_train.mean()),
    )
