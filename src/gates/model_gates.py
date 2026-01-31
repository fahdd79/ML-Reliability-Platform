from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromotionPolicy:
    min_roc_auc_gain: float
    min_pr_auc_gain: float
    max_brier_increase: float  # candidate cannot be worse by more than this


def decide_promotion(prod: Dict[str, float], cand: Dict[str, float], policy: PromotionPolicy) -> Dict[str, object]:
    """
    Returns:
      - decision: PROMOTE or REJECT
      - reasons: list[str]
    """
    reasons = []

    roc_gain = cand["roc_auc"] - prod["roc_auc"]
    pr_gain = cand["pr_auc"] - prod["pr_auc"]
    brier_delta = cand["brier"] - prod["brier"]

    if roc_gain < policy.min_roc_auc_gain:
        reasons.append(f"roc_auc_gain {roc_gain:.4f} < {policy.min_roc_auc_gain:.4f}")
    if pr_gain < policy.min_pr_auc_gain:
        reasons.append(f"pr_auc_gain {pr_gain:.4f} < {policy.min_pr_auc_gain:.4f}")
    if brier_delta > policy.max_brier_increase:
        reasons.append(f"brier_increase {brier_delta:.4f} > {policy.max_brier_increase:.4f}")

    if reasons:
        return {"decision": "REJECT", "reasons": reasons}

    return {"decision": "PROMOTE", "reasons": []}
