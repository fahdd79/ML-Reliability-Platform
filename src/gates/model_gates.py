from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromotionPolicy:
    primary_metric: str
    primary_min_improvement: float
    guardrail_metric: str
    guardrail_max_regression: float  # candidate cannot be worse by more than this (higher is worse for guardrail)


def decide_promotion(prod: Dict[str, float], cand: Dict[str, float], policy: PromotionPolicy) -> Dict[str, object]:
    reasons = []

    p = policy.primary_metric
    g = policy.guardrail_metric

    if p not in prod or p not in cand:
        return {"decision": "REJECT", "reasons": [f"missing primary metric {p}"]}

    if g not in prod or g not in cand:
        return {"decision": "REJECT", "reasons": [f"missing guardrail metric {g}"]}

    primary_gain = cand[p] - prod[p]
    if primary_gain < policy.primary_min_improvement:
        reasons.append(f"{p}_gain {primary_gain:.4f} < {policy.primary_min_improvement:.4f}")

    # Guardrail: allow some regression.
    # For "brier" lower is better -> regression means cand[g] - prod[g] > max_regression
    # For metrics where higher is better (like pr_auc), regression means prod[g] - cand[g] > max_regression
    if g == "brier":
        guardrail_delta = cand[g] - prod[g]
        if guardrail_delta > policy.guardrail_max_regression:
            reasons.append(f"{g}_increase {guardrail_delta:.4f} > {policy.guardrail_max_regression:.4f}")
    else:
        guardrail_regress = prod[g] - cand[g]
        if guardrail_regress > policy.guardrail_max_regression:
            reasons.append(f"{g}_drop {guardrail_regress:.4f} > {policy.guardrail_max_regression:.4f}")

    if reasons:
        return {"decision": "REJECT", "reasons": reasons}

    return {"decision": "PROMOTE", "reasons": []}
