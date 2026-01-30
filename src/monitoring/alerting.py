from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DriftPolicy:
    reference_window: str
    min_fraction_features_breaching: float
    psi_warning: float
    psi_alert: float
    monitored_numeric: List[str]
    monitored_categorical: List[str]


def evaluate_drift_alert(psi_by_feature: Dict[str, float], policy: DriftPolicy) -> Dict[str, object]:
    """
    Returns an alert decision payload:
    - level: none | warning | alert
    - breaching_features: list
    - breach_fraction: float
    """
    monitored = policy.monitored_numeric + policy.monitored_categorical
    monitored = [f for f in monitored if f in psi_by_feature]

    if not monitored:
        return {"level": "none", "breaching_features": [], "breach_fraction": 0.0}

    # Alert if PSI >= psi_alert
    breaching = [f for f in monitored if psi_by_feature.get(f) is not None and psi_by_feature[f] >= policy.psi_alert]
    breach_fraction = len(breaching) / len(monitored)

    if breach_fraction >= policy.min_fraction_features_breaching:
        return {"level": "alert", "breaching_features": breaching, "breach_fraction": breach_fraction}

    # Warning if any feature >= warning
    warn = [f for f in monitored if psi_by_feature.get(f) is not None and psi_by_feature[f] >= policy.psi_warning]
    if warn:
        return {"level": "warning", "breaching_features": warn, "breach_fraction": len(warn) / len(monitored)}

    return {"level": "none", "breaching_features": [], "breach_fraction": 0.0}
