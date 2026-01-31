from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


_LEVEL_ORDER = {"none": 0, "warning": 1, "alert": 2}


@dataclass(frozen=True)
class TriggerPolicy:
    trigger_level: str
    consecutive_windows: int
    cooldown_windows: int
    min_new_samples: int


def max_level(levels: List[str]) -> str:
    return max(levels, key=lambda x: _LEVEL_ORDER.get(x, 0))


def should_retrain(
    recent_window_levels: List[str],
    recent_window_sizes: List[int],
    policy: TriggerPolicy,
    last_retrain_window_index: int | None,
    current_window_index: int,
) -> Dict[str, object]:
    """
    Decide retraining based on:
    - required consecutive windows at or above trigger_level
    - min_new_samples across those windows
    - cooldown since last retrain
    """
    required = policy.trigger_level
    k = policy.consecutive_windows

    if len(recent_window_levels) < k or len(recent_window_sizes) < k:
        return {"trigger": False, "reason": "insufficient_history"}

    # cooldown check
    if last_retrain_window_index is not None:
        if (current_window_index - last_retrain_window_index) <= policy.cooldown_windows:
            return {"trigger": False, "reason": "cooldown_active"}

    # consecutive check: last k windows must be >= required
    required_val = _LEVEL_ORDER[required]
    last_k = recent_window_levels[-k:]
    if not all(_LEVEL_ORDER[l] >= required_val for l in last_k):
        return {"trigger": False, "reason": "not_consecutive"}

    # sample count check
    new_samples = sum(recent_window_sizes[-k:])
    if new_samples < policy.min_new_samples:
        return {"trigger": False, "reason": f"min_new_samples_not_met ({new_samples}<{policy.min_new_samples})"}

    return {"trigger": True, "reason": "trigger_conditions_met", "new_samples": new_samples}
