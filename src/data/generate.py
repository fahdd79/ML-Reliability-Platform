from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataGenConfig:
    n_days: int
    events_per_day: int
    start_day_for_drift: int
    covariate_strength: float
    concept_strength: float
    seed: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_transactions(cfg: DataGenConfig) -> pd.DataFrame:
    """
    Generates a fraud-like transactional dataset with:
    - a timestamp
    - tabular features (amount, merchant category, country, device, etc.)
    - a binary label y (fraud)

    Includes:
    - covariate drift after cfg.start_day_for_drift (feature distributions shift)
    - concept drift after cfg.start_day_for_drift (label-generating process shifts)
    """
    rng = np.random.default_rng(cfg.seed)

    total = cfg.n_days * cfg.events_per_day
    day_index = np.repeat(np.arange(cfg.n_days), cfg.events_per_day)

    # Create timestamps: spread events across each day
    base = pd.Timestamp("2025-01-01")
    seconds_in_day = 24 * 60 * 60
    intra_day_seconds = rng.integers(0, seconds_in_day, size=total)
    timestamps = base + pd.to_timedelta(day_index, unit="D") + pd.to_timedelta(intra_day_seconds, unit="s")

    # Base categorical distributions
    merchant_cats = np.array(["grocery", "fuel", "electronics", "travel", "fashion", "restaurants"])
    countries = np.array(["CA", "US", "GB", "IN", "AE"])
    devices = np.array(["mobile", "web", "pos"])

    # Pre-drift sampling probabilities
    p_cat_pre = np.array([0.28, 0.18, 0.12, 0.08, 0.14, 0.20])
    p_country_pre = np.array([0.55, 0.30, 0.07, 0.05, 0.03])
    p_device_pre = np.array([0.55, 0.35, 0.10])

    # Post-drift sampling probabilities (covariate drift)
    # Strength scales how far we move towards post-drift
    p_cat_post = np.array([0.20, 0.14, 0.18, 0.14, 0.16, 0.18])
    p_country_post = np.array([0.45, 0.34, 0.08, 0.08, 0.05])
    p_device_post = np.array([0.62, 0.28, 0.10])

    drift_mask = day_index >= cfg.start_day_for_drift

    # Interpolate probabilities for covariate drift
    s = np.clip(cfg.covariate_strength, 0.0, 1.0)
    p_cat = (1 - s) * p_cat_pre + s * p_cat_post
    p_country = (1 - s) * p_country_pre + s * p_country_post
    p_device = (1 - s) * p_device_pre + s * p_device_post

    # Sample categoricals with different distributions pre/post drift
    cat = np.empty(total, dtype=object)
    country = np.empty(total, dtype=object)
    device = np.empty(total, dtype=object)

    n_pre = int((~drift_mask).sum())
    n_post = int(drift_mask.sum())

    cat[~drift_mask] = rng.choice(merchant_cats, size=n_pre, p=p_cat_pre)
    country[~drift_mask] = rng.choice(countries, size=n_pre, p=p_country_pre)
    device[~drift_mask] = rng.choice(devices, size=n_pre, p=p_device_pre)

    cat[drift_mask] = rng.choice(merchant_cats, size=n_post, p=p_cat)
    country[drift_mask] = rng.choice(countries, size=n_post, p=p_country)
    device[drift_mask] = rng.choice(devices, size=n_post, p=p_device)

    # Numeric features
    hour = (intra_day_seconds // 3600).astype(int)
    is_night = ((hour <= 5) | (hour >= 22)).astype(int)
    is_new_account = rng.binomial(1, 0.12, size=total)

    # Amount distribution shifts post-drift (covariate drift)
    # Pre: lognormal(mean=3.2, sigma=0.6) ~ typical spending
    # Post: heavier tail
    amount = np.empty(total, dtype=float)
    amount[~drift_mask] = rng.lognormal(mean=3.2, sigma=0.6, size=n_pre)
    amount[drift_mask] = rng.lognormal(mean=3.35, sigma=0.75, size=n_post)

    # Velocity feature (transactions in last hour) - synthetic proxy
    velocity_1h = rng.poisson(lam=0.7, size=total) + is_new_account * rng.poisson(lam=1.2, size=total)

    # Label generation (fraud probability)
    # Pre-drift weights
    # (fraud more likely at night, higher amount, new accounts, electronics/travel, certain countries)
    w_amount_pre = 0.015
    w_night_pre = 0.45
    w_new_pre = 0.65
    w_vel_pre = 0.12

    # Category effects pre
    cat_effect_pre = {
        "grocery": -0.25,
        "fuel": -0.10,
        "electronics": 0.35,
        "travel": 0.55,
        "fashion": 0.10,
        "restaurants": -0.05,
    }
    country_effect_pre = {"CA": -0.15, "US": 0.00, "GB": 0.05, "IN": 0.18, "AE": 0.12}

    # Post-drift concept change: fraudsters shift strategy (concept drift)
    # Strength scales how far we move towards post-drift weights/effects
    t = np.clip(cfg.concept_strength, 0.0, 1.0)
    w_amount_post = 0.010
    w_night_post = 0.35
    w_new_post = 0.55
    w_vel_post = 0.18

    cat_effect_post = {
        "grocery": -0.10,
        "fuel": -0.05,
        "electronics": 0.20,
        "travel": 0.35,
        "fashion": 0.25,
        "restaurants": 0.05,
    }
    country_effect_post = {"CA": -0.10, "US": 0.02, "GB": 0.08, "IN": 0.10, "AE": 0.18}

    # Interpolated weights
    w_amount = (1 - t) * w_amount_pre + t * w_amount_post
    w_night = (1 - t) * w_night_pre + t * w_night_post
    w_new = (1 - t) * w_new_pre + t * w_new_post
    w_vel = (1 - t) * w_vel_pre + t * w_vel_post

    # Build logit
    base_logit = -4.0  # baseline fraud rate a few %
    logit = (
        base_logit
        + w_amount * amount
        + w_night * is_night
        + w_new * is_new_account
        + w_vel * velocity_1h
    )

    # Add categorical effects (pre vs post concept drift)
    # We apply pre effects before drift day, interpolated effects after drift day.
    # This is a clean way to encode concept drift.
    for i in range(total):
        if drift_mask[i]:
            ce = (1 - t) * cat_effect_pre[cat[i]] + t * cat_effect_post[cat[i]]
            coe = (1 - t) * country_effect_pre[country[i]] + t * country_effect_post[country[i]]
        else:
            ce = cat_effect_pre[cat[i]]
            coe = country_effect_pre[country[i]]
        logit[i] += ce + coe

    p_fraud = _sigmoid(logit)
    y = rng.binomial(1, p_fraud)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "day_index": day_index,
            "amount": amount,
            "merchant_cat": cat,
            "country": country,
            "device": device,
            "hour": hour,
            "is_night": is_night,
            "is_new_account": is_new_account,
            "velocity_1h": velocity_1h,
            "y": y,
        }
    ).sort_values("timestamp", ascending=True).reset_index(drop=True)

    return df
