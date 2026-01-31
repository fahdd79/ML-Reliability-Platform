from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib

from src.data.generate import DataGenConfig, generate_synthetic_transactions
from src.data.io import load_yaml
from src.data.label_delay import LabelDelayConfig, apply_label_delay
from src.data.slicing import SliceConfig, slice_by_time
from src.monitoring.drift_metrics import psi_numeric, psi_categorical
from src.monitoring.alerting import DriftPolicy, evaluate_drift_alert
from src.models.train import train_logreg
from src.models.preprocess import NUMERIC, CATEGORICAL
from src.gates.model_gates import PromotionPolicy, decide_promotion


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_dirs(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "incidents").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)


def _window_keys(windows: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(windows.keys(), key=lambda x: int(x[1:]))


def _select_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[NUMERIC + CATEGORICAL].copy()
    y = df["y"].astype(int)
    return x, y


def main() -> None:
    root = _project_root()
    _ensure_dirs(root)

    project_cfg = load_yaml(root / "configs" / "project.yaml")
    policy_cfg = load_yaml(root / "configs" / "policies.yaml")

    # --- project config ---
    env_cfg = project_cfg.get("environment", {}) or {}
    data_cfg = project_cfg.get("data", {}) or {}
    time_cfg = project_cfg.get("time", {}) or {}
    drift_cfg = data_cfg.get("drift", {}) or {}

    seed = int(env_cfg.get("random_seed", 42))
    n_days = int(data_cfg.get("n_days", 60))
    events_per_day = int(data_cfg.get("events_per_day", 800))
    window_days = int(data_cfg.get("window_days", 7))
    delay_windows = int(time_cfg.get("label_delay_windows", 2))

    drift_start_day = int(drift_cfg.get("start_day", 28))
    cov_strength = float(drift_cfg.get("covariate_strength", 1.0))
    con_strength = float(drift_cfg.get("concept_strength", 1.0))

    # --- drift policy ---
    drift_policy_raw = policy_cfg.get("drift", {}) or {}
    monitored = drift_policy_raw.get("monitored_features", {}) or {}
    drift_pol = DriftPolicy(
        reference_window=str(drift_policy_raw.get("reference_window", "T0")),
        min_fraction_features_breaching=float(drift_policy_raw.get("min_fraction_features_breaching", 0.25)),
        psi_warning=float(drift_policy_raw.get("psi_warning", 0.10)),
        psi_alert=float(drift_policy_raw.get("psi_alert", 0.20)),
        monitored_numeric=list((monitored.get("numeric", []) or [])),
        monitored_categorical=list((monitored.get("categorical", []) or [])),
    )

    # --- promotion policy (hardcoded for now; we can move to yaml later) ---
    promo_pol = PromotionPolicy(
        min_roc_auc_gain=0.00,     # allow equal ROC AUC (early phase)
        min_pr_auc_gain=0.00,      # allow equal PR AUC (early phase)
        max_brier_increase=0.002,  # do not get noticeably worse calibrated
    )

    print("=== Phase 4: retraining cycle (trigger + evaluate + gate) ===")

    # --- generate and slice ---
    df = generate_synthetic_transactions(
        DataGenConfig(
            n_days=n_days,
            events_per_day=events_per_day,
            start_day_for_drift=drift_start_day,
            covariate_strength=cov_strength,
            concept_strength=con_strength,
            seed=seed,
        )
    )
    windows = slice_by_time(df, SliceConfig(window_days=window_days))
    delayed = apply_label_delay(windows, LabelDelayConfig(delay_windows=delay_windows), label_col="y")

    keys = _window_keys(windows)
    if drift_pol.reference_window not in windows:
        raise ValueError(f"Reference window {drift_pol.reference_window} not found")

    ref = windows[drift_pol.reference_window]

    # --- compute drift decisions per window ---
    drift_decisions = {}
    for k in keys:
        cur = windows[k]
        psi_map = {}
        for f in drift_pol.monitored_numeric:
            psi_map[f] = psi_numeric(ref[f], cur[f], n_bins=10)
        for f in drift_pol.monitored_categorical:
            psi_map[f] = psi_categorical(ref[f], cur[f])

        drift_decisions[k] = evaluate_drift_alert(psi_map, drift_pol)

    # --- choose windows for prod vs candidate ---
    # Production baseline: early stable labeled windows
    prod_train_keys = ["T0", "T1", "T2"]

    # Candidate retrain: most recent labeled windows BEFORE label delay starts
    # With delay_windows=2 and keys ending at T8, last labeled is T6.
    candidate_train_keys = ["T3", "T4", "T5"]
    eval_key = "T6"

    # Validate label availability
    for k in prod_train_keys + candidate_train_keys + [eval_key]:
        if k not in delayed:
            raise ValueError(f"Missing window {k}")
        if delayed[k]["y"].isna().any():
            raise RuntimeError(f"Cannot use {k}: labels are delayed/hidden")

    # --- training datasets ---
    prod_train = pd.concat([delayed[k] for k in prod_train_keys], ignore_index=True)
    cand_train = pd.concat([delayed[k] for k in candidate_train_keys], ignore_index=True)
    eval_df = delayed[eval_key].copy()

    # --- train models ---
    prod_res = train_logreg(prod_train)
    cand_res = train_logreg(cand_train)

    # --- evaluate both on same eval window ---
    x_eval, y_eval = _select_xy(eval_df)

    prod_probs = prod_res.model.predict_proba(x_eval)[:, 1]
    cand_probs = cand_res.model.predict_proba(x_eval)[:, 1]

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    prod_metrics = {
        "roc_auc": float(roc_auc_score(y_eval, prod_probs)),
        "pr_auc": float(average_precision_score(y_eval, prod_probs)),
        "brier": float(brier_score_loss(y_eval, prod_probs)),
    }
    cand_metrics = {
        "roc_auc": float(roc_auc_score(y_eval, cand_probs)),
        "pr_auc": float(average_precision_score(y_eval, cand_probs)),
        "brier": float(brier_score_loss(y_eval, cand_probs)),
    }

    gate = decide_promotion(prod_metrics, cand_metrics, promo_pol)

    # --- trigger logic: only retrain if we have an ALERT in recent windows ---
    recent_keys = ["T4", "T5", "T6"]
    trigger_level = max((drift_decisions[k]["level"] for k in recent_keys), key=lambda x: {"none": 0, "warning": 1, "alert": 2}[x])

    retrain_triggered = trigger_level == "alert"

    # --- report ---
    report = {
        "trigger_level_recent": trigger_level,
        "retrain_triggered": retrain_triggered,
        "eval_window": eval_key,
        "prod_train_windows": ",".join(prod_train_keys),
        "cand_train_windows": ",".join(candidate_train_keys),
        "prod_metrics": prod_metrics,
        "cand_metrics": cand_metrics,
        "gate_decision": gate["decision"],
        "gate_reasons": gate["reasons"],
    }

    out = root / "reports" / "phase4_retraining_decision.json"
    out.write_text(pd.Series(report).to_json(), encoding="utf-8")
    print(f"Saved: {out}")

    # --- act on decision ---
    if not retrain_triggered:
        incident = root / "incidents" / "incident_010_retrain_blocked_not_alert.md"
        incident.write_text(
            "# Incident 010 — Retraining Blocked\n\n"
            f"## Reason\n"
            f"Retraining requires ALERT in recent windows, but max level was **{trigger_level}**.\n\n"
            "## Notes\n"
            "- This is intentional: warnings do not auto-retrain.\n",
            encoding="utf-8",
        )
        print(f"Saved: {incident}")
        print("Retraining blocked (no alert).")
        return

    if gate["decision"] == "PROMOTE":
        # Save promoted model artifact
        model_path = root / "artifacts" / "model_prod.joblib"
        joblib.dump(cand_res.model, model_path)
        print(f"PROMOTED: saved new production model to {model_path}")
    else:
        incident = root / "incidents" / "incident_011_candidate_rejected.md"
        incident.write_text(
            "# Incident 011 — Candidate Rejected by Gates\n\n"
            f"## Gate decision\n"
            f"{gate['decision']}\n\n"
            f"## Reasons\n"
            + "\n".join([f"- {r}" for r in gate["reasons"]]) +
            "\n",
            encoding="utf-8",
        )
        print(f"Saved: {incident}")
        print("Candidate rejected by gates.")

    print("Phase 4 retraining cycle complete.")


if __name__ == "__main__":
    main()
