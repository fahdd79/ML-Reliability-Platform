from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.generate import DataGenConfig, generate_synthetic_transactions
from src.data.io import load_yaml
from src.data.label_delay import LabelDelayConfig, apply_label_delay
from src.data.slicing import SliceConfig, slice_by_time

from src.monitoring.drift_metrics import psi_numeric, psi_categorical
from src.monitoring.alerting import DriftPolicy, evaluate_drift_alert

from src.models.train import train_logreg
from src.models.preprocess import NUMERIC, CATEGORICAL

from src.gates.model_gates import PromotionPolicy, decide_promotion
from src.orchestration.trigger_logic import TriggerPolicy, should_retrain
from src.serving.model_registry import save_model


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_dirs(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "incidents").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)


def _window_keys(windows: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(windows.keys(), key=lambda x: int(x[1:]))


def _idx(w: str) -> int:
    return int(w[1:])


def _select_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Strict feature selection to match training pipeline
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

    # --- drift policy (Phase 3/4 compatible) ---
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

    # --- new Phase 5 policy blocks ---
    alerts_cfg = policy_cfg.get("alerts", {}) or {}
    retrain_cfg = policy_cfg.get("retraining", {}) or {}
    promo_cfg = policy_cfg.get("promotion", {}) or {}
    rollback_cfg = policy_cfg.get("rollback", {}) or {}

    trigger_policy = TriggerPolicy(
        trigger_level=str(retrain_cfg.get("trigger_level", "alert")),
        consecutive_windows=int(alerts_cfg.get("consecutive_windows", 2)),
        cooldown_windows=int(retrain_cfg.get("cooldown_windows", 2)),
        min_new_samples=int(retrain_cfg.get("min_new_samples", 500)),
    )

    primary_metric = str(promo_cfg.get("primary_metric", "pr_auc"))
    primary_min_improvement = float(promo_cfg.get("primary_metric_min_improvement", 0.0))
    guardrail_metric = str(promo_cfg.get("guardrail_metric", "brier"))
    guardrail_max_regression = float(promo_cfg.get("guardrail_max_regression", 0.002))

    performance_floor = float(rollback_cfg.get("performance_floor_primary_metric", 0.80))

    # NOTE: our gate function currently checks roc/pr/brier.
    # We map your policy onto that gate to keep the implementation minimal and credible.
    promo_pol = PromotionPolicy(
        primary_metric=primary_metric,
        primary_min_improvement=primary_min_improvement,
        guardrail_metric=guardrail_metric,
        guardrail_max_regression=guardrail_max_regression,
    )

    print("=== Phase 5: retraining cycle (policy trigger + gates + registry + rollback floor) ===")
    print(f"trigger_policy: level={trigger_policy.trigger_level}, consecutive={trigger_policy.consecutive_windows}, "
          f"cooldown={trigger_policy.cooldown_windows}, min_new_samples={trigger_policy.min_new_samples}")
    print(f"promotion: primary={primary_metric}, min_improve={primary_min_improvement}, "
          f"guardrail={guardrail_metric}, max_regress={guardrail_max_regression}")
    print(f"rollback: floor_{primary_metric}={performance_floor}")

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
        raise ValueError(f"Reference window {drift_pol.reference_window} not found. Available: {list(windows.keys())}")

    ref = windows[drift_pol.reference_window]

    # --- compute drift decisions per window ---
    drift_decisions: dict[str, dict[str, object]] = {}
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
    # With delay_windows=2 and keys ending at T8, last labeled is T6 (as you observed).
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

    print("prod_metrics:", prod_metrics)
    print("cand_metrics:", cand_metrics)
    print("gate:", gate)

    # --- policy-based trigger ---
    # Use the last 3 windows up to eval_key to decide trigger (configurable later).
    recent_keys = ["T4", "T5", "T6"]
    recent_levels = [str(drift_decisions[k]["level"]) for k in recent_keys]
    recent_sizes = [len(windows[k]) for k in recent_keys]

    trigger = should_retrain(
        recent_window_levels=recent_levels,
        recent_window_sizes=recent_sizes,
        policy=trigger_policy,
        last_retrain_window_index=None,  # Phase 5: not persisted yet
        current_window_index=_idx(eval_key),
    )

    retrain_triggered = bool(trigger["trigger"])
    trigger_reason = str(trigger["reason"])

    # --- report payload ---
    report = {
        "recent_windows": recent_keys,
        "recent_levels": recent_levels,
        "retrain_triggered": retrain_triggered,
        "trigger_reason": trigger_reason,
        "eval_window": eval_key,
        "prod_train_windows": prod_train_keys,
        "candidate_train_windows": candidate_train_keys,
        "prod_metrics": prod_metrics,
        "cand_metrics": cand_metrics,
        "gate": gate,
        "promotion_primary_metric": primary_metric,
        "rollback_floor_primary_metric": performance_floor,
    }

    out = root / "reports" / "phase5_retraining_decision.json"
    out.write_text(pd.Series(report).to_json(), encoding="utf-8")
    print(f"Saved: {out}")

    # --- blocked retraining path ---
    if not retrain_triggered:
        incident = root / "incidents" / "incident_010_retrain_blocked.md"
        incident.write_text(
            "# Incident 010 — Retraining Blocked\n\n"
            "## Trigger policy\n"
            f"- trigger_level: {trigger_policy.trigger_level}\n"
            f"- consecutive_windows: {trigger_policy.consecutive_windows}\n"
            f"- cooldown_windows: {trigger_policy.cooldown_windows}\n"
            f"- min_new_samples: {trigger_policy.min_new_samples}\n\n"
            f"## Recent windows\n- {', '.join([f'{k}:{lvl}' for k, lvl in zip(recent_keys, recent_levels)])}\n\n"
            f"## Reason\n{trigger_reason}\n",
            encoding="utf-8",
        )
        print(f"Saved: {incident}")
        print("Retraining blocked.")
        return

    # --- promotion / registry / rollback floor ---
    if gate["decision"] == "PROMOTE":
        cand_primary = float(cand_metrics.get(primary_metric, float("nan")))
        if cand_primary < performance_floor:
            incident = root / "incidents" / "incident_012_rollback_floor_block.md"
            incident.write_text(
                "# Incident 012 — Promotion Blocked by Performance Floor\n\n"
                f"Candidate {primary_metric}={cand_primary:.4f} below floor {performance_floor:.4f}\n",
                encoding="utf-8",
            )
            print(f"Saved: {incident}")
            print("Promotion blocked by rollback floor.")
            return

        metadata = {
            "primary_metric": primary_metric,
            "candidate_metrics": cand_metrics,
            "production_metrics": prod_metrics,
            "gate": gate,
            "trigger": {"triggered": retrain_triggered, "reason": trigger_reason},
            "train_windows": {"prod": prod_train_keys, "candidate": candidate_train_keys},
            "eval_window": eval_key,
        }
        rec = save_model(root, cand_res.model, metadata)
        print(f"PROMOTED: registered model version v{rec.version:03d} at {rec.model_path}")
        return

    # --- rejected path ---
    incident = root / "incidents" / "incident_011_candidate_rejected.md"
    incident.write_text(
        "# Incident 011 — Candidate Rejected by Gates\n\n"
        f"## Gate decision\n{gate['decision']}\n\n"
        "## Reasons\n"
        + "\n".join([f"- {r}" for r in gate["reasons"]]) +
        "\n",
        encoding="utf-8",
    )
    print(f"Saved: {incident}")
    print("Candidate rejected by gates.")


if __name__ == "__main__":
    main()
