from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.generate import DataGenConfig, generate_synthetic_transactions
from src.data.io import load_yaml
from src.data.label_delay import LabelDelayConfig, apply_label_delay
from src.data.slicing import SliceConfig, slice_by_time
from src.monitoring.drift_metrics import psi_numeric, psi_categorical
from src.monitoring.alerting import DriftPolicy, evaluate_drift_alert


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _project_root()
    project_cfg = load_yaml(root / "configs" / "project.yaml")
    policy_cfg = load_yaml(root / "configs" / "policies.yaml")

    # --- configs ---
    env_cfg = project_cfg.get("environment", {}) or {}
    data_cfg = project_cfg.get("data", {}) or {}
    time_cfg = project_cfg.get("time", {}) or {}
    drift_cfg = data_cfg.get("drift", {}) or {}

    seed = int(env_cfg.get("random_seed", 42))
    n_days = int(data_cfg.get("n_days", 60))
    events_per_day = int(data_cfg.get("events_per_day", 800))
    window_days = int(data_cfg.get("window_days", 7))

    drift_start_day = int(drift_cfg.get("start_day", 28))
    cov_strength = float(drift_cfg.get("covariate_strength", 1.0))
    con_strength = float(drift_cfg.get("concept_strength", 1.0))

    delay_windows = int(time_cfg.get("label_delay_windows", 2))

    drift_policy_raw = policy_cfg.get("drift", {}) or {}
    monitored = drift_policy_raw.get("monitored_features", {}) or {}
    pol = DriftPolicy(
        reference_window=str(drift_policy_raw.get("reference_window", "T0")),
        min_fraction_features_breaching=float(drift_policy_raw.get("min_fraction_features_breaching", 0.25)),
        psi_warning=float(drift_policy_raw.get("psi_warning", 0.10)),
        psi_alert=float(drift_policy_raw.get("psi_alert", 0.20)),
        monitored_numeric=list((monitored.get("numeric", []) or [])),
        monitored_categorical=list((monitored.get("categorical", []) or [])),
    )

    print("=== Phase 3: monitoring cycle (PSI drift + alert) ===")

    # --- data ---
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

    # --- reference ---
    if pol.reference_window not in windows:
        raise ValueError(f"Reference window {pol.reference_window} not found. Available: {list(windows.keys())}")
    ref = windows[pol.reference_window]

    # --- compute PSI per window per feature ---
    rows = []
    for k in sorted(windows.keys(), key=lambda x: int(x[1:])):
        cur = windows[k]
        psi_map = {}

        for f in pol.monitored_numeric:
            psi_map[f] = psi_numeric(ref[f], cur[f], n_bins=10)

        for f in pol.monitored_categorical:
            psi_map[f] = psi_categorical(ref[f], cur[f])

        decision = evaluate_drift_alert(psi_map, pol)

        labels_available = int(delayed[k]["y"].notna().sum())

        rows.append(
            {
                "window": k,
                "labels_available": labels_available,
                "alert_level": decision["level"],
                "breach_fraction": decision["breach_fraction"],
                "breaching_features": ",".join(decision["breaching_features"]),
                **{f"psi_{feat}": psi_map[feat] for feat in psi_map},
            }
        )

    report = pd.DataFrame(rows)
    out = root / "reports" / "phase3_drift_alerts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    print(report[["window", "labels_available", "alert_level", "breach_fraction", "breaching_features"]].to_string(index=False))
    print(f"\nSaved: {out}")

    # Write an incident if any ALERT occurred
    if (report["alert_level"] == "alert").any():
        incident = root / "incidents" / "incident_001_drift_alert.md"
        incident.parent.mkdir(parents=True, exist_ok=True)

        first_alert = report.loc[report["alert_level"] == "alert"].iloc[0]
        incident.write_text(
            "# Incident 001 — Drift Alert Triggered\n\n"
            f"## Summary\n"
            f"Drift alert triggered in window **{first_alert['window']}**.\n\n"
            f"## Evidence\n"
            f"- Breach fraction: {first_alert['breach_fraction']}\n"
            f"- Breaching features: {first_alert['breaching_features']}\n"
            f"- Labels available: {first_alert['labels_available']} (may be delayed)\n\n"
            "## Action\n"
            "- Alert emitted.\n"
            "- No retraining executed in Phase 3.\n",
            encoding="utf-8",
        )
        print(f"Saved: {incident}")

    print("\nPhase 3 monitoring cycle complete.")


if __name__ == "__main__":
    main()
