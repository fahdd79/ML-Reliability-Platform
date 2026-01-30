from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.generate import DataGenConfig, generate_synthetic_transactions
from src.data.io import load_yaml
from src.data.label_delay import LabelDelayConfig, apply_label_delay
from src.data.slicing import SliceConfig, slice_by_time


def _project_root() -> Path:
    # run_end_to_end.py is in src/orchestration/
    # parents[0] = orchestration, [1] = src, [2] = project root
    return Path(__file__).resolve().parents[2]


def _ensure_dirs(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "incidents").mkdir(parents=True, exist_ok=True)


def main() -> None:
    root = _project_root()
    _ensure_dirs(root)

    project_cfg = load_yaml(root / "configs" / "project.yaml")

    # --- Read config safely ---
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

    print("=== Phase 2: data generation + time slicing + delayed labels ===")
    print(f"project_root: {root}")
    print(f"n_days={n_days}, events_per_day={events_per_day}, window_days={window_days}")
    print(f"drift_start_day={drift_start_day}, cov_strength={cov_strength}, concept_strength={con_strength}")
    print(f"label_delay_windows={delay_windows}")

    # --- Generate ---
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

    # --- Slice ---
    windows = slice_by_time(df, SliceConfig(window_days=window_days))

    # --- Apply delayed labels ---
    delayed = apply_label_delay(
        windows,
        LabelDelayConfig(delay_windows=delay_windows),
        label_col="y",
    )

    # --- Summarize ---
    rows = []
    for k in sorted(delayed.keys(), key=lambda x: int(x[1:])):  # T0, T1, ...
        w = delayed[k]
        n = len(w)
        label_available = int(w["y"].notna().sum())
        fraud_rate = float(w["y"].mean()) if label_available > 0 else None
        rows.append(
            {
                "window": k,
                "n_rows": n,
                "labels_available": label_available,
                "labels_hidden": n - label_available,
                "fraud_rate_if_available": fraud_rate,
                "time_min": str(w["timestamp"].min()),
                "time_max": str(w["timestamp"].max()),
            }
        )

    summary = pd.DataFrame(rows)

    print("\n--- Window summary ---")
    print(summary.to_string(index=False))

    out_csv = root / "reports" / "phase2_window_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    incident_path = root / "incidents" / "incident_000_phase2_data_ready.md"
    incident_path.write_text(
        "# Incident 000 — Phase 2 Data Readiness\n\n"
        "## What happened\n"
        "Phase 2 generated a fraud-like transactional dataset, sliced it into time windows, "
        "and applied delayed label availability.\n\n"
        "## Why it matters\n"
        "- Time slicing enables realistic monitoring over time.\n"
        "- Label delay forces decisions under uncertainty (a core production constraint).\n\n"
        "## Output artifacts\n"
        "- `reports/phase2_window_summary.csv`\n",
        encoding="utf-8",
    )
    print(f"Saved: {incident_path}")

    print("\nPhase 2 pipeline ran successfully (data + slicing + label delay).")


if __name__ == "__main__":
    main()
