# Incident 000 — Phase 2 Data Readiness

## What happened
Phase 2 generated a fraud-like transactional dataset, sliced it into time windows, and applied delayed label availability.

## Why it matters
- Time slicing enables realistic monitoring over time.
- Label delay forces decisions under uncertainty (a core production constraint).

## Output artifacts
- `reports/phase2_window_summary.csv`
