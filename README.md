# Production ML Reliability Platform

This project implements a production-style ML reliability system that monitors
distribution shift, raises alerts, and performs gated retraining and deployment
under delayed label feedback.

Core focus:
- Drift detection
- Decision gates
- Safe retraining
- Model promotion / rollback
- Incident-style evaluation

This repository is structured as a multi-phase system build.