from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import re
import joblib
import pandas as pd


_VERSION_RE = re.compile(r"model_v(\d+)\.joblib$")


@dataclass(frozen=True)
class ModelRecord:
    version: int
    model_path: Path
    meta_path: Path


def _registry_dir(root: Path) -> Path:
    d = root / "artifacts" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(root: Path) -> Path:
    p = root / "artifacts" / "registry.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def next_version(root: Path) -> int:
    d = _registry_dir(root)
    versions = []
    for f in d.glob("model_v*.joblib"):
        m = _VERSION_RE.search(f.name)
        if m:
            versions.append(int(m.group(1)))
    return (max(versions) + 1) if versions else 1


def save_model(root: Path, model, metadata: Dict) -> ModelRecord:
    v = next_version(root)
    d = _registry_dir(root)

    model_path = d / f"model_v{v:03d}.joblib"
    meta_path = d / f"model_v{v:03d}.json"

    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Append to registry index
    idx = _index_path(root)
    row = {
        "version": v,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "primary_metric": metadata.get("primary_metric"),
        "primary_value": metadata.get("candidate_metrics", {}).get(metadata.get("primary_metric", ""), None),
        "gate_decision": metadata.get("gate", {}).get("decision"),
    }
    if idx.exists():
        df = pd.read_csv(idx)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(idx, index=False)

    return ModelRecord(version=v, model_path=model_path, meta_path=meta_path)


def latest_model_record(root: Path) -> Optional[ModelRecord]:
    d = _registry_dir(root)
    versions = []
    for f in d.glob("model_v*.joblib"):
        m = _VERSION_RE.search(f.name)
        if m:
            versions.append((int(m.group(1)), f))
    if not versions:
        return None
    v, f = sorted(versions, key=lambda x: x[0])[-1]
    meta = f.with_suffix(".json")
    return ModelRecord(version=v, model_path=f, meta_path=meta)
