"""Versioned, atomic JSON snapshots; never deserialize executable pickle content."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.soccer_dixon_coles import DixonColesEngine
from src.nfl_epa_kalman import NFLKalmanKeyNumberEngine
from src.nba_possession_engine import NBAPossessionEngine


def encode(value):
    if isinstance(value, np.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, (pd.Timestamp, datetime)):
        return {"timestamp": value.isoformat()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"array"}:
            return np.asarray(value["array"])
        if set(value) == {"timestamp"}:
            return pd.Timestamp(value["timestamp"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


class ProductionStateRegistry:
    DEFAULT_STATE_PATH = "state/quant_engines_snapshot.json"
    METADATA_PATH = "state/snapshot_metadata.json"

    @classmethod
    def save_daily_snapshot(cls, soccer_engines, nfl_engine, nba_engine, current_bankroll, path=None):
        target = Path(path or cls.DEFAULT_STATE_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "bankroll_mxn": current_bankroll,
                   "soccer_engines": {k: encode(vars(v)) for k, v in soccer_engines.items()},
                   "nfl_engine": encode(vars(nfl_engine)) if nfl_engine is not None else None,
                   "nba_engine": encode(vars(nba_engine)) if nba_engine is not None else None}
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, target)
        meta = {"last_trained_utc": payload["timestamp_utc"], "bankroll_mxn": current_bankroll,
                "soccer_leagues_ready": list(soccer_engines),
                "nfl_teams_tracked": len(getattr(nfl_engine, "teams", [])),
                "nba_teams_tracked": len(getattr(nba_engine, "team_states", {}))}
        target.with_name("snapshot_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @classmethod
    def load_daily_snapshot(cls, path=None, max_age_hours=36):
        payload = json.loads(Path(path or cls.DEFAULT_STATE_PATH).read_text(encoding="utf-8"))
        if payload.get("schema") != 1:
            raise ValueError("Unsupported snapshot schema")
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(payload["timestamp_utc"])).total_seconds()/3600
        if age < 0 or age > max_age_hours:
            raise ValueError("Snapshot is stale or dated in the future; retrain models")
        def restore(model_type, attributes):
            if attributes is None:
                return None
            model = model_type()
            model.__dict__.update(decode(attributes))
            return model
        payload["soccer_engines"] = {k: restore(DixonColesEngine, v) for k, v in payload["soccer_engines"].items()}
        payload["nfl_engine"] = restore(NFLKalmanKeyNumberEngine, payload["nfl_engine"])
        payload["nba_engine"] = restore(NBAPossessionEngine, payload["nba_engine"])
        return payload
