from __future__ import annotations
import os, yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sqlalchemy.orm import Session
import hashlib

from app.models.monitoring import MonitoringMetric, Alert
from app.models.waiver import Waiver

# Default path: backend/policies/policy.yaml (override with POLICY_PATH env)
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "policies" / "policy.yaml"
_POLICY_PATH = Path(os.getenv("POLICY_PATH", str(_DEFAULT_PATH)))

_cache: Dict[str, Any] | None = None
_cache_mtime: float | None = None

def _hash_bytes(b: bytes) -> str: return hashlib.sha3_256(b).hexdigest()

def _maybe_reload() -> None:
    global _cache, _cache_mtime
    try:
        mtime = _POLICY_PATH.stat().st_mtime
        if _cache is None or _cache_mtime != mtime:
            with _POLICY_PATH.open("r", encoding="utf-8") as f:
                _cache = yaml.safe_load(f) or {}
            _cache_mtime = mtime
    except FileNotFoundError:
        _cache = {"rules": {}}
        _cache_mtime = None

    with _POLICY_PATH.open("rb") as f:
        raw = f.read()
    _cache = yaml.safe_load(raw) or {}
    _cache["__hash"] = _hash_bytes(raw)
    _cache_mtime = mtime

def get_policy() -> Dict[str, Any]:
    _maybe_reload()
    return _cache or {"rules": {}}

def reload_policy() -> Dict[str, Any]:
    global _cache, _cache_mtime
    _cache = None
    _cache_mtime = None
    _maybe_reload()
    return get_policy()

def _active_waiver(db: Session, model_id: int, rule: str, target: Optional[str]) -> Optional[Waiver]:
    now = datetime.utcnow()
    q = (
        db.query(Waiver)
          .filter(Waiver.model_id == model_id,
                  Waiver.rule == rule,
                  Waiver.status == "approved")
          .filter((Waiver.expires_at == None) | (Waiver.expires_at >= now))  # noqa: E711
    )
    if target:
        q = q.filter((Waiver.target == target) | (Waiver.target == None))  # noqa: E711
    return q.order_by(Waiver.created_at.desc()).first()

def explain_policy(db: Session, model_id: int, target: str) -> Dict[str, Any]:

    p = get_policy()
    policy_hash = p.get("__hash")
    rules = p.get("rules", {}) or {}
    results: Dict[str, Dict[str, Any]] = {}
    waivers_used: List[Dict[str, Any]] = []
    failures: List[str] = []

    # Rule: block_on_alerts
    r = rules.get("block_on_alerts") or {}
    severities = set(r.get("severities", ["high"]))
    lookback_hours = int(r.get("lookback_hours", 24))
    if severities:
        since = datetime.utcnow() - timedelta(hours=lookback_hours)
        exists = (
            db.query(Alert)
              .filter(
                  Alert.model_id == model_id,
                  Alert.severity.in_(list(severities)),
                  Alert.status == "open",
                  Alert.created_at >= since
              ).first()
        )
        if exists:
            w = _active_waiver(db, model_id, "block_on_alerts", target)
            if w:
                results["block_on_alerts"] = {"pass": True, "waived": True, "waiver_id": w.id}
                waivers_used.append({"rule": "block_on_alerts", "waiver_id": w.id})
            else:
                results["block_on_alerts"] = {
                    "pass": False,
                    "reason": f"Open alerts in last {lookback_hours}h with severities {sorted(severities)}"
                }
                failures.append(results["block_on_alerts"]["reason"])
        else:
            results["block_on_alerts"] = {"pass": True}

    # Rule: require_baselines
    rb = rules.get("require_baselines") or {}
    feats = rb.get("features", [])
    if feats:
        missing = []
        for feat in feats:
            baseline = (
                db.query(MonitoringMetric)
                  .filter(MonitoringMetric.model_id == model_id,
                          MonitoringMetric.feature == feat,
                          MonitoringMetric.method == "psi_baseline")
                  .order_by(MonitoringMetric.created_at.desc()).first()
            )
            if not baseline:
                missing.append(feat)
        if missing:
            w = _active_waiver(db, model_id, "require_baselines", target)
            if w:
                results["require_baselines"] = {"pass": True, "waived": True, "waiver_id": w.id}
                waivers_used.append({"rule": "require_baselines", "waiver_id": w.id})
            else:
                reason = f"Missing PSI baseline for feature(s): {', '.join(missing)}"
                results["require_baselines"] = {"pass": False, "reason": reason}
                failures.append(reason)
        else:
            results["require_baselines"] = {"pass": True}

    # Combine
    overall_pass = (len(failures) == 0)
    return {"target": target, "policy_hash": policy_hash, "rules": results, "pass": overall_pass, "failures": failures, "waivers_used": waivers_used}

def check_promotion_policy(db: Session, model_id: int, target: str) -> Tuple[bool, List[str]]:
    e = explain_policy(db, model_id, target)
    return e["pass"], e["failures"]

def evaluate_promotion_policy(db: Session, model_id: int, target: str) -> Dict[str, Any]:
    return explain_policy(db, model_id, target)



