# app/utils/policy.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Set

import yaml
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.monitoring import MonitoringMetric, Alert
from app.crud.waiver import list_waivers

# Where to read the policy file (compose sets POLICY_PATH; keep this default as a fallback)
POLICY_PATH = Path(os.getenv("POLICY_PATH", "backend/policies/policy.yaml"))

# cache in memory
_POLICY: Optional[dict] = None


# -------------------------- loading --------------------------

def _load_policy_from_file() -> dict:
    if POLICY_PATH.exists():
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    return {}

def get_policy() -> dict:
    global _POLICY
    if _POLICY is None:
        _POLICY = _load_policy_from_file()
    return _POLICY

def reload_policy() -> dict:
    global _POLICY
    _POLICY = _load_policy_from_file()
    return _POLICY


# -------------------------- helpers --------------------------

def _rule_cfg(pol: dict, key: str) -> Optional[dict]:
    rules = (pol or {}).get("rules") or {}
    val = rules.get(key)
    return val if isinstance(val, dict) else None

def _rule_enabled(cfg: Optional[dict]) -> bool:
    # your YAML doesn’t set "enabled", so treat presence as enabled by default
    if cfg is None:
        return False
    return bool(cfg.get("enabled", True))

def _active_waivers(db: Session, model_id: int, target: Optional[str]) -> Set[str]:
    """Return rule keys that have an active waiver for this model/target."""
    wa = list_waivers(db, model_id=model_id, active_only=True)
    keys: Set[str] = set()
    for w in wa:
        if (w.target is None) or (target is None) or (w.target == target):
            # waiver.rule holds the policy key (e.g., 'block_on_alerts', 'require_baselines', etc.)
            keys.add(w.rule)
    return keys

def _count_open_alerts(db: Session, model_id: int) -> Tuple[int, Dict[str,int], Dict[str,int]]:
    """(total, by_type, by_severity) for all OPEN alerts."""
    by_type: Dict[str,int] = {}
    by_sev: Dict[str,int] = {}
    q = (
        db.query(Alert.type, Alert.severity, func.count(Alert.id))
          .filter(Alert.model_id == model_id, Alert.status == "open")
          .group_by(Alert.type, Alert.severity)
          .all()
    )
    total = 0
    for t, s, c in q:
        total += c
        by_type[t] = by_type.get(t, 0) + c
        by_sev[s] = by_sev.get(s, 0) + c
    return total, by_type, by_sev

def _count_open_alerts_window(db: Session, model_id: int, since: datetime,
                              severities: List[str]) -> int:
    """Open alerts created within window, filtered by severities."""
    if not severities:
        severities = ["high"]
    q = (
        db.query(func.count(Alert.id))
          .filter(Alert.model_id == model_id,
                  Alert.status == "open",
                  Alert.created_at >= since,
                  Alert.severity.in_(severities))
          .scalar()
    )
    return int(q or 0)

def _latest_slo_violation_count(db: Session, model_id: int) -> Optional[int]:
    """Return count of violations in the most-recent SLO check (MonitoringMetric.method == 'slo_check')."""
    m = (
        db.query(MonitoringMetric)
          .filter(MonitoringMetric.model_id == model_id,
                  MonitoringMetric.method == "slo_check")
          .order_by(MonitoringMetric.created_at.desc())
          .first()
    )
    if not m:
        return None
    v = m.details.get("violations")
    if isinstance(v, list):
        return len(v)
    try:
        return int(v or 0)
    except Exception:
        return None

def _has_baseline_for_feature(db: Session, model_id: int, feature: str) -> bool:
    """True if a psi_baseline MonitoringMetric exists for (model, feature)."""
    exists = (
        db.query(func.count(MonitoringMetric.id))
          .filter(MonitoringMetric.model_id == model_id,
                  MonitoringMetric.feature == feature,
                  MonitoringMetric.method == "psi_baseline")
          .scalar()
    )
    return bool(exists)

# rank map for severity comparisons
_SEV_RANK = {"low": 0, "medium": 1, "high": 2}


# -------------------------- main evaluator --------------------------

def explain_policy(db: Session, model_id: int, target: str) -> dict:
    """
    Evaluate policy for a promotion to `target` (e.g., 'staging' or 'production').

    Supports both your existing rules:
      - rules.block_on_alerts: { severities: [...], lookback_hours: N }
      - rules.require_baselines: { features: [...] }

    And the additional rules we introduced (if present in YAML):
      - rules.block_high_alerts: { enabled: true }
      - rules.block_slo_alerts: { enabled: true, min_severity: 'medium'|'high' }
      - rules.require_clean_slo: { enabled: true }
      - rules.require_two_approvers_prod: { enabled: true }  (informational here; quorum is enforced in approvals)
    """
    pol = get_policy()
    rules = (pol or {}).get("rules") or {}

    # facts
    total_open, by_type, by_sev = _count_open_alerts(db, model_id)
    slo_open = by_type.get("slo", 0)
    slo_viol = _latest_slo_violation_count(db, model_id)

    waivers = _active_waivers(db, model_id, target)

    blocks: List[dict] = []

    # ---- Your schema: block_on_alerts (windowed, by severity) ----
    r = _rule_cfg(pol, "block_on_alerts")
    if _rule_enabled(r):
        sev_list = [s.lower() for s in (r.get("severities") or ["high"])]
        lookback_h = float(r.get("lookback_hours", 24))
        since = datetime.utcnow() - timedelta(hours=lookback_h)
        cnt = _count_open_alerts_window(db, model_id, since, sev_list)
        if cnt > 0 and "block_on_alerts" not in waivers:
            blocks.append({
                "rule": "block_on_alerts",
                "reason": f"{cnt} open alert(s) in last {int(lookback_h)}h with severity in {sev_list}",
                "waived": False,
            })

    # ---- Your schema: require_baselines (must have psi_baseline for features) ----
    r = _rule_cfg(pol, "require_baselines")
    if _rule_enabled(r):
        feats = list(r.get("features") or [])
        missing = [f for f in feats if not _has_baseline_for_feature(db, model_id, f)]
        if missing and "require_baselines" not in waivers:
            blocks.append({
                "rule": "require_baselines",
                "reason": f"Missing PSI baselines for features: {', '.join(missing)}",
                "waived": False,
            })

    # ---- Additional optional rules (only if present/enabled) ----

    # Block any HIGH alerts of any type
    r = _rule_cfg(pol, "block_high_alerts")
    if _rule_enabled(r):
        hi = by_sev.get("high", 0)
        if hi > 0 and "block_high_alerts" not in waivers:
            blocks.append({
                "rule": "block_high_alerts",
                "reason": f"{hi} high-severity open alert(s)",
                "waived": False,
            })

    # Block SLO alerts at/above min_severity
    r = _rule_cfg(pol, "block_slo_alerts")
    if _rule_enabled(r):
        min_sev = str(r.get("min_severity", "medium")).lower()
        q = (
            db.query(Alert.severity, func.count(Alert.id))
              .filter(Alert.model_id == model_id,
                      Alert.status == "open",
                      Alert.type == "slo")
              .group_by(Alert.severity)
              .all()
        )
        slo_block_cnt = 0
        for sev, cnt in q:
            if _SEV_RANK.get(sev, 0) >= _SEV_RANK.get(min_sev, 1):
                slo_block_cnt += cnt
        if slo_block_cnt > 0 and "block_slo_alerts" not in waivers:
            blocks.append({
                "rule": "block_slo_alerts",
                "reason": f"{slo_block_cnt} open SLO alert(s) >= {min_sev}",
                "waived": False,
            })

    # Require last SLO check to have zero violations
    r = _rule_cfg(pol, "require_clean_slo")
    if _rule_enabled(r):
        if slo_viol is not None and slo_viol > 0 and "require_clean_slo" not in waivers:
            blocks.append({
                "rule": "require_clean_slo",
                "reason": f"{slo_viol} SLO violation(s) in most recent SLO check",
                "waived": False,
            })

    allow = len(blocks) == 0

    return {
        "allow": allow,
        "target": target,
        "blocks": blocks,
        "snapshot": {
            "open_alerts": {"total": total_open, "by_type": by_type, "by_severity": by_sev},
            "slo": {"open_slo_alerts": slo_open, "last_slo_violations": slo_viol},
        },
        "policy_path": str(POLICY_PATH),
    }
