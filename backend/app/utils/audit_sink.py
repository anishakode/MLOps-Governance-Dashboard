from __future__ import annotations
import os, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Default: backend/var/audit (override with env AUDIT_DIR)
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "var" / "audit"
AUDIT_DIR = Path(os.getenv("AUDIT_DIR", str(_DEFAULT_DIR)))

def _ensure_dir() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def write_event(event: Dict[str, Any]) -> None:
    """
    Append a single audit event to a day-partitioned .jsonl file.
    Each line is a JSON object.
    """
    _ensure_dir()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fp = AUDIT_DIR / f"{day}.jsonl"
    with fp.open("a", encoding="utf-8") as fh:
        json.dump(event, fh, ensure_ascii=False)
        fh.write("\n")
