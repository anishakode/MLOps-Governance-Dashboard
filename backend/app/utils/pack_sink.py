from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Default folder: backend/var/audit-packs (override with AUDIT_PACK_DIR)
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "var" / "audit-packs"
PACK_DIR = Path(os.getenv("AUDIT_PACK_DIR", str(_DEFAULT_DIR)))

def _ensure_dir() -> None:
    PACK_DIR.mkdir(parents=True, exist_ok=True)

def write_pack(model_id: int, pack: Dict[str, Any], prefix: str = "audit-pack") -> Path:
    """
    Save a single audit pack as pretty JSON.
    File name: <prefix>-model-<id>-<UTC timestamp>.json
    """
    _ensure_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{prefix}-model-{model_id}-{ts}.json"
    fp = PACK_DIR / name
    with fp.open("w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return fp
