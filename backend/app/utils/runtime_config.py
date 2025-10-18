import os
from threading import RLock

_lock = RLock()
_state = {
    # seed from environment on boot; can be overridden at runtime
    "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL", "").strip(),
}

def set_slack_webhook(url: str | None) -> None:
    with _lock:
        _state["SLACK_WEBHOOK_URL"] = (url or "").strip()

def get_slack_webhook() -> str:
    with _lock:
        return _state.get("SLACK_WEBHOOK_URL", "")
