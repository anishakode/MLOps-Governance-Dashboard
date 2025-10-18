import os, jwt
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")

# Access/refresh lifetimes (minutes)
ACCESS_TTL_MIN = int(os.getenv("JWT_ACCESS_TTL_MIN", "30"))       
REFRESH_TTL_MIN = int(os.getenv("JWT_REFRESH_TTL_MIN", "10080"))   

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _encode(payload: Dict[str, Any]) -> str:
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def _decode(token: str) -> Dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])

def create_access_token(username: str, role: str) -> str:
    now = _now()
    exp = now + timedelta(minutes=ACCESS_TTL_MIN)
    payload = {
        "sub": username,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return _encode(payload)

def create_refresh_token(username: str, role: str) -> str:
    now = _now()
    exp = now + timedelta(minutes=REFRESH_TTL_MIN)
    payload = {
        "sub": username,
        "role": role,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return _encode(payload)

def decode_token(token: str, expected_type: str | None = None) -> Dict[str, Any]:
    data = _decode(token)
    if expected_type and data.get("type") != expected_type:
        raise jwt.InvalidTokenError(f"unexpected token type: {data.get('type')}")
    return data

def create_token(username: str, role: str) -> str:
    return create_access_token(username, role)
