from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Callable
from app.core.security import decode_token

class CurrentUser(BaseModel):
    username: str
    role: str

def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        data = decode_token(token, expected_type="access")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid/expired token")
    return CurrentUser(username=data.get("sub", "unknown"), role=data.get("role", "viewer"))

def require_role(*allowed: str) -> Callable:
    def checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if allowed and user.role not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return checker
