# backend/__init__.py
import os, sys
_pkg_dir = os.path.dirname(__file__)
if _pkg_dir not in sys.path:
    # let absolute imports like "from app.core..."
    sys.path.insert(0, _pkg_dir)

from .main import app  # re-export FastAPI instance

__all__ = ["app"]