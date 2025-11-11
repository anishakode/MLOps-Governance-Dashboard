from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "tables" in r.json()

def test_list_models_ok():
    r = client.get("/api/models")
    assert r.status_code in (200, 401, 403)  # depends on your auth settings
