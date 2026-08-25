import pytest
from fastapi.testclient import TestClient

ServerFastAPI = pytest.importorskip("ServerFastAPI")
from ServerFastAPI.app import app


client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    j = r.json()
    assert "model" in j


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_api_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, list) and j and "name" in j[0]


def test_v1_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, list) and j and "name" in j[0]


def test_api_generate():
    """Test /api/generate endpoint with simple prompt"""
    r = client.post(
        "/api/generate",
        json={"prompt": "Hello", "max_tokens": 32, "temperature": 0.3}
    )
    assert r.status_code == 200
    j = r.json()
    assert "choices" in j
    assert len(j["choices"]) > 0
    assert "text" in j["choices"][0]


def test_api_generate_stream():
    """Test /api/generate endpoint with streaming enabled"""
    r = client.post(
        "/api/generate",
        json={"prompt": "Hi", "max_tokens": 16, "stream": True}
    )
    assert r.status_code == 200
    # Streaming response is NDJSON
    lines = r.text.strip().split("\n")
    assert len(lines) > 0
    # Last line should be text.complete
    assert "text.complete" in lines[-1]


def test_v1_chat_completions():
    """Test /v1/chat/completions endpoint"""
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 32,
            "temperature": 0.3
        }
    )
    assert r.status_code == 200
    j = r.json()
    assert "choices" in j
    assert len(j["choices"]) > 0
    assert "message" in j["choices"][0]
    assert "content" in j["choices"][0]["message"]


def test_v1_completions():
    """Test /v1/completions endpoint (text completion)"""
    r = client.post(
        "/v1/completions",
        json={"prompt": "The answer is", "max_tokens": 16, "temperature": 0.0}
    )
    assert r.status_code == 200
    j = r.json()
    assert "choices" in j
    assert len(j["choices"]) > 0
    assert "text" in j["choices"][0]
