import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def wait_until_built(client, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = client.get("/api/health").json()
        if health["stage"] in {"ready", "error"}:
            return health
        time.sleep(0.05)
    raise AssertionError("startup did not finish")


@pytest.fixture
def client(settings, llm, embeddings):
    with TestClient(create_app(settings, llm=llm, embeddings=embeddings)) as c:
        assert wait_until_built(c)["stage"] == "ready"
        yield c


def test_health_and_documents(client):
    health = client.get("/api/health").json()
    assert health["provider"] == "ollama" and health["files"] == 2 and health["chunks"] > 3
    assert {d["name"] for d in client.get("/api/documents").json()} == {"manual.txt", "notes.md"}


def test_chat_keeps_a_conversation_per_session(client, llm):
    first = client.post("/api/chat", json={"question": "Tell me about the Falcon drone."}).json()
    assert first["answer"] and first["session_id"] and first["citations_verified"] == [True, False]

    follow_up = client.post("/api/chat", json={"question": "How long is it?", "session_id": first["session_id"]}).json()
    assert follow_up["session_id"] == first["session_id"]
    assert follow_up["standalone_question"] == llm.rewrite       # it had a history to rewrite from

    other = client.post("/api/chat", json={"question": "How long is it?"}).json()
    assert other["session_id"] != first["session_id"]
    assert other["standalone_question"] == "How long is it?"      # a new conversation has no history

    assert client.delete(f"/api/sessions/{first['session_id']}").status_code == 204
    forgotten = client.post("/api/chat", json={"question": "How long is it?", "session_id": first["session_id"]}).json()
    assert forgotten["standalone_question"] == "How long is it?"


def test_responses_are_logged(client, caplog):
    with caplog.at_level("INFO", logger="rag"):
        client.post("/api/chat", json={"question": "How long is the warranty?"})
        client.post("/api/chat", json={"question": ""})
        client.get("/api/health")
    lines = [r.getMessage() for r in caplog.records]
    assert any(l.startswith("Question (session ") and l.endswith("How long is the warranty?") for l in lines)
    assert any(l.startswith("POST /api/chat -> 200 in ") and '"answer":"It is in the manual."' in l for l in lines)
    assert any(l.startswith("POST /api/chat -> 422 in ") for l in lines)      # errors are logged too
    assert not any("/api/health" in l for l in lines)                         # too noisy to log


def test_chat_rejects_bad_input(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "hi", "session_id": "../etc"}).status_code == 422


def test_inspect_and_page(client):
    chunks = client.get("/api/inspect", params={"q": "ZX-CAL", "k": 1}).json()
    assert any("ZX-CAL" in c["text"] for c in chunks) and chunks[0]["source"] == "manual.txt"
    page = client.get("/")
    assert page.status_code == 200 and "Ask your documents" in page.text


def test_empty_folder_is_reported_and_chat_is_refused(settings, llm, embeddings):
    for f in settings.data_dir.iterdir():
        f.unlink()
    with TestClient(create_app(settings, llm=llm, embeddings=embeddings)) as c:
        health = wait_until_built(c)
        assert health["stage"] == "error" and "No text could be extracted" in health["detail"]
        assert c.post("/api/chat", json={"question": "hi"}).status_code == 503
