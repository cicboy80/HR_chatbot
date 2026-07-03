from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main

PDF_MAGIC = b"%PDF-1.7 minimal test payload"


@pytest.fixture
def client(monkeypatch):
    """TestClient with the Weaviate connection mocked out (no network)."""
    fake_weaviate = MagicMock()
    monkeypatch.setattr(main, "connect", lambda *a, **k: fake_weaviate)
    monkeypatch.setattr(main, "ensure_schema", lambda c: None)
    with TestClient(main.app) as tc:
        tc.fake_weaviate = fake_weaviate
        yield tc


def ip(n: int) -> dict:
    """Unique X-Forwarded-For per test so slowapi buckets don't collide."""
    return {"X-Forwarded-For": f"10.9.{n // 256}.{n % 256}"}


_ip_counter = iter(range(1, 10_000))


@pytest.fixture
def headers():
    return ip(next(_ip_counter))


class TestHealth:
    def test_health_reports_weaviate_state(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["weaviate"] in ("connected", "disconnected")


class TestUploadValidation:
    def test_rejects_non_pdf_extension(self, client, headers):
        r = client.post(
            "/upload_pdf",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            headers=headers,
        )
        assert r.status_code == 400
        assert "PDF" in r.json()["detail"]

    def test_rejects_dotfile_name(self, client, headers):
        r = client.post(
            "/upload_pdf",
            files={"file": (".pdf", PDF_MAGIC, "application/pdf")},
            headers=headers,
        )
        assert r.status_code == 400

    def test_rejects_wrong_magic_bytes(self, client, headers):
        r = client.post(
            "/upload_pdf",
            files={"file": ("fake.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")},
            headers=headers,
        )
        assert r.status_code == 400
        assert "not a valid PDF" in r.json()["detail"]

    def test_rejects_oversized_file(self, client, headers, monkeypatch):
        monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
        r = client.post(
            "/upload_pdf",
            files={"file": ("big.pdf", PDF_MAGIC + b"x" * 100, "application/pdf")},
            headers=headers,
        )
        assert r.status_code == 413

    def test_corrupt_pdf_returns_400_and_is_cleaned_up(self, client, headers):
        r = client.post(
            "/upload_pdf",
            files={"file": ("corrupt.pdf", PDF_MAGIC, "application/pdf")},
            headers=headers,
        )
        assert r.status_code == 400
        # nothing left behind in the upload dir for this file
        leftovers = [p for p in main.UPLOAD_DIR.glob("*corrupt.pdf")]
        assert leftovers == []


class TestAskQuestion:
    def test_no_results_message(self, client, headers, monkeypatch):
        monkeypatch.setattr(main, "search_weaviate", lambda *a, **k: [])
        r = client.post("/ask_question", data={"query": "anything"}, headers=headers)
        assert r.status_code == 200
        assert "couldn't find anything relevant" in r.json()["answer"]

    def test_errors_do_not_leak_internals(self, client, headers, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("secret internal detail: password123")

        monkeypatch.setattr(main, "search_weaviate", boom)
        r = client.post("/ask_question", data={"query": "anything"}, headers=headers)
        assert r.status_code == 500
        body = r.text
        assert "trace" not in body.lower()
        assert "password123" not in body
        assert "RuntimeError" not in body

    def test_503_when_weaviate_down(self, monkeypatch, headers):
        def fail_connect(*a, **k):
            raise ConnectionError("no weaviate")

        monkeypatch.setattr(main, "connect", fail_connect)
        monkeypatch.setattr(main, "ensure_schema", lambda c: None)
        with TestClient(main.app) as tc:
            r = tc.post("/ask_question", data={"query": "q"}, headers=headers)
        assert r.status_code == 503


class TestRateLimits:
    def test_upload_rate_limited_per_ip(self, client, monkeypatch):
        limit = int(main.UPLOAD_RATE_LIMIT.split("/")[0])
        my_ip = ip(9999)
        other_ip = ip(9998)
        for _ in range(limit):
            r = client.post(
                "/upload_pdf",
                files={"file": ("x.txt", b"x", "text/plain")},
                headers=my_ip,
            )
            assert r.status_code == 400  # invalid upload, but counts toward the limit
        r = client.post(
            "/upload_pdf",
            files={"file": ("x.txt", b"x", "text/plain")},
            headers=my_ip,
        )
        assert r.status_code == 429
        # a different visitor is unaffected
        r = client.post(
            "/upload_pdf",
            files={"file": ("x.txt", b"x", "text/plain")},
            headers=other_ip,
        )
        assert r.status_code == 400

    def test_ask_rate_limited_per_ip(self, client, monkeypatch):
        monkeypatch.setattr(main, "search_weaviate", lambda *a, **k: [])
        limit = int(main.ASK_RATE_LIMIT.split("/")[0])
        my_ip = ip(9997)
        for _ in range(limit):
            r = client.post("/ask_question", data={"query": "q"}, headers=my_ip)
            assert r.status_code == 200
        r = client.post("/ask_question", data={"query": "q"}, headers=my_ip)
        assert r.status_code == 429
