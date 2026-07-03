import hashlib
from unittest.mock import MagicMock

import app.weaviate_utils as wu


def make_col_with_hashes(stored_hashes):
    """Mock collection whose fetch_objects returns objects for hashes that 'exist'."""
    col = MagicMock()

    def fetch_objects(filters=None, limit=None, return_properties=None):
        res = MagicMock()
        # contains_any filter value is the batch of hashes queried
        queried = filters.value if hasattr(filters, "value") else stored_hashes
        res.objects = [
            MagicMock(properties={"content_hash": h})
            for h in stored_hashes
            if h in queried
        ]
        return res

    col.query.fetch_objects.side_effect = fetch_objects
    return col


class TestChunkHash:
    def test_is_sha256_of_utf8(self):
        text = "holiday policy"
        assert wu.chunk_hash(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_distinct_inputs_distinct_hashes(self):
        assert wu.chunk_hash("a") != wu.chunk_hash("b")


class TestFetchExistingHashes:
    def test_returns_only_stored_hashes(self):
        stored = {wu.chunk_hash("a"), wu.chunk_hash("b")}
        col = make_col_with_hashes(stored)
        queried = [wu.chunk_hash(x) for x in ("a", "b", "c")]
        assert wu.fetch_existing_hashes(col, queried) == stored

    def test_batches_queries(self):
        col = make_col_with_hashes(set())
        hashes = [f"hash-{i}" for i in range(250)]
        wu.fetch_existing_hashes(col, hashes, batch_size=100)
        assert col.query.fetch_objects.call_count == 3  # 100 + 100 + 50

    def test_empty_input_makes_no_queries(self):
        col = make_col_with_hashes(set())
        assert wu.fetch_existing_hashes(col, []) == set()
        col.query.fetch_objects.assert_not_called()


class TestInsertChunks:
    def _client(self, col):
        client = MagicMock()
        client.collections.get.return_value = col
        return client

    def _fake_embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def test_dedupes_within_upload(self, monkeypatch):
        monkeypatch.setattr(wu, "embed_texts", self._fake_embed)
        col = make_col_with_hashes(set())
        col.data.insert_many.return_value = MagicMock(errors=None)

        result = wu.insert_chunks(self._client(col), ["same", "same", "other"], "doc.pdf")

        assert result["unique_in_upload"] == 2
        assert result["inserted"] == 2
        assert result["skipped_existing"] == 0

    def test_skips_chunks_already_in_db(self, monkeypatch):
        monkeypatch.setattr(wu, "embed_texts", self._fake_embed)
        col = make_col_with_hashes({wu.chunk_hash("existing")})
        col.data.insert_many.return_value = MagicMock(errors=None)

        result = wu.insert_chunks(self._client(col), ["existing", "new"], "doc.pdf")

        assert result["inserted"] == 1
        assert result["skipped_existing"] == 1

    def test_all_duplicates_short_circuits_without_embedding(self, monkeypatch):
        called = []
        monkeypatch.setattr(wu, "embed_texts", lambda t: called.append(t))
        col = make_col_with_hashes({wu.chunk_hash("existing")})

        result = wu.insert_chunks(self._client(col), ["existing"], "doc.pdf")

        assert result["inserted"] == 0
        assert called == []
        col.data.insert_many.assert_not_called()

    def test_empty_chunks_raises(self):
        import pytest

        with pytest.raises(ValueError):
            wu.insert_chunks(MagicMock(), [], "doc.pdf")
