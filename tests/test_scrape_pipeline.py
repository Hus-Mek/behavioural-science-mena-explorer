"""Tests for the scrape pipeline additions — run with: pytest tests/ -v

Covers:
- expand_queries(): Arabic -> English query expansion at scrape time
- embed_new_papers(): delta embedding of newly scraped papers
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set dummy API key before importing server (matches test_server.py; setdefault
# so we never clobber the value test_server.py already exported)
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-dummy")

import server
from app import embeddings as app_embeddings


ARABIC_QUERY = "الاقتصاد السلوكي في السعودية"
ENGLISH_TRANSLATION = "behavioral economics Saudi Arabia"


@pytest.fixture
def clean_status():
    """Snapshot scraper_status, hand out a blank output, restore afterwards."""
    prev = dict(server.scraper_status)
    server.scraper_status["output"] = ""
    yield server.scraper_status
    server.scraper_status.update(prev)


# ── Feature A: Arabic -> English query expansion ─────────────────────────────

class TestExpandQueries:
    def test_arabic_query_gets_english_appended(self, clean_status):
        with patch.object(server, "llm_call",
                          return_value={"content": ENGLISH_TRANSLATION}) as mock_llm:
            result = server.expand_queries([ARABIC_QUERY])
        assert result == [ARABIC_QUERY, ENGLISH_TRANSLATION]
        mock_llm.assert_called_once()
        # The Arabic query is inside the prompt the LLM was asked to translate
        assert ARABIC_QUERY in mock_llm.call_args[0][0][0]["content"]
        assert f"Arabic query expanded: {ARABIC_QUERY} -> {ENGLISH_TRANSLATION}" \
            in clean_status["output"]

    def test_non_arabic_queries_do_not_trigger_llm(self, clean_status):
        queries = ["nudge theory", "habit formation"]
        with patch.object(server, "llm_call", MagicMock()) as mock_llm:
            result = server.expand_queries(queries)
        assert result == queries
        mock_llm.assert_not_called()

    def test_llm_error_still_runs_original(self, clean_status):
        with patch.object(server, "llm_call",
                          return_value={"error": "No API key"}):
            result = server.expand_queries([ARABIC_QUERY])
        assert result == [ARABIC_QUERY]
        assert "expansion skipped" in clean_status["output"]

    def test_llm_exception_still_runs_original(self, clean_status):
        with patch.object(server, "llm_call", side_effect=TimeoutError("boom")):
            result = server.expand_queries([ARABIC_QUERY])
        assert result == [ARABIC_QUERY]
        assert "expansion skipped" in clean_status["output"]

    def test_empty_llm_response_skipped(self, clean_status):
        with patch.object(server, "llm_call", return_value={"content": ""}):
            result = server.expand_queries([ARABIC_QUERY])
        assert result == [ARABIC_QUERY]
        assert "expansion skipped" in clean_status["output"]

    def test_reply_still_in_arabic_is_rejected(self, clean_status):
        # A "translation" containing Arabic script means the model failed;
        # queueing it would just re-run an Arabic query against English indexes.
        with patch.object(server, "llm_call",
                          return_value={"content": ARABIC_QUERY}):
            result = server.expand_queries([ARABIC_QUERY])
        assert result == [ARABIC_QUERY]

    def test_duplicate_translation_not_added(self, clean_status):
        queries = [ARABIC_QUERY, "Behavioral Economics Saudi Arabia"]
        with patch.object(server, "llm_call",
                          return_value={"content": ENGLISH_TRANSLATION}) as mock_llm:
            result = server.expand_queries(queries)
        # Case-insensitive match against an already-queued query: nothing added
        assert result == queries
        mock_llm.assert_called_once()

    def test_input_list_is_not_mutated(self, clean_status):
        queries = [ARABIC_QUERY]
        with patch.object(server, "llm_call",
                          return_value={"content": ENGLISH_TRANSLATION}):
            result = server.expand_queries(queries)
        assert queries == [ARABIC_QUERY]
        assert result is not queries

    def test_quotes_stripped_from_translation(self, clean_status):
        with patch.object(server, "llm_call",
                          return_value={"content": f'"{ENGLISH_TRANSLATION}"'}):
            result = server.expand_queries([ARABIC_QUERY])
        assert result == [ARABIC_QUERY, ENGLISH_TRANSLATION]

    def test_mixed_queue_expands_only_arabic(self, clean_status):
        queries = ["nudge theory", ARABIC_QUERY]
        with patch.object(server, "llm_call",
                          return_value={"content": ENGLISH_TRANSLATION}) as mock_llm:
            result = server.expand_queries(queries)
        assert result == ["nudge theory", ARABIC_QUERY, ENGLISH_TRANSLATION]
        assert mock_llm.call_count == 1


# ── Feature B: delta embedding after a scrape ────────────────────────────────

PAPERS = [
    {"id": "p1", "title": "Paper One", "summary": "abstract one"},
    {"id": "p2", "title": "Paper Two", "summary": "abstract two"},
    {"id": "p3", "title": "Paper Three", "summary": "abstract three"},
]


@pytest.fixture
def embedding_env(tmp_path):
    """Redirect embedding caches to tmp and restore the in-memory matrix after."""
    prev_matrix, prev_ids = server.embedding_snapshot()
    emb_cache = tmp_path / "embeddings.json"
    meta_cache = tmp_path / "embeddings_meta.json"
    with patch.object(app_embeddings, "EMBEDDING_CACHE", emb_cache), \
         patch.object(app_embeddings, "EMBEDDING_META_CACHE", meta_cache), \
         patch.object(app_embeddings, "PDF_DIR", tmp_path / "pdfs"):
        yield {"cache": emb_cache, "meta": meta_cache}
    server._set_embeddings(prev_matrix, prev_ids)


class TestEmbedNewPapers:
    def test_embeds_only_the_missing_paper(self, embedding_env):
        server._set_embeddings(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                               ["p1", "p2"])
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(app_embeddings, "get_gemini_api_key", return_value="key"), \
             patch.object(app_embeddings, "_get_embedding_batch_gemini",
                          return_value=[[0.0, 0.0, 1.0]]) as mock_batch:
            count = server.embed_new_papers(PAPERS)

        assert count == 1
        # Only p3's text was sent to the provider
        mock_batch.assert_called_once()
        sent_texts = mock_batch.call_args[0][0]
        assert len(sent_texts) == 1
        assert "Paper Three" in sent_texts[0]
        # ids and matrix grew together, rows still aligned with ids
        matrix, ids = server.embedding_snapshot()
        assert ids == ["p1", "p2", "p3"]
        assert matrix.shape == (3, 3)
        assert matrix[2].tolist() == [0.0, 0.0, 1.0]
        # persisted the same shape build_embeddings writes
        cached = json.loads(embedding_env["cache"].read_text())
        assert cached["ids"] == ["p1", "p2", "p3"]
        assert len(cached["embeddings"]) == 3
        meta = json.loads(embedding_env["meta"].read_text())
        assert meta["count"] == 3
        assert meta["provider"] == "gemini"

    def test_no_api_key_returns_zero_and_changes_nothing(self, embedding_env):
        server._set_embeddings(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                               ["p1", "p2"])
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(app_embeddings, "get_gemini_api_key", return_value=""), \
             patch.object(app_embeddings, "_get_embedding_batch_gemini",
                          MagicMock()) as mock_batch:
            count = server.embed_new_papers(PAPERS)

        assert count == 0
        mock_batch.assert_not_called()
        matrix, ids = server.embedding_snapshot()
        assert ids == ["p1", "p2"]
        assert matrix.shape == (2, 3)
        assert not embedding_env["cache"].exists()
        assert not embedding_env["meta"].exists()

    def test_nothing_new_returns_zero_without_calling_provider(self, embedding_env):
        server._set_embeddings(np.eye(3), ["p1", "p2", "p3"])
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(app_embeddings, "get_gemini_api_key", return_value="key"), \
             patch.object(app_embeddings, "_get_embedding_batch_gemini",
                          MagicMock()) as mock_batch:
            count = server.embed_new_papers(PAPERS)
        assert count == 0
        mock_batch.assert_not_called()
        assert not embedding_env["cache"].exists()

    def test_bootstraps_when_no_matrix_exists(self, embedding_env):
        server._set_embeddings(None, [])
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(app_embeddings, "get_gemini_api_key", return_value="key"), \
             patch.object(app_embeddings, "_get_embedding_batch_gemini",
                          return_value=vectors):
            count = server.embed_new_papers(PAPERS)
        assert count == 3
        matrix, ids = server.embedding_snapshot()
        assert ids == ["p1", "p2", "p3"]
        assert matrix.shape == (3, 2)

    def test_batch_failure_falls_back_to_single_requests(self, embedding_env):
        server._set_embeddings(np.array([[1.0, 0.0], [0.0, 1.0]]), ["p1", "p2"])
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(app_embeddings, "get_gemini_api_key", return_value="key"), \
             patch.object(app_embeddings, "_get_embedding_batch_gemini",
                          return_value=None), \
             patch.object(app_embeddings, "_get_embedding_gemini",
                          return_value=[0.5, 0.5]) as mock_single:
            count = server.embed_new_papers(PAPERS)
        assert count == 1
        mock_single.assert_called_once()
        matrix, ids = server.embedding_snapshot()
        assert ids == ["p1", "p2", "p3"]
        assert matrix.shape == (3, 2)

    def test_dimension_mismatch_refuses_to_append(self, embedding_env):
        # A provider/model switch changes vector size; appending would corrupt
        # every cosine score, so the delta path must refuse.
        server._set_embeddings(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                               ["p1", "p2"])
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(app_embeddings, "get_gemini_api_key", return_value="key"), \
             patch.object(app_embeddings, "_get_embedding_batch_gemini",
                          return_value=[[1.0, 2.0]]):
            count = server.embed_new_papers(PAPERS)
        assert count == 0
        matrix, ids = server.embedding_snapshot()
        assert ids == ["p1", "p2"]
        assert matrix.shape == (2, 3)
        assert not embedding_env["cache"].exists()

    def test_openai_provider_uses_single_requests(self, embedding_env):
        server._set_embeddings(np.array([[1.0, 0.0], [0.0, 1.0]]), ["p1", "p2"])
        with patch.object(app_embeddings, "EMBEDDING_PROVIDER", "openai"), \
             patch.object(app_embeddings, "get_api_key", return_value="key"), \
             patch.object(app_embeddings, "_get_embedding_openai",
                          return_value=[0.7, 0.7]) as mock_openai:
            count = server.embed_new_papers(PAPERS)
        assert count == 1
        mock_openai.assert_called_once()
        _, ids = server.embedding_snapshot()
        assert ids == ["p1", "p2", "p3"]
        meta = json.loads(embedding_env["meta"].read_text())
        assert meta["provider"] == "openai"
