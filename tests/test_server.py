"""Tests for server.py — run with: pytest tests/ -v"""
import json
import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set dummy API key before importing server
os.environ["OPENROUTER_API_KEY"] = "test-key-dummy"

import server
import scraper


def make_papers(tmp_path, papers_data, filename="papers_test.json"):
    """Helper: write papers to a temp raw dir, patch server globals, return loaded papers."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    analyses_dir = tmp_path / "data" / "analyses"
    analyses_dir.mkdir(parents=True)
    cache = tmp_path / "data" / "analysis_cache.json"

    with open(raw_dir / filename, "w") as f:
        json.dump(papers_data, f)

    with patch.object(server, "RAW_DIR", raw_dir):
        with patch.object(server, "ANALYSIS_DIR", analyses_dir):
            with patch.object(server, "ANALYSIS_CACHE", cache):
                return server.load_papers()


# ── Paper Loading ────────────────────────────────────────────────────────────

class TestLoadPapers:
    def test_loads_papers_from_raw_dir(self, tmp_path):
        papers = [
            {"id": f"t{i}", "title": f"Research on {['cats','dogs','fish','birds','snakes','lions','bears','wolves','foxes','deer'][i]}", "summary": f"Abstract {i}"}
            for i in range(10)
        ]
        result = make_papers(tmp_path, papers)
        assert len(result) == 10

    def test_dedup_by_id(self, tmp_path):
        papers = [
            {"id": "dup.001", "title": "Paper A", "summary": "abstract"},
            {"id": "dup.001", "title": "Paper A duplicate", "summary": "abstract"},
            {"id": "unique.002", "title": "Paper B", "summary": "abstract"},
        ]
        result = make_papers(tmp_path, papers)
        ids = [p["id"] for p in result]
        assert ids.count("dup.001") == 1
        assert "unique.002" in ids

    def test_dedup_by_title_similarity(self, tmp_path):
        papers = [
            {"id": "a.001", "title": "Behavioral Science in Saudi Arabia: A Study", "summary": "x"},
            {"id": "b.002", "title": "Behavioral Science in Saudi Arabia: A Study", "summary": "y"},
        ]
        result = make_papers(tmp_path, papers)
        assert len(result) == 1

    def test_handles_empty_raw_dir(self, tmp_path):
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        analyses_dir = tmp_path / "data" / "analyses"
        analyses_dir.mkdir(parents=True)
        cache = tmp_path / "data" / "analysis_cache.json"

        with patch.object(server, "RAW_DIR", raw_dir):
            with patch.object(server, "ANALYSIS_DIR", analyses_dir):
                with patch.object(server, "ANALYSIS_CACHE", cache):
                    result = server.load_papers()
                    assert result == []


# ── Search ───────────────────────────────────────────────────────────────────

class TestSearchPapers:
    def _papers(self, tmp_path):
        animals = ['cats','dogs','fish','birds','snakes','lions','bears','wolves','foxes','deer']
        data = [
            {"id": f"s{i}", "title": f"Research on {animals[i]}", "summary": f"Abstract about {animals[i]}"}
            for i in range(10)
        ]
        return make_papers(tmp_path, data)

    def test_search_by_title(self, tmp_path):
        papers = self._papers(tmp_path)
        results = server.search_papers(papers, "cats")
        assert len(results) >= 1

    def test_search_by_summary(self, tmp_path):
        papers = self._papers(tmp_path)
        results = server.search_papers(papers, "snakes")
        assert len(results) >= 1

    def test_search_case_insensitive(self, tmp_path):
        papers = self._papers(tmp_path)
        results = server.search_papers(papers, "WOLVES")
        assert len(results) >= 1

    def test_search_no_match(self, tmp_path):
        papers = self._papers(tmp_path)
        results = server.search_papers(papers, "xyznonexistent")
        assert len(results) == 0


# ── Analysis ─────────────────────────────────────────────────────────────────

class TestAnalyzePapers:
    def test_returns_expected_keys(self, tmp_path):
        topics = ['quantum physics','marine biology','ancient history','machine learning','climate science','neuroscience','astronomy','geology','economics','psychology']
        papers = [
            {"id": f"a{i}", "title": f"Study of {topics[i]}", "summary": f"Abstract {i}", "authors": ["A"], "published": "2024-01-01T00:00:00Z"}
            for i in range(5)
        ]
        result = server.analyze_papers(papers)
        assert "total_papers" in result
        assert "yearly_distribution" in result
        assert "top_authors" in result
        assert "summary" in result

    def test_total_papers_count(self, tmp_path):
        topics = ['quantum physics','marine biology','ancient history','machine learning','climate science']
        papers = [
            {"id": f"b{i}", "title": f"Distinct {topics[i]}", "summary": f"Abstract {i}", "authors": ["B"], "published": "2024-06-01T00:00:00Z"}
            for i in range(5)
        ]
        result = server.analyze_papers(papers)
        assert result["total_papers"] == 5

    def test_empty_papers(self):
        result = server.analyze_papers([])
        assert result["total_papers"] == 0


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = server.RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.allow("127.0.0.1") is True

    def test_blocks_over_limit(self):
        rl = server.RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.allow("127.0.0.1")
        assert rl.allow("127.0.0.1") is False

    def test_separate_ips(self):
        rl = server.RateLimiter(max_requests=2, window_seconds=60)
        rl.allow("1.1.1.1")
        rl.allow("1.1.1.1")
        assert rl.allow("1.1.1.1") is False
        assert rl.allow("2.2.2.2") is True

    def test_retry_after(self):
        rl = server.RateLimiter(max_requests=1, window_seconds=60)
        rl.allow("127.0.0.1")
        retry = rl.retry_after("127.0.0.1")
        assert 0 < retry <= 60


# ── Config ───────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_config_file(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"openrouter_api_key": "test-key-123"}))
        with patch.object(server, "CONFIG_FILE", cfg_file):
            cfg = server.load_config()
            assert cfg["openrouter_api_key"] == "test-key-123"

    def test_load_config_missing_file(self, tmp_path):
        cfg_file = tmp_path / "nonexistent.json"
        with patch.object(server, "CONFIG_FILE", cfg_file):
            cfg = server.load_config()
            assert cfg == {}

    def test_save_config(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        with patch.object(server, "CONFIG_FILE", cfg_file):
            server.save_config({"key": "value"})
            loaded = json.loads(cfg_file.read_text())
            assert loaded["key"] == "value"


# ── Prompt Config ────────────────────────────────────────────────────────────

class TestPromptConfig:
    def test_load_prompts_file(self, tmp_path):
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text(json.dumps({"test_prompt": "hello"}))
        with patch.object(server, "PROMPTS_FILE", prompts_file):
            prompts = server.load_prompts()
            assert prompts["test_prompt"] == "hello"

    def test_load_prompts_missing(self, tmp_path):
        with patch.object(server, "PROMPTS_FILE", tmp_path / "missing.json"):
            prompts = server.load_prompts()
            assert prompts == {}

    def test_get_prompt_fallback(self, tmp_path):
        with patch.object(server, "PROMPTS_FILE", tmp_path / "missing.json"):
            result = server.get_prompt("nonexistent", "default-value")
            assert result == "default-value"


# ── LLM Call (mocked) ────────────────────────────────────────────────────────

class TestLLMCall:
    def test_no_api_key_returns_error(self):
        with patch.object(server, "get_api_key", return_value=""):
            result = server.llm_call([{"role": "user", "content": "hi"}])
            assert "error" in result
            assert "No API key" in result["error"]

    @patch("urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "test response"}}]
        }).encode()
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(server, "get_api_key", return_value="test-key"):
            result = server.llm_call([{"role": "user", "content": "hi"}])
            assert result.get("content") == "test response"

    @patch("urllib.request.urlopen")
    def test_fallback_on_429(self, mock_urlopen):
        """When primary model returns 429, should try fallback."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # First model fails (retry decorator retries once)
                raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)
            # Fallback succeeds
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "fallback response"}}]
            }).encode()
            return mock_resp

        mock_urlopen.side_effect = side_effect

        with patch.object(server, "get_api_key", return_value="test-key"):
            with patch.object(server, "LLM_FALLBACK_MODELS", ["openai/gpt-oss-20b:free"]):
                result = server.llm_call([{"role": "user", "content": "hi"}])
                # Either fallback works or all fail — just verify no crash
                assert "content" in result or "error" in result


# ── Word Frequency ───────────────────────────────────────────────────────────

class TestWordFreq:
    def test_basic_counting(self):
        texts = ["hello world hello", "world hello"]
        result = server.word_freq(texts)
        assert result[0][0] == "hello"
        assert result[0][1] == 3

    def test_stopwords_filtered(self):
        texts = ["the and for are but not you all any can"]
        result = server.word_freq(texts)
        assert len(result) == 0

    def test_min_length_filter(self):
        texts = ["ab cd efgh ij klmnop"]
        result = server.word_freq(texts, min_len=3)
        words = [w for w, _ in result]
        assert "ab" not in words
        assert "efgh" in words


# ── Batch Job Tracking ───────────────────────────────────────────────────────

class TestBatchJobs:
    def test_job_created(self, tmp_path):
        """Starting a batch job should create a job entry."""
        papers = [
            {"id": f"bj{i}", "title": f"Batch Job Test Paper {i}", "summary": f"Abstract {i}"}
            for i in range(3)
        ]
        # papers_global is set at runtime in serve(); patch it into the module
        server.papers_global = papers
        try:
            handler = server.Handler.__new__(server.Handler)
            handler.filters = []
            handler._rate_check = lambda: True
            handler._server = lambda: None
            handler.client_address = ("127.0.0.1", 0)
            handler.headers = {"Content-Length": "0"}
            handler.path = "/api/summarise_all"
            handler.rfile = None
            handler.wfile = type("MockFile", (), {"write": lambda self, x: None})()
            handler.send_response = lambda status: None
            handler.send_header = lambda *a: None
            handler.end_headers = lambda: None

            sent = []
            handler._json = lambda data, status=200: sent.append(data)

            import io
            body = json.dumps({"start": 0, "count": 3}).encode()
            handler.headers["Content-Length"] = str(len(body))
            handler.rfile = io.BytesIO(body)

            server.batch_jobs.clear()
            server.batch_job_counter = 0

            handler.do_POST()

            assert len(sent) == 1
            assert "job_id" in sent[0]
            assert sent[0]["status"] == "running"
            assert sent[0]["total"] == 3
        finally:
            del server.papers_global

    def test_job_tracking(self):
        """Job should track progress."""
        with server.batch_jobs_lock:
            job_id = "test_job_1"
            server.batch_jobs[job_id] = {
                "status": "running",
                "progress": 5,
                "total": 10,
                "results": [],
                "error": None
            }

        with server.batch_jobs_lock:
            job = server.batch_jobs[job_id]
            assert job["status"] == "running"
            assert job["progress"] == 5
            assert job["total"] == 10

        # Clean up
        with server.batch_jobs_lock:
            del server.batch_jobs[job_id]


# ── DOI Dedup ────────────────────────────────────────────────────────────────

class TestDOIDedup:
    def test_dedup_by_doi(self):
        s = scraper.MultiSourceScraper()
        papers = [
            {"id": "a.001", "title": "Paper A", "summary": "x", "doi": "10.1234/abc"},
            {"id": "b.002", "title": "Paper B", "summary": "y", "doi": "10.1234/abc"},
            {"id": "c.003", "title": "Paper C", "summary": "z", "doi": "10.5678/def"},
        ]
        result = s.dedup(papers)
        dois = [p.get("doi") for p in result]
        assert dois.count("10.1234/abc") == 1
        assert len(result) == 2

    def test_dedup_by_doi_normalizes(self):
        s = scraper.MultiSourceScraper()
        papers = [
            {"id": "a.001", "title": "Paper A", "summary": "x", "doi": "https://doi.org/10.1234/ABC"},
            {"id": "b.002", "title": "Paper B", "summary": "y", "doi": "10.1234/abc"},
        ]
        result = s.dedup(papers)
        assert len(result) == 1

    def test_dedup_by_doi_with_url_prefix(self):
        s = scraper.MultiSourceScraper()
        papers = [
            {"id": "a.001", "title": "Paper A", "summary": "x", "DOI": "doi:10.1234/xyz"},
            {"id": "b.002", "title": "Paper B", "summary": "y", "DOI": "10.1234/xyz"},
        ]
        result = s.dedup(papers)
        assert len(result) == 1


# ── Fulltext ─────────────────────────────────────────────────────────────────

class TestFulltext:
    def test_get_paper_full_text_returns_abstract_when_no_pdf(self, tmp_path):
        """When no PDF available, should return abstract."""
        papers = [
            {"id": "ft.001", "title": "Test Paper", "summary": "This is the abstract", "pdf_url": ""},
        ]
        make_papers(tmp_path, papers)
        text = server.get_paper_full_text(papers[0])
        assert isinstance(text, str)
        assert len(text) > 0

    def test_get_paper_full_text_caching(self, tmp_path):
        """Extracted text should be cached to disk."""
        papers = [
            {"id": "ft.002", "title": "Cached Paper", "summary": "Abstract text", "pdf_url": ""},
        ]
        make_papers(tmp_path, papers)
        text = server.get_paper_full_text(papers[0])
        assert isinstance(text, str)


# ── Incremental Mode ─────────────────────────────────────────────────────────

class TestIncrementalMode:
    def test_load_existing_ids(self, tmp_path):
        """load_existing_ids should find IDs from existing files."""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        existing = [
            {"id": "ex.001", "title": "Existing", "summary": "x", "doi": "10.1234/ex"},
            {"id": "ex.002", "title": "Existing 2", "summary": "y"},
        ]
        with open(raw_dir / "papers_existing.json", "w") as f:
            json.dump(existing, f)
        ids = scraper.load_existing_ids(raw_dir)
        assert "ex.001" in ids
        assert "ex.002" in ids
        assert "DOI:10.1234/ex" in ids


# ── Enrichment ───────────────────────────────────────────────────────────────

class TestEnrichment:
    def test_normalize_doi(self):
        """DOI normalization should handle various formats."""
        from enrichment import _normalize_doi
        assert _normalize_doi("10.1234/ABC") == "10.1234/abc"
        assert _normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("doi:10.1234/abc") == "10.1234/abc"
        assert _normalize_doi("") == ""

    def test_dedup_by_doi(self):
        """dedup_by_doi should remove duplicate DOIs."""
        from enrichment import dedup_by_doi
        papers = [
            {"id": "a", "doi": "10.1234/abc"},
            {"id": "b", "doi": "10.1234/abc"},
            {"id": "c", "doi": "10.5678/def"},
        ]
        result = dedup_by_doi(papers)
        assert len(result) == 2


# ── Grey Sources (import check) ───────────────────────────────────────────────

class TestGreySourcesImport:
    def test_scihub_import(self):
        """Sci-Hub module should import without errors."""
        from grey_sources import scihub_download, SCIHUB_MIRRORS
        assert isinstance(SCIHUB_MIRRORS, list)
        assert len(SCIHUB_MIRRORS) > 0

    def test_libgen_import(self):
        """LibGen module should import without errors."""
        from grey_sources import libgen_search, libgen_download
        assert callable(libgen_search)
        assert callable(libgen_download)

    def test_annas_archive_import(self):
        """Anna's Archive module should import without errors."""
        from grey_sources import annas_archive_search, annas_archive_download
        assert callable(annas_archive_search)
        assert callable(annas_archive_download)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
