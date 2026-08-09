"""Tests for scraper.py sort modes — run with: pytest tests/ -v

Asserts each source sends (or omits) the right sort params in 'relevance'
(default) and 'recent' modes. No network calls: the session's get method is
mocked and every mocked response is empty, so each search returns after its
first request.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import scraper

EMPTY_ATOM_FEED = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
EMPTY_ESEARCH_JSON = {"esearchresult": {"idlist": []}}
EMPTY_CROSSREF_JSON = {"message": {"items": []}}
EMPTY_OPENALEX_JSON = {"results": []}


def make_scraper(sort_mode=None):
    if sort_mode is None:
        return scraper.MultiSourceScraper(delay=0)
    return scraper.MultiSourceScraper(delay=0, sort_mode=sort_mode)


def make_response(text="", json_data=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def first_call_params(scraper_obj, method_name, response):
    """Run a search method with a mocked session and return the first request's params."""
    with patch.object(scraper_obj.session, "get", return_value=response) as mock_get:
        with patch.object(scraper.time, "sleep"):
            getattr(scraper_obj, method_name)("behaviour change", max_results=5)
    return mock_get.call_args_list[0].kwargs["params"]


# ── Default mode ─────────────────────────────────────────────────────────────

class TestSortModeDefault:
    def test_init_defaults_to_relevance(self):
        # Arrange / Act
        s = make_scraper()

        # Assert
        assert s.sort_mode == "relevance"

    def test_init_accepts_recent(self):
        # Arrange / Act
        s = make_scraper("recent")

        # Assert
        assert s.sort_mode == "recent"


# ── arXiv ────────────────────────────────────────────────────────────────────

class TestArxivSort:
    def test_relevance_mode_sorts_by_relevance(self):
        # Arrange
        s = make_scraper("relevance")

        # Act
        params = first_call_params(s, "search_arxiv", make_response(text=EMPTY_ATOM_FEED))

        # Assert
        assert params["sortBy"] == "relevance"

    def test_recent_mode_sorts_by_submitted_date_descending(self):
        # Arrange
        s = make_scraper("recent")

        # Act
        params = first_call_params(s, "search_arxiv", make_response(text=EMPTY_ATOM_FEED))

        # Assert
        assert params["sortBy"] == "submittedDate"
        assert params["sortOrder"] == "descending"


# ── PubMed ───────────────────────────────────────────────────────────────────

class TestPubmedSort:
    def test_relevance_mode_adds_sort_param_to_esearch(self):
        # Arrange
        s = make_scraper("relevance")

        # Act
        params = first_call_params(s, "search_pubmed", make_response(json_data=EMPTY_ESEARCH_JSON))

        # Assert
        assert params["sort"] == "relevance"

    def test_recent_mode_omits_sort_param(self):
        # Arrange
        s = make_scraper("recent")

        # Act
        params = first_call_params(s, "search_pubmed", make_response(json_data=EMPTY_ESEARCH_JSON))

        # Assert
        assert "sort" not in params


# ── PubMed Central ───────────────────────────────────────────────────────────

class TestPubmedCentralSort:
    def test_relevance_mode_sorts_esearch_by_relevance(self):
        # Arrange
        s = make_scraper("relevance")

        # Act
        params = first_call_params(s, "search_pubmedcentral", make_response(json_data=EMPTY_ESEARCH_JSON))

        # Assert
        assert params["sort"] == "relevance"

    def test_recent_mode_keeps_date_sort(self):
        # Arrange
        s = make_scraper("recent")

        # Act
        params = first_call_params(s, "search_pubmedcentral", make_response(json_data=EMPTY_ESEARCH_JSON))

        # Assert
        assert params["sort"] == "date"


# ── CrossRef ─────────────────────────────────────────────────────────────────

class TestCrossrefSort:
    def test_relevance_mode_omits_sort_and_order(self):
        # Arrange
        s = make_scraper("relevance")

        # Act
        params = first_call_params(s, "search_crossref", make_response(json_data=EMPTY_CROSSREF_JSON))

        # Assert
        assert "sort" not in params
        assert "order" not in params

    def test_recent_mode_sorts_by_published_descending(self):
        # Arrange
        s = make_scraper("recent")

        # Act
        params = first_call_params(s, "search_crossref", make_response(json_data=EMPTY_CROSSREF_JSON))

        # Assert
        assert params["sort"] == "published"
        assert params["order"] == "desc"


# ── OpenAlex ─────────────────────────────────────────────────────────────────

class TestOpenalexSort:
    def test_relevance_mode_omits_sort_param(self):
        # Arrange
        s = make_scraper("relevance")

        # Act
        params = first_call_params(s, "search_openalex", make_response(json_data=EMPTY_OPENALEX_JSON))

        # Assert
        assert "sort" not in params

    def test_recent_mode_sorts_by_publication_date_descending(self):
        # Arrange
        s = make_scraper("recent")

        # Act
        params = first_call_params(s, "search_openalex", make_response(json_data=EMPTY_OPENALEX_JSON))

        # Assert
        assert params["sort"] == "publication_date:desc"
