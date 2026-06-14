'''enrichment.py
Utility functions for DOI deduplication and Unpaywall API integration.
Only Python standard library is used.
'''

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Set

# Global email for Unpaywall API (can be set via set_unpaywall_email)
_UNPAYWALL_EMAIL: str | None = None


def set_unpaywall_email(email: str) -> None:
    """Set the global email used for Unpaywall API requests.

    The Unpaywall API requires a valid contact email. This function stores it in a module‑level
    variable that other functions will use when ``email`` is not supplied directly.
    """
    global _UNPAYWALL_EMAIL
    _UNPAYWALL_EMAIL = email


def _build_unpaywall_url(doi: str, email: str | None) -> str:
    """Construct the Unpaywall API URL for a given DOI.

    Parameters
    ----------
    doi: str
        DOI string (can include ``https://doi.org/`` or ``doi:`` prefixes).
    email: str | None
        Email address to embed in the query string. If ``None`` the global email is used.
    """
    if email is None:
        email = _UNPAYWALL_EMAIL
    if not email:
        raise ValueError("An email address must be provided for Unpaywall API requests.")
    # Normalise DOI for URL insertion – remove any surrounding ``doi:``, ``http`` etc.
    norm = _normalize_doi(doi)
    return f"https://api.unpaywall.org/v2/{norm}?email={urllib.parse.quote(email)}"


def enrich_unpaywall(doi: str, email: str | None = None) -> Dict[str, Any]:
    """Query the Unpaywall API for a DOI and return a simplified dict.

    The returned dictionary always contains the keys:
        - doi (original DOI string)
        - is_oa (bool)
        - oa_status (str | None)
        - title (str | None)
        - journal (str | None)
        - published (str | None)
        - pdf_url (str | None)
        - pdf_source (str | None, currently ``"unpaywall"`` when a PDF URL is present)

    Errors (network issues, HTTP 404, malformed JSON, missing email) are handled
    gracefully – the dict will contain ``is_oa`` set to ``False`` and the other
    fields set to ``None``.
    """
    result: Dict[str, Any] = {
        "doi": doi,
        "is_oa": False,
        "oa_status": None,
        "title": None,
        "journal": None,
        "published": None,
        "pdf_url": None,
        "pdf_source": None,
    }

    try:
        url = _build_unpaywall_url(doi, email)
    except ValueError as e:
        # Missing email – return early with the default result.
        return result

    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                # Non‑200 responses are treated as failures.
                return result
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as http_err:
        # 404 means the DOI is not known to Unpaywall – we keep defaults.
        if http_err.code == 404:
            return result
        return result
    except (urllib.error.URLError, json.JSONDecodeError):
        # Network problems or malformed JSON – return defaults.
        return result

    # Populate fields from the API response.
    result["is_oa"] = bool(data.get("is_oa"))
    result["oa_status"] = data.get("oa_status")
    result["title"] = data.get("title")
    result["journal"] = data.get("journal_name")
    result["published"] = data.get("published_date")

    # Best OA location may contain a PDF URL.
    best_location = data.get("best_oa_location")
    if isinstance(best_location, dict):
        pdf_url = best_location.get("url")
        if pdf_url:
            result["pdf_url"] = pdf_url
            result["pdf_source"] = "unpaywall"
    return result


def enrich_papers_unpaywall(papers: List[Dict[str, Any]], delay: float = 0.5) -> None:
    """Enrich a list of paper dictionaries in‑place using Unpaywall.

    For each paper that contains a ``doi`` or ``DOI`` key, the function queries the
    Unpaywall API (respecting the optional ``delay`` between requests) and stores the
    response under a new ``unpaywall`` key.

    If Unpaywall provides a ``pdf_url`` and the original paper does not already have a
    ``pdf_url`` field, the function adds ``pdf_url`` and ``pdf_source`` (set to
    ``"unpaywall"``) to the paper dictionary.
    """
    for paper in papers:
        doi = paper.get("doi") or paper.get("DOI")
        if not doi:
            continue
        info = enrich_unpaywall(doi)
        paper["unpaywall"] = info
        if info.get("pdf_url") and not paper.get("pdf_url"):
            paper["pdf_url"] = info["pdf_url"]
            paper["pdf_source"] = "unpaywall"
        if delay:
            time.sleep(delay)


def _normalize_doi(doi: str) -> str:
    """Return a lowercase DOI without URL or ``doi:`` prefixes.

    Example::
        >>> _normalize_doi('https://doi.org/10.1234/ABC')
        '10.1234/abc'
    """
    # Strip leading whitespace and make lowercase.
    doi = doi.strip().lower()
    # Remove common prefixes.
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi


def load_existing_ids(data_dir: str | Path) -> Set[str]:
    """Load existing paper identifiers from ``data/raw/papers_*.json`` files.

    The function returns a set containing:
        * ``id`` or ``entry_id`` values (as‑is)
        * Normalised DOIs prefixed with ``"DOI:"`` (e.g. ``"DOI:10.1234/abc"``)
    """
    base = Path(data_dir) / "raw"
    ids: Set[str] = set()
    for file_path in base.glob("papers_*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                papers = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(papers, list):
            continue
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            # Direct identifiers.
            for key in ("id", "entry_id"):
                val = paper.get(key)
                if isinstance(val, str):
                    ids.add(val)
            # DOI handling.
            doi = paper.get("doi") or paper.get("DOI")
            if isinstance(doi, str):
                norm = _normalize_doi(doi)
                ids.add(f"DOI:{norm}")
    return ids


def dedup_by_doi(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate a list of paper dicts based on DOI.

    Normalisation steps:
        * Lower‑case
        * Strip ``https://doi.org/`` and ``doi:`` prefixes
    Papers without a DOI are kept in their original order.
    """
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for paper in papers:
        doi = paper.get("doi") or paper.get("DOI")
        if isinstance(doi, str):
            norm = _normalize_doi(doi)
            if norm in seen:
                continue
            seen.add(norm)
        deduped.append(paper)
    return deduped
