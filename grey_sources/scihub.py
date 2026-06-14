"""Sci-Hub integration utilities (std lib only).

Provides a list of known Sci-Hub mirrors and utilities to resolve a DOI or title to a
PDF URL, then download the PDF.
"""

import urllib.request
import urllib.parse
import re
import time
from pathlib import Path
from typing import Optional, List

# Known Sci-Hub mirrors (as of writing). Users may add/remove as needed.
SCIHUB_MIRRORS: List[str] = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.ren",
    "https://sci-hub.shop",
    "https://sci-hub.ee",
]


def _extract_pdf_url(html: str) -> Optional[str]:
    """Attempt to extract a direct PDF URL from a Sci‑Hub HTML page.

    Several patterns are tried:
    1. <iframe src="...pdf"> or <embed src="...pdf">
    2. JavaScript location.href assignment to a .pdf URL
    3. Any quoted .pdf URL in the page
    """
    patterns = [
        r"<iframe[^>]+src=[\"']([^\"'>]+\.pdf)[\"']",
        r"<embed[^>]+src=[\"']([^\"'>]+\.pdf)[\"']",
        r"location\.href\s*=\s*[\"']([^\"']+\.pdf)[\"']",
        r"[\"'](https?://[^\"'>]+\.pdf)[\"']",
    ]
    for pat in patterns:
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _try_mirror(mirror: str, doi: Optional[str] = None, title: Optional[str] = None) -> Optional[str]:
    """Resolve a single Sci‑Hub mirror to a possible PDF URL.

    Returns the direct PDF URL if found, otherwise ``None``.
    """
    if doi:
        target = f"{mirror}/{urllib.parse.quote(doi)}"
    elif title:
        target = f"{mirror}/{urllib.parse.quote(title)}"
    else:
        return None
    try:
        with urllib.request.urlopen(target, timeout=15) as response:
            html_bytes = response.read()
            html = html_bytes.decode(errors="ignore")
            pdf_url = _extract_pdf_url(html)
            return pdf_url
    except Exception:
        return None


def scihub_download(
    doi: Optional[str] = None,
    title: Optional[str] = None,
    paper_id: Optional[str] = None,
    pdf_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Download a PDF from Sci‑Hub.

    Parameters
    ----------
    doi: DOI string (e.g. ``10.1038/s41586-020-03051-8``). If supplied, it is tried first.
    title: Paper title – used when DOI is not available.
    paper_id: Optional identifier used only for naming the saved file when neither DOI nor title
        is supplied.
    pdf_dir: Directory where the PDF should be stored. If omitted, a ``pdfs`` directory under the
        current working directory is used.

    Returns
    -------
    pathlib.Path of the saved PDF, or ``None`` on failure.
    """
    if pdf_dir is None:
        pdf_dir = Path.cwd() / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Create a safe base name for the file.
    base_id = doi or title or paper_id or "unknown"
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", base_id)
    dest_path = pdf_dir / f"{safe_id}_scihub.pdf"
    if dest_path.exists():
        return dest_path

    # Try DOI first, then title.
    for mirror in SCIHUB_MIRRORS:
        pdf_url = None
        if doi:
            pdf_url = _try_mirror(mirror, doi=doi)
        if not pdf_url and title:
            pdf_url = _try_mirror(mirror, title=title)
        if pdf_url:
            try:
                with urllib.request.urlopen(pdf_url, timeout=30) as resp:
                    data = resp.read()
                    dest_path.write_bytes(data)
                    return dest_path
            except Exception:
                # Failed to download – continue trying other mirrors.
                pass
        # Be polite to the server.
        time.sleep(1)
    return None
