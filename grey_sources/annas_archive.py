"""Anna's Archive integration utilities (std lib only).

Provides simple search and download functions for Anna's Archive.
"""

import urllib.request
import urllib.parse
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

ANNAS_ARCHIVE_URL = "https://annas-archive.org"


def _parse_search_results(html: str) -> List[Dict]:
    """Parse Anna's Archive search results.

    Looks for entries with /md5/<md5> links. Returns list of dicts with keys:
    title, md5, download_url, source.
    """
    results: List[Dict] = []
    # Each result row contains a link like /md5/<md5>. We'll capture that and the title.
    pattern = re.compile(r"<a[^>]+href=\"(/md5/([a-fA-F0-9]{32})[^\"]*)\"[^>]*>([^<]{1,200})</a>", re.IGNORECASE)
    for match in pattern.finditer(html):
        path = match.group(1)
        md5 = match.group(2)
        title_raw = match.group(3)
        title = re.sub(r"<.*?>", "", title_raw).strip()
        download_url = f"{ANNAS_ARCHIVE_URL}{path}"
        results.append({
            "title": title,
            "md5": md5,
            "download_url": download_url,
            "source": ANNAS_ARCHIVE_URL,
        })
    return results


def annas_archive_search(query: str, max_results: int = 10) -> List[Dict]:
    """Search Anna's Archive for a query.

    Returns a list of result dictionaries.
    """
    search_url = f"{ANNAS_ARCHIVE_URL}/search?search={urllib.parse.quote(query)}"
    try:
        with urllib.request.urlopen(search_url, timeout=15) as resp:
            html = resp.read().decode(errors="ignore")
            results = _parse_search_results(html)
            return results[:max_results]
    except Exception:
        return []


def annas_archive_download(
    md5: str,
    title: str = "",
    paper_id: Optional[str] = None,
    pdf_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Download a PDF from Anna's Archive by its MD5 hash.

    Returns the path to the saved PDF or ``None`` on failure.
    """
    if pdf_dir is None:
        pdf_dir = Path.cwd() / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    base_id = title or paper_id or md5
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", base_id)
    dest_path = pdf_dir / f"{safe_id}_annas.pdf"
    if dest_path.exists():
        return dest_path

    download_url = f"{ANNAS_ARCHIVE_URL}/md5/{md5}"
    try:
        with urllib.request.urlopen(download_url, timeout=30) as resp:
            data = resp.read()
            dest_path.write_bytes(data)
            return dest_path
    except Exception:
        return None
