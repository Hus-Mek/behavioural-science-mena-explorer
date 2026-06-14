"""LibGen integration utilities (std lib only).

Provides functions to search the Library Genesis (LibGen) catalog and download PDFs by MD5 hash.
"""

import urllib.request
import urllib.parse
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

# Known LibGen mirrors.
LIBGEN_MIRRORS: List[str] = [
    "https://libgen.is",
    "https://libgen.rs",
    "https://libgen.st",
]


def _parse_search_results(html: str, base_url: str) -> List[Dict]:
    """Parse a LibGen search results page.

    Returns a list of dictionaries with keys: title, authors, md5, download_url, source.
    The implementation uses simple regexes that work for the typical LibGen table layout.
    """
    results: List[Dict] = []
    # LibGen tables have rows like:
    # <td><a href="/md5/<md5>">Download</a></td> ... <td>Title</td> ... <td>Authors</td>
    # We'll try to capture md5, title, authors.
    # First capture rows.
    row_pattern = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    md5_pattern = re.compile(r"/md5/([a-fA-F0-9]{32})", re.IGNORECASE)
    title_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
    for row_match in row_pattern.finditer(html):
        row_html = row_match.group(1)
        md5_match = md5_pattern.search(row_html)
        if not md5_match:
            continue
        md5 = md5_match.group(1)
        # Extract all <td> contents.
        cells = title_pattern.findall(row_html)
        # LibGen tables generally have many columns; title is often the 2nd or 3rd.
        # We'll heuristically look for a cell that contains a link ending with .pdf or a reasonable title.
        title = ""
        authors = ""
        if len(cells) >= 2:
            # Remove any HTML tags from the cell.
            clean = re.sub(r"<.*?>", "", cells[1])
            title = clean.strip()
        if len(cells) >= 3:
            authors = re.sub(r"<.*?>", "", cells[2]).strip()
        download_url = f"{base_url}/download.php?md5={md5}"
        results.append({
            "title": title,
            "authors": authors,
            "md5": md5,
            "download_url": download_url,
            "source": base_url,
        })
    return results


def libgen_search(query: str, max_results: int = 10) -> List[Dict]:
    """Search Library Genesis for a query string.

    Parameters
    ----------
    query: Search terms.
    max_results: Maximum number of result dictionaries to return.
    """
    all_results: List[Dict] = []
    for mirror in LIBGEN_MIRRORS:
        search_url = f"{mirror}/search.php?req={urllib.parse.quote(query)}&open=0&res=25&view=simple&phrase=1&column=def"
        try:
            with urllib.request.urlopen(search_url, timeout=15) as resp:
                html = resp.read().decode(errors="ignore")
                results = _parse_search_results(html, mirror)
                all_results.extend(results)
                if len(all_results) >= max_results:
                    return all_results[:max_results]
        except Exception:
            # Silently ignore failed mirror and continue.
            pass
        time.sleep(1)
    return all_results[:max_results]


def libgen_download(
    md5: str,
    title: str = "",
    paper_id: Optional[str] = None,
    pdf_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Download a PDF from LibGen given its MD5 hash.

    Returns the path to the saved PDF or ``None`` on failure.
    """
    if pdf_dir is None:
        pdf_dir = Path.cwd() / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    base_id = title or paper_id or md5
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", base_id)
    dest_path = pdf_dir / f"{safe_id}_libgen.pdf"
    if dest_path.exists():
        return dest_path

    for mirror in LIBGEN_MIRRORS:
        download_url = f"{mirror}/download.php?md5={md5}"
        try:
            with urllib.request.urlopen(download_url, timeout=30) as resp:
                data = resp.read()
                dest_path.write_bytes(data)
                return dest_path
        except Exception:
            # Try next mirror.
            pass
        time.sleep(1)
    return None
