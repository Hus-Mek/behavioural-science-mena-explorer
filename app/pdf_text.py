"""PDF download, SSRF-safe URL vetting, and full-text extraction/caching."""
import ipaddress
import re
import socket
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None
try:
    from enrichment import enrich_unpaywall
except Exception:
    enrich_unpaywall = None
try:
    from grey_sources.scihub import scihub_download
except Exception:
    scihub_download = None

ROOT = Path(__file__).parent.parent.resolve()

PDF_DIR = ROOT / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

MAX_PDF_PAGES = 80
MAX_PDF_TEXT_CHARS = 400_000


def extract_pdf_text(pdf_path):
    # Bounded: there was no page or character cap, and MAX_PDF_BYTES permits a 60MB
    # PDF whose extracted text can run to hundreds of MB held in one handler thread.
    if DocumentConverter is not None:
        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
            if len(text.strip()) > 100:
                return text.strip()
        except Exception:
            pass
    try:
        import fitz
        doc = fitz.open(pdf_path)
        parts = []          # was `text += ...`, quadratic string building
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                break
            parts.append(page.get_text())
            if sum(map(len, parts)) > MAX_PDF_TEXT_CHARS:
                break
        doc.close()
        text = "".join(parts)[:MAX_PDF_TEXT_CHARS]
        if text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "[No PDF extraction tool available. Install pymupdf: pip install pymupdf]"

MAX_PDF_BYTES = 60 * 1024 * 1024


def is_fetchable_url(url):
    """Reject anything that isn't a plain http(s) URL to a public host.

    Every pdf_url in the corpus is copied verbatim from a third-party API response
    or scraped HTML, and download_pdf() fetches it unattended. urllib honours
    file:// and ftp://, so an unvalidated URL was a local-file read and an SSRF
    into loopback/LAN services -- including this server's own API. The direct
    branch of download_pdf_with_fallback() at least required a .pdf suffix; the
    Unpaywall branch did not, and "file:///c:/x.pdf" satisfies it anyway.

    Returns (ok, reason).
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception:
        return False, "unparseable URL"
    if parts.scheme not in ("http", "https"):
        return False, f"scheme {parts.scheme or '(none)'} not allowed"
    host = parts.hostname
    if not host:
        return False, "no host"
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return False, f"DNS failure: {e}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False, "unresolvable address"
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False, f"non-public address {ip}"
    return True, ""


def download_pdf(url, paper_id):
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
    pdf_path = PDF_DIR / f"{safe_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    ok, reason = is_fetchable_url(url)
    if not ok:
        print(f"  PDF download refused for {paper_id}: {reason}")
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            # Bounded read: the old resp.read() had no cap, so one oversized or
            # endless response could exhaust memory.
            data = resp.read(MAX_PDF_BYTES + 1)
            if len(data) > MAX_PDF_BYTES:
                print(f"  PDF download refused for {paper_id}: larger than {MAX_PDF_BYTES} bytes")
                return None
            if len(data) < 1000:
                return None
            if not data.startswith(b"%PDF-"):
                # Mirrors and paywalls serve HTML block pages with a .pdf URL; storing
                # one means the extractor and the LLM later treat markup as a paper.
                print(f"  PDF download refused for {paper_id}: not a PDF (no %PDF- header)")
                return None
            with open(pdf_path, "wb") as f:
                f.write(data)
            return pdf_path
    except Exception as e:
        print(f"  PDF download failed for {paper_id}: {e}")
        return None

def download_pdf_with_fallback(paper, paper_id, grey=False):
    pdf_url = paper.get("pdf_url") or paper.get("url", "")
    if pdf_url and pdf_url.lower().endswith('.pdf'):
        path = download_pdf(pdf_url, paper_id)
        if path:
            return path, "direct"

    if enrich_unpaywall:
        doi = paper.get("doi") or paper.get("DOI")
        if doi:
            info = enrich_unpaywall(doi)
            unpay_pdf = info.get("pdf_url")
            if unpay_pdf:
                path = download_pdf(unpay_pdf, paper_id)
                if path:
                    return path, "unpaywall"

    if grey and scihub_download:
        doi = paper.get("doi") or paper.get("DOI")
        title = paper.get("title")
        path = scihub_download(doi=doi, title=title, paper_id=paper_id, pdf_dir=PDF_DIR)
        if path:
            return path, "scihub"

    return None, None

def get_paper_full_text(paper, grey=False):
    paper_id = paper.get("id") or paper.get("entry_id") or ""
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
    if not safe_id.strip("_"):
        # An id-less paper produced cache paths of ".txt"/".pdf", so EVERY such
        # paper shared one slot and the first one cached became sticky -- returning
        # the wrong paper's text to the UI and into LLM prompts.
        return {"error": "Paper has no usable id; cannot fetch full text."}
    cache_path = PDF_DIR / f"{safe_id}.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass

    pdf_path, source = download_pdf_with_fallback(paper, safe_id, grey=grey)
    if pdf_path:
        text = extract_pdf_text(pdf_path)
        if text and not text.startswith("["):
            try:
                cache_path.write_text(text, encoding="utf-8")
            except Exception:
                pass
        return text
    return paper.get("summary") or ""
