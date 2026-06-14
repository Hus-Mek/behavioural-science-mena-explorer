import json
import re
import os
import sys
import subprocess
import threading
import urllib.request
import urllib.error
import urllib.parse
import csv
import io
import time
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler

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

class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}
        self._lock = threading.Lock()

    def allow(self, ip):
        now = time.time()
        with self._lock:
            window_start = now - self.window_seconds
            if ip not in self._requests:
                self._requests[ip] = []
            self._requests[ip] = [t for t in self._requests[ip] if t > window_start]
            if len(self._requests[ip]) >= self.max_requests:
                return False
            self._requests[ip].append(now)
            return True

    def retry_after(self, ip):
        now = time.time()
        with self._lock:
            timestamps = self._requests.get(ip, [])
            if len(timestamps) < self.max_requests:
                return 0
            oldest = timestamps[0]
            return max(1, int(oldest + self.window_seconds - now))

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

ROOT = Path(__file__).parent.resolve()
CONFIG_FILE = ROOT / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.load(open(CONFIG_FILE))
        except Exception:
            pass
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

def get_api_key():
    cfg = load_config()
    return cfg.get("openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", ""))

LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "poolside/laguna-m.1:free"
LLM_FALLBACK_MODELS = [
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-coder:free",
    "google/gemini-2.0-flash-001",
]

PROMPTS_FILE = ROOT / "prompts.json"

def load_prompts():
    if PROMPTS_FILE.exists():
        try:
            return json.load(open(PROMPTS_FILE))
        except Exception:
            pass
    return {}

def get_prompt(name, default):
    prompts = load_prompts()
    return prompts.get(name, default)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"

for d in [RAW_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PDF_DIR = ROOT / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf_text(pdf_path):
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
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
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

def download_pdf(url, paper_id):
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
    pdf_path = PDF_DIR / f"{safe_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 1000:
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

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CACHE = ROOT / "data" / "embeddings.json"
_paper_embeddings = None
_paper_embedding_ids = []

def _load_embeddings():
    global _paper_embeddings, _paper_embedding_ids
    import numpy as np
    if _paper_embeddings is not None:
        return
    if EMBEDDING_CACHE.exists():
        try:
            cached = json.load(open(EMBEDDING_CACHE))
            _paper_embeddings = np.array(cached["embeddings"])
            _paper_embedding_ids = cached["ids"]
            return
        except Exception:
            pass
    _paper_embeddings = None
    _paper_embedding_ids = []

def _get_embedding(text):
    api_key = get_api_key()
    if not api_key:
        return None
    payload = json.dumps({"model": EMBEDDING_MODEL, "input": text[:8000]}).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/embeddings", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"  Embedding error: {e}")
        return None

def build_embeddings(papers):
    api_key = get_api_key()
    if not api_key:
        print("  Skipping embeddings: no API key configured.")
        return
    global _paper_embeddings, _paper_embedding_ids
    import numpy as np
    _load_embeddings()
    paper_ids = [p.get("id") or p.get("entry_id") for p in papers]
    if _paper_embedding_ids == paper_ids and _paper_embeddings is not None:
        return
    print("  Building embeddings for semantic search...")
    embeddings, ids = [], []
    fail_fast = False
    for i, p in enumerate(papers):
        if fail_fast:
            break
        safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(p.get("id") or p.get("entry_id") or ""))
        cache_path = PDF_DIR / f"{safe_id}.txt"
        if cache_path.exists():
            try:
                full_text = cache_path.read_text(encoding="utf-8")[:8000]
            except Exception:
                full_text = ""
            text = full_text
        else:
            text = (p.get("title") or "") + " " + (p.get("summary") or "")
        emb = _get_embedding(text)
        if emb:
            embeddings.append(emb)
            ids.append(p.get("id") or p.get("entry_id"))
        elif i == 0:
            fail_fast = True
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(papers)} embedded")
    if embeddings:
        _paper_embeddings = np.array(embeddings)
        _paper_embedding_ids = ids
        try:
            json.dump({"embeddings": _paper_embeddings.tolist(), "ids": ids}, open(EMBEDDING_CACHE, "w"))
        except Exception:
            pass
        print(f"  Embeddings built: {len(ids)} papers")

def semantic_search(query, papers, top_k=15):
    import numpy as np
    _load_embeddings()
    if _paper_embeddings is None or len(_paper_embeddings) == 0:
        return []
    query_emb = _get_embedding(query)
    if query_emb is None:
        return []
    query_vec = np.array(query_emb)
    norms = np.linalg.norm(_paper_embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    similarities = np.dot(_paper_embeddings, query_vec) / (norms * query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        paper_id = _paper_embedding_ids[idx]
        paper = next((p for p in papers if (p.get("id") or p.get("entry_id")) == paper_id), None)
        if paper:
            results.append({"paper": paper, "score": float(similarities[idx])})
    return results

ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
ARABIC_KEYWORDS = [
    "arabic", "arab", "عرب", "السعودية", "سعودي", "الإمارات", "إماراتي",
    "قطر", "قطري", "الكويت", "كويتي", "البحرين", "بحريني", "عمان", "عماني",
    "مصر", "مصري", "الأردن", "أردني", "لبنان", "لبناني", "العراق", "عراقي",
    "إيران", "إيراني", "فلسطين", "فلسطيني", "غزة", "القدس",
    "تركيا", "تركي", "اليمن", "يمني", "سوريا", "سوري",
    "المغرب", "مغربي", "تونس", "تونسي", "الجزائر", "جزائري", "ليبيا", "ليبي",
    "السودان", "سوداني", "موريتانيا", "موريتاني",
    "إسلام", "إسلامي", "مسلم", "قرآن", "حديث", "فقه", "شريعة",
    "عربية", "العربية",
]

def detect_arabic_content(text):
    if not text:
        return False, []
    has_script = bool(ARABIC_PATTERN.search(text))
    text_lower = text.lower()
    found_keywords = [kw for kw in ARABIC_KEYWORDS if kw in text_lower]
    return has_script or len(found_keywords) > 0, found_keywords

def score_arabic_relevance(paper):
    title = (paper.get("title") or "").lower()
    summary = (paper.get("summary") or "").lower()
    text = title + " " + summary
    score = 0
    details = []
    has_script, script_words = detect_arabic_content(text)
    if has_script:
        score += 5
        details.append("arabic_script")
    for kw in ARABIC_KEYWORDS:
        if kw in text:
            score += 1
            if kw not in details:
                details.append(kw)
    country_terms = {
        "saudi": 3, "arabia": 3, "uae": 3, "emirati": 3, "qatar": 3, "kuwait": 3,
        "bahrain": 3, "oman": 3, "egypt": 3, "jordan": 3, "lebanon": 3, "iraq": 3,
        "iran": 2, "israel": 2, "palestine": 3, "gaza": 3, "turkey": 2, "yemen": 3,
        "syria": 3, "morocco": 3, "tunisia": 3, "algeria": 3, "libya": 3, "sudan": 3,
        "mena": 4, "middle east": 4, "gulf": 3, "arabian": 3,
    }
    for term, boost in country_terms.items():
        if term in text:
            score += boost
            details.append(term)
    return score, list(set(details))

BEHAVIOURAL_TERMS = [
    "nudge","nudging","behavior","behaviour","cognitive","decision",
    "choice","heuristic","bias","motivation","incentive","reward",
    "punishment","feedback","social norm","conformity","compliance",
    "attitude","perception","learning","memory","emotion","affect",
    "risk","trust","cooperation","altruism","fairness","justice",
    "intervention","policy","regulation","adherence","frame",
    "framing","priming","anchoring","loss aversion","prospect",
    "utility","preference","gamification","habit",
    "automaticity","self-control","willpower","attention","salience",
    "default","opt-in","opt-out","commitment","consistency",
    "reciprocity","authority","scarcity","social proof","liking",
]

MIDDLE_EAST_TERMS = [
    "saudi","arabia","uae","emirati","dubai","abu dhabi","riyadh",
    "jeddah","mecca","medina","qatar","doha","kuwait","bahrain",
    "oman","muscat","egypt","cairo","jordan","amman","lebanon",
    "beirut","iraq","baghdad","iran","israel","palestine","gaza",
    "middle east","mena","gulf","arab","islamic","muslim",
    "conservative","liberal","tribe","tribal","honor","shame",
    "collectivism","individualism","religion","cultural","context",
    "hijab","veil","gender","women","youth","unemployment",
    "vision 2030","neom","diversification","oil","expat","foreign worker",
]

SCRAPER_QUERIES = {
    "broad_behavioural": {"source": "arXiv", "desc": "Behavioral Science (broad)"},
    "broad_me": {"source": "arXiv", "desc": "Middle East (broad)"},
    "saudi": {"source": "arXiv", "desc": "Saudi Arabia"},
    "arab_psychology": {"source": "arXiv", "desc": "Arab Psychology"},
    "mena_health": {"source": "arXiv", "desc": "MENA Health"},
    "behavioural_economics": {"source": "arXiv", "desc": "Behavioral Economics"},
    "digital_behaviour": {"source": "arXiv", "desc": "Digital Behavior"},
    "nudge_policy": {"source": "arXiv", "desc": "Nudge Policy"},
    "arabic_health": {"source": "PubMed", "desc": "Arabic Health (PubMed)"},
    "cultural_psychology": {"source": "PubMed", "desc": "Cultural Psychology (PubMed)"},
    "com_b": {"source": "arXiv", "desc": "COM-B / Behaviour Change Wheel"},
    "tpb": {"source": "arXiv", "desc": "Theory of Planned Behaviour"},
    "hbm": {"source": "arXiv", "desc": "Health Belief Model"},
    "sct": {"source": "arXiv", "desc": "Social Cognitive Theory"},
    "sdt": {"source": "arXiv", "desc": "Self-Determination Theory"},
    "dual_process": {"source": "arXiv", "desc": "Dual Process / Kahneman"},
    "social_norms": {"source": "arXiv", "desc": "Social Norms"},
    "habit_formation": {"source": "arXiv", "desc": "Habit Formation"},
    "health_psychology": {"source": "arXiv", "desc": "Health Psychology"},
    "cultural_psych": {"source": "arXiv", "desc": "Cultural Psychology"},
    "mena_nudge": {"source": "arXiv", "desc": "Nudge + MENA"},
    "mena_health_behaviour": {"source": "arXiv", "desc": "Health Behaviour + MENA"},
    "mena_mental_health": {"source": "arXiv", "desc": "Mental Health + MENA"},
    "mena_women": {"source": "arXiv", "desc": "Women/Gender + MENA"},
    "mena_youth": {"source": "arXiv", "desc": "Youth + MENA"},
    "arabic_transliterated": {"source": "arXiv", "desc": "Arabic transliterated terms"},
    "mena_education": {"source": "arXiv", "desc": "Education + MENA"},
    "mena_business": {"source": "arXiv", "desc": "Business/Management + MENA"},
    "mena_technology": {"source": "arXiv", "desc": "Technology/Digital + MENA"},
    "mena_migration": {"source": "arXiv", "desc": "Migration/Refugee + MENA"},
    "crossref_psychology": {"source": "CrossRef", "desc": "Psychology journals (CrossRef)"},
    "crossref_health": {"source": "CrossRef", "desc": "Health behaviour (CrossRef)"},
    "crossref_mena": {"source": "CrossRef", "desc": "MENA + behaviour (CrossRef)"},
    "semanticscholar_broad": {"source": "SemanticScholar", "desc": "Behavioural science (SS)"},
    "semanticscholar_mena": {"source": "SemanticScholar", "desc": "MENA behavioural (SS)"},
}

ANALYSIS_CACHE = ROOT / "data" / "analysis_cache.json"
ANALYSIS_DIR = ROOT / "data" / "analyses"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def save_analysis(analysis):
    try:
        with open(ANALYSIS_CACHE, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: could not save analysis cache: {e}")

def load_analysis():
    if ANALYSIS_CACHE.exists():
        try:
            return json.load(open(ANALYSIS_CACHE, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_paper_analysis(paper_id, analysis):
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
    path = ANALYSIS_DIR / f"{safe_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: could not save analysis for {paper_id}: {e}")

def load_paper_analysis(paper_id):
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
    path = ANALYSIS_DIR / f"{safe_id}.json"
    if path.exists():
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            pass
    return None

batch_jobs = {}
batch_job_counter = 0
batch_jobs_lock = threading.Lock()

def _run_batch_job(job_id, papers):
    total = len(papers)
    results = []
    for i, paper in enumerate(papers):
        try:
            result = llm_summarise(paper)
            results.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "analysis": result
            })
        except Exception as e:
            results.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "analysis": {"error": str(e)}
            })
            with batch_jobs_lock:
                batch_jobs[job_id]["error"] = str(e)
        with batch_jobs_lock:
            batch_jobs[job_id]["progress"] = i + 1
            batch_jobs[job_id]["results"] = results
    with batch_jobs_lock:
        batch_jobs[job_id]["status"] = "done"
        batch_jobs[job_id]["progress"] = total

def load_papers():
    files = sorted(RAW_DIR.glob("papers_*.json"))
    all_papers = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    all_papers.extend(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    seen_ids = set()
    unique = []
    for p in all_papers:
        pid = p.get("id") or p.get("entry_id")
        if pid and pid in seen_ids:
            continue
        if pid:
            seen_ids.add(pid)
        unique.append(p)

    def _norm_title(t):
        if not t:
            return frozenset()
        return frozenset(re.sub(r'[^a-z0-9 ]', '', t.lower()).strip().split())

    deduped = []
    deduped_tokens = []
    for p in unique:
        title = p.get("title", "")
        tokens = _norm_title(title)
        if not tokens:
            deduped.append(p)
            deduped_tokens.append(tokens)
            continue
        is_dup = False
        for existing_tokens in deduped_tokens:
            if not existing_tokens:
                continue
            overlap = tokens & existing_tokens
            min_len = min(len(tokens), len(existing_tokens))
            if min_len > 0 and len(overlap) / min_len > 0.85:
                is_dup = True
                break
        if not is_dup:
            deduped.append(p)
            deduped_tokens.append(tokens)

    removed = len(unique) - len(deduped)
    if removed > 0:
        print(f"  Title dedup removed {removed} near-duplicate papers")
    return deduped

def word_freq(texts, stopwords=None, min_len=3):
    if stopwords is None:
        stopwords = {"the","and","for","are","but","not","you","all","any","can","had",
            "her","was","one","our","out","day","get","has","him","his","how",
            "its","may","new","now","old","see","two","who","did","man","men",
            "put","too","use","that","this","with","from","which","their","have",
            "been","will","would","could","should","about","when","make","like",
            "time","just","than","also","into","more","some","these","each","they",
            "being","were","them","such","only","over","very","what","where",
            "much","many","well","still","most","those","using","based","paper",
            "study","research","result","used","show","found","find","present",
            "propose","model","approach","method","data","experiment","analysis"}
    words = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{%d,}\b" % min_len, text.lower())
        words.extend([w for w in tokens if w not in stopwords])
    return Counter(words).most_common(200)

def retry(max_attempts=3, base_delay=1, backoff=2):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (backoff ** attempt)
                        print(f"  Retry {attempt+1}/{max_attempts} for {fn.__name__} after {delay}s: {e}")
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

@retry(max_attempts=2, base_delay=1, backoff=2)
def _llm_call_single(messages, model, max_tokens=600, temperature=0.3):
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key. Set it in the GUI Scraper tab or OPENROUTER_API_KEY env var."}
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or message.get("reasoning") or ""
        if not content and message.get("reasoning_details"):
            parts = []
            for rd in message.get("reasoning_details", []):
                if rd.get("text"):
                    parts.append(rd.get("text"))
            content = "\n".join(parts)
        return {"content": content.strip(), "model_used": model}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 429:
            raise
        if 400 <= e.code < 500:
            return {"error": f"HTTP {e.code}: {body}", "model_used": model}
        raise
    except Exception:
        raise

def llm_call(messages, max_tokens=600, temperature=0.3):
    models_to_try = [LLM_MODEL] + LLM_FALLBACK_MODELS
    last_error = None
    for model in models_to_try:
        try:
            result = _llm_call_single(messages, model, max_tokens, temperature)
            if "error" in result:
                last_error = result["error"]
                print(f"  Model {model} failed: {last_error}. Trying fallback...")
                continue
            if result.get("model_used") != LLM_MODEL:
                print(f"  Fallback to {result['model_used']} succeeded")
            return result
        except Exception as e:
            last_error = str(e)
            print(f"  Model {model} error: {last_error}. Trying fallback...")
            continue
    return {"error": f"All models failed. Last error: {last_error}"}

def llm_summarise(paper):
    paper_id = paper.get("id") or paper.get("entry_id") or ""
    cached = load_paper_analysis(paper_id)
    if cached is not None:
        return cached
    prompt = (
        "You are a behavioural science research assistant specialising in MENA region studies. "
        "Analyse this academic paper and provide a structured JSON analysis.\n\n"
        f"Title: {paper.get('title', '')}\n"
        f"Abstract: {(paper.get('summary') or '')[:2500]}\n\n"
        "Respond in JSON with these exact keys:\n"
        '- "behavioural_model": primary model/theory used (e.g. "COM-B", "Theory of Planned Behaviour", "Health Belief Model", "Social Cognitive Theory", "Self-Determination Theory", "Dual Process Theory", "Nudge", "Transtheoretical Model", "Social Norms Theory", "None/not explicit")\n'
        '- "key_findings": 3-5 bullet points of main findings as a list of strings\n'
        '- "methodology": research method (e.g. "RCT", "survey", "qualitative interview", "computational model", "systematic review", "meta-analysis", "mixed methods", "case study")\n'
        '- "mena_relevance": "direct study" (paper studies MENA population), "some relevance" (mentions MENA or has cultural implications), or "general/theoretical" (no MENA-specific content)\n'
        '- "behavioural_domain": primary domain (e.g. "health behaviour", "decision making", "technology adoption", "financial behaviour", "environmental behaviour", "education", "organisational behaviour", "social behaviour", "consumer behaviour", "political behaviour", "other")\n'
        '- "summary": 2-3 sentence plain-language summary of what this paper is about and why it matters\n'
        '- "arabic_terms": list of Arabic/MENA-specific terms, countries, or cultural contexts mentioned (empty list if none)\n'
        '- "limitations": list of key limitations mentioned by authors (empty list if none)\n'
        '- "future_research": list of future research directions mentioned (empty list if none)\n\n'
        "Respond ONLY with valid JSON, no markdown fences."
    )
    result = llm_call([{"role": "user", "content": prompt}], max_tokens=2000)
    if "error" in result:
        return result
    content = result.get("content", "")
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    try:
        parsed = json.loads(content)
        if paper_id and "error" not in parsed:
            save_paper_analysis(paper_id, parsed)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response", "raw": content[:300]}

def llm_cluster_papers(papers, max_n=50, existing_clusters=None):
    papers = papers[:max_n]
    existing_indices = set()
    if existing_clusters:
        for cl in existing_clusters:
            existing_indices.update(cl.get("paper_indices", []))
        papers = [p for i, p in enumerate(papers) if i not in existing_indices]
    if len(papers) <= 50:
        return _cluster_batch(papers, offset=0)

    all_clusters = []
    seen_indices = set()
    batch_size, overlap, step = 50, 10, 40
    for start in range(0, len(papers), step):
        batch = papers[start:start + batch_size]
        if len(batch) < 10:
            break
        result = _cluster_batch(batch, offset=start)
        if "error" in result:
            return result
        for cl in result.get("clusters", []):
            existing = next((c for c in all_clusters if c["name"] == cl["name"]), None)
            if existing:
                for idx in cl.get("paper_indices", []):
                    if idx not in seen_indices:
                        existing["paper_indices"].append(idx)
                        seen_indices.add(idx)
            else:
                new_cl = dict(cl)
                for idx in cl.get("paper_indices", []):
                    seen_indices.add(idx)
                all_clusters.append(new_cl)

    all_unclustered = set()
    for start in range(0, len(papers), step):
        batch = papers[start:start + batch_size]
        if len(batch) < 10:
            break
        result = _cluster_batch(batch, offset=start)
        if "error" not in result:
            all_unclustered.update(result.get("unclustered", []))
    all_unclustered -= seen_indices
    return {"clusters": all_clusters, "unclustered": list(all_unclustered)}

def _cluster_batch(papers, offset=0):
    paper_summaries = []
    for i, p in enumerate(papers):
        title = p.get('title', '')
        abstract = (p.get('summary') or '')[:200]
        paper_summaries.append(f"[{i}] {title}\n{abstract}")
    context = "\n\n".join(paper_summaries)
    prompt = (
        "You are a behavioural science research assistant. Below are paper titles and abstracts. "
        "Group them into 4-10 conceptual clusters based on shared themes, topics, or research areas. "
        "Each cluster should have a descriptive name (2-4 words) and list the paper indices.\n\n"
        f"Papers:\n{context}\n\n"
        "Respond in JSON: {\"clusters\": [{\"name\": \"...\", \"description\": \"...\", \"paper_indices\": [0,3]}], \"unclustered\": [1,5]}\n"
        "Respond ONLY with valid JSON, no markdown fences."
    )
    result = llm_call([{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.2)
    if "error" in result:
        return result
    content = result.get("content") or result.get("reasoning") or ""
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response", "raw": content[:300]}
    for cl in parsed.get("clusters", []):
        cl["paper_indices"] = [i + offset for i in cl.get("paper_indices", [])]
    parsed["unclustered"] = [i + offset for i in parsed.get("unclustered", [])]
    return parsed

def llm_batch_summarise(papers):
    results = []
    for paper in papers:
        result = llm_summarise(paper)
        results.append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "analysis": result
        })
        time.sleep(0.5)
    return results

def llm_rag_chat(query, papers_context, history=None):
    context_parts = []
    for i, p in enumerate(papers_context[:30]):
        abstract = (p.get('summary') or '')[:500]
        context_parts.append(f"[Paper {i+1}] {p.get('title', '')}\n{abstract}")
    context = "\n\n".join(context_parts)

    system_msg = (
        "You are a research assistant specialising in behavioural science with focus on the "
        "Middle East and North Africa (MENA) region. You answer questions based ONLY on the "
        "provided paper abstracts.\n\n"
        "CRITICAL RULES:\n"
        "- DO NOT invent, fabricate, or guess any paper titles, authors, findings, or facts.\n"
        "- If the provided abstracts do not contain information to answer the question, "
        "say exactly: 'No relevant papers found in the current dataset.'\n"
        "- Only cite papers from the provided context using [Paper N] references.\n"
        "- If you are unsure, say 'The available papers do not cover this topic.'\n"
        "Be concise but thorough. Use bullet points where appropriate."
    )
    user_msg = (
        f"Based on the following {len(papers_context)} paper abstracts, answer this question:\n\n"
        f"Question: {query}\n\n"
        f"Paper abstracts:\n{context}"
    )
    msgs = [{"role": "system", "content": system_msg}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_msg})
    result = llm_call(msgs, max_tokens=2000, temperature=0.2)
    return result

def analyze_papers(papers):
    results = {"generated_at": datetime.now().isoformat(), "total_papers": len(papers)}
    years = Counter()
    months = Counter()
    dates_raw = []
    for p in papers:
        pub = p.get("published", "")
        if pub:
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                dt = dt.replace(tzinfo=None)
                years[dt.year] += 1
                months[f"{dt.year}-{dt.month:02d}"] += 1
                dates_raw.append(dt)
            except Exception:
                pass
    results["yearly_distribution"] = dict(years.most_common())
    results["monthly_distribution"] = dict(sorted(months.items()))
    results["date_range"] = {
        "earliest": str(min(dates_raw).date()) if dates_raw else None,
        "latest": str(max(dates_raw).date()) if dates_raw else None,
    }
    all_authors = []
    author_counts = Counter()
    for p in papers:
        for a in p.get("authors", []):
            all_authors.append(a)
            author_counts[a] += 1
    results["total_authors"] = len(set(all_authors))
    results["avg_authors_per_paper"] = round(
        sum(len(p.get("authors", [])) for p in papers) / len(papers), 2
    ) if papers else 0
    results["top_authors"] = author_counts.most_common(20)
    results["top_title_keywords"] = word_freq([(p.get("title") or "") for p in papers], min_len=3)[:30]
    results["top_abstract_keywords"] = word_freq([(p.get("summary") or "") for p in papers], min_len=4)[:30]
    combined = " ".join(((p.get("title") or "") + " " + (p.get("summary") or "")).lower() for p in papers)
    results["behavioural_term_freq"] = sorted(
        {t: combined.count(t) for t in BEHAVIOURAL_TERMS if combined.count(t) > 0}.items(), key=lambda x: -x[1])
    results["region_term_freq"] = sorted(
        {t: combined.count(t) for t in MIDDLE_EAST_TERMS if combined.count(t) > 0}.items(), key=lambda x: -x[1])
    clusters = {}
    for term in BEHAVIOURAL_TERMS:
        related = [{"id": p["id"], "title": p["title"],
                    "year": p.get("published","")[:4], "authors": p.get("authors",[])}
                   for p in papers if term in ((p.get("title") or "") + " " + (p.get("summary") or "")).lower()]
        if len(related) >= 2:
            clusters[term] = related
    results["concept_clusters"] = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    results["most_collaborative"] = sorted(
        [{"title": p["title"], "authors": p.get("authors",[]),
          "count": len(p.get("authors",[])), "year": p.get("published","")[:4]}
         for p in papers], key=lambda x: -x["count"])[:10]
    results["summary"] = {
        "total_papers": len(papers),
        "date_range": f"{results['date_range']['earliest']} -> {results['date_range']['latest']}",
        "unique_authors": len(set(all_authors)),
        "avg_authors": results["avg_authors_per_paper"],
        "top_behavioural_term": results["behavioural_term_freq"][0] if results["behavioural_term_freq"] else ("N/A",0),
        "top_region_term": results["region_term_freq"][0] if results["region_term_freq"] else ("N/A",0),
        "concept_clusters_count": len(results["concept_clusters"]),
        "most_studied_concept": max(results["concept_clusters"].items(), key=lambda x: len(x[1]))[0]
            if results["concept_clusters"] else "N/A",
        "years_active": len(years),
    }
    return results

def search_papers(papers, query, fields=None):
    if fields is None:
        fields = ["title", "summary", "authors"]
    ql = query.lower()
    results = []
    for p in papers:
        for field in fields:
            val = p.get(field, "")
            if isinstance(val, list):
                val = " ".join(val)
            if val is None:
                val = ""
            if ql in val.lower():
                results.append(p)
                break
    return results

def run_scraper(query_key, count):
    global scraper_status, papers_global, analysis_global
    scraper_status["running"] = True
    scraper_status["output"] = f"Starting scrape: {query_key} ({count} papers)...\n"
    scraper_status["returncode"] = None
    try:
        result = subprocess.run(
            [sys.executable, "scraper.py", "-q", query_key, "-n", str(count)],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT)
        )
        scraper_status["output"] += result.stdout + "\n" + result.stderr
        scraper_status["returncode"] = result.returncode
        if result.returncode == 0:
            papers_global = load_papers()
            analysis_global = analyze_papers(papers_global)
            save_analysis(analysis_global)
            scraper_status["output"] += f"\nDone. Total papers: {len(papers_global)}"
    except subprocess.TimeoutExpired:
        scraper_status["output"] += "\nScraper timed out after 5 minutes."
        scraper_status["returncode"] = -1
    except Exception as e:
        scraper_status["output"] += f"\nError: {e}"
        scraper_status["returncode"] = -1
    scraper_status["running"] = False

scraper_status = {"running": False, "output": "", "returncode": None}

class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Drop a connection that opens but never sends a full request (browsers
    # routinely open speculative/preconnect sockets). Without this a single
    # blocked socket read would otherwise pin a worker thread indefinitely.
    timeout = 30

    def log_message(self, *a): pass

    def _get_client_ip(self):
        return self.client_address[0]

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _rate_check(self):
        ip = self._get_client_ip()
        if not rate_limiter.allow(ip):
            retry_after = rate_limiter.retry_after(ip)
            self.send_response(429)
            self.send_header("Retry-After", str(retry_after))
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Rate limit exceeded. Try again later."}).encode())
            return False
        return True

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/api/init":
                papers_meta = []
                for p in papers_global:
                    papers_meta.append({
                        "id": p.get("id") or p.get("entry_id"),
                        "title": p.get("title", ""),
                        "year": (p.get("published") or "")[:4]
                    })
                cfg = load_config()
                has_key = bool(cfg.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY", ""))
                self._json({
                    "papers_meta": papers_meta,
                    "papers_count": len(papers_global),
                    "analysis": analysis_global,
                    "key": {"has_key": has_key}
                })
                return

            elif path == "/api/papers":
                q = params.get("q", [""])[0]
                year = params.get("year", [""])[0]
                term = params.get("term", [""])[0]
                result = list(papers_global)
                if q and len(q) >= 2:
                    result = search_papers(result, q)
                if year:
                    result = [p for p in result if p.get("published","").startswith(year)]
                if term:
                    tl = term.lower()
                    result = [p for p in result if tl in (p.get("title","")+p.get("summary","")).lower()]
                self._json({"papers": result, "count": len(result)})
                return

            elif path == "/api/analysis":
                self._json(analysis_global)
                return

            elif path == "/api/years":
                self._json(list(sorted(analysis_global.get("yearly_distribution", {}).keys())))
                return

            elif path == "/api/search":
                q = params.get("q", [""])[0]
                if q and len(q) >= 2:
                    r = search_papers(papers_global, q)
                    self._json({"papers": r[:50], "count": len(r)})
                else:
                    self._json({"papers": [], "count": 0})
                return

            elif path == "/api/export":
                return self._handle_export(params)

            elif path == "/api/config/key":
                cfg = load_config()
                self._json({"key": cfg.get("openrouter_api_key", "")})
                return

            elif path == "/api/scraper/status":
                self._json({
                    "running": scraper_status.get("running", False),
                    "output": scraper_status.get("output", ""),
                    "returncode": scraper_status.get("returncode")
                })
                return

            elif path == "/api/shutdown":
                self._json({"ok": True, "message": "Shutting down..."})
                threading.Timer(0.5, self.server.shutdown).start()
                return

            elif path == "/":
                path = "/index.html"

            paper_graph_path = ROOT / "graphify-out" / "paper_graph.html"
            if path == "/paper_graph.html" and paper_graph_path.exists():
                local_path = paper_graph_path
                ct = "text/html; charset=utf-8"
            else:
                try:
                    local_path = ROOT / path.lstrip("/")
                    if path.endswith(".html"):
                        ct = "text/html; charset=utf-8"
                    elif path.endswith(".js"):
                        ct = "application/javascript; charset=utf-8"
                    elif path.endswith(".css"):
                        ct = "text/css; charset=utf-8"
                    else:
                        ct = "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", ct)
                    self.send_header("Cache-Control", "no-store, must-revalidate")
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    with open(local_path, "rb") as f:
                        self.wfile.write(f.read())
                except FileNotFoundError:
                    self.send_error(404)

        except ConnectionAbortedError:
            pass
        except Exception as e:
            with open(ROOT / "server_errors.log", 'a') as ef:
                ef.write(f"[{datetime.now().isoformat()}] GET {self.path} ERROR: {type(e).__name__}: {e}\n")
            self._json({"error": f"Internal error: {str(e)}"}, status=500)

    def _handle_export(self, params):
        _err = open(os.path.join(tempfile.gettempdir(), 'export_debug.log'), 'w')
        try:
            fmt = params.get("format", ["csv"])[0].lower()
            _err.write(f"format={fmt}\n")
            q = params.get("q", [""])[0]
            _err.write(f"q={q}\n")
            ids_param = params.get("ids", [""])[0]
            _err.write(f"ids={ids_param}\n")

            result = list(papers_global)
            _err.write(f"papers_global len={len(result)}\n")

            if q and len(q) >= 2:
                result = search_papers(result, q)
                _err.write(f"after search: {len(result)} papers\n")

            if ids_param:
                ids_set = {i.strip() for i in ids_param.split(",") if i.strip()}
                result = [p for p in result if str(p.get("id") or p.get("entry_id", "")) in ids_set]
                _err.write(f"after id filter: {len(result)} papers\n")

            _err.write(f"building {fmt} export for {len(result)} papers\n")

            if fmt == "json":
                body = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
                ct = "application/json"
                ext = "json"
            elif fmt == "bibtex":
                entries = []
                for p in result:
                    pid = str(p.get("id") or p.get("entry_id") or "")
                    title = (p.get("title") or "").replace("{", "\\{").replace("}", "\\}")
                    authors_list = p.get("authors") or []
                    author = " and ".join(authors_list)
                    year = ""
                    pub = p.get("published", "")
                    if pub:
                        try:
                            year = str(datetime.fromisoformat(pub.replace("Z", "+00:00")).year)
                        except Exception:
                            year = pub[:4]
                    abstract = (p.get("summary") or "").replace("{", "\\{").replace("}", "\\}")
                    url = p.get("url", "")
                    entry = f"@article{{{pid},\\n  title={{ {title} }},\\n  author={{ {author} }},\\n  year={{ {year} }},\\n  abstract={{ {abstract} }},\\n  url={{ {url} }}\\n}}"
                    entries.append(entry)
                body = "\\n\\n".join(entries).encode("utf-8")
                ct = "application/x-bibtex"
                ext = "bib"
            else:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["id", "title", "authors", "year", "abstract", "url"])
                for p in result:
                    pid = str(p.get("id") or p.get("entry_id") or "")
                    title = (p.get("title") or "").replace("\n", " ").replace("\r", " ")
                    authors = "; ".join(p.get("authors") or [])
                    year = ""
                    pub = p.get("published", "")
                    if pub:
                        try:
                            year = str(datetime.fromisoformat(pub.replace("Z", "+00:00")).year)
                        except Exception:
                            year = pub[:4]
                    abstract = (p.get("summary") or "").replace("\n", " ").replace("\r", " ")
                    url = p.get("url", "")
                    writer.writerow([pid, title, authors, year, abstract, url])
                body = output.getvalue().encode("utf-8")
                ct = "text/csv"
                ext = "csv"

            fname = "papers_export_{0}_{1}.{2}".format(
                len(result),
                q.replace(" ", "_")[:20] if q else "all",
                ext
            )
            _err.write(f"sending response, body len={len(body)}\n")
            self.send_response(200)
            self.send_header("Content-Type", ct + "; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionAbortedError:
            pass
        except Exception as e:
            _err.write(f"ERROR: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc(file=_err)
            _err.close()
            self._json({"error": f"Export failed: {str(e)}"}, status=500)
            return
        finally:
            _err.close()
        return

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        try:
            if path == "/api/config/key":
                key = data.get("key", "").strip()
                cfg = load_config()
                cfg["openrouter_api_key"] = key
                save_config(cfg)
                self._json({"ok": True, "message": "API key saved." if key else "Key cleared."})

            elif path == "/api/scraper/run":
                queries = data.get("queries", [])
                max_count = int(data.get("max", 20))
                if not queries:
                    self._json({"error": "No queries provided."}, status=400)
                    return
                if scraper_status.get("running", False):
                    self._json({"error": "Scraper is already running."}, status=409)
                    return
                for q in queries:
                    t = threading.Thread(target=run_scraper, args=(q, max_count), daemon=True)
                    t.start()
                self._json({"ok": True, "message": f"Started scraper for {len(queries)} query group(s) ({max_count} papers each)."})

            elif path == "/api/chat":
                if not self._rate_check():
                    return
                query = data.get("query", "").strip()
                if not query:
                    self._json({"error": "No query provided."}, status=400)
                    return
                history = data.get("history", [])
                paper_ids = data.get("paper_ids", [])
                if paper_ids:
                    papers = [p for p in papers_global if p.get("id") in paper_ids]
                else:
                    import re as _re
                    query_lower = query.lower()
                    query_phrases = [query_lower]
                    query_words = set(_re.findall(r'\b[a-zA-Z]{4,}\b', query_lower))
                    scored = []
                    for p in papers_global:
                        text = ((p.get('title') or '') + ' ' + (p.get('summary') or '')).lower()
                        phrase_pattern = r'\b' + _re.escape(query_phrases[0]) + r'\b'
                        score = 3 if _re.search(phrase_pattern, text) else 0
                        score += sum(1 for w in query_words if w in text)
                        if score > 0:
                            scored.append((score, p))
                    scored.sort(key=lambda x: -x[0])
                    papers = [p for _, p in scored[:30]]
                    if not papers:
                        papers = papers_global[:30]
                result = llm_rag_chat(query, papers, history=history)
                if "content" in result and result["content"]:
                    import re as _rev
                    cited = _rev.findall(r'\[Paper (\d+)\]', result["content"])
                    max_idx = len(papers)
                    invalid = [c for c in cited if int(c) < 1 or int(c) > max_idx]
                    if invalid:
                        result["content"] += "\n\n[Warning: citations " + ", ".join(invalid) + " refer to papers not in the context. They may be hallucinated.]"
                papers_meta = [{"id": p.get("id"), "title": p.get("title", "")} for p in papers[:30]]
                self._json({"query": query, "papers_used": papers_meta, "response": result})

            elif path == "/api/summarise":
                if not self._rate_check():
                    return
                idx = data.get("idx", -1)
                try:
                    idx = int(idx)
                    paper = papers_global[idx] if 0 <= idx < len(papers_global) else None
                except (ValueError, TypeError):
                    paper = None
                if not paper:
                    pid = data.get("id", "")
                    for p in papers_global:
                        if p.get("id") == pid:
                            paper = p
                            break
                if not paper:
                    self._json({"error": "Paper not found. Provide idx (index) or id.", "idx": data.get("idx"), "id": data.get("id")}, status=400)
                    return
                result = llm_summarise(paper)
                self._json({"paper_id": paper.get("id"), "title": paper.get("title"), "analysis": result})

            elif path == "/api/cluster":
                if not self._rate_check():
                    return
                try:
                    max_n = int(data.get("max", 50))
                except (ValueError, TypeError):
                    self._json({"error": "Invalid 'max' parameter. Must be a positive integer.", "max": data.get("max")}, status=400)
                    return
                max_n = max(1, min(max_n, 500))
                papers_subset = papers_global[:max_n]
                result = llm_cluster_papers(papers_subset, max_n=max_n, existing_clusters=data.get("existing"))
                self._json({"papers_analyzed": len(papers_subset), "clusters": result, "merged": data.get("existing") is not None})

            elif path == "/api/summarise_all":
                if not self._rate_check():
                    return
                try:
                    start = int(data.get("start", 0))
                    count = int(data.get("count", 10))
                except (ValueError, TypeError):
                    self._json({"error": "Invalid 'start' or 'count' parameter. Must be positive integers.",
                                "start": data.get("start"), "count": data.get("count")}, status=400)
                    return
                if start < 0:
                    self._json({"error": f"'start' must be >= 0, got {start}", "start": start}, status=400)
                    return
                count = max(1, min(count, 50))
                if start >= len(papers_global):
                    self._json({"error": f"'start' exceeds paper count ({len(papers_global)})",
                                "start": start, "total": len(papers_global)}, status=400)
                    return
                batch = papers_global[start:start+count]
                global batch_job_counter
                with batch_jobs_lock:
                    batch_job_counter += 1
                    job_id = f"batch_{batch_job_counter}"
                    batch_jobs[job_id] = {
                        "status": "running",
                        "progress": 0,
                        "total": len(batch),
                        "results": [],
                        "error": None
                    }
                thread = threading.Thread(target=_run_batch_job, args=(job_id, batch), daemon=True)
                thread.start()
                self._json({"job_id": job_id, "status": "running", "total": len(batch)})

            elif path == "/api/summarise_all/status":
                job_id = data.get("job_id", "")
                with batch_jobs_lock:
                    job = batch_jobs.get(job_id)
                if not job:
                    self._json({"error": f"Job '{job_id}' not found."}, status=404)
                    return
                self._json({
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "total": job["total"],
                    "results": job["results"],
                    "error": job["error"]
                })

            elif path == "/api/paper_graph":
                graph_path = ROOT / "graphify-out" / "paper_graph.json"
                if graph_path.exists():
                    try:
                        graph_data = json.load(open(graph_path, encoding="utf-8"))
                        self._json(graph_data)
                    except Exception as e:
                        self._json({"error": str(e)})
                else:
                    self._json({"error": "Paper graph not found. Build with: python build_paper_graph.py"})
                return

            elif path == "/api/shutdown":
                self._json({"ok": True, "message": "Shutting down..."})
                threading.Timer(0.5, self.server.shutdown).start()

            elif path == "/api/semantic_search":
                if not self._rate_check():
                    return
                query = data.get("query", "").strip()
                if not query or len(query) < 2:
                    self._json({"error": "Query must be at least 2 characters."}, status=400)
                    return
                top_k = int(data.get("top_k", 15))
                results = semantic_search(query, papers_global, top_k=top_k)
                papers_out = [{"id": r["paper"].get("id"), "title": r["paper"].get("title", ""),
                               "summary": (r["paper"].get("summary") or "")[:300],
                               "score": round(r["score"], 4)} for r in results]
                self._json({"query": query, "results": papers_out, "count": len(papers_out)})

            elif path == "/api/arabic_papers":
                if not self._rate_check():
                    return
                min_score = int(data.get("min_score", 3))
                scored = []
                for p in papers_global:
                    score, details = score_arabic_relevance(p)
                    if score >= min_score:
                        scored.append({
                            "id": p.get("id"),
                            "title": p.get("title", ""),
                            "summary": (p.get("summary") or "")[:300],
                            "score": score,
                            "details": details[:10]
                        })
                scored.sort(key=lambda x: -x["score"])
                self._json({"results": scored, "count": len(scored), "min_score": min_score})

            elif path == "/api/build_embeddings":
                if not self._rate_check():
                    return
                thread = threading.Thread(target=build_embeddings, args=(papers_global,), daemon=True)
                thread.start()
                self._json({"ok": True, "message": "Building embeddings in background..."})

            elif path == "/api/fulltext":
                if not self._rate_check():
                    return
                paper_id = data.get("id", "")
                force = data.get("force", False)
                paper = None
                for p in papers_global:
                    if p.get("id") == paper_id or p.get("entry_id") == paper_id:
                        paper = p
                        break
                if not paper:
                    self._json({"error": "Paper not found", "id": paper_id}, status=404)
                    return
                safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
                cache_path = PDF_DIR / f"{safe_id}.txt"
                if not force and cache_path.exists():
                    try:
                        text = cache_path.read_text(encoding="utf-8")
                        self._json({
                            "id": paper_id,
                            "title": paper.get("title", ""),
                            "text": text[:50000],
                            "chars": len(text),
                            "cached": True,
                            "source": "cache",
                        })
                        return
                    except Exception:
                        pass
                text = ""
                source = "none"
                pdf_url = paper.get("pdf_url", "")
                if pdf_url:
                    pdf_path = download_pdf(pdf_url, paper_id)
                    if pdf_path:
                        text = extract_pdf_text(pdf_path)
                        source = "pdf"
                if not text or text.startswith("["):
                    text = paper.get("summary") or ""
                    source = "abstract"
                if text and not text.startswith("["):
                    try:
                        cache_path.write_text(text, encoding="utf-8")
                    except Exception:
                        pass
                self._json({
                    "id": paper_id,
                    "title": paper.get("title", ""),
                    "text": text[:50000],
                    "chars": len(text),
                    "cached": False,
                    "source": source,
                })

            else:
                self.send_error(404)

        except ConnectionAbortedError:
            pass
        except Exception as e:
            with open(ROOT / "server_errors.log", 'a') as ef:
                ef.write(f"[{datetime.now().isoformat()}] POST {self.path} ERROR: {type(e).__name__}: {e}\n")
            self._json({"error": f"Internal error: {str(e)}"}, status=500)

def serve(port=3000):
    global papers_global, analysis_global
    error_log = ROOT / "server_errors.log"
    print(f"  ROOT: {ROOT}")
    print("Loading papers...")
    papers_global = load_papers()
    print(f"  Found {len(papers_global)} papers")
    if papers_global:
        analysis_global = analyze_papers(papers_global)
        save_analysis(analysis_global)
        s = analysis_global["summary"]
        print(f"  Analysis: {s['total_papers']} papers, {s['unique_authors']} authors, {s['concept_clusters_count']} clusters")
        print("  Embeddings deferred. POST /api/build_embeddings to build semantic search.")
    else:
        analysis_global = {"total_papers":0,"summary":{},"yearly_distribution":{},
                           "top_title_keywords":[],"top_abstract_keywords":[],
                           "behavioural_term_freq":[],"region_term_freq":[],
                           "concept_clusters":{},"most_collaborative":[],
                           "generated_at":datetime.now().isoformat(),
                           "total_authors":0,"avg_authors_per_paper":0,
                           "top_authors":[],"date_range":{"earliest":None,"latest":None},
                           "monthly_distribution":{}}

    # ThreadingHTTPServer: each request runs on its own thread so concurrent
    # browser requests (init / papers / analysis / status polls) and any
    # long-running call (LLM, scraper, PDF extraction) can never block one
    # another. Single-threaded HTTPServer froze the whole server whenever one
    # socket stalled, which surfaced as "Failed to connect to server".
    server = ThreadingHTTPServer(("", port), Handler)
    server.daemon_threads = True
    server.allow_reuse_address = True

    print(f"\n  Running at http://localhost:{port}")
    print(f"  Rate limit: 10 requests/60s per IP on LLM endpoints")
    print(f"  {len(papers_global)} papers loaded\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
    finally:
        server.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    serve(port)
