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
import collections
import hashlib
import ipaddress
import socket
import tempfile
import logging
import queue
from pathlib import Path
from datetime import datetime
from collections import Counter
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler


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

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "server.log"

# The console on Windows defaults to cp1252, so the StreamHandler raised
# UnicodeEncodeError on every log line containing an Arabic query (the file
# handler was already UTF-8). Same fix scraper.py applies to its own stdout.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from app.pdf_text import (
    PDF_DIR, MAX_PDF_PAGES, MAX_PDF_TEXT_CHARS, MAX_PDF_BYTES,
    extract_pdf_text, is_fetchable_url, download_pdf,
    download_pdf_with_fallback, get_paper_full_text,
)
from app.terms import (
    ARABIC_PATTERN, ARABIC_KEYWORDS, detect_arabic_content, score_arabic_relevance,
    BEHAVIOURAL_TERMS, MENA_PLACE_TERMS, MENA_CONTEXT_TERMS, MIDDLE_EAST_TERMS,
    SCRAPER_QUERIES, BEHAVIOURAL_QUERY_PREFIX, _term_pattern, _count_terms,
)

scraper_log_queue = queue.Queue()
# Bounded. This was a plain list that log_scraper appended to forever and nothing
# ever trimmed, so a long-lived server grew it without limit.
scraper_logs = collections.deque(maxlen=2000)

def log_scraper(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    scraper_logs.append(entry)
    logger.info(f"SCRAPER: {msg}")

def get_scraper_logs():
    return list(scraper_logs)   # a deque is not JSON-serialisable

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

def get_gemini_api_key():
    cfg = load_config()
    return cfg.get("gemini_api_key", os.environ.get("GEMINI_API_KEY", ""))

EMBEDDING_PROVIDER = "gemini"
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_MODEL_GEMINI = "models/gemini-embedding-2"
EMBEDDING_BATCH_SIZE = 100
GEMINI_EMBEDDING_DIM = 3072

LLM_BASE_URL = "https://openrouter.ai/api/v1"
# Every model below was verified to return a completion on 2026-08-05.
# The previous chain had three dead entries out of four: laguna-m.1 and
# gemini-2.0-flash-001 no longer exist ("No endpoints found"), and qwen3-coder:free
# became paid-only. Only gpt-oss-20b still answered, so every LLM call burned a
# 404 round trip on the dead primary before falling through to it.
# laguna-s-2.1 is the successor to the laguna-m.1 this project was built around.
# Chat context limits. One constant so the prompt, the citation validator and the
# papers_used payload can never describe different lists.
MAX_CONTEXT_PAPERS = 30
MAX_ABSTRACT_CHARS = 2000
# Neighbours each paper keeps in the similarity graph. See build_embedding_graph
# for why this replaced an absolute cosine threshold.
GRAPH_TOP_K = 8
MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHARS = 8000

LLM_MODEL = "poolside/laguna-s-2.1:free"
LLM_FALLBACK_MODELS = [
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "inclusionai/ling-3.0-flash:free",
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

# PDF download and full-text extraction: app/pdf_text.py

EMBEDDING_CACHE = ROOT / "data" / "embeddings.json"
EMBEDDING_META_CACHE = ROOT / "data" / "embeddings_meta.json"
_paper_embeddings = None
_paper_embedding_ids = []
# Guards replacement of the (matrix, ids) pair. They are two separate globals that
# must agree by position: row i of the matrix belongs to ids[i]. Readers that
# touched them in separate statements could pair a NEW matrix with OLD ids, which
# silently maps similarity scores onto the wrong papers -- a wrong-answer bug, not
# a crash. Writers hold this lock; readers take a coherent pair via
# embedding_snapshot().
_embedding_lock = threading.Lock()


def embedding_snapshot():
    """Return (matrix, ids) as a pair that agree with each other."""
    with _embedding_lock:
        return _paper_embeddings, _paper_embedding_ids


def _set_embeddings(matrix, ids):
    global _paper_embeddings, _paper_embedding_ids
    with _embedding_lock:
        _paper_embeddings = matrix
        _paper_embedding_ids = ids


def _load_embeddings():
    import numpy as np
    if _paper_embeddings is not None:
        return
    if EMBEDDING_CACHE.exists():
        try:
            cached = json.load(open(EMBEDDING_CACHE))
            _set_embeddings(np.array(cached["embeddings"]), cached["ids"])
            return
        except Exception:
            pass
    _set_embeddings(None, [])

def _get_embedding_openai(text):
    api_key = get_api_key()
    if not api_key:
        return None
    payload = json.dumps({"model": EMBEDDING_MODEL_OPENAI, "input": text[:8000]}).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/embeddings", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"  OpenAI embedding error: {e}")
        return None

def _get_embedding_gemini(text):
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL_GEMINI}:embedContent?key={api_key}"
    payload = json.dumps({"content": {"parts": [{"text": text[:8000]}]}}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["embedding"]["values"]
    except Exception as e:
        print(f"  Gemini embedding error: {e}")
        return None

def _get_embedding_batch_gemini(texts):
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL_GEMINI}:batchEmbedContents?key={api_key}"
    requests_list = [{"model": EMBEDDING_MODEL_GEMINI, "content": {"parts": [{"text": t[:8000]}]}} for t in texts]
    payload = json.dumps({"requests": requests_list}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            embeddings = data.get("embeddings", [])
            return [e["values"] for e in embeddings]
    except Exception as e:
        print(f"  Gemini batch embedding error: {e}")
        return None

def _get_embedding(text):
    if EMBEDDING_PROVIDER == "gemini":
        return _get_embedding_gemini(text)
    return _get_embedding_openai(text)

def _embedding_text(paper):
    """Text to embed for one paper: cached PDF full text if present, else title+abstract."""
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper.get("id") or paper.get("entry_id") or ""))
    cache_path = PDF_DIR / f"{safe_id}.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")[:8000]
        except Exception:
            return ""
    return (paper.get("title") or "") + " " + (paper.get("summary") or "")

def build_embeddings(papers, provider=None, batch=True):
    # No `global` here: the pair is replaced only via _set_embeddings(), under the lock.
    import numpy as np

    if provider is None:
        provider = EMBEDDING_PROVIDER

    if provider == "gemini":
        api_key = get_gemini_api_key()
    else:
        api_key = get_api_key()
    if not api_key:
        print(f"  Skipping embeddings: no API key for provider '{provider}'.")
        return

    _load_embeddings()
    paper_ids = [p.get("id") or p.get("entry_id") for p in papers]

    # Check cache: if same IDs and same provider, skip
    cached_matrix, cached_ids = embedding_snapshot()
    if cached_ids == paper_ids and cached_matrix is not None:
        meta = {}
        if EMBEDDING_META_CACHE.exists():
            try:
                meta = json.load(open(EMBEDDING_META_CACHE))
            except Exception:
                pass
        if meta.get("provider") == provider:
            print(f"  Embeddings already built ({len(ids)} papers, provider={provider}).")
            return

    print(f"  Building embeddings (provider={provider}, batch={batch}, papers={len(papers)})...")
    embeddings, ids = [], []

    if provider == "gemini" and batch:
        # Collect texts
        texts = []
        id_map = []
        for p in papers:
            texts.append(_embedding_text(p))
            id_map.append(p.get("id") or p.get("entry_id"))

        # Process in batches of EMBEDDING_BATCH_SIZE
        batch_size = EMBEDDING_BATCH_SIZE
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_ids = id_map[batch_start:batch_end]
            print(f"    Embedding batch {batch_start+1}-{batch_end}/{len(texts)}...")
            results = _get_embedding_batch_gemini(batch_texts)
            if results:
                for emb, pid in zip(results, batch_ids):
                    if emb:
                        embeddings.append(emb)
                        ids.append(pid)
            else:
                # Fallback: try one-by-one for this batch
                print(f"    Batch failed, falling back to single requests...")
                for t, pid in zip(batch_texts, batch_ids):
                    emb = _get_embedding_gemini(t)
                    if emb:
                        embeddings.append(emb)
                        ids.append(pid)
    else:
        # Single-request mode (original behavior, works for both providers)
        fail_fast = False
        for i, p in enumerate(papers):
            if fail_fast:
                break
            text = _embedding_text(p)
            if provider == "gemini":
                emb = _get_embedding_gemini(text)
            else:
                emb = _get_embedding_openai(text)
            if emb:
                embeddings.append(emb)
                ids.append(p.get("id") or p.get("entry_id"))
            elif i == 0:
                fail_fast = True
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(papers)} embedded")

    if embeddings:
        _set_embeddings(np.array(embeddings), ids)
        # Was json.dump(..., open(path, "w")) -- the handle was never closed, so a
        # 71MB write had no guaranteed flush, and it truncated in place while other
        # threads could be reading it.
        atomic_write_json(EMBEDDING_CACHE,
                          {"embeddings": np.array(embeddings).tolist(), "ids": ids})
        atomic_write_json(EMBEDDING_META_CACHE, {
            "provider": provider,
            "model": EMBEDDING_MODEL_GEMINI if provider == "gemini" else EMBEDDING_MODEL_OPENAI,
            "count": len(ids),
        })
        print(f"  Embeddings built: {len(ids)} papers (provider={provider})")
    else:
        print("  No embeddings were generated.")

def embed_new_papers(papers):
    """Embed only the papers that have no vector yet and append them to the matrix.

    Delta counterpart to build_embeddings(): after a scrape, the handful of new
    papers get vectors immediately instead of being invisible to semantic search
    until a full manual rebuild. Returns the number of papers embedded. No API
    key, nothing new, or a failed provider call all return 0 and leave the
    in-memory matrix and on-disk caches untouched.
    """
    import numpy as np

    provider = EMBEDDING_PROVIDER
    # Same provider selection and key check as build_embeddings.
    api_key = get_gemini_api_key() if provider == "gemini" else get_api_key()
    if not api_key:
        return 0

    _load_embeddings()
    matrix, ids = embedding_snapshot()
    embedded_ids = set(ids or [])
    new_papers = [p for p in papers
                  if (p.get("id") or p.get("entry_id")) not in embedded_ids]
    if not new_papers:
        return 0

    texts = [_embedding_text(p) for p in new_papers]
    id_map = [p.get("id") or p.get("entry_id") for p in new_papers]

    embeddings, new_ids = [], []
    if provider == "gemini":
        for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
            batch_ids = id_map[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
            results = _get_embedding_batch_gemini(batch_texts)
            if results:
                for emb, pid in zip(results, batch_ids):
                    if emb:
                        embeddings.append(emb)
                        new_ids.append(pid)
            else:
                # Same fallback build_embeddings uses: retry the batch one by one.
                for t, pid in zip(batch_texts, batch_ids):
                    emb = _get_embedding_gemini(t)
                    if emb:
                        embeddings.append(emb)
                        new_ids.append(pid)
    else:
        for t, pid in zip(texts, id_map):
            emb = _get_embedding_openai(t)
            if emb:
                embeddings.append(emb)
                new_ids.append(pid)

    if not embeddings:
        return 0

    new_matrix = np.array(embeddings)
    if matrix is not None and len(ids) > 0:
        # A provider/model switch changes vector dimensions; appending mixed
        # dimensions would corrupt every cosine score, so refuse and require a
        # full rebuild instead.
        if matrix.shape[1] != new_matrix.shape[1]:
            print(f"  Skipping delta embed: dimension mismatch with existing matrix "
                  f"({matrix.shape[1]} vs {new_matrix.shape[1]}); rebuild embeddings.")
            return 0
        combined_matrix = np.vstack([matrix, new_matrix])
        # New list, never mutate the snapshot: row i must keep belonging to ids[i].
        combined_ids = list(ids) + new_ids
    else:
        combined_matrix = new_matrix
        combined_ids = list(new_ids)

    _set_embeddings(combined_matrix, combined_ids)
    # Persist exactly like build_embeddings: matrix cache, ids, then meta.
    atomic_write_json(EMBEDDING_CACHE,
                      {"embeddings": combined_matrix.tolist(), "ids": combined_ids})
    atomic_write_json(EMBEDDING_META_CACHE, {
        "provider": provider,
        "model": EMBEDDING_MODEL_GEMINI if provider == "gemini" else EMBEDDING_MODEL_OPENAI,
        "count": len(combined_ids),
    })
    return len(new_ids)

def semantic_search(query, papers, top_k=15):
    import numpy as np
    _load_embeddings()
    emb_matrix, emb_ids = embedding_snapshot()
    if emb_matrix is None or len(emb_matrix) == 0:
        return []
    query_emb = _get_embedding(query)
    if query_emb is None:
        return []
    query_vec = np.array(query_emb)
    norms = np.linalg.norm(emb_matrix, axis=1)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    similarities = np.dot(emb_matrix, query_vec) / (norms * query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        paper_id = emb_ids[idx]
        paper = next((p for p in papers if (p.get("id") or p.get("entry_id")) == paper_id), None)
        if paper:
            results.append({"paper": paper, "score": float(similarities[idx])})
    return results

def score_papers_by_embedding(papers, query, top_k=None):
    _load_embeddings()
    emb_matrix, emb_ids = embedding_snapshot()
    if emb_matrix is None or len(emb_matrix) == 0:
        return []
    query_emb = _get_embedding(query)
    if query_emb is None:
        return []
    import numpy as np
    query_vec = np.array(query_emb)
    norms = np.linalg.norm(emb_matrix, axis=1)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    similarities = np.dot(emb_matrix, query_vec) / (norms * query_norm)
    scored = []
    for i, sim in enumerate(similarities):
        pid = emb_ids[i]
        paper = next((p for p in papers if (p.get("id") or p.get("entry_id")) == pid), None)
        if paper:
            scored.append((paper, float(sim)))
    scored.sort(key=lambda x: -x[1])
    if top_k:
        return scored[:top_k]
    return scored

def expand_query_by_embeddings(query, papers):
    top_papers = score_papers_by_embedding(papers, query, top_k=5)
    if not top_papers:
        return []
    titles = [p.get("title", "") for p, _ in top_papers]
    freq = word_freq(titles, min_len=4)
    query_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
    expanded = [w for w, _ in freq if w not in query_words][:5]
    return expanded

EMBEDDING_GRAPH_CACHE = ROOT / "data" / "embedding_graph.json"
KEYWORD_GRAPH_PATH = ROOT / "graphify-out" / "paper_graph.json"
MERGED_GRAPH_CACHE = ROOT / "data" / "merged_graph.json"

def build_embedding_graph(threshold=None, top_k=None):
    """Build a k-nearest-neighbour similarity graph over the paper embeddings.

    Absolute thresholds do not work on these vectors. Measured over 179,700 random
    pairs of the live corpus, cosine similarity is: p25 0.602, median 0.626, p95
    0.708. Unrelated papers already sit at ~0.63, so the old default of 0.65 kept
    28% of ALL pairs (avg degree 474) and the 0.70 the handler passed kept 6.5%
    (avg degree ~102, 76k edges). That is not a graph -- the d3 force layout never
    produced a single SVG element, and at that density it could only ever render as
    one blob.

    So each paper keeps its own top_k most similar neighbours instead. Rank is
    meaningful even when the absolute scale is not, every node gets edges, and the
    result does not need re-tuning if the embedding provider changes. `threshold`
    is kept as an optional floor for callers that want one.
    """
    import numpy as np
    _load_embeddings()
    embs, paper_ids = embedding_snapshot()
    if embs is None or len(embs) < 2:
        print("  Not enough embeddings to build graph.")
        return None

    k = GRAPH_TOP_K if top_k is None else max(1, min(int(top_k), 50))
    n = len(paper_ids)
    k = min(k, n - 1)
    print(f"  Building kNN similarity graph (n={n}, k={k})...")

    # float32 and an in-place normalise: the old code held three n x n float64
    # arrays at once (dot product, norms*norms.T, and the quotient), which is
    # ~2.4GB of transient at 10k papers. Normalising once means the dot product
    # alone is the cosine similarity.
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    np.divide(embs, np.maximum(norms, 1e-12), out=embs)
    zero_rows = int((norms.ravel() <= 1e-12).sum())
    if zero_rows:
        print(f"  Warning: {zero_rows} zero-norm embedding(s) excluded.")

    sim_matrix = embs @ embs.T
    np.fill_diagonal(sim_matrix, -1.0)  # never let a paper be its own neighbour

    # Adjacency is accumulated as edges are chosen, so a weight always belongs to
    # the pair it was computed from. Union-find went away with the switch to label
    # propagation: on a kNN graph everything lands in one connected component, so
    # counting components told us nothing.
    adjacency = [[] for _ in range(n)]

    # Each row keeps its k best neighbours. argpartition is O(n) per row rather
    # than a full sort. Pairs are de-duplicated so an edge appears once even when
    # both endpoints choose each other.
    floor = float(threshold) if threshold is not None else None
    seen_pairs = set()
    edges = []
    neighbour_idx = np.argpartition(-sim_matrix, k - 1, axis=1)[:, :k]
    for ri in range(n):
        for ci in neighbour_idx[ri]:
            ci = int(ci)
            if ci == ri:
                continue
            sim = float(sim_matrix[ri, ci])
            if not np.isfinite(sim) or sim <= -1.0:
                continue
            if floor is not None and sim < floor:
                continue
            pair = (ri, ci) if ri < ci else (ci, ri)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            adjacency[pair[0]].append((pair[1], sim))
            adjacency[pair[1]].append((pair[0], sim))
            edges.append({
                "source": paper_ids[pair[0]],
                "target": paper_ids[pair[1]],
                "weight": round(sim, 4),
                "type": "semantic"
            })

    # Communities by label propagation, not connected components.
    #
    # A kNN graph is essentially always connected, so counting components now
    # returns 1 -- and since the renderer colours nodes by community, every node
    # would come out the same colour. Label propagation finds actual clusters
    # inside a connected graph: each node repeatedly adopts the commonest label
    # among its neighbours. Deterministic here because the visit order is a fixed
    # permutation seeded off the node count rather than reshuffled per pass.
    labels = list(range(n))
    order = sorted(range(n), key=lambda i: (len(adjacency[i]), i))
    for _ in range(12):  # converges well before this on graphs of this size
        changed = 0
        for i in order:
            if not adjacency[i]:
                continue
            tally = {}
            for j, w in adjacency[i]:
                tally[labels[j]] = tally.get(labels[j], 0.0) + w
            # Highest weighted vote; lowest label id breaks ties for determinism.
            best = min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if labels[i] != best:
                labels[i] = best
                changed += 1
        if not changed:
            break

    # Compact the surviving labels to 0..m-1 so the colour scale stays small.
    remap = {}
    for i, lab in enumerate(labels):
        if lab not in remap:
            remap[lab] = len(remap)
        labels[i] = remap[lab]
    n_components = len(remap)
    print(f"  {len(edges)} edges, {n_components} communities "
          f"(avg degree {2 * len(edges) / max(1, n):.1f})")

    # Build paper lookup for title/year
    paper_lookup = {}
    for p in papers_global:
        pid = p.get("id") or p.get("entry_id")
        if pid:
            paper_lookup[pid] = p

    nodes = []
    for i, pid in enumerate(paper_ids):
        p = paper_lookup.get(pid, {})
        nodes.append({
            "id": pid,
            "title": p.get("title", ""),
            "year": (p.get("published") or "")[:4],
            "domain": "paper",
            "community": int(labels[i])
        })

    graph = {
        "directed": False,
        "multigraph": True,
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "type": "embedding_knn",
            "top_k": k,
            "threshold": floor,
            "embedding_provider": EMBEDDING_PROVIDER,
            "total_papers": len(paper_ids),
            "total_edges": len(edges),
            "avg_degree": round(2 * len(edges) / max(1, len(paper_ids)), 2),
            "communities": int(n_components),
            # Recorded so a cache built from a different corpus or a different
            # embedding set is detected instead of being served forever as current.
            # The on-disk cache said total_papers=1501 against a 1681-paper corpus.
            "corpus_papers": len(papers_global) if papers_global else 0,
            "embeddings_mtime": (EMBEDDING_CACHE.stat().st_mtime
                                 if EMBEDDING_CACHE.exists() else 0),
        }
    }

    if atomic_write_json(EMBEDDING_GRAPH_CACHE, graph):
        print(f"  Embedding graph saved: {len(nodes)} nodes, {len(edges)} edges, {n_components} communities")

    return graph


def _graph_cache_is_current(graph):
    """Reject a cache built from a different corpus, provider or edge rule.

    Caches used to be trusted on existence alone, so the on-disk graph (1501
    papers, threshold 0.7, no communities) was served indefinitely against a
    1681-paper corpus -- ~180 papers silently missing from what the user was shown.
    """
    meta = (graph or {}).get("meta") or {}
    if meta.get("type") != "embedding_knn":
        return False  # built by the old threshold rule
    if meta.get("embedding_provider") != EMBEDDING_PROVIDER:
        return False
    if meta.get("corpus_papers") != (len(papers_global) if papers_global else 0):
        return False
    on_disk = EMBEDDING_CACHE.stat().st_mtime if EMBEDDING_CACHE.exists() else 0
    return abs(float(meta.get("embeddings_mtime") or 0) - on_disk) < 1.0


def load_or_build_embedding_graph(threshold=None, top_k=None):
    if EMBEDDING_GRAPH_CACHE.exists():
        try:
            cached = json.load(open(EMBEDDING_GRAPH_CACHE, encoding="utf-8"))
            if _graph_cache_is_current(cached):
                return cached
            print("  Embedding graph cache is stale; rebuilding.")
        except Exception:
            pass
    return build_embedding_graph(threshold=threshold, top_k=top_k)


def load_or_build_merged_graph():
    if MERGED_GRAPH_CACHE.exists():
        try:
            return json.load(open(MERGED_GRAPH_CACHE))
        except Exception:
            pass
    merged = merge_graphs()
    if merged and "error" not in merged:
        atomic_write_json(MERGED_GRAPH_CACHE, merged)
    return merged


MAX_KEYWORD_GRAPH_BYTES = 64 * 1024 * 1024
_keyword_graph_cache = {"mtime": None, "data": None}


def _load_keyword_graph():
    """Load graphify's keyword graph, cached on mtime.

    It is a 22MB file that was re-read and re-parsed on every request that touched
    it, with the handle never closed.
    """
    if not KEYWORD_GRAPH_PATH.exists():
        return None
    try:
        stat = KEYWORD_GRAPH_PATH.stat()
        if stat.st_size > MAX_KEYWORD_GRAPH_BYTES:
            print(f"  Keyword graph too large ({stat.st_size} bytes); ignoring.")
            return None
        if _keyword_graph_cache["mtime"] == stat.st_mtime:
            return _keyword_graph_cache["data"]
        with open(KEYWORD_GRAPH_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        _keyword_graph_cache.update(mtime=stat.st_mtime, data=data)
        return data
    except Exception as e:
        print(f"  Could not load keyword graph: {e}")
        return None


def merge_graphs():
    emb_graph = load_or_build_embedding_graph()
    kw_graph = _load_keyword_graph()

    if emb_graph is None and kw_graph is None:
        return {"error": "No graphs available"}

    if emb_graph is None:
        merged = dict(kw_graph)
        for e in merged.get("edges", []):
            e["type"] = "keyword"
        return merged

    if kw_graph is None:
        return emb_graph

    # The external keyword graph is third-party output; validate rather than
    # subscript. A missing "id"/"source"/"target" used to raise KeyError straight
    # out of the handler.
    def _valid_nodes(graph):
        return [n for n in (graph.get("nodes") or [])
                if isinstance(n, dict) and isinstance(n.get("id"), str)]

    def _valid_edges(graph):
        return [e for e in (graph.get("edges") or [])
                if isinstance(e, dict) and isinstance(e.get("source"), str)
                and isinstance(e.get("target"), str)]

    all_nodes = {}
    for n in _valid_nodes(emb_graph):
        all_nodes[n["id"]] = n
    for n in _valid_nodes(kw_graph):
        if n["id"] not in all_nodes:
            # Normalise to the embedding node schema; keyword nodes carry
            # label/abstract and no community, and the renderer colours by
            # community. The abstract is dropped too -- copying it verbatim was
            # most of the 18MB payload.
            all_nodes[n["id"]] = {
                "id": n["id"],
                "title": n.get("title") or n.get("label") or "",
                "year": n.get("year") or "",
                "domain": n.get("domain") or "paper",
                "community": n.get("community", -1),
            }

    all_edges = []
    # id -> edge, so a duplicate is an O(1) lookup. This used to scan all_edges
    # linearly for every duplicate keyword edge: 13,525 duplicates against ~70-135k
    # edges is ~1e9 tuple comparisons, inside a GET handler.
    edge_index = {}

    def _key(e):
        return (e["source"], e["target"]) if e["source"] <= e["target"] else (e["target"], e["source"])

    for e in _valid_edges(emb_graph):
        k = _key(e)
        if k not in edge_index:
            e["type"] = "semantic"
            edge_index[k] = e
            all_edges.append(e)

    # The keyword graph is as dense as the old threshold-based semantic graph was:
    # 71,215 edges here, which makes the merged view exactly the hairball that kept
    # the semantic view from rendering. Apply the same rule -- each node keeps its
    # strongest keyword links -- so "merged" stays drawable. Edges that duplicate a
    # semantic edge are always kept, since those are the interesting ones.
    kw_edges = _valid_edges(kw_graph)
    kept_kw, per_node = [], collections.Counter()
    for e in sorted(kw_edges, key=lambda x: -(x.get("weight") or 0)):
        k = _key(e)
        if k in edge_index:
            kept_kw.append(e)      # overlaps a semantic edge; keep regardless
            continue
        if per_node[e["source"]] >= GRAPH_TOP_K or per_node[e["target"]] >= GRAPH_TOP_K:
            continue
        per_node[e["source"]] += 1
        per_node[e["target"]] += 1
        kept_kw.append(e)

    for e in kept_kw:
        k = _key(e)
        existing = edge_index.get(k)
        if existing is None:
            e["type"] = "keyword"
            edge_index[k] = e
            all_edges.append(e)
        elif existing.get("type") == "semantic":
            existing["type"] = "both"
            # Keep BOTH weights. Only `type` was upgraded before, so the keyword
            # weight and shared_concepts were discarded on all 13,525 shared edges
            # and the merged view's weights were purely semantic despite the label.
            existing["semantic_weight"] = existing.get("weight")
            existing["keyword_weight"] = e.get("weight")
            if e.get("shared_concepts"):
                existing["shared_concepts"] = e["shared_concepts"]

    # Drop edges whose endpoints aren't nodes; dangling ids reach the renderer.
    all_edges = [e for e in all_edges if e["source"] in all_nodes and e["target"] in all_nodes]

    return {
        "directed": False,
        "multigraph": True,
        "nodes": list(all_nodes.values()),
        "edges": all_edges,
        "meta": {
            "semantic_edges": len(emb_graph.get("edges", [])),
            "keyword_edges": len(kw_graph.get("edges", [])),
            "total_edges": len(all_edges),
            "total_nodes": len(all_nodes)
        }
    }


def hybrid_search(query, papers, top_k=30):
    import numpy as np
    import re as _re
    ql = query.lower()
    query_words = set(_re.findall(r'\b[a-zA-Z]{4,}\b', ql))

    keyword_scores = {}
    for p in papers:
        text = ((p.get('title') or '') + ' ' + (p.get('summary') or '')).lower()
        phrase_present = ql in text
        word_overlap = sum(1 for w in query_words if w in text)
        score = (3 if phrase_present else 0) + word_overlap
        if score > 0:
            keyword_scores[p.get("id") or p.get("entry_id")] = score

    _load_embeddings()
    semantic_scores = {}
    emb_matrix, emb_ids = embedding_snapshot()
    if emb_matrix is not None and len(emb_matrix) > 0:
        query_emb = _get_embedding(query)
        if query_emb is not None:
            query_vec = np.array(query_emb)
            norms = np.linalg.norm(emb_matrix, axis=1)
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                similarities = np.dot(emb_matrix, query_vec) / (norms * query_norm)
                for idx, pid in enumerate(emb_ids):
                    sim = float(similarities[idx])
                    if sim > 0:
                        semantic_scores[pid] = sim

    # Reciprocal Rank Fusion instead of weighted raw scores.
    #
    # The old formula was kw*0.4 + (sem*3.0)*0.6, but the two scales are not
    # comparable: semantic is a cosine so sem*3.0 is capped at 3.0, while keyword
    # is 3 + word_overlap and unbounded. On a six-word query keyword reaches ~9
    # (weighted 3.6) against semantic's maximum 1.8 -- so the nominal 40/60 split
    # INVERTED to keyword-dominant exactly when the query was most specific. It is
    # worse here than usual because these embeddings sit on a narrow band (median
    # pairwise cosine 0.626), so the semantic magnitudes barely vary at all.
    #
    # RRF combines by RANK, discarding the incomparable magnitudes, and needs no
    # weights to tune. k=60 is the standard constant.
    RRF_K = 60
    fused = {}

    def _fuse(scores, label):
        for rank, (pid, _s) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
            entry = fused.setdefault(pid, {"score": 0.0, "keyword_rank": None, "semantic_rank": None})
            entry["score"] += 1.0 / (RRF_K + rank + 1)
            entry[label] = rank + 1

    _fuse(keyword_scores, "keyword_rank")
    _fuse(semantic_scores, "semantic_rank")
    if not fused:
        return []

    known = {(p.get("id") or p.get("entry_id")) for p in papers}
    ranked = sorted(((pid, e) for pid, e in fused.items() if pid in known),
                    key=lambda x: (-x[1]["score"], str(x[0])))[:top_k]
    # Always 3-tuples. The old no-embeddings fallback returned 2-tuples while the
    # normal path returned 3, so the caller's `for pid, score, scores in ...`
    # raised ValueError whenever embeddings were missing but keywords matched.
    return [(pid, entry["score"], dict(entry,
                                       keyword_score=keyword_scores.get(pid, 0),
                                       semantic_score=semantic_scores.get(pid, 0)))
            for pid, entry in ranked]


# Term banks, Arabic detection and scraper query catalog: app/terms.py

ANALYSIS_CACHE = ROOT / "data" / "analysis_cache.json"
ANALYSIS_DIR = ROOT / "data" / "analyses"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Export field sanitisers ────────────────────────────────────────────────
# Every field below originates in a third-party API response or scraped HTML.

_CSV_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell(value):
    """Neutralise spreadsheet formula injection and flatten newlines.

    A cell beginning =, +, -, @, TAB or CR is executed as a formula by Excel and
    Google Sheets, so a scraped paper title of `=cmd|' /c calc'!A1` runs as the
    user who opens the export. Prefixing with an apostrophe forces text.
    """
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    if text[:1] in _CSV_FORMULA_LEAD:
        text = "'" + text
    return text


def _tex_escape(value):
    """Escape the LaTeX specials that break or silently truncate a .bib entry."""
    text = "" if value is None else str(value)
    text = text.replace("\\", r"\textbackslash{}")  # first, or it re-escapes the rest
    for ch, rep in (("{", r"\{"), ("}", r"\}"), ("%", r"\%"), ("$", r"\$"),
                    ("&", r"\&"), ("#", r"\#"), ("_", r"\_"),
                    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(ch, rep)
    return " ".join(text.split())


def _bibtex_key(paper, index):
    """A cite key that is unique and legal even when the paper has no id."""
    raw = str(paper.get("id") or paper.get("entry_id") or "")
    # "/" is legal in a BibTeX key and is part of every DOI, so keep it -- stripping
    # it silently rewrote 10.1016/j.tra as 10.1016j.tra and broke key/DOI identity.
    # BibTeX cannot handle , { } = or whitespace in a key; those go.
    key = re.sub(r"[^A-Za-z0-9:./_-]", "", raw)
    return key or f"paper{index}"


def _export_year(paper):
    pub = paper.get("published") or ""
    if not pub:
        return ""
    try:
        return str(datetime.fromisoformat(str(pub).replace("Z", "+00:00")).year)
    except Exception:
        return str(pub)[:4]


def atomic_write_json(path, obj, **dump_kwargs):
    """Write JSON via a temp file in the same directory, then os.replace.

    Every cache here was written in place with mode "w", which truncates first.
    Readers are on other threads (ThreadingHTTPServer) and readers of a
    half-written file get a JSONDecodeError that the surrounding `except:
    pass` turns into "no cache" -- so a concurrent read during a save silently
    discarded the analysis or, worse, the embeddings. The window is long: the
    analysis cache is ~1.2MB and embeddings.json ~71MB. os.replace is atomic, so
    a reader sees either the whole old file or the whole new one, and a crash
    mid-write leaves the previous file intact.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"Warning: could not write {path.name}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def save_analysis(analysis):
    atomic_write_json(ANALYSIS_CACHE, analysis, ensure_ascii=False)

def load_analysis():
    if ANALYSIS_CACHE.exists():
        try:
            return json.load(open(ANALYSIS_CACHE, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_paper_analysis(paper_id, analysis):
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))
    atomic_write_json(ANALYSIS_DIR / f"{safe_id}.json", analysis,
                      ensure_ascii=False, indent=2)

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
            # A per-paper failure is recorded on the paper, NOT on the job. Setting
            # job["error"] while status was still "running" made the frontend poller
            # (which checks d.error first) declare the whole run failed and stop
            # polling -- discarding the progress bar and the analyses already
            # completed, while this thread carried on spending on the rest.
            results.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "analysis": {"error": str(e)}
            })
            logger.warning(f"batch {job_id}: paper {paper.get('id')} failed: {e}")
        with batch_jobs_lock:
            batch_jobs[job_id]["progress"] = i + 1
            batch_jobs[job_id]["results"] = results
    failed = sum(1 for r in results if isinstance(r.get("analysis"), dict) and "error" in r["analysis"])
    with batch_jobs_lock:
        batch_jobs[job_id]["status"] = "done"
        batch_jobs[job_id]["progress"] = total
        batch_jobs[job_id]["failed"] = failed
        batch_jobs[job_id]["finished_at"] = time.time()
    _prune_batch_jobs()


BATCH_JOB_TTL = 3600


def _prune_batch_jobs():
    """Drop finished jobs after a TTL.

    Each entry holds the full results list for its papers, and nothing ever removed
    them, so a long-lived server accumulated every analysis it had ever run in RAM.
    """
    cutoff = time.time() - BATCH_JOB_TTL
    with batch_jobs_lock:
        stale = [jid for jid, j in batch_jobs.items()
                 if j.get("status") == "done" and (j.get("finished_at") or 0) < cutoff]
        for jid in stale:
            batch_jobs.pop(jid, None)


def _batch_job_running():
    with batch_jobs_lock:
        return any(j.get("status") == "running" for j in batch_jobs.values())

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

    _load_embeddings()
    emb_matrix, emb_ids = embedding_snapshot()
    if emb_matrix is not None and len(emb_ids) > 0:
        import numpy as np
        deduped_ids = [(p.get("id") or p.get("entry_id")) for p in deduped]
        deduped_to_emb = {}
        emb_pos = {pid: i for i, pid in enumerate(emb_ids)}  # was .index() per paper: O(n^2)
        for di, pid in enumerate(deduped_ids):
            if pid in emb_pos:
                deduped_to_emb[di] = emb_pos[pid]
        if len(deduped_to_emb) > 1:
            indices = list(deduped_to_emb.values())
            emb_vectors = emb_matrix[indices]
            norms = np.linalg.norm(emb_vectors, axis=1, keepdims=True)
            sim_matrix = np.dot(emb_vectors, emb_vectors.T) / (norms * norms.T)
            deduped_indices = list(deduped_to_emb.keys())
            remove = set()
            for i_idx in range(len(deduped_indices)):
                for j_idx in range(i_idx + 1, len(deduped_indices)):
                    if sim_matrix[i_idx][j_idx] > 0.92:
                        di = deduped_indices[i_idx]
                        dj = deduped_indices[j_idx]
                        pi = deduped[di]
                        pj = deduped[dj]
                        if len(pi.get("summary") or "") >= len(pj.get("summary") or ""):
                            sources_i = pi.get("sources") or [pi.get("source") or pi.get("id")]
                            sources_j = pj.get("sources") or [pj.get("source") or pj.get("id")]
                            pi["sources"] = list(set(sources_i + sources_j))
                            remove.add(dj)
                        else:
                            sources_i = pi.get("sources") or [pi.get("source") or pi.get("id")]
                            sources_j = pj.get("sources") or [pj.get("source") or pj.get("id")]
                            pj["sources"] = list(set(sources_i + sources_j))
                            remove.add(di)
            if remove:
                removed_emb = len(remove)
                deduped = [p for i, p in enumerate(deduped) if i not in remove]
                print(f"  Embedding dedup removed {removed_emb} cross-source duplicates")
    return deduped


def load_latest_scrape():
    """Load and deduplicate papers from the most recent scrape file."""
    files = list(RAW_DIR.glob("papers_*.json"))
    if not files:
        return None, None, None
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    try:
        with open(latest_file, "r", encoding="utf-8", errors="replace") as fp:
            data = json.load(fp)
            if not isinstance(data, list):
                return None, None, None
            papers = data
    except Exception as e:
        print(f"Warning: Could not load {latest_file}: {e}")
        return None, None, None

    seen_ids = set()
    unique = []
    for p in papers:
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
        print(f"  Title dedup removed {removed} near-duplicate papers from latest scrape")

    _load_embeddings()
    emb_matrix, emb_ids = embedding_snapshot()
    if emb_matrix is not None and len(emb_ids) > 0:
        import numpy as np
        deduped_ids = [(p.get("id") or p.get("entry_id")) for p in deduped]
        deduped_to_emb = {}
        emb_pos = {pid: i for i, pid in enumerate(emb_ids)}  # was .index() per paper: O(n^2)
        for di, pid in enumerate(deduped_ids):
            if pid in emb_pos:
                deduped_to_emb[di] = emb_pos[pid]
        if len(deduped_to_emb) > 1:
            indices = list(deduped_to_emb.values())
            emb_vectors = emb_matrix[indices]
            norms = np.linalg.norm(emb_vectors, axis=1, keepdims=True)
            sim_matrix = np.dot(emb_vectors, emb_vectors.T) / (norms * norms.T)
            deduped_indices = list(deduped_to_emb.keys())
            remove = set()
            for i_idx in range(len(deduped_indices)):
                for j_idx in range(i_idx + 1, len(deduped_indices)):
                    if sim_matrix[i_idx][j_idx] > 0.92:
                        di = deduped_indices[i_idx]
                        dj = deduped_indices[j_idx]
                        pi = deduped[di]
                        pj = deduped[dj]
                        if len(pi.get("summary") or "") >= len(pj.get("summary") or ""):
                            sources_i = pi.get("sources") or [pi.get("source") or pi.get("id")]
                            sources_j = pj.get("sources") or [pj.get("source") or pj.get("id")]
                            pi["sources"] = list(set(sources_i + sources_j))
                            remove.add(dj)
                        else:
                            sources_i = pi.get("sources") or [pi.get("source") or pi.get("id")]
                            sources_j = pj.get("sources") or [pj.get("source") or pj.get("id")]
                            pj["sources"] = list(set(sources_i + sources_j))
                            remove.add(di)
            if remove:
                removed_emb = len(remove)
                deduped = [p for i, p in enumerate(deduped) if i not in remove]
                print(f"  Embedding dedup removed {removed_emb} cross-source duplicates")

    timestamp = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
    return deduped, latest_file.name, timestamp

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
        stopwords = stopwords | ARABIC_STOPWORDS
    # [a-zA-Z] matched ASCII letters only, so every Arabic token was silently
    # dropped and this -- the headline keyword analysis of a MENA-focused tool --
    # was structurally English-only. "naive"/"cafe" lost their accented forms too.
    # [^\W\d_] is "any unicode letter", so Arabic and accented Latin both survive.
    pattern = re.compile(r"[^\W\d_]{%d,}" % min_len, re.UNICODE)
    words = []
    for text in texts:
        for token in pattern.findall((text or "").lower()):
            token = _normalise_arabic(token)
            if token and token not in stopwords:
                words.append(token)
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

# The frontend abandons a POST at 330s. retry(2) x a 300s per-call timeout x 4
# models in the fallback chain was ~2400s worst case, so the browser gave up, the
# user retried and started a SECOND full chain, while the abandoned first one kept
# billing and holding a worker thread for ~40 minutes.
LLM_DEADLINE_SECONDS = 240
LLM_CALL_TIMEOUT = 110


@retry(max_attempts=2, base_delay=1, backoff=2)
def _llm_call_single(messages, model, max_tokens=600, temperature=0.3, timeout=None):
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
        with urllib.request.urlopen(req, timeout=timeout or LLM_CALL_TIMEOUT) as resp:
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
    deadline = time.monotonic() + LLM_DEADLINE_SECONDS
    for model in models_to_try:
        remaining = deadline - time.monotonic()
        if remaining <= 5:
            last_error = last_error or "LLM deadline exceeded"
            print(f"  LLM deadline reached; not trying {model}.")
            break
        try:
            result = _llm_call_single(messages, model, max_tokens, temperature,
                                      timeout=min(LLM_CALL_TIMEOUT, remaining))
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
    prompt = get_prompt("summarise", prompt)   # prompts.json override
    result = llm_call([{"role": "user", "content": prompt}], max_tokens=2000)
    if "error" in result:
        return result
    # `or`, not a get default: the reasoning model returns content=None, and
    # .get("content", "") hands back that None rather than the default.
    content = result.get("content") or result.get("reasoning") or ""
    parsed = _parse_json_object(content)
    if parsed is None:
        return {"error": "Failed to parse LLM response", "raw": content[:300]}
    if paper_id and "error" not in parsed:
        save_paper_analysis(paper_id, parsed)
    return parsed

def _parse_json_object(content):
    """Pull a JSON object out of an LLM reply, or None.

    The old strip was anchored (^``` and ```$), so a reasoning model prefixing
    "Here is the JSON:" or adding a trailing note failed to parse and burned the
    whole call. json.loads("null") or "42" also used to sail past the
    JSONDecodeError guard and then raise AttributeError on .get(), turning a bad
    reply into a 500 rather than a handled error.
    """
    if not content:
        return None
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _existing_cluster_list(existing):
    """Accept either the cluster list or the whole {clusters, unclustered} object.

    The frontend sends its entire stored object as `existing`, but this used to do
    `for cl in existing_clusters`, which iterates a dict's KEYS -- so the first
    iteration called "clusters".get(...) and raised AttributeError. Clustering has
    therefore never succeeded on a second run; the button 500'd every time.
    """
    if isinstance(existing, dict):
        existing = existing.get("clusters")
    if not isinstance(existing, list):
        return []
    return [c for c in existing if isinstance(c, dict)]


def llm_cluster_papers(papers, max_n=50, existing_clusters=None):
    papers = papers[:max_n]
    # Carry each paper's ORIGINAL index. Filtering out already-clustered papers
    # renumbered the survivors 0..m-1, while the frontend maps the returned indices
    # into the unfiltered PAPERS array -- so every index after the first run pointed
    # at the wrong paper.
    indexed = list(enumerate(papers))
    already = set()
    for cl in _existing_cluster_list(existing_clusters):
        already.update(i for i in cl.get("paper_indices", []) if isinstance(i, int))
    if already:
        indexed = [(i, p) for i, p in indexed if i not in already]
    if not indexed:
        return {"clusters": [], "unclustered": []}

    if len(indexed) <= 50:
        return _cluster_batch(indexed)

    all_clusters = []
    seen_indices = set()
    unclustered = set()
    batch_size, step = 50, 40
    for start in range(0, len(indexed), step):
        batch = indexed[start:start + batch_size]
        if len(batch) < 10:
            break
        result = _cluster_batch(batch)
        if "error" in result:
            return result
        for cl in result.get("clusters", []):
            match = next((c for c in all_clusters if c["name"] == cl["name"]), None)
            target = match if match else dict(cl, paper_indices=[])
            for idx in cl.get("paper_indices", []):
                if idx not in seen_indices:
                    target["paper_indices"].append(idx)
                    seen_indices.add(idx)
            if match is None:
                all_clusters.append(target)
        # Collected in the SAME pass. This used to run the whole batch loop a second
        # time purely to gather `unclustered`, doubling the LLM calls (and the bill)
        # on the most expensive endpoint, with a fresh non-deterministic generation
        # that disagreed with the clusters just returned.
        unclustered.update(result.get("unclustered", []))
    return {"clusters": all_clusters, "unclustered": sorted(unclustered - seen_indices)}

def _cluster_batch(indexed_papers):
    """Cluster one batch. `indexed_papers` is [(original_index, paper), ...].

    The model is shown local positions 0..n-1 and its answer is mapped back onto
    the original corpus indices, so filtering earlier in the pipeline can no longer
    shift what a returned index refers to.
    """
    local_to_original = [orig for orig, _ in indexed_papers]
    paper_summaries = []
    for i, (_orig, p) in enumerate(indexed_papers):
        title = p.get('title') or ''
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
    prompt = get_prompt("cluster", prompt)     # prompts.json override
    result = llm_call([{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.2)
    if "error" in result:
        return result
    content = result.get("content") or result.get("reasoning") or ""
    parsed = _parse_json_object(content)
    if parsed is None:
        return {"error": "Failed to parse LLM response", "raw": content[:300]}

    n = len(local_to_original)

    def _map(indices):
        """Map model-supplied local positions to original indices, dropping junk.

        Indices came straight back from the model with no bounds check, so a
        hallucinated index sailed through and the frontend used it to subscript
        PAPERS.
        """
        out = []
        for i in indices if isinstance(indices, list) else []:
            if isinstance(i, bool) or not isinstance(i, int):
                continue
            if 0 <= i < n:
                out.append(local_to_original[i])
        return out

    clusters = []
    for cl in (parsed.get("clusters") or []):
        if not isinstance(cl, dict):
            continue
        mapped = _map(cl.get("paper_indices"))
        if not mapped:
            continue
        clusters.append({
            "name": str(cl.get("name") or "Unnamed cluster")[:80],
            "description": str(cl.get("description") or "")[:400],
            "paper_indices": mapped,
        })
    return {"clusters": clusters, "unclustered": _map(parsed.get("unclustered"))}

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
    papers_context = list(papers_context)[:MAX_CONTEXT_PAPERS]
    context_parts = []
    for i, p in enumerate(papers_context):
        # Was 500, which is roughly the first third of a typical 1200-1800 char
        # abstract -- cut mid-sentence, usually before the findings. MAX_ABSTRACT_CHARS
        # keeps the whole abstract for almost every paper while still bounding the prompt.
        abstract = (p.get('summary') or '')[:MAX_ABSTRACT_CHARS]
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
    system_msg = get_prompt("chat_system", system_msg)   # prompts.json override
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

MIN_PLAUSIBLE_YEAR = 1900

# Common Arabic function words. Without these the Arabic keyword ranking would be
# dominated by "من/في/على" exactly as an English one would be by "the/and/of".
ARABIC_STOPWORDS = {
    "من", "في", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "كان", "كانت", "قد", "لم", "لا", "ما", "أن", "إن", "او", "أو",
    "ثم", "كما", "بين", "بعد", "قبل", "عند", "حيث", "كل", "بعض", "غير",
    "هو", "هي", "هم", "نحن", "انت", "أنت", "لكن", "حتى", "اذا", "إذا",
}

# Alef and yeh/teh-marbuta variants are written inconsistently across sources, so
# the same word arrives in several forms and would otherwise be counted separately.
_ARABIC_NORMALISE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_HAS_ARABIC = re.compile(r"[؀-ۿ]")


def _normalise_arabic(token):
    if not _HAS_ARABIC.search(token):
        return token
    return _ARABIC_DIACRITICS.sub("", token).translate(_ARABIC_NORMALISE)


def _author_names(paper):
    """Authors as a clean list of strings, whatever shape the source produced."""
    raw = paper.get("authors")
    if isinstance(raw, str):
        return [a.strip() for a in raw.split(";") if a.strip()]
    if not isinstance(raw, list):
        return []
    names = []
    for a in raw:
        if isinstance(a, str) and a.strip():
            names.append(a.strip())
        elif isinstance(a, dict):
            name = a.get("name") or f"{a.get('given','')} {a.get('family','')}".strip()
            if name:
                names.append(name)
    return names


def _paper_brief(paper):
    """Summary record used by concept_clusters and most_collaborative.

    Both used to subscript p["id"] and p["title"] directly. analyze_papers runs at
    startup BEFORE the socket binds, so one record missing either key took the whole
    server down with a traceback and no port ever opened.
    """
    names = _author_names(paper)
    return {
        "id": paper.get("id") or paper.get("entry_id") or "",
        "title": paper.get("title") or "",
        "year": (paper.get("published") or "")[:4],
        "authors": names,
        "count": len(names),
    }


def _plausible_year(year):
    """Papers dated in the future are scraper artefacts, not publications.

    The live corpus carries 2116, 2117 and 2121 dates, which put phantom buckets on
    the timeline and made date_range read "2008-02-01 -> 2121-01-01".
    """
    return MIN_PLAUSIBLE_YEAR <= year <= datetime.now().year + 1


def analyze_papers(papers):
    results = {"generated_at": datetime.now().isoformat(), "total_papers": len(papers)}
    years = Counter()
    months = Counter()
    dates_raw = []
    implausible = 0
    for p in papers:
        pub = p.get("published") or ""   # .get(k, "") returns None on a JSON null
        if pub:
            try:
                dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                dt = dt.replace(tzinfo=None)
                if not _plausible_year(dt.year):
                    implausible += 1
                    continue
                years[dt.year] += 1
                months[f"{dt.year}-{dt.month:02d}"] += 1
                dates_raw.append(dt)
            except Exception:
                pass
    if implausible:
        results["implausible_dates"] = implausible
    results["yearly_distribution"] = dict(years.most_common())
    results["monthly_distribution"] = dict(sorted(months.items()))
    results["date_range"] = {
        "earliest": str(min(dates_raw).date()) if dates_raw else None,
        "latest": str(max(dates_raw).date()) if dates_raw else None,
    }
    # authors can arrive as null (TypeError on iteration), as a "A; B" string
    # (iterates characters, filling top_authors with single letters), or as a list
    # of dicts (unhashable -> TypeError). Coerce once, here.
    all_authors = []
    author_counts = Counter()
    author_totals = []
    for p in papers:
        names = _author_names(p)
        author_totals.append(len(names))
        for a in names:
            all_authors.append(a)
            author_counts[a] += 1
    results["total_authors"] = len(set(all_authors))
    results["avg_authors_per_paper"] = round(
        sum(author_totals) / len(papers), 2
    ) if papers else 0
    results["top_authors"] = author_counts.most_common(20)
    results["top_title_keywords"] = word_freq([(p.get("title") or "") for p in papers], min_len=3)[:30]
    results["top_abstract_keywords"] = word_freq([(p.get("summary") or "") for p in papers], min_len=4)[:30]
    combined = " ".join(((p.get("title") or "") + " " + (p.get("summary") or "")).lower() for p in papers)
    # Word-boundary counts, not substring: "mena" was matching phenoMENA and "arab"
    # matched ARABidopsis, inflating every regional figure.
    results["behavioural_term_freq"] = _count_terms(combined, BEHAVIOURAL_TERMS)
    # Place names only. MIDDLE_EAST_TERMS also carries generic sociology vocabulary
    # (context, cultural, gender, women, religion, oil, youth...), which appears in
    # behavioural-science abstracts constantly and swamped the panel: the live top
    # six read context 272, women 145, cultural 132, arab 101, oil 68, gender 65 --
    # measuring vocabulary, not the region.
    results["region_term_freq"] = _count_terms(combined, MENA_PLACE_TERMS)
    results["region_context_term_freq"] = _count_terms(combined, MENA_CONTEXT_TERMS)

    clusters = {}
    for term in BEHAVIOURAL_TERMS:
        pattern = _term_pattern(term)
        related = [_paper_brief(p) for p in papers
                   if pattern.search(((p.get("title") or "") + " " + (p.get("summary") or "")).lower())]
        if len(related) >= 2:
            clusters[term] = related
    results["concept_clusters"] = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    results["most_collaborative"] = sorted(
        (_paper_brief(p) for p in papers), key=lambda x: -x["count"])[:10]
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

# Tashkeel (harakat), dagger alif, and tatweel: presentation marks that make the
# same Arabic word compare unequal byte-for-byte.
_ARABIC_NOISE = re.compile(r'[ً-ْٰـ]')
# Any letter in the Arabic blocks (base + supplement + extended-A).
_ARABIC_CHARS = re.compile(r'[؀-ۿݐ-ݿࢠ-ࣿ]')
# Unicode-aware word: letters only, 2+ chars. \w alone would admit digits/underscore.
_SEARCH_WORD = re.compile(r'[^\W\d_]{2,}')

def _normalize_arabic(text):
    """Fold the Arabic orthographic variants users type interchangeably."""
    text = _ARABIC_NOISE.sub('', text)
    return (text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
                .replace('ة', 'ه').replace('ى', 'ي'))

def search_papers(papers, query, fields=None):
    if fields is None:
        fields = ["title", "summary", "authors"]
    ql = _normalize_arabic(query.lower()).strip()
    # \b[a-zA-Z]\b tokenising meant an Arabic query produced ZERO query words and
    # scored nothing; and bare substring matching let "art" hit "particle".
    query_words = set(_SEARCH_WORD.findall(ql))
    phrase_re = re.compile(r'(?<!\w)' + re.escape(ql) + r'(?!\w)') if ql else None
    # One shared word was enough to include a paper, so "nudge theory in
    # healthcare" returned every abstract containing "theory" (~1500 of 1700).
    # Require at least half the query words before a keyword match counts.
    required_overlap = max(1, (len(query_words) + 1) // 2)
    keyword_scores = {}
    for p in papers:
        text = _normalize_arabic(" ".join(str(p.get(f, "") or "") for f in fields).lower())
        score = 2 if (phrase_re and phrase_re.search(text)) else 0
        text_words = set(_SEARCH_WORD.findall(text))
        word_overlap = len(query_words & text_words)
        if word_overlap >= required_overlap:
            score += min(word_overlap, 3)
        score = min(score, 3)
        if score > 0:
            keyword_scores[p.get("id") or p.get("entry_id")] = score
    _load_embeddings()
    embedding_scores = {}
    emb_matrix, emb_ids = embedding_snapshot()
    if emb_matrix is not None and len(emb_matrix) > 0:
        query_emb = _get_embedding(query)
        if query_emb is not None:
            import numpy as np
            query_vec = np.array(query_emb)
            norms = np.linalg.norm(emb_matrix, axis=1)
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                similarities = np.dot(emb_matrix, query_vec) / (norms * query_norm)
                # `sim > 0` admitted the entire corpus: with these embeddings even
                # an off-topic query scores ~0.5 cosine against every paper
                # (measured: min 0.32, median 0.50 for "quantum computing" on a
                # behavioural-science corpus), so every search returned ~1700
                # "results". Absolute thresholds don't separate on- from off-topic
                # here; the margin above the query's own corpus median does. Keep
                # only papers at least halfway from the median to the best match.
                med = float(np.median(similarities))
                spread = max(float(similarities.max()) - med, 1e-6)
                for idx, pid in enumerate(emb_ids):
                    rel = (float(similarities[idx]) - med) / spread
                    if rel >= 0.5:
                        embedding_scores[pid] = rel * 3.0
    combined = []
    for p in papers:
        pid = p.get("id") or p.get("entry_id")
        kw = keyword_scores.get(pid, 0)
        emb = embedding_scores.get(pid, None)
        if emb is not None:
            final = kw * 0.3 + emb * 0.7
        else:
            final = kw
        if final > 0:
            d = dict(p)
            d["_search_score"] = round(final, 4)
            combined.append((final, d))
    if combined:
        combined.sort(key=lambda x: -x[0])
        return [p for _, p in combined]
    # Fallback: word-start prefix matching. Anchoring at a word boundary keeps
    # partial typing working ("behav" finds "behaviour") without the mid-word
    # false positives bare substring gave ("art" matched "particle").
    prefix_re = re.compile(r'(?<!\w)' + re.escape(ql)) if ql else None
    results = []
    for p in papers:
        for field in fields:
            val = p.get(field, "")
            if isinstance(val, list):
                val = " ".join(val)
            if val is None:
                val = ""
            if prefix_re and prefix_re.search(_normalize_arabic(val.lower())):
                d = dict(p)
                d["_search_score"] = 1.0
                results.append(d)
                break
    return results


def _ensure_behavioural_query(query_key):
    if query_key in SCRAPER_QUERIES:
        return query_key
    # An Arabic query is already region-specific, and no source indexes papers
    # containing BOTH Arabic script and these English terms — the AND wrap turned
    # every Arabic scrape into a guaranteed zero-result run.
    if _ARABIC_CHARS.search(query_key):
        return query_key
    return f"({BEHAVIOURAL_QUERY_PREFIX}) AND ({query_key})"

def run_scraper(query_key, count, sources=None, manage_flag=True):
    """Scrape one query group.

    manage_flag=False when called from _run_scraper_queue, which owns the shared
    running flag and the accumulated output across a multi-query run.
    """
    global scraper_status, papers_global, analysis_global
    behavioural_query = _ensure_behavioural_query(query_key)
    header = f"Starting scrape: {query_key} -> {behavioural_query[:100]}... ({count} papers)\n"
    if manage_flag:
        scraper_status["running"] = True
        scraper_status["output"] = header
    else:
        scraper_status["output"] += header
    scraper_status["returncode"] = None
    log_scraper(f"Starting scrape: {query_key} ({count} papers)")
    log_scraper(f"Behavioural query: {behavioural_query}")
    try:
        cmd = [sys.executable, "-u", "scraper.py", "-q", behavioural_query, "-n", str(count)]
        if sources:
            cmd.extend(["--sources", ",".join(sources)])
        # encoding= is mandatory here: scraper.py reconfigures its stdout to UTF-8,
        # but text=True alone decodes the pipe with the locale codec (cp1252 on
        # Windows), which raises UnicodeDecodeError on bytes like 0x81 — i.e. the
        # letter ف — killing the whole scrape for most Arabic queries.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            bufsize=1,
        )
        output_lines = []
        start_time = time.time()
        timeout = 300
        while True:
            if proc.poll() is not None:
                break
            if time.time() - start_time > timeout:
                proc.kill()
                log_scraper("Scraper timed out after 5 minutes")
                scraper_status["output"] += "\nScraper timed out after 5 minutes."
                scraper_status["returncode"] = -1
                scraper_status["running"] = False
                return
            line = proc.stdout.readline()
            if line:
                line = line.rstrip()
                output_lines.append(line)
                scraper_status["output"] += line + "\n"
                log_scraper(line)
            else:
                time.sleep(0.1)
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            scraper_status["output"] += line + "\n"
            log_scraper(line)
        returncode = proc.wait()
        scraper_status["returncode"] = returncode
        if returncode == 0:
            log_scraper("Scraper completed successfully, reloading papers...")
            papers_global = load_papers()
            analysis_global = analyze_papers(papers_global)
            save_analysis(analysis_global)
            msg = f"\nDone. Total papers: {len(papers_global)}"
            scraper_status["output"] += msg
            log_scraper(msg.strip())
            # Delta-embed before the relevance report below so the new papers'
            # vectors are already in the matrix it scores against. Wrapped so a
            # provider hiccup can never turn a successful scrape into a failure.
            try:
                embedded_count = embed_new_papers(papers_global)
                if embedded_count:
                    emb_msg = f"Embedded {embedded_count} new papers"
                    scraper_status["output"] += "\n" + emb_msg
                    log_scraper(emb_msg)
            except Exception as e:
                log_scraper(f"Auto-embed of new papers failed: {e}")
            try:
                _load_embeddings()
                _emb, _eids = embedding_snapshot()
                if _emb is not None and len(_eids) > 0:
                    scored = score_papers_by_embedding(papers_global, query_key)
                    high_rel = sum(1 for _, s in scored if s > 0.7)
                    if high_rel > 0:
                        rel_msg = f"Embedding relevance: {high_rel} papers with similarity > 0.7"
                        scraper_status["output"] += "\n" + rel_msg
                        log_scraper(rel_msg)
            except Exception:
                pass
        else:
            err_msg = f"\nScraper failed with return code {returncode}"
            scraper_status["output"] += err_msg
            log_scraper(err_msg.strip())
    except Exception as e:
        err_msg = f"\nError: {e}"
        scraper_status["output"] += err_msg
        scraper_status["returncode"] = -1
        log_scraper(f"Error: {e}")
    if manage_flag:
        scraper_status["running"] = False


def _translate_query_to_english(query):
    """One-shot LLM translation of an Arabic query into a short English academic
    search query. Returns the English string, or None on any failure (no API
    key, timeout, empty reply, reply still in Arabic).
    """
    prompt = (
        "Translate this Arabic academic search query into a short English "
        "academic search query (a few keywords, no boolean operators, no "
        "quotes). Reply with ONLY the English query, nothing else.\n\n"
        f"Arabic query: {query}"
    )
    try:
        # max_tokens=2000: the reasoning model returns content=None when the
        # token budget is too small, even for a one-line answer (see CLAUDE.md).
        result = llm_call([{"role": "user", "content": prompt}],
                          max_tokens=2000, temperature=0.0)
    except Exception:
        return None
    if "error" in result:
        return None
    content = (result.get("content") or "").strip()
    # Reasoning models sometimes wrap the answer in extra prose or quotes;
    # keep only the first non-empty line.
    lines = [ln.strip().strip('"\'') for ln in content.splitlines() if ln.strip()]
    english = lines[0] if lines else ""
    # A reply that still contains Arabic script (or rambles) is a failed
    # translation, not a usable search query.
    if not english or len(english) > 300 or _ARABIC_CHARS.search(english):
        return None
    return english


def expand_queries(queries):
    """Return a NEW queue: the input queries plus an English translation
    appended for each Arabic-script query.

    arXiv and PubMed index English only, so an Arabic query silently gets zero
    results from them. Expanding — never replacing — keeps the Arabic query's
    OpenAlex/CrossRef/SemanticScholar coverage AND adds English-index coverage.
    Any translation failure is non-fatal: the Arabic query still runs on its
    own. The input list is not mutated.
    """
    expanded = list(queries)
    for q in queries:
        if not _ARABIC_CHARS.search(q):
            continue
        english = _translate_query_to_english(q)
        if not english:
            note = f"Arabic query expansion skipped for: {q}"
            scraper_status["output"] += note + "\n"
            log_scraper(note)
            continue
        # Don't queue a translation that duplicates an existing query.
        if any(english.casefold() == existing.casefold() for existing in expanded):
            continue
        note = f"Arabic query expanded: {q} -> {english}"
        scraper_status["output"] += note + "\n"
        log_scraper(note)
        expanded.append(english)
    return expanded


def _run_scraper_queue(queries, count, sources=None):
    """Run query groups one at a time on a single worker thread.

    /api/scraper/run used to start one thread per query, all sharing the single
    global scraper_status dict. Consequences, with 17 presets selectable in the UI:
      - output interleaved and returncode raced between threads
      - the first query to finish set running=False, so the UI reported "Scraper
        finished" and stopped polling while the others were still going
      - N concurrent scraper.py subprocesses hammered the same APIs, which defeats
        any per-process rate limiting and invites 429s
      - N concurrent rebuilds of papers_global / analysis_global for ~1697 papers
    """
    scraper_status["running"] = True
    scraper_status["output"] = ""
    scraper_status["returncode"] = None
    # After the output reset, so the expansion notes survive in the status log.
    queries = expand_queries(queries)
    failed = 0
    try:
        for i, q in enumerate(queries, 1):
            banner = f"\n=== [{i}/{len(queries)}] {q} ===\n"
            scraper_status["output"] += banner
            log_scraper(banner.strip())
            try:
                run_scraper(q, count, sources, manage_flag=False)
            except Exception as e:
                failed += 1
                msg = f"\nQuery {q!r} failed: {type(e).__name__}: {e}"
                scraper_status["output"] += msg
                log_scraper(msg.strip())
                continue
            if scraper_status.get("returncode") not in (0, None):
                failed += 1
        # Don't let the UI claim success when some groups failed.
        if failed:
            scraper_status["output"] += f"\n{failed} of {len(queries)} query group(s) failed."
            if scraper_status.get("returncode") == 0:
                scraper_status["returncode"] = 1
    finally:
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
        # No Access-Control-Allow-Origin. The UI is served by this same server, so
        # it is same-origin and needs none; the wildcard that used to be here let
        # ANY page the user visited read every endpoint cross-origin -- including
        # the corpus, the scraper logs, and (before the fix above) the API key.
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._headers_sent = True
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            raise

    def _rate_check(self):
        ip = self._get_client_ip()
        if not rate_limiter.allow(ip):
            retry_after = rate_limiter.retry_after(ip)
            self.send_response(429)
            self.send_header("Retry-After", str(retry_after))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Rate limit exceeded. Try again later."}).encode())
            return False
        return True

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)
        logger.info(f"GET {path} from {self.client_address[0]}")
        try:
            if path == "/api/latest_scrape":
                papers, source_file, timestamp = load_latest_scrape()
                if papers is None:
                    self._json({"papers": [], "count": 0, "error": "No scrape files found"})
                    return
                self._json({
                    "papers": papers,
                    "source_file": source_file,
                    "count": len(papers),
                    "timestamp": timestamp
                })
                return

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

            elif path == "/api/embedding_status":
                meta = {}
                if EMBEDDING_META_CACHE.exists():
                    try:
                        meta = json.load(open(EMBEDDING_META_CACHE))
                    except Exception:
                        pass
                _load_embeddings()
                emb_matrix, emb_ids = embedding_snapshot()
                has_embeddings = emb_matrix is not None and len(emb_ids) > 0
                total = len(papers_global) if papers_global else 0
                count = len(emb_ids) if has_embeddings else 0
                # Only "ready" if we have embeddings for ALL papers
                ready = has_embeddings and count == total and total > 0
                self._json({
                    "ready": ready,
                    "stale": has_embeddings and count < total,
                    "provider": meta.get("provider", EMBEDDING_PROVIDER),
                    "model": meta.get("model", ""),
                    "count": count,
                    "total_papers": total,
                })
                return

            elif path == "/api/embedding_graph":
                try:
                    logger.info("  embedding_graph handler ENTERED")
                    merged = params.get("merged", ["0"])[0] == "1"
                    rebuild = params.get("rebuild", ["0"])[0] == "1"
                    # Edge count is governed by top_k now, not a cosine cutoff.
                    try:
                        top_k = int(params.get("top_k", [str(GRAPH_TOP_K)])[0])
                    except (TypeError, ValueError):
                        top_k = GRAPH_TOP_K
                    logger.info(f"  merged={merged}, rebuild={rebuild}, top_k={top_k}")
                    if rebuild:
                        graph = build_embedding_graph(top_k=top_k)
                        if MERGED_GRAPH_CACHE.exists():
                            MERGED_GRAPH_CACHE.unlink()
                        if graph is None:
                            self._json({"error": "Not enough embeddings to build graph."})
                            return
                    else:
                        graph = load_or_build_embedding_graph()
                    logger.info(f"  graph loaded: type={type(graph).__name__}")
                    if graph is None:
                        self._json({"error": "Embedding graph not available. Build embeddings first."})
                        return
                    if merged:
                        result = load_or_build_merged_graph()
                    else:
                        result = graph
                    logger.info(f"  result type={type(result).__name__}, keys={list(result.keys())}")
                    self._json(result)
                    logger.info(f"  _json completed")
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    raise
                except Exception as e:
                    logger.error(f"  embedding_graph handler error: {type(e).__name__}: {e}")
                    if not getattr(self, '_headers_sent', False):
                        self._json({"error": f"Graph error: {str(e)}"}, status=500)
                return

            elif path == "/api/expand_query":
                q = params.get("q", [""])[0]
                if not q or len(q) < 2:
                    self._json({"query": q, "expanded_terms": []})
                    return
                expanded = expand_query_by_embeddings(q, papers_global)
                self._json({"query": q, "expanded_terms": expanded})
                return

            elif path == "/api/graph_stats":
                # Cache-only. This used to call load_or_build_embedding_graph(), so a
                # stats request on a cold cache synchronously ran the full graph
                # build with no timeout -- and every refresh spawned another thread
                # starting its own build.
                graph = None
                if EMBEDDING_GRAPH_CACHE.exists():
                    try:
                        cached = json.load(open(EMBEDDING_GRAPH_CACHE, encoding="utf-8"))
                        if _graph_cache_is_current(cached):
                            graph = cached
                    except Exception:
                        graph = None
                if graph is None or "error" in graph:
                    self._json({"ready": False, "nodes": 0, "edges": 0,
                                "communities": 0, "avg_degree": 0})
                    return
                nodes = len(graph.get("nodes", []))
                edges = len(graph.get("edges", []))
                communities = graph.get("meta", {}).get("communities", 0)
                avg_degree = round(2 * edges / nodes, 2) if nodes > 0 else 0
                self._json({"ready": True, "nodes": nodes, "edges": edges,
                            "communities": communities, "avg_degree": avg_degree})
                return

            elif path == "/api/papers":
                q = params.get("q", [""])[0]
                year = params.get("year", [""])[0]
                term = params.get("term", [""])[0]
                hybrid = params.get("hybrid", ["0"])[0] == "1"
                result = list(papers_global)
                if q and len(q) >= 2:
                    if hybrid:
                        _load_embeddings()
                        _emb, _ = embedding_snapshot()
                        if _emb is not None and len(_emb) > 0:
                            hybrid_results = hybrid_search(q, result, top_k=50)
                            # A bare next() raised StopIteration -> 500 whenever a
                            # ranked id was no longer in the corpus (stale embeddings
                            # vs a deduped corpus -- exactly what embedding_status
                            # calls "stale"). Build one lookup and skip misses.
                            by_id = {(p.get("id") or p.get("entry_id")): p for p in result}
                            result = [by_id[pid] for pid, _, _ in hybrid_results if pid in by_id]
                        else:
                            result = search_papers(result, q)
                    else:
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
                # Never return the key itself. This used to send
                # {"key": "sk-or-v1-..."} in full, and with the wildcard CORS
                # header that was below, any page the user visited could read it
                # with a plain fetch() -- a simple GET needs no preflight. The UI
                # only ever consumes has_key and model.
                cfg = load_config()
                key = cfg.get("openrouter_api_key", "") or ""
                self._json({
                    "has_key": bool(key),
                    "key_suffix": key[-4:] if len(key) >= 4 else "",
                    "model": LLM_MODEL,
                })
                return

            elif path == "/api/scraper/status":
                self._json({
                    "running": scraper_status.get("running", False),
                    "output": scraper_status.get("output", ""),
                    "returncode": scraper_status.get("returncode")
                })
                return

            elif path == "/api/paper_graph":
                # Moved here from do_POST, where it was unreachable by GET and 404'd
                # through the static handler. Cached rather than re-reading and
                # re-serialising a 22MB file per request.
                graph_data = _load_keyword_graph()
                if graph_data is None:
                    self._json({"error": "Paper graph not found. Build with: "
                                         "python build_paper_graph.py"}, status=404)
                else:
                    self._json(graph_data)
                return

            elif path == "/api/health":
                # Tiny liveness probe. The Tools tab used to ping by fetching
                # /api/papers, downloading the entire ~2.2MB corpus to measure
                # latency.
                self._json({"ok": True, "papers": len(papers_global) if papers_global else 0})
                return

            elif path == "/api/logs":
                self._json({
                    "scraper_logs": get_scraper_logs(),
                    # Name only: the absolute path disclosed the OS username.
                    "server_log_path": LOG_FILE.name,
                })
                return

            elif path == "/api/shutdown":
                self._json({"ok": True, "message": "Shutting down..."})
                threading.Timer(0.5, self.server.shutdown).start()
                return

            elif path == "/":
                path = "/index.html"

            # Static files come from an explicit allow-list.
            #
            # This replaces `local_path = ROOT / path.lstrip("/")`, which had no
            # containment check and served any file the process could read:
            #   GET /config.json          -> both API keys in plaintext
            #   GET /server.py            -> full source
            #   GET /.git/config          -> repo internals
            #   GET /../../../Windows/... -> arbitrary file read
            # (Browsers normalise "..", so this needed a socket client to reach --
            # which is exactly what an attacker uses.)
            #
            # It also fixes a hang: the old /paper_graph.html branch assigned
            # local_path and ct but the send lived in the else, so the Graph tab's
            # default view got an empty response (HTTP 000, 0 bytes).
            static = {
                "/index.html":           ROOT / "index.html",
                "/embedding_graph.html": ROOT / "embedding_graph.html",
                "/paper_graph.html":     ROOT / "graphify-out" / "paper_graph.html",
            }
            target = static.get(path)
            if target is None:
                self.send_error(404)
                return
            try:
                # Belt and braces: even an allow-listed entry must resolve inside ROOT.
                resolved = target.resolve()
                if not resolved.is_file() or ROOT.resolve() not in resolved.parents:
                    self.send_error(404)
                    return
                body = resolved.read_bytes()
            except (FileNotFoundError, OSError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            self._headers_sent = True
            self.wfile.write(body)
            return

        except ConnectionAbortedError:
            pass
        except Exception as e:
            with open(ROOT / "server_errors.log", 'a') as ef:
                ef.write(f"[{datetime.now().isoformat()}] GET {self.path} ERROR: {type(e).__name__}: {e}\n")
            if not getattr(self, '_headers_sent', False):
                self._json({"error": f"Internal error: {str(e)}"}, status=500)

    def _handle_export(self, params):
        # Debug instrumentation removed: this used to open a fixed-name log in the
        # system temp dir on every export (mode "w", so concurrent exports on
        # ThreadingHTTPServer clobbered each other) and write the user's search
        # terms into it. logger already covers what it was for.
        try:
            fmt = params.get("format", ["csv"])[0].lower()
            q = params.get("q", [""])[0]
            ids_param = params.get("ids", [""])[0]

            result = list(papers_global)

            if q and len(q) >= 2:
                result = search_papers(result, q)

            if ids_param:
                ids_set = {i.strip() for i in ids_param.split(",") if i.strip()}
                result = [p for p in result if str(p.get("id") or p.get("entry_id", "")) in ids_set]

            logger.info(f"export fmt={fmt} papers={len(result)}")

            if fmt == "json":
                body = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
                ct = "application/json"
                ext = "json"
            elif fmt == "bibtex":
                entries = []
                for i, p in enumerate(result):
                    pid = _bibtex_key(p, i)
                    title = _tex_escape(p.get("title"))
                    author = " and ".join(
                        _tex_escape(a) for a in (p.get("authors") or []) if isinstance(a, str))
                    year = _export_year(p)
                    abstract = _tex_escape(p.get("summary"))
                    url = _tex_escape(p.get("url") or p.get("pdf_url"))
                    # Real newlines. These were "\\n" in the source, so the whole
                    # .bib came out as one line and "\n  title=" is not a parseable
                    # field name -- no BibTeX or Zotero import could ever have
                    # succeeded.
                    entries.append(
                        f"@article{{{pid},\n"
                        f"  title = {{{title}}},\n"
                        f"  author = {{{author}}},\n"
                        f"  year = {{{year}}},\n"
                        f"  abstract = {{{abstract}}},\n"
                        f"  url = {{{url}}}\n"
                        f"}}"
                    )
                body = ("\n\n".join(entries) + "\n").encode("utf-8")
                ct = "application/x-bibtex"
                ext = "bib"
            else:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["id", "title", "authors", "year", "abstract", "url"])
                for p in result:
                    writer.writerow([
                        _csv_cell(p.get("id") or p.get("entry_id")),
                        _csv_cell(p.get("title")),
                        _csv_cell("; ".join(a for a in (p.get("authors") or []) if isinstance(a, str))),
                        _csv_cell(_export_year(p)),
                        _csv_cell(p.get("summary")),
                        _csv_cell(p.get("url") or p.get("pdf_url")),
                    ])
                # utf-8-sig: Excel on Windows decodes a BOM-less UTF-8 CSV as the
                # ANSI codepage, which mangles every Arabic and accented title.
                body = output.getvalue().encode("utf-8-sig")
                ct = "text/csv"
                ext = "csv"

            # The filename went into the header raw, so an Arabic query raised
            # UnicodeEncodeError (send_header encodes latin-1 strict) -> HTTP 500 on
            # a MENA-focused tool, and a quote in the query broke the header value.
            slug = re.sub(r'[^A-Za-z0-9_-]', '', (q or "all").replace(" ", "_"))[:20] or "all"
            fname = f"papers_export_{len(result)}_{slug}.{ext}"
            self.send_response(200)
            self.send_header("Content-Type", ct + "; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._headers_sent = True
            self.wfile.write(body)
        except ConnectionAbortedError:
            pass
        except Exception as e:
            logger.exception("export failed")
            # Only send an error response if no status line went out already;
            # otherwise a second send_response embeds an HTTP status line as a
            # header and the client stores the JSON error as the "downloaded file".
            if not getattr(self, '_headers_sent', False):
                self._json({"error": f"Export failed: {str(e)}"}, status=500)
        return

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        # int() on the raw header sat outside every try block: "Content-Length: abc"
        # raised ValueError, killing the worker thread with no response at all. An
        # oversized value was also an unbounded read straight into memory.
        MAX_BODY = 32 * 1024 * 1024
        try:
            content_length = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            self._json({"error": "Malformed Content-Length header"}, status=400)
            return
        if content_length < 0 or content_length > MAX_BODY:
            self._json({"error": f"Request body too large (max {MAX_BODY} bytes)"}, status=413)
            return
        body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}
        logger.info(f"POST {path} from {self.client_address[0]}")
        try:
            if path == "/api/config/key":
                if not self._rate_check():
                    return
                key = data.get("key")
                # Anything at all used to be written straight to config.json, and a
                # body that failed to parse left data={} -> key="" -> the key was
                # silently cleared.
                if not isinstance(key, str) or not key.strip():
                    self._json({"error": "A non-empty key string is required."}, status=400)
                    return
                key = key.strip()
                if len(key) < 20 or any(c.isspace() for c in key):
                    self._json({"error": "That does not look like an API key."}, status=400)
                    return
                cfg = load_config()
                cfg["openrouter_api_key"] = key
                save_config(cfg)
                self._json({"ok": True, "message": "API key saved." if key else "Key cleared."})

            elif path == "/api/scraper/run":
                queries = data.get("queries", [])
                max_count = int(data.get("max", 20))
                sources = data.get("sources")
                if not queries:
                    self._json({"error": "No queries provided."}, status=400)
                    return
                if scraper_status.get("running", False):
                    self._json({"error": "Scraper is already running."}, status=409)
                    return
                # One worker, queries sequential -- see _run_scraper_queue for why.
                t = threading.Thread(target=_run_scraper_queue,
                                     args=(list(queries), max_count, sources), daemon=True)
                t.start()
                self._json({"ok": True, "message": f"Queued {len(queries)} query group(s), {max_count} papers each, running one at a time."})

            elif path == "/api/papers_by_ids":
                ids = data.get("ids", [])
                if not isinstance(ids, list):
                    self._json({"error": "'ids' must be a list"}, status=400)
                    return
                id_set = set(str(i) for i in ids)
                found = []
                missing = []
                for pid in id_set:
                    paper = next((p for p in papers_global if str(p.get("id") or p.get("entry_id") or "") == pid), None)
                    if paper:
                        found.append(paper)
                    else:
                        missing.append(pid)
                self._json({
                    "papers": found,
                    "count": len(found),
                    "found": len(found),
                    "missing": missing
                })
                return

            elif path == "/api/chat":
                if not self._rate_check():
                    return
                query = data.get("query", "").strip()
                if not query:
                    self._json({"error": "No query provided."}, status=400)
                    return
                # history came straight from the client into the OpenRouter messages
                # array with no validation, so a caller could inject a "system" turn
                # or relay megabytes to the paid API. The 20-turn cap was client-side
                # only.
                history = []
                for turn in (data.get("history") or [])[-MAX_HISTORY_TURNS:]:
                    if not isinstance(turn, dict):
                        continue
                    role, content = turn.get("role"), turn.get("content")
                    if role in ("user", "assistant") and isinstance(content, str):
                        history.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})

                paper_ids = [str(i) for i in (data.get("paper_ids") or [])
                             if isinstance(i, (str, int))][:MAX_CONTEXT_PAPERS * 4]
                source = data.get("source", "all")
                if source not in ("all", "latest", "bookmarks", "selected"):
                    # Was left to fall through with `papers` unbound, so any typo or
                    # older client got a 500 reading "cannot access local variable
                    # 'papers'" and the chat bubble showed only "Error: HTTP 500".
                    source = "all"
                papers = []
                warning = None

                if source == "latest":
                    latest_papers, source_file, timestamp = load_latest_scrape()
                    if latest_papers is None:
                        warning = "No latest scrape file found, falling back to all papers"
                        source = "all"
                    else:
                        papers = latest_papers[:30]
                        if paper_ids:
                            id_set = set(str(i) for i in paper_ids)
                            papers = [p for p in papers if str(p.get("id") or p.get("entry_id") or "") in id_set]

                if source == "bookmarks" or source == "selected":
                    if paper_ids:
                        id_set = set(str(i) for i in paper_ids)
                        papers = [p for p in papers_global if str(p.get("id") or p.get("entry_id") or "") in id_set]
                    else:
                        papers = []

                if source == "all":
                    results = semantic_search(query, papers_global, top_k=20)
                    if results:
                        papers = [r["paper"] for r in results]
                    else:
                        papers = search_papers(papers_global, query)[:30]
                        if not papers:
                            papers = papers_global[:30]

                # Slice ONCE, here, so the prompt, the citation validator and
                # papers_used all describe the same list. Previously the prompt was
                # built from papers[:30] while the validator used len(papers), so with
                # 100 selected papers a "[Paper 63]" citation passed validation, had no
                # entry in papers_used, and the frontend rendered it as literal text --
                # a citation-shaped token with no source and no warning.
                papers = papers[:MAX_CONTEXT_PAPERS]
                result = llm_rag_chat(query, papers, history=history)
                if isinstance(result, dict) and result.get("content"):
                    cited = re.findall(r'\[Paper (\d+)\]', result["content"])
                    invalid = [c for c in cited if not (1 <= int(c) <= len(papers))]
                    if invalid:
                        result["content"] += "\n\n[Warning: citations " + ", ".join(invalid) + " refer to papers not in the context. They may be hallucinated.]"
                papers_meta = [{"id": p.get("id"), "title": p.get("title", "")} for p in papers]
                response_data = {"query": query, "papers_used": papers_meta, "response": result}
                if warning:
                    response_data["warning"] = warning
                self._json(response_data)

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
                # One batch at a time. There was no guard at all, unlike
                # /api/scraper/run, and the rate limiter allows 10 POSTs/minute --
                # so 10 rapid calls (or a page-reload loop) started 10 threads of up
                # to 50 papers each: 500 uncached LLM analyses per minute,
                # repeatable, with overlapping ranges racing on the same
                # data/analyses/{id}.json files.
                if _batch_job_running():
                    self._json({"error": "A batch analysis is already running."}, status=409)
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

                return

            elif path == "/api/embedding_graph":
                rebuild = data.get("rebuild", False)
                if rebuild:
                    try:
                        _tk = int(data.get("top_k", GRAPH_TOP_K))
                    except (TypeError, ValueError):
                        _tk = GRAPH_TOP_K
                    graph = build_embedding_graph(top_k=_tk)
                    if graph is None:
                        self._json({"error": "Not enough embeddings to build graph."})
                        return
                else:
                    graph = load_or_build_embedding_graph()
                if graph is None:
                    self._json({"error": "Embedding graph not available. Build embeddings first."})
                    return
                merged = data.get("merged", False)
                if merged:
                    result = merge_graphs()
                else:
                    result = graph
                self._json(result)
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
                try:
                    top_k = int(data.get("top_k", 15))
                except (TypeError, ValueError):
                    top_k = 15
                top_k = max(1, min(top_k, 100))  # was unguarded: 'abc' 500'd, -5 silently truncated
                hybrid = data.get("hybrid", False)
                if hybrid:
                    results = hybrid_search(query, papers_global, top_k=top_k)
                    papers_out = []
                    for pid, score, scores in results:
                        paper = next((p for p in papers_global if (p.get("id") or p.get("entry_id")) == pid), None)
                        if paper:
                            papers_out.append({
                                "id": pid,
                                "title": paper.get("title", ""),
                                "summary": (paper.get("summary") or "")[:300],
                                "score": round(score, 4),
                                "keyword_score": scores["keyword_score"],
                                "semantic_score": round(scores["semantic_score"], 4)
                            })
                else:
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

            elif path == "/api/clear_embeddings":
                _set_embeddings(None, [])
                try:
                    if EMBEDDING_CACHE.exists():
                        EMBEDDING_CACHE.unlink()
                except Exception:
                    pass
                try:
                    if EMBEDDING_META_CACHE.exists():
                        EMBEDDING_META_CACHE.unlink()
                except Exception:
                    pass
                self._json({"ok": True, "message": "Embeddings cleared"})

            elif path == "/api/build_embeddings":
                if not self._rate_check():
                    return
                provider = data.get("provider", EMBEDDING_PROVIDER)
                use_batch = data.get("batch", True)
                thread = threading.Thread(target=build_embeddings, args=(papers_global,), kwargs={"provider": provider, "batch": use_batch}, daemon=True)
                thread.start()
                self._json({"ok": True, "message": f"Building embeddings in background (provider={provider}, batch={use_batch})..."})

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

            elif path == "/api/expand_query":
                query = data.get("query", "").strip()
                if not query or len(query) < 2:
                    self._json({"query": query, "expanded_terms": []})
                    return
                expanded = expand_query_by_embeddings(query, papers_global)
                self._json({"query": query, "expanded_terms": expanded})
                return

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
    # Bind loopback only. ("", port) binds 0.0.0.0, which exposed this server --
    # and every file it would serve, plus unauthenticated /api/shutdown and
    # LLM-spending endpoints -- to every host on the local network.
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
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
