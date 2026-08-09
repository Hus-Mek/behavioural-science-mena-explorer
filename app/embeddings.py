"""Embedding vectors: providers, on-disk caches, delta builds, semantic scoring.

Owns the (matrix, ids) module state and the lock that keeps row i of the
matrix paired with ids[i]. Never import server.py from here.
"""
import json
import re
import threading
import urllib.request
from pathlib import Path

from app.config import get_api_key, get_gemini_api_key
from app.pdf_text import PDF_DIR
from app.util import atomic_write_json

ROOT = Path(__file__).parent.parent.resolve()

EMBEDDING_PROVIDER = "gemini"
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_MODEL_GEMINI = "models/gemini-embedding-2"
EMBEDDING_BATCH_SIZE = 100
GEMINI_EMBEDDING_DIM = 3072

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
