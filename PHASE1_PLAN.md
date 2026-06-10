# Phase 1: Critical Fixes — Implementation Plan

## Status
- **Created:** 2026-06-10
- **Status:** Pending approval

## Overview
Phase 1 addresses reliability, UX during loading, input validation, and abuse prevention.

---

## 1. Dedup Papers Across Scraper Runs
**Problem:** Multiple scraper runs create timestamped files (`papers_COM-B_1780999630.json`, `papers_COM-B_1780999669.json`, etc.). `load_papers()` deduplicates by `id`/`entry_id` only, but some sources (CrossRef, OpenAlex) use different ID schemes, so same paper can appear with different IDs across runs.

**Fix:**
- Add title-based fuzzy dedup in `load_papers()`: normalize titles (lowercase, strip punctuation), check for near-duplicates (>0.9 similarity via simple token overlap)
- Keep existing ID-based dedup as primary pass
- Add a `--dedup` CLI flag to `scraper.py` for explicit dedup runs

**Files:** `server.py` (load_papers), `scraper.py` (dedup method)

---

## 2. Loading Spinners / Skeleton UI
**Problem:** Page renders blank until all 3 API calls (papers, analysis, config) complete. No visual feedback.

**Fix:**
- Add a loading overlay/spinner on initial page load
- Show skeleton cards in paper/concept lists while loading
- Add per-section spinners for async operations (clustering, summarising)
- The existing "Thinking..." message in chat is good — extend pattern elsewhere

**Files:** `index.html` (CSS + JS)

---

## 3. Error Retry with Exponential Backoff
**Problem:** `llm_call()` fails silently on transient errors (rate limits, timeouts). No retry.

**Fix:**
- Add retry decorator/function with exponential backoff (3 attempts, 1s/2s/4s delays)
- Apply to `llm_call()`, scraper HTTP calls (arXiv, PubMed, etc.)
- Make max retries configurable (default 3)

**Files:** `server.py` (llm_call + run_scraper), `scraper.py` (search methods)

---

## 4. Request Validation
**Problem:** API endpoints accept arbitrary values with no bounds checking:
- `/api/cluster?max=99999` — tries to cluster 99k papers
- `/api/summarise_all?start=-5&count=999` — invalid range
- `/api/summarise?idx=abc` — non-integer index
- No validation on paper ID format

**Fix:**
- Add `max` clamp: `min(max_n, len(papers_global))` and reasonable upper bound (500)
- Validate `start >= 0`, `count > 0 && count <= 50`
- Return proper 400 errors with messages for invalid params
- Add paper ID format validation (reject empty/special chars)

**Files:** `server.py` (Handler.do_GET, Handler.do_POST)

---

## 5. Rate Limiting
**Problem:** No limits on API calls. A user could spam `/api/summarise` and exhaust the LLM API quota.

**Fix:**
- Simple token bucket rate limiter: 10 requests per 60 seconds per IP for LLM endpoints
- Queue-based: if rate limited, return 429 with Retry-After header
- Apply to `/api/summarise`, `/api/cluster`, `/api/summarise_all`, `/api/chat`
- Non-LLM endpoints (`/api/papers`, `/api/analysis`) exempt or higher limit

**Files:** `server.py` (new rate_limiter module/class)

---

## Implementation Order
1. Request validation (safest, no side effects)
2. Dedup (data quality)
3. Error retry (reliability)
4. Loading spinners (UX)
5. Rate limiting (abuse prevention)

## Risk
- Low risk for all changes — mostly additive, validation tightening
- Dedup needs care to not accidentally drop legitimate papers
- Rate limiter state must be thread-safe (use threading.Lock)