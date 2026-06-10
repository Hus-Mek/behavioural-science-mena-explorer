# Handover Prompt — Behavioural Science MENA Explorer

Copy this entire block as your first message to start a new session:

---

## Session Handover — Behavioural Science MENA Explorer

### Project Location
`C:\Users\USER\localled` — Python HTTP server on port 3000.
GitHub: https://github.com/Hus-Mek/behavioural-science-mena-explorer

### First Actions (do these before anything else)
1. Load Obsidian skill: `skill_view(name='obsidian')`
2. Read vault context:
   - `C:\Users\USER\Documents\Obsidian Vault\CRITICAL_FACTS.md`
   - `C:\Users\USER\Documents\Obsidian Vault\01_Projects\Behavioural Science MENA Explorer.md`
   - `C:\Users\USER\Documents\Obsidian Vault\Sessions\` (latest session notes)
3. Start the server: `cd C:\Users\USER\localled && python server.py`
4. Open http://localhost:3000

### What Was Done Last Session (2026-06-09)
- Fixed chatbot hallucination: stricter system prompt, phrase matching for keyword pre-filter, citation validation
- Implemented cluster merging: re-running clustering merges with existing clusters instead of replacing
- Added paper count input (10-200) on Concepts tab with batch mode for >50 papers
- Replaced `[Paper N]` inline citations with actual paper titles + references table
- Added conversation history to chat (last 20 messages sent as context)
- Increased LLM timeout from 120s to 300s (reasoning model needs time)
- Set up Obsidian vault bridge via AGENTS.md in workspace
- Created GitHub repo and pushed all code
- Fixed datetime comparison bug (naive vs aware) and None summary crash
- Added analysis persistence (analysis_cache.json + per-paper analyses/)

### Architecture
- **Backend:** server.py (834 lines) — SimpleHTTP server, REST API, LLM calls via urllib
- **Frontend:** index.html (1275 lines) — single-file, dark mode, Linear design, 5 tabs
- **Scraper:** scraper.py (616 lines) — arXiv, PubMed, Semantic Scholar, CrossRef
- **Data:** data/raw/papers_*.json (334 papers), data/analyses/ (per-paper LLM cache)
- **LLM:** poolside/laguna-m.1:free via OpenRouter (REASONING MODEL)
- **API key:** stored in config.json, set via POST /api/config/key from GUI

### Key Gotchas
- Reasoning model returns `content: null` when max_tokens too low — always check `reasoning` field fallback
- max_tokens must be 2000+ for structured JSON responses
- Server auto-shutdown on tab close via `navigator.sendBeacon('/api/shutdown')`
- SO_LINGER(1,0) + SO_REUSEADDR prevents zombie ports
- `p.get("summary") or ""` not `p.get("summary","")` — the latter returns None if key exists with None value
- AGENTS.md has priority over CLAUDE.md in Hermes context loading

### API Endpoints
- `GET /api/papers?q=&year=&term=` — search/filter papers
- `GET /api/analysis` — dashboard statistics
- `POST /api/summarise` — per-paper AI analysis {idx or id}
- `POST /api/chat` — RAG chat {query, paper_ids[], history[]}
- `POST /api/cluster` — AI clustering {max, existing?}
- `POST /api/summarise_all` — batch analysis {start, count}
- `GET /api/scraper/status` — scraper status
- `GET /api/scraper/run?q=&n=` — start scraper
- `POST /api/config/key` — save API key {key}
- `GET /api/shutdown` / `POST /api/shutdown` — shutdown server

### Frontend Tabs
1. **Dashboard** — stats cards, publication timeline, top authors, term frequencies
2. **Papers** — search, filter by year/concept, paginated list, click to open reader
3. **Concepts** — AI clustering with paper count input, cluster cards, unclustered section
4. **Chat** — RAG chat with history, clickable citations, references table
5. **Tools** — API key input, scraper controls (preset grid + custom queries), batch analysis, server controls

### Obsidian Vault Rules (mandatory)
After every meaningful work session:
1. Write session note to `Sessions/YYYY-MM-DD — Description.md`
2. Update `01_Projects/Behavioural Science MENA Explorer.md`
3. Append to `Logs/YYYY-MM-DD.md`

Format: frontmatter (date/type/tags/ai-first), "For future Claude" summary, Summary, What Was Done, Decisions, Files Modified, Next Steps.

### Pending Roadmap
**Phase 1 (Critical):** Dedup papers, loading spinners, error retry, request validation, rate limiting
**Phase 2 (UX):** Export CSV/BibTeX, bookmarks, search history, keyboard shortcuts
**Phase 3 (AI):** Live batch progress, model fallback, more context papers, prompt versioning
**Phase 4 (Scale):** Docker, CI/CD, tests, PDF extraction, semantic search, Arabic detection

### Current Model
openrouter/owl-alpha (switched from inclusionai/ring-2.6-1t this session)

### User Preferences
- Concise, direct responses without fluff
- Plan first, then delegate coding to subagents
- Dark mode UI, Linear design system
- No emoji, no gradients, monochrome + single accent color (#5e6ad2)
- Always write to Obsidian vault after work sessions
