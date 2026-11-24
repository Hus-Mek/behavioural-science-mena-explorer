# CLAUDE.md — Behavioural Science MENA Explorer

## Obsidian Vault (non-negotiable)

**Vault path:** `C:\Users\USER\Documents\Obsidian Vault`

### Rules
1. **Read vault context first.** Before starting work, read `CRITICAL_FACTS.md` and `01_Projects/Behavioural Science MENA Explorer.md`.
2. **Write session notes.** After every meaningful work session, write to `Sessions/YYYY-MM-DD — Description.md`.
3. **Update project note.** After each session, update `01_Projects/Behavioural Science MENA Explorer.md`.

### Session Note Format
```markdown
---
date: YYYY-MM-DD
type: session
tags: [session, <topics>]
ai-first: true
---

## For future Claude
Session note from YYYY-MM-DD. [Brief summary]

## Summary
[2-3 sentence overview]

## What Was Done
- [Itemized actions]

## Decisions Made
- **[Decision]**: [Rationale]

## Files Created/Modified
- `path/to/file` — what changed

## Next Steps
- [ ] Action items
```

## Project Context
- **Location:** `C:\Users\USER\localled`
- **Port:** 3000 (binds 127.0.0.1 only — do NOT change back to `("", port)`, that exposed config.json and both API keys to the LAN)
- **LLM:** poolside/laguna-s-2.1:free (reasoning model — needs max_tokens=2000+, timeout=300s). Verified live 2026-08-05; the previous `laguna-m.1:free` 404s ("No endpoints found"), as do `qwen3-coder:free` (paid-only now) and `gemini-2.0-flash-001`. Re-verify the chain against `https://openrouter.ai/api/v1/models` when AI features misbehave — free model IDs rot without warning.

### Key Gotchas
- Reasoning model returns `content: null` when max_tokens too low — always check `reasoning` field fallback
- Server auto-shutdown on tab close via `navigator.sendBeacon('/api/shutdown')`
- Analysis cached to `data/analysis_cache.json` and `data/analyses/{paper_id}.json`
- Clustering: batch mode for >50 papers, merge with existing clusters
- Chat: phrase matching for keyword pre-filter (prevents "com-b" matching "combinations")

## Done means it works for the user

Green CI doesn't make a feature, a fix, or any code change done. Before you call it done, use it
the way its user will meet it, and read what they would see:

- Run `python server.py`, open http://127.0.0.1:3000 in a browser (use the Claude in Chrome extension),
  and click through the screen you changed: search, analysis, clustering, chat.
- Read the rendered page and the browser console, not only the server log.
- Put what you ran and what the user saw in the pull request body. If you couldn't try a change
  this way, say so and why.
