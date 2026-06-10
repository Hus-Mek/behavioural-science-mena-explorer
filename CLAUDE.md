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
- **Port:** 3000
- **LLM:** poolside/laguna-m.1:free (reasoning model — needs max_tokens=2000+, timeout=300s)

### Key Gotchas
- Reasoning model returns `content: null` when max_tokens too low — always check `reasoning` field fallback
- Server auto-shutdown on tab close via `navigator.sendBeacon('/api/shutdown')`
- Analysis cached to `data/analysis_cache.json` and `data/analyses/{paper_id}.json`
- Clustering: batch mode for >50 papers, merge with existing clusters
- Chat: phrase matching for keyword pre-filter (prevents "com-b" matching "combinations")
