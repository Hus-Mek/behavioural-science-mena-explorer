# AGENTS.md — Behavioural Science MENA Explorer

> Auto-loaded by Hermes Agent from the working directory at session start.
> This file has priority over CLAUDE.md and .cursorrules.

---

## First Action: Load Obsidian Skill + Read Vault

At the start of every session, before doing ANYTHING else:
1. `skill_view(name='obsidian')` — loads full vault operation rules
2. Read `C:\Users\USER\Documents\Obsidian Vault\CRITICAL_FACTS.md`
3. Read `C:\Users\USER\Documents\Obsidian Vault\01_Projects\Behavioural Science MENA Explorer.md`

This is non-negotiable.

---

## Obsidian Vault — MANDATORY

**Vault path:** `C:\Users\USER\Documents\Obsidian Vault`
**Skill:** `obsidian` (loaded via skill_view above)

### After Every Session
Write a session note to `Sessions/YYYY-MM-DD — Description.md`:
```
---
date: YYYY-MM-DD
type: session
tags: [session, behavioural-science-mena]
ai-first: true
---

## For future Claude
Session note from YYYY-MM-DD. [One-line summary]

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

Also update `01_Projects/Behavioural Science MENA Explorer.md` and append to `Logs/YYYY-MM-DD.md`.

---

## Delegation Pattern — MANDATORY

For ALL coding tasks (anything touching server.py, index.html, scraper.py, or any file):

1. **PLAN first.** Write out the approach: files to change, exact edits, edge cases. Do NOT write any code yet.
2. **DELEGATE implementation.** Use `delegate_task` with `goal` (what to build), `context` (the plan + all relevant background), `file paths to modify`, `edge cases`, `verification steps`.
3. **REVIEW results.** Read the changed files, verify they compile/parse, then report to user.

**Exception:** Simple one-liner fixes (typos, single-line config changes, trivial edits) can be done directly. If the task requires patching more than 1 file or more than 10 lines, delegate.

Before delegating, read the relevant code to understand the current state. Pass all context the subagent needs in the `context` field.

---

## Project: Behavioural Science MENA Explorer

**Location:** `C:\Users\USER\localled`
**Port:** 3000
**LLM:** poolside/laguna-m.1:free (reasoning model — needs max_tokens=2000+, timeout=300s)
**Stack:** server.py (Python HTTP), index.html (single-file dark mode frontend), scraper.py
**Data:** `data/raw/papers_*.json`

### Key Gotchas
- Reasoning model returns `content: null` when max_tokens too low — always check `reasoning` field fallback
- Server auto-shutdown on tab close via `navigator.sendBeacon('/api/shutdown')`
- Analysis cached: `data/analysis_cache.json` + `data/analyses/{paper_id}.json`
- Clustering: batch mode for >50 papers, merge with existing clusters
- Chat: phrase matching for keyword pre-filter (prevents "com-b" matching "combinations")

### System Prompt Priority
Hermes loads ONE project context file: `.hermes.md` > `AGENTS.md` > `CLAUDE.md` > `.cursorrules`.
This AGENTS.md is the one that wins. Put ALL persistent rules here.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
