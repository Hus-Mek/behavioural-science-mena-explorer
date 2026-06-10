# Phase 2 Plan: Export, Bookmarks, Search History, Keyboard Shortcuts

## Overview
Add 4 UX features to the Behavioural Science MENA Explorer app.

## Feature Breakdown

### 1. Export CSV/BibTeX (server + frontend)
**server.py:**
- New GET endpoint `/api/export?format=csv|bibtex|json`
- Optional `?q=` filter, `?ids=` comma-separated paper IDs
- CSV: columns = id, title, authors, year, abstract, url
- BibTeX: @article entries with standard fields
- JSON: full paper objects
- Returns as downloadable file with proper Content-Disposition header

**index.html:**
- Export section in Tools tab with format selector (CSV / BibTeX / JSON)
- "Export All" and "Export Selected" buttons
- Filter toggle: export current search results vs all papers
- Download triggers via fetch blob

### 2. Bookmarks (frontend, localStorage)
**index.html:**
- Star/bookmark button on each paper card (toggle)
- Bookmarks panel in a new "Bookmarks" tab or sidebar section
- localStorage key: `bs_mena_bookmarks` → array of paper IDs
- Persists across sessions (same browser)
- Visual indicator on bookmarked cards

**server.py:** No changes needed (pure client-side)

### 3. Search History (frontend, localStorage)
**index.html:**
- Track all search queries in localStorage key `bs_mena_search_history`
- Max 50 entries, deduplicated
- Show recent searches (clickable to re-run)
- Clear history button
- Timestamp per entry

**server.py:** No changes needed (pure client-side)

### 4. Keyboard Shortcuts (frontend)
**index.html:**
- `/` → focus search bar (papers tab)
- `Ctrl+Enter` in chat → send message
- `g` then `d` → go to Dashboard
- `g` then `p` → go to Papers
- `g` then `c` → go to Concepts
- `g` then `t` → go to Chat
- `g` then `o` → go to Tools
- `Escape` → close reader overlay, clear search focus
- `?` → show shortcuts cheat sheet (modal)

## Files to Modify
- `server.py` — add `/api/export` endpoint (~60 lines)
- `index.html` — add export UI, bookmarks, search history, keyboard shortcuts (~300 lines JS + HTML)

## Constraints
- localStorage limited to ~5MB, store only paper IDs for bookmarks, strings for history
- Export endpoint must handle 333+ papers efficiently (streaming not needed at this scale)
- All features backward compatible — no changes to existing API responses
- Use existing CSS patterns and design system