"""Small shared utilities."""
import json
import os
from pathlib import Path


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
