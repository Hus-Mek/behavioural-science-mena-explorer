"""Fail when commit messages or the PR body carry AI-tool attribution."""

import argparse
import os
import re
import subprocess
import sys

ATTRIBUTION_RE = re.compile(
    r"co-authored-by:[^\n]*(claude|anthropic)|generated with \[?claude|noreply@anthropic\.com",
    re.IGNORECASE,
)
RECORD_SEP = "\x1e"


def find_attribution(texts: dict[str, str]) -> tuple[str, ...]:
    return tuple(label for label, text in texts.items() if ATTRIBUTION_RE.search(text))


def parse_log(raw: str) -> dict[str, str]:
    records = (r.lstrip("\n") for r in raw.split(RECORD_SEP) if r.strip())
    return dict(r.split("\x00", 1) for r in records)


def commit_messages(rev_range: str) -> dict[str, str]:
    raw = subprocess.run(
        ["git", "log", "--no-merges", f"--format=%h%x00%B{RECORD_SEP}", rev_range],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return parse_log(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", required=True, help="git revision range, for example origin/main..HEAD")
    args = parser.parse_args()
    try:
        texts = commit_messages(args.range)
    except subprocess.CalledProcessError as err:
        print(f"check_attribution: cannot read {args.range}: {err.stderr}", file=sys.stderr)
        return 2
    if os.environ.get("PR_BODY"):
        texts = {**texts, "PR body": os.environ["PR_BODY"]}
    hits = find_attribution(texts)
    for label in hits:
        print(f"::error::AI attribution found in {label}; remove the trailer or footer and push again")
    print(f"checked {len(texts)} messages, {len(hits)} with attribution")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
