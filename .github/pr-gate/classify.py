"""Deterministic PR risk classifier: git diff in, tier (0-3) and reasons out."""

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

POLICY_PATH = Path(__file__).with_name("risk-policy.toml")
RENAME_RE = re.compile(r"\{([^{}]*) => ([^{}]*)\}")
FIX_SUBJECT_RE = r"^(fix|revert)"
COMMENT_MARKER = "<!-- pr-gate -->"
TIER_ACTIONS = {
    0: "auto-merge when CI is green",
    1: "AI review; Claude may merge when the gate is green",
    2: "three-pass human review; a human merges",
    3: "human review plus a delayed second pass; a human merges",
}


@dataclass(frozen=True)
class Change:
    path: str
    added: int
    deleted: int


@dataclass(frozen=True)
class Policy:
    max_t1_lines: int
    window_days: int
    min_fix_commits: int
    trivial: tuple[str, ...]
    dependency: tuple[str, ...]
    critical: tuple[str, ...]


@dataclass(frozen=True)
class Classification:
    tier: int
    reasons: tuple[str, ...]
    empty: bool = False


def load_policy(path: Path) -> Policy:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    return Policy(
        max_t1_lines=raw["size"]["max_t1_lines"],
        window_days=raw["hotspot"]["window_days"],
        min_fix_commits=raw["hotspot"]["min_fix_commits"],
        trivial=tuple(paths["trivial"]),
        dependency=tuple(paths["dependency"]),
        critical=tuple(paths["critical"]),
    )


DEFAULT_POLICY = load_policy(POLICY_PATH)


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    name = PurePosixPath(path).name
    return any(fnmatch(path if "/" in p else name, p) for p in patterns)


def parse_numstat(raw: str) -> tuple[Change, ...]:
    changes = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        added, deleted, path = line.split("\t", 2)
        path = RENAME_RE.sub(r"\2", path).split(" => ")[-1].replace("//", "/")
        changes.append(Change(path, int(added) if added != "-" else 0, int(deleted) if deleted != "-" else 0))
    return tuple(changes)


def classify(changes: tuple[Change, ...], hotspots: frozenset[str], policy: Policy) -> Classification:
    if not changes:
        return Classification(0, ("empty diff: the merge would change nothing",), empty=True)
    source = [c for c in changes if not matches(c.path, policy.trivial)]
    lines = sum(c.added + c.deleted for c in source)
    rules = (
        (2, lines > policy.max_t1_lines, f"{lines} changed lines exceed the T1 cap of {policy.max_t1_lines}"),
        (2, _hits(changes, policy.dependency), "dependency manifest changed: "),
        (2, sorted(hotspots & {c.path for c in changes}), "hotspot file (repeated fixes): "),
        (3, _hits(changes, policy.critical), "critical path changed: "),
    )
    fired = [(tier, _reason(text, hit)) for tier, hit, text in rules if hit]
    base = 1 if source else 0
    reasons = [r for _, r in fired] or [("source change within T1 limits" if base else "docs/tests only")]
    return Classification(max([base, *(t for t, _ in fired)]), tuple(reasons))


def _hits(changes: tuple[Change, ...], patterns: tuple[str, ...]) -> list[str]:
    return sorted(c.path for c in changes if matches(c.path, patterns))


def _reason(text: str, hit: bool | list[str]) -> str:
    return text + ", ".join(hit) if isinstance(hit, list) else text


def render_markdown(result: Classification) -> str:
    reasons = "\n".join(f"- {r}" for r in result.reasons)
    return f"{COMMENT_MARKER}\n### pr-gate: **T{result.tier}**, {TIER_ACTIONS[result.tier]}\n\n{reasons}\n"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def hotspot_files(base: str, policy: Policy) -> frozenset[str]:
    log = git("log", f"--since={policy.window_days}.days", "--format=", "--name-only",
              "-i", "-E", f"--grep={FIX_SUBJECT_RE}", base)
    counts = Counter(line for line in log.splitlines() if line)
    return frozenset(path for path, n in counts.items() if n >= policy.min_fix_commits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base ref, for example origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--markdown", type=Path, help="also write the PR comment body to this file")
    args = parser.parse_args()
    try:
        policy = load_policy(args.policy)
        changes = parse_numstat(git("diff", "--numstat", f"{args.base}...{args.head}"))
        result = classify(changes, hotspot_files(args.base, policy), policy)
    except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as err:
        detail = getattr(err, "stderr", "") or err
        print(f"classify: cannot classify {args.base}...{args.head}: {detail}", file=sys.stderr)
        return 2
    if args.markdown:
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(asdict(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
