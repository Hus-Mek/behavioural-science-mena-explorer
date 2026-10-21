"""Keep only review findings that two independent passes agree on."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SEVERITIES = frozenset({"blocking", "non-blocking"})
DEFAULT_LINE_WINDOW = 3


class FindingsError(ValueError):
    """A findings file is missing, is not JSON, or does not match the schema."""


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    severity: str
    body: str

    @property
    def is_blocking(self) -> bool:
        return self.severity == "blocking"


def parse_finding(raw: object, index: int) -> Finding:
    if not isinstance(raw, dict):
        raise FindingsError(f"finding {index} is not an object")
    path, line, severity, body = (raw.get(k) for k in ("path", "line", "severity", "body"))
    if not isinstance(path, str) or not path:
        raise FindingsError(f"finding {index}: 'path' must be a non-empty string")
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise FindingsError(f"finding {index}: 'line' must be a positive integer")
    if severity not in SEVERITIES:
        raise FindingsError(f"finding {index}: 'severity' must be one of {sorted(SEVERITIES)}")
    if not isinstance(body, str) or not body.strip():
        raise FindingsError(f"finding {index}: 'body' must be a non-empty string")
    return Finding(path, line, severity, body)


def parse_findings(data: object) -> list[Finding]:
    if not isinstance(data, list):
        raise FindingsError("findings must be a JSON list")
    return [parse_finding(item, i) for i, item in enumerate(data)]


def merged(a: Finding, b: Finding) -> Finding:
    """a's location; blocking if either pass said so, with the body that carries that label."""
    if a.is_blocking or not b.is_blocking:
        return a
    return Finding(a.path, a.line, "blocking", b.body)


def agree(
    a: list[Finding], b: list[Finding], line_window: int = DEFAULT_LINE_WINDOW
) -> tuple[list[Finding], int]:
    """Findings of `a` matched one-to-one to `b` (same path, nearest line within the window).

    The second value counts findings, from either pass, that found no partner.
    Closest pairs are matched first, so a nearby finding is not stolen by a farther one.
    """
    candidates = sorted(
        (abs(x.line - y.line), i, j)
        for i, x in enumerate(a)
        for j, y in enumerate(b)
        if x.path == y.path and abs(x.line - y.line) <= line_window
    )
    pairs: dict[int, int] = {}
    used_b: set[int] = set()
    for _, i, j in candidates:
        if i not in pairs and j not in used_b:
            pairs[i] = j
            used_b.add(j)
    agreed = [merged(a[i], b[pairs[i]]) for i in sorted(pairs)]
    dropped = len(a) + len(b) - 2 * len(agreed)
    return agreed, dropped


def load_findings(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise FindingsError(f"cannot read {path}: {err.strerror or err}") from err
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise FindingsError(f"{path} is not valid JSON: {err}") from err
    try:
        return parse_findings(data)
    except FindingsError as err:
        raise FindingsError(f"{path}: {err}") from err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="findings.json from the first pass")
    parser.add_argument("second", type=Path, help="findings.json from the second pass")
    parser.add_argument("--out", type=Path, required=True, help="where to write the agreed findings")
    parser.add_argument("--line-window", type=int, default=DEFAULT_LINE_WINDOW)
    args = parser.parse_args(argv)
    try:
        first, second = load_findings(args.first), load_findings(args.second)
    except FindingsError as err:
        print(f"agree: {err}", file=sys.stderr)
        return 2
    agreed, dropped = agree(first, second, args.line_window)
    args.out.write_text(json.dumps([asdict(f) for f in agreed], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"agreed": len(agreed), "dropped": dropped}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
