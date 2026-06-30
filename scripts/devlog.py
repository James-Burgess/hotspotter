#!/usr/bin/env python3
"""Query and submit entries in docs/development/devlog.md."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEVLOG = ROOT / "docs" / "development" / "devlog.md"
ENTRY_RE = re.compile(r"^## (?P<date>\d{4}-\d{2}-\d{2})\s+[—-]\s+(?P<title>.+)$")


@dataclass(frozen=True)
class Entry:
    date: str
    title: str
    text: str
    start_line: int

    @property
    def heading(self) -> str:
        return f"## {self.date} — {self.title}"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"devlog not found: {path}")


def _parse_entries(text: str) -> list[Entry]:
    lines = text.splitlines()
    starts: list[tuple[int, re.Match[str]]] = []
    for idx, line in enumerate(lines):
        match = ENTRY_RE.match(line)
        if match:
            starts.append((idx, match))

    entries: list[Entry] = []
    for pos, (idx, match) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        entry_lines = lines[idx:end]
        while entry_lines and not entry_lines[-1].strip():
            entry_lines.pop()
        if entry_lines and entry_lines[-1].strip() == "---":
            entry_lines.pop()
        while entry_lines and not entry_lines[-1].strip():
            entry_lines.pop()
        entries.append(
            Entry(
                date=match.group("date"),
                title=match.group("title").strip(),
                text="\n".join(entry_lines).rstrip() + "\n",
                start_line=idx + 1,
            )
        )
    return entries


def _entry_insert_offset(text: str) -> int:
    """Return character offset after the intro separator."""
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            return sum(len(part) for part in lines[: idx + 1])
    return 0


def _print_entry(entry: Entry, show_line: bool = False) -> None:
    if show_line:
        print(f"# {entry.start_line}: {entry.heading}")
        print(entry.text.split("\n", 1)[1].rstrip())
    else:
        print(entry.text.rstrip())


def _split_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        for line in value.splitlines():
            stripped = line.strip()
            if stripped:
                items.append(stripped)
    return items


def _numbered_section(title: str, items: list[str]) -> str:
    if not items:
        return ""
    body = "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))
    return f"### {title}\n\n{body}\n"


def _plain_section(title: str, value: str | None) -> str:
    if not value:
        return ""
    return f"### {title}\n\n{value.strip()}\n"


def _code_section(title: str, commands: list[str]) -> str:
    if not commands:
        return ""
    body = "\n".join(commands)
    return f"### {title}\n\n```bash\n{body}\n```\n"


def _build_structured_entry(args: argparse.Namespace) -> str:
    title = args.title.strip()
    if not title:
        sys.exit("--title cannot be empty")

    parts = [f"## {args.date} — {title}\n"]
    parts.append(_plain_section("Summary", args.summary))
    parts.append(_numbered_section("What was done", _split_items(args.done)))
    parts.append(_plain_section("Results", args.results))
    parts.append(_code_section("Verification", _split_items(args.verify)))
    parts.append(_numbered_section("Files changed", _split_items(args.files)))

    body = "\n".join(part.rstrip() for part in parts if part).rstrip()
    if body == parts[0].strip():
        sys.exit(
            "provide --body or at least one of --summary/--done/--results/--verify/--files"
        )
    return body + "\n"


def _build_entry(args: argparse.Namespace) -> str:
    if args.body:
        try:
            body = args.body.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            sys.exit(f"body file not found: {args.body}")
        if not body:
            sys.exit(f"body file is empty: {args.body}")
        if body.startswith("## "):
            return body + "\n"
        title = args.title.strip()
        if not title:
            sys.exit("--title is required when --body does not include a heading")
        return f"## {args.date} — {title}\n\n{body}\n"

    return _build_structured_entry(args)


def cmd_latest(args: argparse.Namespace) -> int:
    entries = _parse_entries(_read(args.file))
    if not entries:
        print("No dated devlog entries found.")
        return 1
    latest = max(entries, key=lambda entry: (entry.date, -entry.start_line))
    _print_entry(latest, show_line=args.line)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    entries = _parse_entries(_read(args.file))
    terms = [term.lower() for term in args.terms]
    matches: list[Entry] = []

    for entry in entries:
        haystack = entry.text.lower()
        if args.date and entry.date != args.date:
            continue
        if terms and not all(term in haystack for term in terms):
            continue
        matches.append(entry)

    for idx, entry in enumerate(matches):
        if idx:
            print("\n---\n")
        _print_entry(entry, show_line=args.line)

    if not matches:
        print("No matching devlog entries found.")
        return 1
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    existing = _read(args.file)
    entry = _build_entry(args)
    offset = _entry_insert_offset(existing)
    prefix = existing[:offset].rstrip()
    suffix = existing[offset:].lstrip()
    updated = f"{prefix}\n\n{entry}\n---\n\n{suffix}"
    args.file.write_text(updated, encoding="utf-8")
    first_line = entry.splitlines()[0]
    print(f"Added devlog entry to {args.file}: {first_line}")
    return 0


def _valid_date(value: str) -> str:
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_DEVLOG,
        help=f"devlog path (default: {DEFAULT_DEVLOG.relative_to(ROOT)})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    latest = subparsers.add_parser("latest", help="print the most recent dated entry")
    latest.add_argument(
        "--line", action="store_true", help="include source line number"
    )
    latest.set_defaults(func=cmd_latest)

    query = subparsers.add_parser("query", help="search dated entries")
    query.add_argument(
        "terms", nargs="*", help="case-insensitive terms; all must match"
    )
    query.add_argument("--date", type=_valid_date, help="only entries for YYYY-MM-DD")
    query.add_argument("--line", action="store_true", help="include source line number")
    query.set_defaults(func=cmd_query)

    submit = subparsers.add_parser("submit", help="prepend a new entry")
    submit.add_argument("--date", type=_valid_date, default=dt.date.today().isoformat())
    submit.add_argument("--title", default="", help="entry title")
    submit.add_argument("--body", type=Path, help="markdown file for the entry body")
    submit.add_argument("--summary", help="summary paragraph")
    submit.add_argument(
        "--done", action="append", help="completed item; repeat or pass multiline text"
    )
    submit.add_argument("--results", help="results paragraph or markdown table")
    submit.add_argument(
        "--verify",
        action="append",
        help="verification command; repeat or pass multiline text",
    )
    submit.add_argument(
        "--files",
        action="append",
        help="changed file item; repeat or pass multiline text",
    )
    submit.set_defaults(func=cmd_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
