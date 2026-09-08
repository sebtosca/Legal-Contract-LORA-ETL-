"""Pure function: raw SEC exhibit document -> labeled clause records.

Header-bootstrap labeling, consistent with the LEDGAR methodology: a clause's
label is derived from its section header text, matched against the fixed
10-category taxonomy by keyword. No Spark dependency - this operates on one
document's raw text so it's unit-testable on its own, then wrapped for Spark
in Phase 2 via mapPartitions.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

TAXONOMY = (
    "Indemnification",
    "Limitation of Liability",
    "Termination",
    "Governing Law",
    "Confidentiality",
    "Assignment",
    "Force Majeure",
    "Payment Terms",
    "Warranty/Representations",
    "Dispute Resolution/Arbitration",
)

# Ordered so a more specific phrase (e.g. "limitation of liability") is
# checked before a shorter one that could otherwise be mistaken for it.
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "Limitation of Liability",
        [
            "limitation of liability",
            "limitation on liability",
            "limitations of liability",
            "limitations on liability",
            "limit of liability",
            "liability limitation",
        ],
    ),
    ("Dispute Resolution/Arbitration", ["arbitrat", "dispute resolution"]),
    ("Governing Law", ["governing law", "choice of law", "applicable law"]),
    ("Force Majeure", ["force majeure"]),
    ("Confidentiality", ["confidential", "non-disclosure", "nondisclosure"]),
    ("Indemnification", ["indemnif"]),
    ("Termination", ["terminat"]),
    ("Assignment", ["assignment", "assign"]),
    ("Payment Terms", ["payment"]),
    ("Warranty/Representations", ["warrant", "representation"]),
]

_MIN_CLAUSE_CHARS = 30
_MAX_HEADER_CHARS = 100

# Table-of-contents lines end in a tab/dot-leader/non-breaking-space-run +
# page number (SEC's HTML export renders the leader as any of these); a real
# section header in the body doesn't.
_TOC_SUFFIX_RE = re.compile(r"(\t+|[\xa0]{2,}|\.{2,})\s*\d{1,4}\s*$")

# Captures the numbering token so its dot-depth can become a heading level,
# e.g. "5.1 Orders and Forecasts" -> number="5.1", level=2.
_NUMBERED_HEADER_RE = re.compile(r"^(?:Section\s+|Article\s+)?(\d+(?:\.\d+)*)\.?\s+\S")
_ARTICLE_HEADER_RE = re.compile(r"^ARTICLE\s+[IVXLC\d]+\b", re.IGNORECASE)

# A numbering token alone on its own line, e.g. "5.1" - HTML export commonly
# splits "5.1 Orders and Forecasts" into separate spans/lines for the number
# and the title.
_STANDALONE_NUMBER_RE = re.compile(r"^\(?\d+(?:\.\d+)*\)?\.?$")

# Short lowercase connectors that commonly appear inside an otherwise
# title-case header, e.g. "Limitation of Liability", "Limitations on Exercise".
_TITLE_CASE_CONNECTORS = {"of", "on", "in", "for", "and", "to", "the", "a", "an", "or", "by", "with", "from"}
_MAX_TITLE_CASE_WORDS = 8


def _is_title_case_header(line: str) -> bool:
    words = re.split(r"[\s,/\-]+", line.strip())
    words = [w for w in words if w]
    if not words or len(words) > _MAX_TITLE_CASE_WORDS:
        return False
    if not words[0][0].isupper() or not words[-1][0].isupper():
        return False
    return all(w[0].isupper() or w.lower() in _TITLE_CASE_CONNECTORS for w in words)


@dataclass(frozen=True)
class Clause:
    source_id: str
    label: str
    section_header: str
    text: str


def _strip_sgml_preamble(raw: str) -> str:
    """SEC exhibit files are wrapped in an SGML <DOCUMENT>/<TYPE>/<SEQUENCE>/
    <FILENAME>/<DESCRIPTION>/<TEXT> preamble before the actual document
    content starts; keep only what's inside <TEXT>."""
    match = re.search(r"<TEXT>", raw, re.IGNORECASE)
    return raw[match.end():] if match else raw


def _extract_lines(raw: str) -> list[str]:
    body = _strip_sgml_preamble(raw)
    text = BeautifulSoup(body, "lxml").get_text("\n")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_all_caps(line: str) -> bool:
    alpha = [c for c in line if c.isalpha()]
    return bool(alpha) and sum(c.isupper() for c in alpha) / len(alpha) > 0.9


def _is_plausible_header_line(line: str) -> bool:
    """Structural check only (length, not a TOC line) - used both for
    top-level candidates and to validate what follows a standalone number."""
    return bool(line) and len(line) <= _MAX_HEADER_CHARS and not _TOC_SUFFIX_RE.search(line)


def _number_depth(token: str) -> int:
    """'5' -> 1, '5.1' -> 2, '5.1.2' -> 3."""
    return token.strip("()").rstrip(".").count(".") + 1


@dataclass(frozen=True)
class _HeaderMatch:
    start_idx: int  # first line consumed by this header (number line, if any)
    title_idx: int  # line holding the actual header text
    header_text: str
    level: int


def _find_headers(lines: list[str]) -> list[_HeaderMatch]:
    """Locate section headers and assign each a nesting level inferred from
    its numbering depth (e.g. "5" = level 1, "5.1" = level 2). Headers with
    no explicit numbering (ALL-CAPS titles, ARTICLE headers, bare title-case
    names like "Force Majeure") are level 1: in this corpus, a subsection
    virtually always carries its own number, so the absence of one is itself
    the signal that a header is top-level.
    """
    matches: list[_HeaderMatch] = []
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if not _is_plausible_header_line(line):
            i += 1
            continue

        # Case 1: standalone number line, e.g. "5.1", immediately followed by
        # its title on the next line (a common HTML-export split). The next
        # line must itself look like a short header phrase, not a sentence
        # (a numbered paragraph like "9.3\nLicensee will make such payment..."
        # would otherwise be misread as a header).
        if _STANDALONE_NUMBER_RE.match(line) and i + 1 < n and _is_plausible_header_line(lines[i + 1]):
            title = lines[i + 1]
            if _is_all_caps(title) or _is_title_case_header(title):
                matches.append(_HeaderMatch(i, i + 1, title, _number_depth(line)))
                i += 2
                continue

        # Case 2: numbered header on a single line, e.g. "5.1 Orders and Forecasts".
        m = _NUMBERED_HEADER_RE.match(line)
        if m:
            matches.append(_HeaderMatch(i, i, line, _number_depth(m.group(1))))
            i += 1
            continue

        # Case 3: no numbering at all - ALL-CAPS, ARTICLE, or bare title-case.
        if _is_all_caps(line) or _ARTICLE_HEADER_RE.match(line) or _is_title_case_header(line):
            matches.append(_HeaderMatch(i, i, line, 1))
            i += 1
            continue

        i += 1
    return matches


_KEYWORD_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (label, [re.compile(r"\b" + re.escape(keyword)) for keyword in keywords])
    for label, keywords in _CATEGORY_KEYWORDS
]


def _classify_header(header_text: str) -> str | None:
    lowered = header_text.lower()
    for label, patterns in _KEYWORD_PATTERNS:
        if any(pattern.search(lowered) for pattern in patterns):
            return label
    return None


def extract_clauses(raw: str, source_id: str) -> list[Clause]:
    """Extract taxonomy-labeled clauses from one filing's raw exhibit text."""
    lines = _extract_lines(raw)
    headers = _find_headers(lines)

    candidates: dict[tuple[str, str], Clause] = {}
    for pos, header in enumerate(headers):
        label = _classify_header(header.header_text)
        if label is None:
            continue
        # A clause runs until the next header of equal-or-shallower nesting
        # level, so a deeper subsection (e.g. "5.1 Orders and Forecasts")
        # doesn't truncate the top-level clause it belongs to.
        end = len(lines)
        for later in headers[pos + 1:]:
            if later.level <= header.level:
                end = later.start_idx
                break
        body = "\n".join(lines[header.title_idx + 1:end]).strip()
        if len(body) < _MIN_CLAUSE_CHARS:
            continue
        # A section header can appear twice: once in a table of contents
        # (where the "body" up to the next TOC line is meaningless), once at
        # the real section (with substantive body text). Keep the longer one.
        # Normalize whitespace around hyphens for the key only (not display)
        # since "ARTICLE VI - Foo" vs "ARTICLE VI- Foo" are the same header.
        normalized = re.sub(r"\s+", " ", header.header_text.strip().lower())
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        key = (label, normalized)
        existing = candidates.get(key)
        if existing is None or len(body) > len(existing.text):
            candidates[key] = Clause(source_id=source_id, label=label, section_header=header.header_text, text=body)
    return list(candidates.values())
