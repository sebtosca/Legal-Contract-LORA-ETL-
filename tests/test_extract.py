"""Unit tests for the pure extraction/labeling function against hand-verified
SEC filings (see tests/fixtures/filings/). Per the PRD's Testing Decisions,
this is the highest-value seam: it must be correct independent of Spark
before Phase 2 wraps it in a distributed job.
"""

import json
from pathlib import Path

import pytest

from etl.extract import TAXONOMY, extract_clauses

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "filings"
FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.htm"))


@pytest.mark.parametrize("htm_path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_extraction_matches_hand_verified_labels(htm_path: Path):
    expected = json.loads(htm_path.with_suffix(".expected.json").read_text())
    raw = htm_path.read_text(encoding="utf-8", errors="replace")

    clauses = extract_clauses(raw, source_id=htm_path.stem)
    actual = [{"label": c.label, "section_header": c.section_header} for c in clauses]

    assert actual == expected


def test_all_extracted_labels_are_in_taxonomy():
    for htm_path in FIXTURE_FILES:
        raw = htm_path.read_text(encoding="utf-8", errors="replace")
        for clause in extract_clauses(raw, source_id=htm_path.stem):
            assert clause.label in TAXONOMY


def test_short_document_yields_no_clauses():
    htm_path = FIXTURES_DIR / "04_short_doc_zero_clauses.htm"
    raw = htm_path.read_text(encoding="utf-8", errors="replace")
    assert extract_clauses(raw, source_id="short") == []
