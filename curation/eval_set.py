"""Build a candidate pool for the hand-picked evaluation set, biased toward
genuinely ambiguous/edge cases rather than a plain random sample - per the
PRD, the eval set should deliberately include hard cases, since that's what
makes it a meaningful benchmark rather than an easy rubber stamp.

This produces *candidates* (more than the target 30-50), not the final set -
the final selection is a manual review pass, the same way the Phase 1
fixture filings were hand-picked.

Usage:
    uv run python -m curation.eval_set --per-category 8
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from curation.sample import CIK_SPLITS_PATH, _load_bucket_ciks
from etl.extract import TAXONOMY, _KEYWORD_PATTERNS
from indexing.schema import INDEX_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _matching_categories(header_text: str) -> list[str]:
    """All taxonomy categories whose keywords appear in this header, not just
    the first match - a header matching 2+ categories (e.g. "Indemnification;
    Limitation of Liability; Insurance") is inherently ambiguous."""
    lowered = header_text.lower()
    return [label for label, patterns in _KEYWORD_PATTERNS if any(p.search(lowered) for p in patterns)]


def _ambiguity_score(row: dict) -> int:
    score = len(_matching_categories(row["section_header"])) - 1  # 0 if only its own category matched
    if row["text_len"] < 60:  # near the extraction function's own minimum-length floor
        score += 1
    if row["section_header"].strip().rstrip(".:").lower() in {"confidential", "termination", "payment"}:
        score += 1  # bare single-word headers, prone to the watermark-style false positive we found in Phase 2
    return score


def build_candidates(bucket: str, per_category: int, es_host: str) -> pd.DataFrame:
    ciks = _load_bucket_ciks(bucket, CIK_SPLITS_PATH)
    es = Elasticsearch(es_host)
    query = {"query": {"terms": {"cik": ciks}}}
    df = pd.DataFrame([hit["_source"] for hit in scan(es, index=INDEX_NAME, query=query)])
    df["text_len"] = df["text"].str.len()
    df["ambiguity_score"] = df.apply(lambda r: _ambiguity_score(r.to_dict()), axis=1)

    pools = []
    for label in TAXONOMY:
        subset = df[df["label"] == label].sort_values("ambiguity_score", ascending=False)
        pools.append(subset.head(per_category))
    candidates = pd.concat(pools, ignore_index=True)
    logger.info("Built %d candidates (%d per category) from the '%s' bucket", len(candidates), per_category, bucket)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="test")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--output", type=Path, default=Path("data/curated/eval_candidates.jsonl"))
    args = parser.parse_args()

    candidates = build_candidates(args.bucket, args.per_category, args.es_host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_json(args.output, orient="records", lines=True)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
