"""Stratified sampling of the curated training/validation sets from the ES
index, restricted to one split's CIKs (see curation.split) and balanced
across the 10 taxonomy categories.

A per-company cap is applied on top of the category balance: without it, a
high-volume filer (e.g. an auto-loan securitization issuer with 1,000+
near-identical exhibits) could still dominate the curated set even after
restricting to its assigned split - it would just dominate *within* that
split instead of leaking across splits.

Usage:
    uv run python -m curation.sample --bucket train --per-category 100 --company-cap 10
    uv run python -m curation.sample --bucket val --per-category 15 --company-cap 5
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from etl.extract import TAXONOMY
from indexing.schema import INDEX_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CIK_SPLITS_PATH = Path("data/curated/cik_splits.json")
SAMPLE_SEED = 42


def _load_bucket_ciks(bucket: str, cik_splits_path: Path = CIK_SPLITS_PATH) -> list[str]:
    assignments = json.loads(cik_splits_path.read_text())
    return [cik for cik, b in assignments.items() if b == bucket]


def _fetch_bucket_docs(es: Elasticsearch, ciks: list[str]) -> pd.DataFrame:
    # ES terms filters are capped (index.max_terms_count, default 65536) -
    # comfortably above our CIK counts, so one query is fine.
    query = {"query": {"terms": {"cik": ciks}}}
    docs = [hit["_source"] for hit in scan(es, index=INDEX_NAME, query=query)]
    return pd.DataFrame(docs)


def stratified_sample(
    df: pd.DataFrame, per_category: int, company_cap: int, seed: int = SAMPLE_SEED
) -> pd.DataFrame:
    rng = random.Random(seed)
    company_usage: Counter[str] = Counter()
    selected_rows = []

    for label in TAXONOMY:
        candidates = df[df["label"] == label].to_dict(orient="records")
        rng.shuffle(candidates)

        count = 0
        for row in candidates:
            if company_usage[row["cik"]] >= company_cap:
                continue
            selected_rows.append(row)
            company_usage[row["cik"]] += 1
            count += 1
            if count >= per_category:
                break

        if count < per_category:
            logger.warning(
                "%s: only found %d/%d examples within the per-company cap (company_cap=%d)",
                label, count, per_category, company_cap,
            )
        else:
            logger.info("%s: sampled %d examples", label, count)

    return pd.DataFrame(selected_rows)


def run(bucket: str, per_category: int, company_cap: int, es_host: str, output_path: Path) -> pd.DataFrame:
    ciks = _load_bucket_ciks(bucket)
    logger.info("Bucket '%s': %d companies", bucket, len(ciks))

    es = Elasticsearch(es_host)
    df = _fetch_bucket_docs(es, ciks)
    logger.info("Fetched %d candidate clauses from this bucket", len(df))

    sampled = stratified_sample(df, per_category=per_category, company_cap=company_cap)
    logger.info(
        "Final sample: %d examples across %d companies",
        len(sampled), sampled["cik"].nunique() if len(sampled) else 0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_json(output_path, orient="records", lines=True)
    logger.info("Wrote %s", output_path)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, choices=["train", "val", "test"])
    parser.add_argument("--per-category", type=int, required=True)
    parser.add_argument("--company-cap", type=int, required=True)
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"data/curated/{args.bucket}.jsonl")
    run(bucket=args.bucket, per_category=args.per_category, company_cap=args.company_cap,
        es_host=args.es_host, output_path=output)


if __name__ == "__main__":
    main()
