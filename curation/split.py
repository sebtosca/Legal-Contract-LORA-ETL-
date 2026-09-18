"""Partition every company (CIK) in the corpus into train/val/test buckets.

Splitting by CIK rather than by individual clause is the leakage guard the
PRD calls for: a single filer's near-identical boilerplate, reused across
many of its own contracts, would otherwise leak across splits and inflate
apparent accuracy. The split is a deterministic hash of the CIK, so it's
reproducible without needing to persist any random state - but the
resulting mapping is still written to disk so it's auditable and so the
leakage test can check it directly.

Usage:
    uv run python -m curation.split
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from indexing.schema import INDEX_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/curated/cik_splits.json")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = "lora-clause-split-v1"


def assign_bucket(cik: str, seed: str = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest()
    frac = (int(digest, 16) % 10_000) / 10_000
    cumulative = 0.0
    for bucket, ratio in SPLIT_RATIOS.items():
        cumulative += ratio
        if frac < cumulative:
            return bucket
    return "test"


def get_unique_ciks(es: Elasticsearch) -> list[str]:
    ciks = set()
    for hit in scan(es, index=INDEX_NAME, query={"query": {"match_all": {}}}, _source=["cik"]):
        ciks.add(hit["_source"]["cik"])
    return sorted(ciks)


def run(es_host: str, output_path: Path) -> dict[str, str]:
    es = Elasticsearch(es_host)
    ciks = get_unique_ciks(es)
    logger.info("Splitting %d unique companies", len(ciks))

    assignments = {cik: assign_bucket(cik) for cik in ciks}
    counts = Counter(assignments.values())
    logger.info("Bucket sizes (companies): %s", dict(counts))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(assignments, indent=2))
    logger.info("Wrote CIK split assignments to %s", output_path)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(es_host=args.es_host, output_path=args.output)


if __name__ == "__main__":
    main()
