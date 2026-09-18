"""Bulk-index the extracted clause corpus into Elasticsearch, joining in
SIC code and exact filing date from the Phase 3 company-metadata backfill.

Usage:
    uv run python -m indexing.bulk_index --input data/clauses.parquet
"""

import argparse
import hashlib
import logging

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from etl.company_metadata import DEFAULT_DB_PATH, load_metadata
from indexing.schema import INDEX_MAPPING, INDEX_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _clause_id(row: dict) -> str:
    raw = f"{row['accession']}:{row['filename']}:{row['label']}:{row['section_header']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _iter_actions(df: pd.DataFrame, companies: dict, filing_dates: dict):
    for row in df.to_dict(orient="records"):
        company_meta = companies.get(row["cik"], {})
        yield {
            "_index": INDEX_NAME,
            "_id": _clause_id(row),
            "_source": {
                "clause_id": _clause_id(row),
                "label": row["label"],
                "section_header": row["section_header"],
                "text": row["text"],
                "cik": row["cik"],
                "company": row["company"],
                "form_type": row["form_type"],
                "file_type": row["file_type"],
                "accession": row["accession"],
                "filename": row["filename"],
                "filing_date": filing_dates.get((row["cik"], row["accession"])),
                "filing_year": row["filing_year"],
                "sic": company_meta.get("sic"),
                "sic_description": company_meta.get("sic_description"),
                "text_len": len(row["text"]),
            },
        }


def run(input_path: str, es_host: str, recreate: bool) -> None:
    es = Elasticsearch(es_host)

    if recreate and es.indices.exists(index=INDEX_NAME):
        logger.info("Deleting existing index %s", INDEX_NAME)
        es.indices.delete(index=INDEX_NAME)
    if not es.indices.exists(index=INDEX_NAME):
        logger.info("Creating index %s", INDEX_NAME)
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

    df = pd.read_parquet(input_path)
    companies, filing_dates = load_metadata(DEFAULT_DB_PATH)
    logger.info("Loaded metadata for %d companies, %d filing dates", len(companies), len(filing_dates))

    ok_count = 0
    error_count = 0
    for ok, item in streaming_bulk(es, _iter_actions(df, companies, filing_dates), chunk_size=2000):
        if ok:
            ok_count += 1
        else:
            error_count += 1
            logger.warning("Index failure: %s", item)

    es.indices.refresh(index=INDEX_NAME)
    logger.info("Indexed %d documents (%d failures)", ok_count, error_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/clauses.parquet")
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the index first")
    args = parser.parse_args()
    run(input_path=args.input, es_host=args.es_host, recreate=args.recreate)


if __name__ == "__main__":
    main()
