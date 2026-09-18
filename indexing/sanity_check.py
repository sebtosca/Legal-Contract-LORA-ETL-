"""Phase 3 verification: does the ES index match the Spark output, and do
basic filtered queries behave as expected?

Usage:
    uv run python -m indexing.sanity_check
"""

import logging

import pandas as pd
from elasticsearch import Elasticsearch

from indexing.schema import INDEX_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(es_host: str = "http://localhost:9200", clauses_path: str = "data/clauses.parquet") -> None:
    es = Elasticsearch(es_host)

    es_count = es.count(index=INDEX_NAME)["count"]
    parquet_count = len(pd.read_parquet(clauses_path, columns=["label"]))
    logger.info("ES doc count: %d | Parquet row count: %d", es_count, parquet_count)
    assert es_count == parquet_count, "ES doc count does not match Spark output row count"

    logger.info("Counts per category (via ES aggregation):")
    agg = es.search(
        index=INDEX_NAME,
        size=0,
        aggs={"by_label": {"terms": {"field": "label", "size": 20}}},
    )
    for bucket in agg["aggregations"]["by_label"]["buckets"]:
        logger.info("  %s: %d", bucket["key"], bucket["doc_count"])

    logger.info("Sample filtered query: Indemnification clauses with a known SIC code, filing_date in 2023")
    result = es.search(
        index=INDEX_NAME,
        size=3,
        query={
            "bool": {
                "filter": [
                    {"term": {"label": "Indemnification"}},
                    {"exists": {"field": "sic"}},
                    {"range": {"filing_date": {"gte": "2023-01-01", "lt": "2024-01-01"}}},
                ]
            }
        },
    )
    logger.info("  matched %d docs, showing up to 3:", result["hits"]["total"]["value"])
    for hit in result["hits"]["hits"]:
        src = hit["_source"]
        logger.info("    %s | %s | sic=%s | filing_date=%s", src["company"], src["section_header"], src["sic"], src["filing_date"])

    logger.info("All sanity checks passed.")


if __name__ == "__main__":
    run()
