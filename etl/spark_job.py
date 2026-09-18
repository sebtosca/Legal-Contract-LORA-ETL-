"""Distributed clause extraction over the full downloaded corpus.

Wraps the Phase 1 pure function (etl.extract.extract_clauses) via
mapPartitions so the actual extraction/labeling logic stays identical to
what's unit-tested in tests/test_extract.py - Spark here is purely a
parallel-execution layer, not a place where new extraction logic lives.

Usage:
    uv run python -m etl.spark_job --input data/raw --output data/clauses.parquet
"""

import argparse
import logging
import sqlite3
from pathlib import Path

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from etl.checkpoint import DEFAULT_CHECKPOINT_PATH
from etl.extract import extract_clauses

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_SCHEMA = StructType(
    [
        StructField("label", StringType(), False),
        StructField("section_header", StringType(), False),
        StructField("text", StringType(), False),
        StructField("cik", StringType(), False),
        StructField("company", StringType(), True),
        StructField("form_type", StringType(), True),
        StructField("file_type", StringType(), True),
        StructField("accession", StringType(), False),
        StructField("filename", StringType(), False),
        # Coarse (year-only) proxy derived from the accession number's
        # embedded 2-digit year, e.g. "0001493152-23-002944" -> 2023. Exact
        # filing dates and SIC codes aren't in this metadata yet - see
        # Phase 3, where they're joined in for Elasticsearch indexing.
        StructField("filing_year", StringType(), True),
    ]
)


def _load_filing_metadata(checkpoint_path: Path) -> dict[tuple[str, str], dict]:
    """Read-only snapshot of (accession, filename) -> metadata from the
    checkpoint DB. Uses a plain SELECT so it's safe to run even while the
    downloader is still writing to the same file in the background."""
    conn = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT accession, filename, cik, company, form_type, file_type FROM downloads WHERE status = 'done'"
        ).fetchall()
    finally:
        conn.close()
    return {
        (accession, filename): {"cik": cik, "company": company, "form_type": form_type, "file_type": file_type}
        for accession, filename, cik, company, form_type, file_type in rows
    }


def _filing_year(accession: str) -> str | None:
    parts = accession.split("-")
    if len(parts) == 3 and len(parts[1]) == 2 and parts[1].isdigit():
        return f"20{parts[1]}"
    return None


def _process_partition(rows, metadata: dict[tuple[str, str], dict]):
    for file_path, raw_text in rows:
        path = Path(file_path)
        accession = path.parent.name
        filename = path.name
        cik = path.parent.parent.name
        meta = metadata.get((accession, filename))
        if meta is None:
            continue  # not a completed download per the checkpoint; skip

        try:
            clauses = extract_clauses(raw_text, source_id=f"{accession}/{filename}")
        except Exception:
            logger.exception("Failed to extract clauses from %s, skipping", file_path)
            continue

        for clause in clauses:
            yield Row(
                label=clause.label,
                section_header=clause.section_header,
                text=clause.text,
                cik=cik,
                company=meta["company"],
                form_type=meta["form_type"],
                file_type=meta["file_type"],
                accession=accession,
                filename=filename,
                filing_year=_filing_year(accession),
            )


def run(input_dir: str, output_path: str, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    metadata = _load_filing_metadata(checkpoint_path)
    logger.info("Loaded metadata for %d completed downloads", len(metadata))

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("legal-clause-extraction")
        # wholeTextFiles buffers each partition's raw file contents in the
        # driver heap; the default (~2 partitions regardless of core count)
        # tried to hold a ~25,000-file share in a 1g heap and OOM'd. Bump
        # both so a full ~50k-file corpus fits comfortably.
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        # data/raw/{cik}/{accession}/{filename}.htm - Hadoop's glob doesn't
        # support "**" recursion, so the depth has to be spelled out.
        num_partitions = spark.sparkContext.defaultParallelism * 4
        files_rdd = spark.sparkContext.wholeTextFiles(f"{input_dir}/*/*/*.htm", minPartitions=num_partitions)
        clauses_rdd = files_rdd.mapPartitions(lambda rows: _process_partition(rows, metadata))
        df = spark.createDataFrame(clauses_rdd, schema=OUTPUT_SCHEMA)

        count = df.count()
        logger.info("Extracted %d clauses", count)

        df.write.mode("overwrite").parquet(output_path)
        logger.info("Wrote output to %s", output_path)
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/clauses.parquet")
    args = parser.parse_args()
    run(input_dir=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
