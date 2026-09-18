"""Backfill SIC industry codes and exact filing dates, keyed by CIK/accession.

The Spark ETL job (Phase 2) only has what the EDGAR full-index and filing
index pages expose directly: company, CIK, form type, accession. SIC code
and exact filing date aren't in either of those - they come from SEC's
per-company submissions API instead, which also happens to list each
company's filings (with accessionNumber + filingDate), so one rate-limited
call per unique CIK backfills both at once.

Usage:
    uv run python -m etl.company_metadata --input data/clauses.parquet
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from etl.edgar_client import EdgarClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("etl/.company_metadata.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik TEXT PRIMARY KEY,
    sic TEXT,
    sic_description TEXT
);
CREATE TABLE IF NOT EXISTS filing_dates (
    cik TEXT NOT NULL,
    accession TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    PRIMARY KEY (cik, accession)
);
"""


def _fetch_one(client: EdgarClient, cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    return client.get_json(url)


def backfill(clauses_path: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    df = pd.read_parquet(clauses_path, columns=["cik"])
    unique_ciks = sorted(df["cik"].unique())
    logger.info("Backfilling SIC + filing dates for %d unique companies", len(unique_ciks))

    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    already_done = {row[0] for row in conn.execute("SELECT cik FROM companies").fetchall()}

    client = EdgarClient()
    for i, cik in enumerate(unique_ciks, 1):
        if cik in already_done:
            continue
        try:
            data = _fetch_one(client, cik)
        except Exception:
            logger.exception("Failed to fetch submissions for CIK %s, skipping", cik)
            continue

        conn.execute(
            "INSERT OR REPLACE INTO companies (cik, sic, sic_description) VALUES (?, ?, ?)",
            (cik, data.get("sic"), data.get("sicDescription")),
        )
        recent = data.get("filings", {}).get("recent", {})
        rows = list(zip(recent.get("accessionNumber", []), recent.get("filingDate", [])))
        conn.executemany(
            "INSERT OR REPLACE INTO filing_dates (cik, accession, filing_date) VALUES (?, ?, ?)",
            [(cik, accession, filing_date) for accession, filing_date in rows],
        )
        conn.commit()

        if i % 200 == 0:
            logger.info("[%d/%d] companies done", i, len(unique_ciks))

    logger.info("Done. %d companies in metadata DB", conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0])
    conn.close()


def load_metadata(db_path: Path = DEFAULT_DB_PATH) -> tuple[dict[str, dict], dict[tuple[str, str], str]]:
    """Read-only load for the indexing step: cik -> {sic, sic_description},
    (cik, accession) -> filing_date."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        companies = {
            cik: {"sic": sic, "sic_description": sic_description}
            for cik, sic, sic_description in conn.execute("SELECT cik, sic, sic_description FROM companies")
        }
        filing_dates = {
            (cik, accession): filing_date
            for cik, accession, filing_date in conn.execute("SELECT cik, accession, filing_date FROM filing_dates")
        }
    finally:
        conn.close()
    return companies, filing_dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/clauses.parquet")
    args = parser.parse_args()
    backfill(args.input)


if __name__ == "__main__":
    main()
