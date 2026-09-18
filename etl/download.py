"""Bulk-download SEC EDGAR Exhibit-10 documents, respecting the fair-access
rate limit and resuming from the checkpoint on restart.

Usage:
    uv run python -m etl.download --limit 300
    uv run python -m etl.download --limit 300 --start-year 2023 --start-quarter 4
"""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

from etl.checkpoint import open_checkpoint
from etl.discover import Exhibit10Doc, find_ex10_documents, iter_filings
from etl.edgar_client import EdgarClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")


def iter_quarters_backward(start_year: int, start_quarter: int) -> Iterator[tuple[int, int]]:
    year, quarter = start_year, start_quarter
    while year > 2000:
        yield year, quarter
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1


def download_document(client: EdgarClient, doc: Exhibit10Doc) -> Path:
    dest_dir = RAW_DATA_DIR / doc.filing.cik / doc.filing.accession
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / doc.filename
    dest_path.write_text(client.get_text(doc.url), encoding="utf-8")
    return dest_path


def run(limit: int, start_year: int, start_quarter: int) -> None:
    client = EdgarClient()
    with open_checkpoint() as checkpoint:
        downloaded_this_run = 0
        already_done = checkpoint.count_done()
        logger.info("Resuming: %d documents already downloaded", already_done)

        for year, quarter in iter_quarters_backward(start_year, start_quarter):
            if downloaded_this_run >= limit:
                break
            logger.info("Scanning %d QTR%d", year, quarter)
            for filing in iter_filings(client, year, quarter):
                if downloaded_this_run >= limit:
                    break
                try:
                    docs = find_ex10_documents(client, filing)
                except Exception:
                    logger.exception("Failed to read index for %s, skipping", filing.accession)
                    continue

                for doc in docs:
                    if downloaded_this_run >= limit:
                        break
                    if checkpoint.is_done(doc.filing.accession, doc.filename):
                        continue
                    try:
                        download_document(client, doc)
                    except Exception:
                        logger.exception("Failed to download %s/%s, skipping", doc.filing.accession, doc.filename)
                        continue
                    checkpoint.mark_done(
                        accession=doc.filing.accession,
                        filename=doc.filename,
                        cik=doc.filing.cik,
                        company=doc.filing.company,
                        form_type=doc.filing.form_type,
                        file_type=doc.exhibit_type,
                    )
                    downloaded_this_run += 1
                    logger.info(
                        "[%d/%d] %s %s (%s)",
                        downloaded_this_run, limit, doc.filing.company, doc.filename, doc.exhibit_type,
                    )

        logger.info(
            "Run complete: %d new documents downloaded, %d total in checkpoint",
            downloaded_this_run, checkpoint.count_done(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, required=True, help="stop after downloading this many new EX-10 documents")
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--start-quarter", type=int, default=4)
    args = parser.parse_args()
    run(limit=args.limit, start_year=args.start_year, start_quarter=args.start_quarter)


if __name__ == "__main__":
    main()
