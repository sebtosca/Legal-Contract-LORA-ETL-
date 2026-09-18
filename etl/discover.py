"""Discover Exhibit-10 documents attached to SEC EDGAR filings.

Two-step process:
1. Walk the quarterly full-index (form.idx) to enumerate filings of the
   target root form types (10-K, 10-Q, 8-K, S-1 - the forms that carry
   Exhibit-10 material contracts).
2. For each filing, parse its human-readable index page's "Document Format
   Files" table and pick out documents whose Type is EX-10.* (a material
   contract exhibit) - NOT EX-101.* (XBRL taxonomy files, a false-positive
   trap since "EX-101..." also starts with "EX-10").
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from bs4 import BeautifulSoup

from etl.edgar_client import EdgarClient

TARGET_ROOT_FORMS = ("10-K", "10-Q", "8-K", "S-1")

_EX10_TYPE_RE = re.compile(r"^EX-10(?!\d)", re.IGNORECASE)


@dataclass(frozen=True)
class FilingRef:
    form_type: str
    company: str
    cik: str
    date_filed: str
    accession: str


@dataclass(frozen=True)
class Exhibit10Doc:
    filing: FilingRef
    filename: str
    exhibit_type: str
    url: str


def iter_filings(client: EdgarClient, year: int, quarter: int) -> Iterator[FilingRef]:
    """Yield every filing of a target root form type from one quarterly index."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
    text = client.get_text(url)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("---")) + 1
    for line in lines[start:]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) != 5:
            continue
        form_type, company, cik, date_filed, file_name = parts
        if form_type not in TARGET_ROOT_FORMS:
            continue
        accession = file_name.rsplit("/", 1)[-1].removesuffix(".txt")
        yield FilingRef(
            form_type=form_type,
            company=company,
            cik=cik,
            date_filed=date_filed,
            accession=accession,
        )


def find_ex10_documents(client: EdgarClient, filing: FilingRef) -> list[Exhibit10Doc]:
    """Fetch a filing's index page and return its Exhibit-10 documents, if any."""
    accession_nodash = filing.accession.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{filing.cik}/{accession_nodash}/"
        f"{filing.accession}-index.htm"
    )
    html = client.get_text(index_url)
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="tableFile", summary="Document Format Files")
    if table is None:
        return []

    docs = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        link = cells[2].find("a")
        exhibit_type = cells[3].get_text(strip=True)
        if link is None or not _EX10_TYPE_RE.match(exhibit_type):
            continue
        filename = link.get_text(strip=True)
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{filing.cik}/{accession_nodash}/{filename}"
        docs.append(
            Exhibit10Doc(filing=filing, filename=filename, exhibit_type=exhibit_type, url=doc_url)
        )
    return docs
