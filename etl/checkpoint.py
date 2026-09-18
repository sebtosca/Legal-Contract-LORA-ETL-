"""SQLite-backed checkpoint store so a killed/interrupted download resumes
instead of restarting from zero."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CHECKPOINT_PATH = Path("etl/.download_checkpoint.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS downloads (
    accession   TEXT NOT NULL,
    filename    TEXT NOT NULL,
    cik         TEXT NOT NULL,
    company     TEXT,
    form_type   TEXT,
    file_type   TEXT,
    status      TEXT NOT NULL,
    downloaded_at TEXT,
    PRIMARY KEY (accession, filename)
);
"""


class Checkpoint:
    def __init__(self, path: Path = DEFAULT_CHECKPOINT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def is_done(self, accession: str, filename: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM downloads WHERE accession = ? AND filename = ? AND status = 'done'",
            (accession, filename),
        ).fetchone()
        return row is not None

    def mark_done(self, *, accession, filename, cik, company, form_type, file_type):
        self._conn.execute(
            """
            INSERT INTO downloads (accession, filename, cik, company, form_type, file_type, status, downloaded_at)
            VALUES (?, ?, ?, ?, ?, ?, 'done', datetime('now'))
            ON CONFLICT (accession, filename) DO UPDATE SET
                status = 'done', downloaded_at = datetime('now')
            """,
            (accession, filename, cik, company, form_type, file_type),
        )
        self._conn.commit()

    def count_done(self) -> int:
        (count,) = self._conn.execute(
            "SELECT COUNT(*) FROM downloads WHERE status = 'done'"
        ).fetchone()
        return count

    def close(self):
        self._conn.close()


@contextmanager
def open_checkpoint(path: Path = DEFAULT_CHECKPOINT_PATH):
    checkpoint = Checkpoint(path)
    try:
        yield checkpoint
    finally:
        checkpoint.close()
