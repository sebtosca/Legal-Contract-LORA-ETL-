"""Orchestrates Phase 4 end-to-end: CIK split, stratified train/val sampling,
eval-set candidate generation, and a final report of split sizes and
per-category distribution.

The eval set's final 30-50 examples are a manual selection from the
candidates this produces (see data/curated/eval_candidates.jsonl) - written
out separately as data/curated/eval.jsonl once reviewed, the same way the
Phase 1 fixture filings were hand-picked rather than auto-selected.

Usage:
    uv run python -m curation.build_dataset
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from curation.eval_set import build_candidates
from curation.sample import run as sample_run
from curation.split import run as split_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CURATED_DIR = Path("data/curated")


def _report(train: pd.DataFrame, val: pd.DataFrame, eval_path: Path) -> None:
    lines = ["# Phase 4 curated dataset report", ""]
    for name, df in [("train", train), ("val", val)]:
        lines.append(f"## {name} ({len(df)} examples, {df['cik'].nunique()} companies)")
        lines.append("")
        lines.append(df["label"].value_counts().to_string())
        lines.append("")

    if eval_path.exists():
        eval_df = pd.read_json(eval_path, lines=True)
        lines.append(f"## eval ({len(eval_df)} examples, {eval_df['cik'].nunique()} companies)")
        lines.append("")
        lines.append(eval_df["label"].value_counts().to_string())
    else:
        lines.append(
            "## eval\n\nNot yet hand-selected - see data/curated/eval_candidates.jsonl "
            "for the candidate pool to review."
        )

    report_path = CURATED_DIR / "dataset_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", report_path)
    print("\n".join(lines))


def run(es_host: str, train_per_category: int, train_company_cap: int,
        val_per_category: int, val_company_cap: int, eval_per_category: int) -> None:
    split_run(es_host=es_host, output_path=CURATED_DIR / "cik_splits.json")

    train = sample_run(bucket="train", per_category=train_per_category, company_cap=train_company_cap,
                        es_host=es_host, output_path=CURATED_DIR / "train.jsonl")
    val = sample_run(bucket="val", per_category=val_per_category, company_cap=val_company_cap,
                      es_host=es_host, output_path=CURATED_DIR / "val.jsonl")

    candidates = build_candidates(bucket="test", per_category=eval_per_category, es_host=es_host)
    candidates_path = CURATED_DIR / "eval_candidates.jsonl"
    candidates.to_json(candidates_path, orient="records", lines=True)
    logger.info("Wrote %s - hand-select the final 30-50 into eval.jsonl", candidates_path)

    _report(train, val, CURATED_DIR / "eval.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--train-per-category", type=int, default=100)
    parser.add_argument("--train-company-cap", type=int, default=10)
    parser.add_argument("--val-per-category", type=int, default=15)
    parser.add_argument("--val-company-cap", type=int, default=5)
    parser.add_argument("--eval-per-category", type=int, default=8)
    args = parser.parse_args()
    run(
        es_host=args.es_host,
        train_per_category=args.train_per_category,
        train_company_cap=args.train_company_cap,
        val_per_category=args.val_per_category,
        val_company_cap=args.val_company_cap,
        eval_per_category=args.eval_per_category,
    )


if __name__ == "__main__":
    main()
