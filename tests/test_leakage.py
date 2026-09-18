"""Integration test: the main defense against inflated eval numbers is that
no company (CIK) appears in more than one of train/val/test - otherwise a
filer's near-duplicate boilerplate could leak across splits."""

import json
from pathlib import Path

import pandas as pd

CURATED_DIR = Path("data/curated")


def test_cik_splits_assignment_is_a_partition():
    assignments = json.loads((CURATED_DIR / "cik_splits.json").read_text())
    buckets = set(assignments.values())
    assert buckets <= {"train", "val", "test"}


def test_no_cik_appears_in_more_than_one_curated_set():
    train = pd.read_json(CURATED_DIR / "train.jsonl", lines=True)
    val = pd.read_json(CURATED_DIR / "val.jsonl", lines=True)
    eval_ = pd.read_json(CURATED_DIR / "eval.jsonl", lines=True)

    train_ciks = set(train["cik"])
    val_ciks = set(val["cik"])
    eval_ciks = set(eval_["cik"])

    assert not (train_ciks & val_ciks), f"CIKs in both train and val: {train_ciks & val_ciks}"
    assert not (train_ciks & eval_ciks), f"CIKs in both train and eval: {train_ciks & eval_ciks}"
    assert not (val_ciks & eval_ciks), f"CIKs in both val and eval: {val_ciks & eval_ciks}"


def test_curated_sets_agree_with_the_cik_split_assignment():
    assignments = json.loads((CURATED_DIR / "cik_splits.json").read_text())
    train = pd.read_json(CURATED_DIR / "train.jsonl", lines=True)
    val = pd.read_json(CURATED_DIR / "val.jsonl", lines=True)
    eval_ = pd.read_json(CURATED_DIR / "eval.jsonl", lines=True)

    for df, expected_bucket in [(train, "train"), (val, "val"), (eval_, "test")]:
        actual_buckets = {assignments[cik] for cik in df["cik"].astype(str)}
        assert actual_buckets == {expected_bucket}, (
            f"expected all CIKs assigned to '{expected_bucket}', found {actual_buckets}"
        )
