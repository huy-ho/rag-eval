"""
data_loader.py
--------------
Loads evaluation datasets from various sources.

    load_dataset(path) -> list[dict]

    path=None  → returns MOCK_DATASET from mock_data.py
    .csv       → pandas; expects columns:
                   question, answer, contexts (pipe-separated), ground_truth
    .json      → top-level list of record dicts
    .jsonl     → one JSON object per line

All loaders validate required keys and skip invalid records with a warning.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from mock_data import MOCK_DATASET

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"question", "answer", "contexts", "ground_truth"}


def _validate_record(record: dict, index: int) -> bool:
    missing = REQUIRED_KEYS - set(record.keys())
    if missing:
        logger.warning("Record %d missing required keys %s — skipped", index, missing)
        return False
    if not isinstance(record["contexts"], list):
        logger.warning("Record %d: 'contexts' must be a list — skipped", index)
        return False
    return True


def load_dataset(path: str | None = None) -> list[dict]:
    """Return a list of validated dataset records.

    Args:
        path: File path to load from, or None to use the built-in mock dataset.

    Returns:
        List of dicts with keys: question, answer, contexts, ground_truth.

    Raises:
        ValueError: If the file format is unsupported or required columns/keys
                    are missing.
        FileNotFoundError: If the specified path does not exist.
    """
    if path is None:
        logger.info(
            "No data path specified — using mock dataset (%d records)", len(MOCK_DATASET)
        )
        return list(MOCK_DATASET)

    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    elif suffix == ".json":
        return _load_json(path, jsonl=False)
    elif suffix == ".jsonl":
        return _load_json(path, jsonl=True)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Supported: .csv, .json, .jsonl"
        )


def _load_csv(path: str) -> list[dict]:
    logger.info("Loading CSV dataset from %s", path)
    df = pd.read_csv(path)

    required_cols = {"question", "answer", "contexts", "ground_truth"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}. "
            f"Got: {list(df.columns)}"
        )

    records = []
    for i, row in df.iterrows():
        record = {
            "question":    str(row["question"]),
            "answer":      str(row["answer"]),
            "contexts":    [c.strip() for c in str(row["contexts"]).split("|")],
            "ground_truth": str(row["ground_truth"]),
        }
        if _validate_record(record, i):
            records.append(record)

    logger.info("Loaded %d valid records from CSV (skipped %d)", len(records), len(df) - len(records))
    return records


def _load_json(path: str, jsonl: bool = False) -> list[dict]:
    fmt = "JSONL" if jsonl else "JSON"
    logger.info("Loading %s dataset from %s", fmt, path)

    with open(path, "r", encoding="utf-8") as f:
        if jsonl:
            raw = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"{fmt} file must contain a top-level list of record dicts. "
            f"Got: {type(raw).__name__}"
        )

    records = []
    for i, record in enumerate(raw):
        if not isinstance(record, dict):
            logger.warning("Record %d is not a dict (%s) — skipped", i, type(record).__name__)
            continue
        if _validate_record(record, i):
            records.append(record)

    logger.info(
        "Loaded %d valid records from %s (skipped %d)",
        len(records), fmt, len(raw) - len(records),
    )
    return records
