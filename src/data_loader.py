"""
data_loader.py — Load raw CSV files (train, test, sample_submission)
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_train() -> pd.DataFrame:
    """Load raw train CSV."""
    path = cfg.data.raw_data_dir / cfg.data.train_file
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded train      → shape: {df.shape}")
    return df


def load_test() -> pd.DataFrame:
    """Load raw test CSV."""
    path = cfg.data.raw_data_dir / cfg.data.test_file
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded test       → shape: {df.shape}")
    return df


def load_sample_submission() -> pd.DataFrame:
    """Load sample submission CSV."""
    path = cfg.data.raw_data_dir / "sample_submission.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sample submission not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded submission  → shape: {df.shape}")
    return df


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three raw CSVs at once.

    Returns:
        train_df, test_df, sample_df
    """
    train_df  = load_train()
    test_df   = load_test()
    sample_df = load_sample_submission()
    return train_df, test_df, sample_df


if __name__ == "__main__":
    train_df, test_df, sample_df = load_all()

    print(f"\nTrain      : {train_df.shape}")
    print(f"Test       : {test_df.shape}")
    print(f"Submission : {sample_df.shape}")

    print(f"\nTrain columns : {train_df.columns.tolist()}")
    print(f"Test columns  : {test_df.columns.tolist()}")