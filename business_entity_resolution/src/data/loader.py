"""
Data loading and validation utilities.
"""
import pandas as pd
from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
)


def load_tsv(path, nrows=None):
    """Load a tab-separated file into a DataFrame."""
    return pd.read_csv(path, sep="\t", dtype=str, nrows=nrows, keep_default_na=False)


def load_train_data(nrows=None):
    """Load all training source files and ground truth."""
    s1 = load_tsv(TRAIN_S1, nrows=nrows)
    s2 = load_tsv(TRAIN_S2, nrows=nrows)
    s3 = load_tsv(TRAIN_S3, nrows=nrows)
    gt = load_tsv(TRAIN_GT, nrows=nrows)
    return s1, s2, s3, gt


def load_test_data(nrows=None):
    """Load all test source files."""
    s1 = load_tsv(TEST_S1, nrows=nrows)
    s2 = load_tsv(TEST_S2, nrows=nrows)
    s3 = load_tsv(TEST_S3, nrows=nrows)
    return s1, s2, s3


def parse_ground_truth(gt_df):
    """
    Parse ground truth into a dict: {s1_id: set of matched_ids}.
    Empty matches → empty set (singleton).
    """
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or str(matched).strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(","))
    return gt_dict


def validate_ground_truth(gt_dict, s1_df, s2_df, s3_df):
    """Validate that all GT IDs exist in the correct source files."""
    s1_ids = set(s1_df["entity_id"])
    s2_ids = set(s2_df["entity_id"])
    s3_ids = set(s3_df["entity_id"])

    issues = []
    for s1_id, matched in gt_dict.items():
        if s1_id not in s1_ids:
            issues.append(f"GT references unknown S1 ID: {s1_id}")
        for mid in matched:
            if mid.startswith("S2-") and mid not in s2_ids:
                issues.append(f"GT references unknown S2 ID: {mid}")
            elif mid.startswith("S3-") and mid not in s3_ids:
                issues.append(f"GT references unknown S3 ID: {mid}")
            elif not mid.startswith(("S2-", "S3-")):
                issues.append(f"GT contains invalid ID prefix: {mid}")
    return issues
