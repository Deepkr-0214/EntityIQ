"""
Tests for candidate generation and blocking index mechanisms.
"""
import pytest
import pandas as pd
from src.blocking.candidate_generator import (
    build_inverted_index_first_token,
    build_inverted_index_tokens,
    block_by_first_token,
    block_by_token_overlap,
)
from src.preprocessing.normalization import normalize_record


def test_inverted_index_and_blocking():
    data = [
        {"entity_id": "S2-1", "name_first_token": "acme", "name_tokens": {"acme", "tech"}},
        {"entity_id": "S2-2", "name_first_token": "acme", "name_tokens": {"acme", "logistics"}},
        {"entity_id": "S2-3", "name_first_token": "beta", "name_tokens": {"beta", "corp"}},
    ]
    df = pd.DataFrame(data)

    idx_ft = build_inverted_index_first_token(df)
    assert "acme" in idx_ft
    assert len(idx_ft["acme"]) == 2

    s1_row = {"name_first_token": "acme", "name_tokens": {"acme", "ventures"}}
    cands = block_by_first_token(s1_row, idx_ft)
    assert "S2-1" in cands
    assert "S2-2" in cands
    assert "S2-3" not in cands


def test_token_overlap_blocking():
    data = [
        {"entity_id": "S3-1", "name_tokens": {"apex", "global", "solutions"}},
        {"entity_id": "S3-2", "name_tokens": {"apex", "motors"}},
    ]
    df = pd.DataFrame(data)
    idx_tok = build_inverted_index_tokens(df)

    s1_row = {"name_tokens": {"global", "solutions"}}
    cands = block_by_token_overlap(s1_row, idx_tok, min_overlap=1)
    assert "S3-1" in cands
