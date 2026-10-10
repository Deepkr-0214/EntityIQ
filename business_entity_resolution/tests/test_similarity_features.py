"""
Tests for pairwise similarity features and numeric matching.
"""
import pytest
from src.features.similarity_features import (
    jaccard_token,
    norm_lev,
    tok_sort,
    tok_set,
    monge_elkan,
    acronym_match,
    numeric_features,
    compute_pairwise_features,
)
from src.preprocessing.normalization import normalize_record


def test_jaccard_token():
    assert jaccard_token("acme global tech", "acme tech services") == pytest.approx(2 / 4)
    assert jaccard_token("", "acme") == 0.0
    assert jaccard_token("apple", "apple") == 1.0


def test_norm_lev():
    assert norm_lev("google", "google") == 1.0
    assert norm_lev("google", "googla") == pytest.approx(5 / 6)
    assert norm_lev("", "abc") == 0.0


def test_tok_sort_and_set():
    assert tok_sort("smith john", "john smith") == 1.0
    assert tok_set("acme incorporated global", "acme global") == 1.0


def test_acronym_match():
    assert acronym_match("ibm", "international business machines") == 1.0
    assert acronym_match("tcs", "tata consultancy services") == 1.0
    assert acronym_match("xyz", "random company name") == 0.0


def test_numeric_features():
    has_match, frac, max_len = numeric_features(["560001", "102"], ["560001", "404"])
    assert has_match == 1
    assert max_len == 6
    assert frac > 0


def test_compute_pairwise_features():
    rec_a = normalize_record("S1-1", "Tata Consultancy Services Ltd", "Whitefield Road, Bangalore 560066", "India")
    rec_b = normalize_record("S2-1", "Tata Consultancy Services", "560066 Whitefield, Bengaluru", "India")
    
    feats = compute_pairwise_features(rec_a, rec_b)
    assert isinstance(feats, dict)
    assert len(feats) >= 20
    assert feats["name_jaccard"] > 0.8  # name jaccard / similarity should be high
