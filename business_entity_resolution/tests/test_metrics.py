"""
Tests for F0.5 macro evaluation metric and singleton handling.
"""
import pytest
from src.evaluation.metrics import entity_f05, macro_f05, f_beta_score


def test_f_beta_score():
    assert f_beta_score(1.0, 1.0, beta=0.5) == 1.0
    assert f_beta_score(0.0, 0.0, beta=0.5) == 0.0
    # Precision weighted 2x recall: P=1.0, R=0.5 => F0.5 = 1.25 * 0.5 / (0.25*1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    assert f_beta_score(1.0, 0.5, beta=0.5) == pytest.approx(0.83333333)


def test_entity_f05_singletons():
    # Correct singleton identification
    f05, prec, rec = entity_f05(set(), set())
    assert f05 == 1.0
    assert prec == 1.0
    assert rec == 1.0

    # False merge on singleton
    f05, prec, rec = entity_f05({"S2-100"}, set())
    assert f05 == 0.0
    assert prec == 0.0

    # Missed match on non-singleton
    f05, prec, rec = entity_f05(set(), {"S2-100"})
    assert f05 == 0.0


def test_macro_f05():
    gt = {
        "S1-1": {"S2-1", "S3-1"},  # multi-match
        "S1-2": set(),              # singleton
        "S1-3": {"S2-3"},          # single-match
    }
    preds = {
        "S1-1": {"S2-1", "S3-1"},  # perfect
        "S1-2": set(),              # perfect singleton
        "S1-3": {"S2-3"},          # perfect
    }
    res = macro_f05(preds, gt)
    assert res["f05_macro"] == 1.0
    assert res["singleton_f05"] == 1.0
    assert res["n_entities"] == 3
