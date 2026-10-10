"""
Tests for text normalization, cleaning, and token extraction.
"""
import pytest
from src.preprocessing.normalization import (
    normalize_light,
    normalize_aggressive_name,
    normalize_aggressive_address,
    strip_domain,
    consonant_skeleton,
    transliterate_if_needed,
    extract_first_token,
    extract_name_tokens,
    extract_numeric_tokens,
    normalize_record,
)


def test_normalize_light():
    assert normalize_light("Acme & Sons Ltd.") == "acme and sons ltd."
    assert normalize_light("  MULTIPLE   SPACES  ") == "multiple spaces"
    assert normalize_light("") == ""


def test_strip_domain():
    assert strip_domain("acmecorp.com") == "acmecorp"
    assert strip_domain("mycompany.co.in") == "mycompany"
    assert strip_domain("standardbusiness") == "standardbusiness"


def test_normalize_aggressive_name():
    assert normalize_aggressive_name("Acme Corporation Pvt Ltd") == "acme"
    assert normalize_aggressive_name("maurewilliamscolombier.com") == "maurewilliamscolombier"
    assert normalize_aggressive_name("Raj Enterprises LLC") == "raj"


def test_transliterate_and_indic_scripts():
    # Hindi / Devanagari
    hindi_text = "राज इन्वेस्टमेंट्स"
    translit = transliterate_if_needed(hindi_text)
    assert len(translit) > 0
    assert all(ord(c) < 128 for c in translit)


def test_consonant_skeleton():
    assert consonant_skeleton("royal") == "ryl"
    assert consonant_skeleton("shakti") == "shkt"
    assert consonant_skeleton("") == ""


def test_extract_first_token():
    assert extract_first_token("the royal enterprises") == "royal"
    assert extract_first_token("dr sharma clinic") == "sharma"
    assert extract_first_token("acme group") == "acme"


def test_extract_numeric_tokens():
    nums = extract_numeric_tokens("Plot 42, Sector 18, PIN 560001, Building 104")
    assert "42" in nums
    assert "560001" in nums


def test_normalize_record():
    rec = normalize_record(
        entity_id="S1-001",
        name="Royal Enterprises Pvt. Ltd.",
        address="123 Main Road, PIN 560001",
        country="India"
    )
    assert rec["id"] == "S1-001"
    assert rec["country"] == "india"
    assert rec["ft"] == "royal"
    assert "560001" in rec["anums"]
