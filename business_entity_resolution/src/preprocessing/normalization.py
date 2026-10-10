"""
Data cleaning and normalization module.

Two normalization tiers:
  - Light: lowercase, whitespace/punctuation normalization (for exact-match blocking keys)
  - Aggressive: legal-suffix stripping, stopword removal, domain stripping (for similarity features)

Handles domain names, multi-script business names, and dataset-specific patterns.
Includes Unicode transliteration for cross-script matching (Indic -> Latin).
"""
import re
from unidecode import unidecode


# ── Detect non-Latin text ────────────────────────────────────────────────────
_NON_LATIN_RE = re.compile(r'[^\u0000-\u024F\u1E00-\u1EFF]')

def transliterate_if_needed(text: str) -> str:
    """Transliterate non-Latin scripts (Hindi, Bengali, Kannada, etc.) to ASCII."""
    if not text:
        return ""
    if _NON_LATIN_RE.search(text):
        return unidecode(text)
    return text


# ── Consonant skeleton for phonetic blocking ─────────────────────────────────
_VOWELS = frozenset('aeiou')

def consonant_skeleton(text: str) -> str:
    """
    Reduce text to consonant skeleton: strip vowels, collapse repeated chars.
    'royal' -> 'ryl', 'ronyl' -> 'rnyl' (close enough for 2-3 char prefix match)
    'shakti' -> 'shkt', 'shkti' -> 'shkt'
    """
    if not text:
        return ""
    # Remove spaces, keep only alphanumeric
    t = re.sub(r'[^a-z0-9]', '', text.lower())
    # Remove vowels
    t = ''.join(c for c in t if c not in _VOWELS)
    # Collapse consecutive duplicates
    if not t:
        return ""
    result = [t[0]]
    for c in t[1:]:
        if c != result[-1]:
            result.append(c)
    return ''.join(result)


# ── Legal suffix patterns ────────────────────────────────────────────────────
LEGAL_SUFFIXES_RE = re.compile(
    r'\b(?:'
    # English
    r'private\s+limited|pvt\.?\s*ltd\.?|ltd\.?|limited|'
    r'llc\.?|llp\.?|inc\.?|incorporated|'
    r'corp\.?|corporation|co\.?|company|'
    r'enterprises?|industries|group|holdings?|'
    r'partnership|associates?|foundation|trust|society|'
    r'ventures?|solutions?|services?|technolog(?:y|ies)|'
    r'systems?|international|global|'
    # Indian
    r'udyog|udhyog|sansthan|'
    # French
    r'sarl\.?|sas|eurl|société|societe|cie\.?'
    r')\b',
    re.IGNORECASE
)

# Transliterated legal suffixes that appear after unidecode
TRANSLIT_LEGAL_SUFFIXES_RE = re.compile(
    r'\b(?:'
    r'praaivett?\s*limittedd?|praaivet\s*limitedd?|'
    r'praa?\s*li\.?|'
    r'praaibhett?\s*limittedd?|'
    r'limittedd?|limitedd?|'
    r'praivett?|praaivett?|praaibhett?|'
    r'privett?\s*limitedd?'
    r')\b',
    re.IGNORECASE
)

# Domain TLDs to strip
DOMAIN_TLD_RE = re.compile(
    r'\.(com|net|org|in|co|fr|io|biz|info|edu|gov|us|uk|de|co\.in|co\.uk|co\.fr)$',
    re.IGNORECASE
)

# ── Address abbreviation mapping ─────────────────────────────────────────────
_ADDR_MAP = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'pkwy': 'parkway', 'hwy': 'highway', 'apt': 'apartment', 'ste': 'suite',
    'bldg': 'building', 'dist': 'district', 'nr': 'near', 'opp': 'opposite',
}
_ADDR_ABBREVS_RE = re.compile(
    r'\b(rd|st|ave|blvd|dr|ln|ct|pl|pkwy|hwy|apt|ste|bldg|dist|nr|opp)\.?\b',
    re.IGNORECASE
)

def _replace_addr_abbrev(m):
    return _ADDR_MAP.get(m.group(1).lower(), m.group(0))


# Words to skip when extracting the "first significant token"
_SKIP_FIRST = frozenset({
    'the', 'a', 'an', 'shri', 'sri', 'shree', 'smt',
    'mr', 'mrs', 'dr', 'prof',
    'le', 'la', 'les', 'de', 'du', 'des', 'l', 'el',
})

# Stop words removed during aggressive normalisation / token-set extraction
_NAME_STOPS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'for', 'to', 'at',
    'by', 'on', 'with', 'from', 'is', 'it', 'its', 'as',
    'shri', 'sri', 'shree', 'smt', 'mr', 'mrs', 'dr', 'prof',
    'le', 'la', 'les', 'de', 'du', 'des', 'et', 'un', 'une', 'l', 'el',
})

_ADDR_STOPS = frozenset({
    'road', 'street', 'near', 'opposite', 'floor', 'building',
    'number', 'block', 'plot', 'sector', 'phase', 'area',
    'nagar', 'colony', 'village', 'town', 'city', 'state',
    'district', 'taluka', 'tehsil', 'post', 'office',
    'rue', 'avenue', 'boulevard', 'place', 'square',
    'north', 'south', 'east', 'west', 'new', 'old',
    'suite', 'apartment', 'unit', 'room', 'dept',
    'no', 'po', 'box', 'via', 'to', 'at', 'the', 'of',
})

_WS_RE = re.compile(r'\s+')
_NON_ALNUM_RE = re.compile(r'[^\w\s]')
_DIGITS_RE = re.compile(r'\b\d+\b')


# ══════════════════════════════════════════════════════════════════════════════
#  Public helpers — operate on individual strings, no DataFrame dependency
# ══════════════════════════════════════════════════════════════════════════════

def normalize_light(text: str) -> str:
    """Lowercase, collapse whitespace, replace '&' with 'and', strip basic punct.
    Also transliterates non-Latin scripts to ASCII."""
    if not text:
        return ""
    t = transliterate_if_needed(text)
    t = t.lower()
    t = t.replace('&', ' and ')
    t = re.sub(r'[^\w\s,.\-/]', ' ', t)
    return _WS_RE.sub(' ', t).strip()


def strip_domain(text: str) -> str:
    """Remove a trailing TLD so 'acmecorp.com' -> 'acmecorp'."""
    return DOMAIN_TLD_RE.sub('', text)


def normalize_aggressive_name(text: str) -> str:
    """
    Heavy-duty name normalisation used for similarity features.
    Strips domain TLDs, legal suffixes, stop words, and all punctuation.
    Includes transliteration for cross-script matching.
    """
    if not text:
        return ""
    t = normalize_light(text)
    t = strip_domain(t)                       # acmecorp.com -> acmecorp
    t = LEGAL_SUFFIXES_RE.sub(' ', t)          # legal suffixes
    t = TRANSLIT_LEGAL_SUFFIXES_RE.sub(' ', t) # transliterated legal suffixes
    t = _NON_ALNUM_RE.sub(' ', t)              # remaining punctuation
    return _WS_RE.sub(' ', t).strip()


def normalize_aggressive_address(text: str) -> str:
    """Expand abbreviations, remove noise words, strip punctuation.
    Includes transliteration for cross-script matching."""
    if not text:
        return ""
    t = normalize_light(text)
    t = _ADDR_ABBREVS_RE.sub(_replace_addr_abbrev, t)
    t = _NON_ALNUM_RE.sub(' ', t)
    return _WS_RE.sub(' ', t).strip()


def extract_first_token(name_light: str) -> str:
    """First significant word from a lightly-normalised name (for blocking key)."""
    if not name_light:
        return ""
    for tok in name_light.split():
        if tok not in _SKIP_FIRST and len(tok) > 1:
            return tok
    parts = name_light.split()
    return parts[0] if parts else ""


def extract_name_prefix(name_light: str, n: int = 4) -> str:
    """First *n* chars of the lightly-normalised name (catches domain-name squash)."""
    if not name_light:
        return ""
    cleaned = _NON_ALNUM_RE.sub('', name_light.replace(' ', ''))
    return cleaned[:n] if len(cleaned) >= n else cleaned


def extract_name_tokens(name_agg: str):
    """Significant tokens from aggressively-normalised name (for token-set blocking)."""
    if not name_agg:
        return set()
    return {t for t in name_agg.split() if t not in _NAME_STOPS and len(t) > 1}


def extract_addr_significant_tokens(addr_agg: str):
    """Significant address tokens (non-stopword, length>2, non-numeric)."""
    if not addr_agg:
        return set()
    return {t for t in addr_agg.split()
            if t not in _ADDR_STOPS and len(t) > 2 and not t.isdigit()}


def extract_numeric_tokens(text: str):
    """All purely-numeric tokens (PIN codes, building numbers ...)."""
    if not text:
        return []
    # Transliterate first to handle non-Latin digits
    t = transliterate_if_needed(text)
    return _DIGITS_RE.findall(t)


def get_char_ngrams(text: str, n: int = 3):
    """Character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def get_consonant_ngrams(name_agg: str, n: int = 3):
    """Character n-grams from consonant skeleton of name (phonetic blocking)."""
    skel = consonant_skeleton(name_agg)
    if not skel or len(skel) < n:
        return set()
    return {skel[i:i + n] for i in range(len(skel) - n + 1)}


# ==============================================================================
#  Record-level normalisation (returns a compact dict of derived fields)
# ==============================================================================

def normalize_record(entity_id: str, name: str, address: str, country: str) -> dict:
    """
    Produce all normalised / derived fields for a single record.

    Returns a flat dict that is compact enough to keep millions in memory.
    """
    nl = normalize_light(name)
    na = normalize_aggressive_name(name)
    al = normalize_light(address)
    aa = normalize_aggressive_address(address)

    # Consonant skeleton of aggressive name for phonetic blocking
    cskel = consonant_skeleton(na)

    return {
        'id':           entity_id,
        'country':      country.strip().lower() if country else '',
        'nl':           nl,             # name light
        'na':           na,             # name aggressive
        'al':           al,             # addr light
        'aa':           aa,             # addr aggressive
        'ft':           extract_first_token(nl),           # first token
        'npfx':         extract_name_prefix(nl, 4),        # 4-char name prefix
        'ntoks':        extract_name_tokens(na),           # significant name tokens
        'atoks':        extract_addr_significant_tokens(aa),  # significant addr tokens
        'anums':        [n for n in extract_numeric_tokens(address) if len(n) >= 4],
        'cskel':        cskel,          # consonant skeleton
        'cskel3':       cskel[:3] if len(cskel) >= 3 else cskel,  # 3-char consonant prefix
        'cngrams':      get_consonant_ngrams(na, 3),       # consonant trigrams
    }
