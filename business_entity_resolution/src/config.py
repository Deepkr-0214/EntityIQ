"""
Configuration module - all paths and constants in one place.
"""
import os

# ── Base paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_ROOT
STUDENT_RESOURCE = os.path.join(os.path.dirname(PROJECT_ROOT), "student_resource")

# Dataset paths
DATASET_DIR = os.path.join(STUDENT_RESOURCE, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Train files
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# Test files
TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Reports
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Models
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2  # fraction of S1 entities held out for validation

# Blocking parameters
BLOCKING_NGRAM_SIZE = 3
BLOCKING_TOP_K = 50  # max candidates per blocking strategy per S1 entity
MAX_CANDIDATES_PER_S1 = 100  # hard cap after union of all blocking strategies

# Feature engineering
TFIDF_MAX_FEATURES = 50000

# Model
MATCH_THRESHOLD = 0.5  # will be optimized in Phase 11

# Ensure output directories exist
for d in [OUTPUT_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
