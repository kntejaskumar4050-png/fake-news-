"""
Central configuration for the Fake News Detection System.

Every path is derived from the project root so the application can be started
from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
FAKE_CSV = DATASET_DIR / "Fake.csv"
TRUE_CSV = DATASET_DIR / "True.csv"

MODELS_DIR = PROJECT_ROOT / "models"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
BILSTM_MODEL_PATH = MODELS_DIR / "bilstm_model.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
PREPROCESSOR_CONFIG_PATH = MODELS_DIR / "preprocessor_config.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
HISTORY_DIR = MODELS_DIR / "history"

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = Path(os.environ.get("FND_DATABASE", DATABASE_DIR / "predictions.db"))

STATIC_DIR = PROJECT_ROOT / "static"
IMAGES_DIR = STATIC_DIR / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"

NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Data split (split happens BEFORE the tokenizer is fitted -> no leakage)
# --------------------------------------------------------------------------- #
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
MAX_WORDS = 20_000      # vocabulary size
MAX_LEN = 256           # sequence length (tokens)
OOV_TOKEN = "<OOV>"
MIN_TEXT_CHARS = 25     # articles shorter than this are discarded as unusable

# Negations are meaningful for stance detection, so they are never removed.
KEEP_TOKENS = {
    "no", "nor", "not", "never", "none", "cannot", "n't", "without",
    "against", "why", "how", "who", "what", "when", "where",
}

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 32
SPATIAL_DROPOUT = 0.20
DROPOUT = 0.40
RECURRENT_DROPOUT = 0.0   # kept at 0 for CPU speed (documented in README)
LEARNING_RATE = 5e-4
GRADIENT_CLIPNORM = 1.0  # stabilises recurrent training
BATCH_SIZE = 128
EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
LR_PATIENCE = 2

# Label convention used across training, evaluation, API and database.
# 0 -> FAKE, 1 -> REAL
LABEL_NAMES = {0: "FAKE", 1: "REAL"}
DISPLAY_NAMES = {"lstm": "LSTM", "bilstm": "BiLSTM", "ensemble": "LSTM + BiLSTM Ensemble"}

DEFAULT_MODEL = "bilstm"

# --------------------------------------------------------------------------- #
# API / UI limits
# --------------------------------------------------------------------------- #
MIN_INPUT_CHARS = 15
MAX_INPUT_CHARS = 20_000
MAX_HISTORY_ROWS = 500

# --------------------------------------------------------------------------- #
# Ensure runtime folders exist
# --------------------------------------------------------------------------- #
for _directory in (MODELS_DIR, HISTORY_DIR, DATABASE_DIR, IMAGES_DIR,
                   REPORTS_DIR, NLTK_DATA_DIR, STATIC_DIR / "images" / "screenshots"):
    _directory.mkdir(parents=True, exist_ok=True)
