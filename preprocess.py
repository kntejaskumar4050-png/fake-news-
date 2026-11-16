"""
Text preprocessing utilities for the Fake News Detection System.

Pipeline
--------
1. Load Fake.csv / True.csv, drop missing values and duplicates.
2. Clean each article: HTML, URLs, entities, punctuation, digits, casing.
3. Tokenise with a unicode-aware regex tokenizer and remove NLTK stopwords
   (negations are preserved because they carry stance information).
4. Split the dataset into train / validation / test *before* fitting the
   Keras Tokenizer, so no information from validation or test leaks into the
   vocabulary.
5. Fit the tokenizer on the training split only, then pad sequences.

The cleaned corpus is cached on disk (dataset/cleaned_corpus.pkl) because the
cleaning step is deterministic; the cache key is derived from the raw files so
a changed dataset automatically invalidates it.

Run directly to (re)build the cleaned corpus and inspect dataset statistics:

    python preprocess.py
"""
from __future__ import annotations

import html
import json
import os
import pickle
import re
import string
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------- #
# NLTK bootstrap (project-local data folder, no global install required)
# --------------------------------------------------------------------------- #
def ensure_nltk_data() -> None:
    """Make the bundled `nltk_data` folder visible and verify the resources."""
    os.environ.setdefault("NLTK_DATA", str(config.NLTK_DATA_DIR))
    import nltk

    if str(config.NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(config.NLTK_DATA_DIR))
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:  # pragma: no cover - only on a fresh checkout
        nltk.download("stopwords", download_dir=str(config.NLTK_DATA_DIR), quiet=True)
        nltk.download("punkt", download_dir=str(config.NLTK_DATA_DIR), quiet=True)
        nltk.download("punkt_tab", download_dir=str(config.NLTK_DATA_DIR), quiet=True)


ensure_nltk_data()
from nltk.corpus import stopwords  # noqa: E402  (after path bootstrap)

STOPWORDS: set[str] = set(stopwords.words("english")) - config.KEEP_TOKENS

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+")
_REUTERS_BYLINE_RE = re.compile(r"^\s*[A-Z\s,]{2,40}\(reuters\)\s*-\s*", re.IGNORECASE)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities such as ``&amp;`` / ``&#39;``."""
    text = _HTML_RE.sub(" ", text)
    text = html.unescape(text)
    # A second pass handles escaped markup like &lt;p&gt;
    return _HTML_RE.sub(" ", text)


def clean_text(text: object, remove_stopwords: bool = True) -> str:
    """Normalise one article into a space-separated token string.

    Steps: HTML removal, URL/email removal, unicode normalisation, lowercasing,
    punctuation/digit removal, regex tokenisation, stopword removal.
    The function is intentionally stateless so it can be applied before the
    train/test split without leaking information.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text)
    text = strip_html(text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _REUTERS_BYLINE_RE.sub(" ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)
    text = _WHITESPACE_RE.sub(" ", text)

    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
@dataclass
class DatasetStats:
    fake_raw: int
    real_raw: int
    duplicates_removed: int
    missing_removed: int
    too_short_removed: int
    total_clean: int
    fake_clean: int
    real_clean: int
    train_size: int
    val_size: int
    test_size: int


def _read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download the Kaggle 'Fake and Real News Dataset' and place "
            "Fake.csv and True.csv in the dataset/ folder. See README.md."
        )
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def load_raw_dataset() -> pd.DataFrame:
    """Load, validate, deduplicate and label both CSV files."""
    fake_df = _read_csv_safely(config.FAKE_CSV)
    true_df = _read_csv_safely(config.TRUE_CSV)

    for name, frame in (("Fake.csv", fake_df), ("True.csv", true_df)):
        if not {"title", "text"}.issubset(frame.columns):
            raise ValueError(f"{name} must contain 'title' and 'text' columns.")

    fake_df = fake_df.assign(label=0, source="fake")
    true_df = true_df.assign(label=1, source="real")
    df = pd.concat([fake_df, true_df], ignore_index=True)

    raw_total = len(df)
    df = df.dropna(subset=["text", "title"])
    df["text"] = df["text"].astype(str)
    df["title"] = df["title"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    duplicates_removed = before - len(df)

    df["content"] = (
        df["title"].str.strip() + ". " + df["text"].str.strip()
    ).str.strip()
    df = df[df["content"].str.len() >= config.MIN_TEXT_CHARS]
    too_short = before - duplicates_removed - len(df)

    df = df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    df.attrs["raw_total"] = raw_total
    df.attrs["duplicates_removed"] = duplicates_removed
    df.attrs["too_short_removed"] = max(too_short, 0)
    return df


def _cache_is_fresh(cache: Path, meta_cache: Path) -> bool:
    """A cache is valid only if it was built from the current CSV files."""
    try:
        meta = json.loads(meta_cache.read_text())
        sources = meta.get("sources", {})
        return (
            int(sources.get("fake_size", -1)) == config.FAKE_CSV.stat().st_size
            and int(sources.get("true_size", -1)) == config.TRUE_CSV.stat().st_size
            and float(sources.get("fake_mtime", -1)) == config.FAKE_CSV.stat().st_mtime
            and float(sources.get("true_mtime", -1)) == config.TRUE_CSV.stat().st_mtime
        )
    except Exception:
        return False


def load_clean_dataset(use_cache: bool = True) -> pd.DataFrame:
    """Return a DataFrame with a ``clean`` column holding normalised text."""
    dataset_dir = config.DATASET_DIR
    cache = dataset_dir / "cleaned_corpus.pkl"
    meta_cache = dataset_dir / "cleaned_corpus_meta.json"

    if use_cache and cache.exists() and _cache_is_fresh(cache, meta_cache):
        try:
            df = pd.read_pickle(cache)
            if len(df) > 1000:
                return df
        except Exception:
            pass  # fall through and rebuild

    df = load_raw_dataset()
    df["clean"] = [clean_text(t) for t in df["content"].tolist()]
    df = df[df["clean"].str.split().str.len() >= 5].reset_index(drop=True)

    points = (df["label"].value_counts().to_dict(), int(len(df)))
    try:
        df.to_pickle(cache)
        meta_cache.write_text(
            json.dumps(
                {
                    "rows": points[1],
                    "labels": {str(k): int(v) for k, v in points[0].items()},
                    "sources": {
                        "fake_mtime": config.FAKE_CSV.stat().st_mtime,
                        "true_mtime": config.TRUE_CSV.stat().st_mtime,
                        "fake_size": config.FAKE_CSV.stat().st_size,
                        "true_size": config.TRUE_CSV.stat().st_size,
                    },
                },
                indent=2,
            )
        )
    except Exception as exc:  # cache is an optimisation, never fatal
        print(f"[warn] could not write corpus cache: {exc}")
    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_dataset(df: pd.DataFrame):
    """Stratified train/validation/test split (70 / 15 / 15)."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        stratify=df["label"],
        random_state=config.SEED,
    )
    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=config.SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Tokenizer wrapper
# --------------------------------------------------------------------------- #
class TextPreprocessor:
    """Keras tokenizer + padding configuration, persisted as pickle + JSON."""

    def __init__(self, tokenizer=None, max_len: int = config.MAX_LEN,
                 max_words: int = config.MAX_WORDS):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_words = max_words
        self._fitted = tokenizer is not None

    # -- fitting -----------------------------------------------------------
    def fit(self, texts: Sequence[str]) -> "TextPreprocessor":
        from tensorflow.keras.preprocessing.text import Tokenizer

        self.tokenizer = Tokenizer(
            num_words=self.max_words,
            oov_token=config.OOV_TOKEN,
            filters="",
            lower=False,          # cleaning already lowercases the text
            split=" ",
        )
        self.tokenizer.fit_on_texts(texts)
        self._fitted = True
        return self

    # -- transforms --------------------------------------------------------
    @property
    def word_index(self) -> dict[str, int]:
        if not self._fitted:
            raise RuntimeError("Tokenizer is not fitted yet.")
        return self.tokenizer.word_index

    @property
    def vocab_size(self) -> int:
        return min(self.max_words, len(self.word_index) + 1)

    def transform_text(self, text: str) -> np.ndarray:
        """Clean, tokenise and pad a single article for inference."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned = clean_text(text)
        seq = self.tokenizer.texts_to_sequences([cleaned])
        return pad_sequences(
            seq, maxlen=self.max_len, padding="post", truncating="post"
        )

    def transform(self, texts: Iterable[str], padding="post", truncating="post"):
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        sequences = self.tokenizer.texts_to_sequences(list(texts))
        return pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding=padding,
            truncating=truncating,
            dtype="int32",
        )

    # -- persistence -------------------------------------------------------
    def save(self, tokenizer_path: Path | str = config.TOKENIZER_PATH,
             config_path: Path | str = config.PREPROCESSOR_CONFIG_PATH) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tokenizer_path, "wb") as fh:
            pickle.dump(self.tokenizer, fh, protocol=4)
        Path(config_path).write_text(
            json.dumps(
                {
                    "max_len": self.max_len,
                    "max_words": self.max_words,
                    "vocab_size": self.vocab_size,
                    "oov_token": config.OOV_TOKEN,
                    "remove_stopwords": True,
                    "keep_tokens": sorted(config.KEEP_TOKENS),
                    "cleaning": "html,urls,emails,entities,unicode,lowercase,"
                                "punctuation,digits,stopwords",
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, tokenizer_path: Path | str = config.TOKENIZER_PATH,
             config_path: Path | str = config.PREPROCESSOR_CONFIG_PATH):
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. Run 'python train_model.py' first."
            )
        with open(tokenizer_path, "rb") as fh:
            tokenizer = pickle.load(fh)
        max_len, max_words = config.MAX_LEN, config.MAX_WORDS
        config_path = Path(config_path)
        if config_path.exists():
            meta = json.loads(config_path.read_text())
            max_len = int(meta.get("max_len", max_len))
            max_words = int(meta.get("max_words", max_words))
        return cls(tokenizer=tokenizer, max_len=max_len, max_words=max_words)


# --------------------------------------------------------------------------- #
# CLI helper
# --------------------------------------------------------------------------- #
def main() -> int:
    print("Loading and cleaning dataset …")
    df = load_clean_dataset(use_cache=False)
    train_df, val_df, test_df = split_dataset(df)
    stats = DatasetStats(
        fake_raw=int((df["label"] == 0).sum()),
        real_raw=int((df["label"] == 1).sum()),
        duplicates_removed=int(df.attrs.get("duplicates_removed", 0)),
        missing_removed=int(df.attrs.get("raw_total", 0) - len(df)),
        too_short_removed=int(df.attrs.get("too_short_removed", 0)),
        total_clean=len(df),
        fake_clean=int((df["label"] == 0).sum()),
        real_clean=int((df["label"] == 1).sum()),
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
    )
    print(json.dumps(asdict(stats), indent=2))
    print("\nExample cleaned article:\n", df["clean"].iloc[0][:400])

    pre = TextPreprocessor().fit(train_df["clean"].tolist())
    pre.save()
    print(f"\nTokenizer fitted on {len(train_df)} training articles "
          f"| vocabulary: {pre.vocab_size} tokens | max_len: {pre.max_len}")
    print(f"Saved tokenizer -> {config.TOKENIZER_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
