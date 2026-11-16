"""
Inference layer: loads the trained models once and serves predictions to the
Flask API and the CLI.

    python predict.py "Your news article text ..."

The confidence score returned here is the model's certainty about *its own
class*, based on patterns learned from the training corpus. It is NOT the
probability that an article is factually true. Human-readable percentages are
capped at 99.9% so a saturated output is never shown as absolute certainty.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

import config
from preprocess import TextPreprocessor, clean_text

DISCLAIMER = (
    "This prediction is produced by a deep-learning model trained on a labelled "
    "news corpus. The confidence score reflects the model's certainty in its own "
    "classification based on learned writing patterns; it is not a fact-check and "
    "does not prove that an article is true or false."
)


class ModelNotReadyError(RuntimeError):
    """Raised when the trained artifacts are missing."""


# TensorFlow must be imported exactly once, from a single thread. Importing it
# concurrently (e.g. from a background warm-up thread and a request handler)
# corrupts TensorFlow's op/gradient registries, so every import goes through
# this lock.
_TF_IMPORT_LOCK = threading.Lock()
_TF_MODULE = None


def _tensorflow():
    global _TF_MODULE
    with _TF_IMPORT_LOCK:
        if _TF_MODULE is None:
            import tensorflow as tf

            _TF_MODULE = tf
    return _TF_MODULE


_LOAD_LOCK = threading.Lock()


class Predictor:
    """Thread-safe singleton that owns the tokenizer and both Keras models."""

    def __init__(self, models_dir: Path | str = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._lock = threading.RLock()
        self._predict_lock = threading.Lock()
        self._models: dict = {}
        self._preprocessor: TextPreprocessor | None = None
        self._metrics_cache: dict | None = None
        self._metrics_mtime: float | None = None
        self._loaded = False
        self.load_error: str | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def available_models(self) -> list[str]:
        return sorted(self._models.keys())

    def load(self, force: bool = False) -> "Predictor":
        with _LOAD_LOCK:
            if self._loaded and not force:
                return self
            errors: list[str] = []
            self._preprocessor = TextPreprocessor.load(
                self.models_dir / config.TOKENIZER_PATH.name,
                self.models_dir / config.PREPROCESSOR_CONFIG_PATH.name,
            )

            tf = _tensorflow()  # single, locked import

            paths = {
                "lstm": self.models_dir / config.LSTM_MODEL_PATH.name,
                "bilstm": self.models_dir / config.BILSTM_MODEL_PATH.name,
            }
            models = {}
            for key, path in paths.items():
                if not path.exists():
                    errors.append(f"{path.name} is missing")
                    continue
                try:
                    models[key] = tf.keras.models.load_model(path, compile=False)
                except Exception as exc:  # corrupted file / version mismatch
                    errors.append(f"{path.name}: {exc}")

            self._models = models
            if not models:
                self._loaded = False
                self.load_error = (
                    "No trained model could be loaded. Run: python train_model.py. "
                    + " | ".join(errors)
                )
                raise ModelNotReadyError(self.load_error)

            self._loaded = True
            self.load_error = None
            return self

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    @property
    def metrics(self) -> dict:
        path = self.models_dir / config.METRICS_PATH.name
        if not path.exists():
            return {}
        mtime = path.stat().st_mtime
        if self._metrics_cache is None or self._metrics_mtime != mtime:
            try:
                self._metrics_cache = json.loads(path.read_text())
                self._metrics_mtime = mtime
            except Exception:
                return {}
        return self._metrics_cache or {}

    def model_summary(self) -> dict:
        metrics = self.metrics
        models = metrics.get("models", {})
        summary = {}
        for key in self.available_models:
            item = models.get(key, {})
            summary[key] = {
                "display_name": config.DISPLAY_NAMES.get(key, key),
                "accuracy": item.get("accuracy"),
                "precision": item.get("precision"),
                "recall": item.get("recall"),
                "f1_score": item.get("f1_score"),
                "roc_auc": item.get("roc_auc"),
                "parameters": item.get("parameters"),
                "epochs_trained": item.get("epochs_trained"),
            }
        return {
            "models": summary,
            "best_model": metrics.get("best_model"),
            "dataset": metrics.get("dataset", {}),
            "training": metrics.get("training", {}),
            "generated_at": metrics.get("generated_at"),
        }

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, text: str, model_key: str = config.DEFAULT_MODEL) -> dict:
        if not self._loaded:
            self.load()
        if model_key not in ("lstm", "bilstm", "ensemble"):
            raise ValueError("Unknown model. Choose 'lstm', 'bilstm' or 'ensemble'.")
        if model_key not in self._models and model_key != "ensemble":
            raise ModelNotReadyError(
                f"The {config.DISPLAY_NAMES.get(model_key, model_key)} model is not available. "
                "Retrain with: python train_model.py"
            )
        if not text or not str(text).strip():
            raise ValueError("Please provide some news text to analyse.")

        started = time.perf_counter()
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError(
                "No usable words found in the input. Please paste a news headline "
                "or article containing readable English text."
            )

        sequence = self._preprocessor.transform_text(text)
        raw_tokens = len(cleaned.split())
        max_len = self._preprocessor.max_len
        known = sum(1 for token in cleaned.split()
                    if token in self._preprocessor.word_index
                    and self._preprocessor.word_index[token] < self._preprocessor.max_words)

        with self._predict_lock:
            probabilities = {
                key: float(model.predict(sequence, verbose=0).ravel()[0])
                for key, model in self._models.items()
            }

        if model_key == "ensemble" and len(probabilities) > 1:
            probability_real = float(np.mean(list(probabilities.values())))
        elif model_key in probabilities:
            probability_real = probabilities[model_key]
        else:  # requested model missing but another is available
            model_key = next(iter(probabilities))
            probability_real = probabilities[model_key]

        label_code = 1 if probability_real >= 0.5 else 0
        label = config.LABEL_NAMES[label_code]
        confidence = probability_real if label_code == 1 else 1.0 - probability_real
        coverage = known / raw_tokens if raw_tokens else 0.0
        processing_ms = (time.perf_counter() - started) * 1000

        scores = []
        for key, prob in probabilities.items():
            code = 1 if prob >= 0.5 else 0
            scores.append(
                {
                    "key": key,
                    "label": config.DISPLAY_NAMES.get(key, key),
                    "prediction": config.LABEL_NAMES[code],
                    "probability_real": round(prob, 4),
                    "probability_fake": round(1 - prob, 4),
                    "confidence": round(prob if code == 1 else 1 - prob, 4),
                }
            )

        return {
            "prediction": label,
            "label_code": label_code,
            "confidence": round(float(confidence), 4),
            # Never publish 100.0%: a softmax/sigmoid score is model certainty,
            # not proof, and 100% reads as infallibility.
            "confidence_percent": round(min(float(confidence) * 100, 99.9), 2),
            "model_key": model_key,
            "model_used": config.DISPLAY_NAMES.get(model_key, model_key),
            "probability_real": round(float(probability_real), 4),
            "probability_fake": round(float(1 - probability_real), 4),
            "model_scores": scores,
            "explanation": self._explain(label, confidence, model_key, probability_real,
                                          tokens=raw_tokens),
            "disclaimer": DISCLAIMER,
            "text_stats": {
                "characters": len(str(text)),
                "words_raw": len(str(text).split()),
                "tokens_used": min(raw_tokens, max_len),
                "vocabulary_coverage": round(coverage, 4),
                "truncated": raw_tokens > max_len,
                "max_len": max_len,
            },
            "reliability": self._reliability(raw_tokens, coverage),
            "processing_ms": round(processing_ms, 1),
            "threshold": 0.5,
        }

    # ------------------------------------------------------------------ #
    # Human-readable explanations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _explain(label: str, confidence: float, model_key: str,
                 probability_real: float, tokens: int = config.MAX_LEN) -> str:
        model = config.DISPLAY_NAMES.get(model_key, model_key)
        other = "REAL" if label == "FAKE" else "FAKE"
        if confidence >= 0.90:
            strength = ("The wording, tone and vocabulary are strongly consistent with "
                        f"patterns the {model} learned from {label.lower()} articles.")
        elif confidence >= 0.70:
            strength = ("The text shows a clear but not extreme similarity to the "
                        f"{label.lower()} examples the {model} was trained on.")
        elif confidence >= 0.55:
            strength = ("The article leans towards the "
                        f"{label.lower()} class, but the signal is weak and mixed.")
        else:
            strength = ("The article sits close to the decision boundary. The model sees "
                        "conflicting stylistic signals, so treat this result as inconclusive.")
        margin = abs(probability_real - 0.5) * 2
        # Displayed scores are capped at 99.9% so a saturated network output is
        # never presented as absolute certainty.
        shown = min(confidence * 100, 99.9)
        complement = 100 - shown
        shown_margin = min(margin * 100, 99.9)
        length_note = ""
        if tokens < 80:
            length_note = (
                f" Only {tokens} analysable tokens remained after cleaning, which gives "
                "the network limited context; short inputs are noticeably less reliable "
                "than full articles, so treat this confidence with caution."
            )
        return (
            f"The {model} classified this text as {label} NEWS with "
            f"{shown:.1f}% confidence (decision margin {shown_margin:.1f}%). "
            f"{strength} A confidence of {shown:.1f}% means the network's output "
            f"is {shown:.1f}% aligned with the {label} class and "
            f"{complement:.1f}% aligned with {other} — it does not verify facts."
            f"{length_note}"
        )

    @staticmethod
    def _reliability(tokens: int, coverage: float) -> dict:
        if tokens < 25:
            return {
                "level": "low",
                "message": "Very short input. Stylistic models need context — try pasting a few sentences or the full article.",
            }
        if tokens < 80 or coverage < 0.70:
            return {
                "level": "medium",
                "message": "Moderate input length or many unseen words. Treat the score as indicative rather than definitive.",
            }
        return {
            "level": "high",
            "message": "Input length and vocabulary coverage are sufficient for a stable pattern-based signal.",
        }


# --------------------------------------------------------------------------- #
# Shared singleton used by Flask
# --------------------------------------------------------------------------- #
_predictor: Predictor | None = None
_predictor_lock = threading.Lock()


def get_predictor() -> Predictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = Predictor()
        return _predictor


def warm_up(background: bool = True) -> None:
    """Load models eagerly so the first API call is fast."""
    def _run():
        try:
            predictor = get_predictor()
            predictor.load()
            predictor.predict("The government announced a new budget proposal on Monday.",
                              config.DEFAULT_MODEL)
            print("[predictor] models warmed up successfully")
        except Exception as exc:  # pragma: no cover
            print(f"[predictor] warm-up failed: {exc}")

    if background:
        threading.Thread(target=_run, daemon=True, name="model-warmup").start()
    else:
        _run()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify a news article")
    parser.add_argument("text", nargs="?", help="Article text (reads stdin if omitted)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        choices=["lstm", "bilstm", "ensemble"])
    args = parser.parse_args()

    import sys

    text = args.text or sys.stdin.read()
    result = get_predictor().load().predict(text, args.model)
    print(f"\nPrediction : {result['prediction']} NEWS")
    print(f"Confidence : {result['confidence_percent']}%")
    print(f"Model      : {result['model_used']}")
    print(f"P(real)    : {result['probability_real']} | P(fake): {result['probability_fake']}")
    print(f"Time       : {result['processing_ms']} ms")
    print(f"\n{result['explanation']}\n\n{result['disclaimer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
