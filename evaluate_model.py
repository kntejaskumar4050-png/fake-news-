"""
Independent evaluation of the saved model artifacts.

This script reloads `lstm_model.keras` and `bilstm_model.keras` from disk,
re-runs inference on the held-out test split and regenerates every metric,
confusion matrix and comparison report. Use it to verify the numbers shown in
the web app were genuinely produced by the shipped models.

    python evaluate_model.py
    python evaluate_model.py --threshold 0.5 --model bilstm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import config
from evaluation import (
    compute_metrics,
    dump_json,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    plot_training_curves,
)
from preprocess import TextPreprocessor, load_clean_dataset, split_dataset

MODEL_FILES = {"lstm": config.LSTM_MODEL_PATH, "bilstm": config.BILSTM_MODEL_PATH}


def load_reference_metrics() -> dict:
    if config.METRICS_PATH.exists():
        try:
            return json.loads(config.METRICS_PATH.read_text())
        except Exception:
            return {}
    return {}


def main(threshold: float = 0.5, selected: list[str] | None = None) -> int:
    import tensorflow as tf

    print("=" * 74)
    print(" FAKE NEWS DETECTOR — INDEPENDENT MODEL EVALUATION")
    print("=" * 74)

    missing = [str(path.name) for path in MODEL_FILES.values() if not path.exists()]
    if missing:
        print(f"ERROR: missing model files: {', '.join(missing)}")
        print("Train the models first:  python train_model.py")
        return 1

    df = load_clean_dataset()
    _, _, test_df = split_dataset(df)
    preprocessor = TextPreprocessor.load()
    X_test = preprocessor.transform(test_df["clean"])
    y_test = test_df["label"].to_numpy().astype(int)
    print(f"Test split: {len(test_df):,} articles | vocabulary: {preprocessor.vocab_size:,}")
    print(f"Decision threshold: {threshold:.2f}\n")

    reference = load_reference_metrics()
    keys = selected or [k for k in ("lstm", "bilstm") if MODEL_FILES[k].exists()]
    results: dict[str, dict] = {}
    roc_curves: dict[str, tuple] = {}

    for key in keys:
        path = MODEL_FILES[key]
        print(f"Evaluating {config.DISPLAY_NAMES[key]} ({path.name}) …")
        model = tf.keras.models.load_model(path, compile=False)
        started = time.perf_counter()
        probabilities = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0).ravel()
        elapsed = time.perf_counter() - started

        metrics = compute_metrics(y_test, probabilities, threshold=threshold)
        metrics["display_name"] = config.DISPLAY_NAMES[key]
        metrics["model_file"] = path.name
        metrics["parameters"] = int(model.count_params())
        metrics["inference_seconds"] = round(elapsed, 2)
        metrics["ms_per_article"] = round(elapsed / max(len(y_test), 1) * 1000, 3)
        results[key] = metrics
        roc_curves[key] = (y_test, probabilities)

        stored = reference.get("models", {}).get(key, {})
        delta = (
            abs(metrics["accuracy"] - stored["accuracy"])
            if stored.get("accuracy") is not None else None
        )
        metrics["matches_stored_metrics"] = bool(delta is not None and delta < 1e-6)

        print(f"  accuracy {metrics['accuracy']:.4f} | precision {metrics['precision']:.4f} "
              f"| recall {metrics['recall']:.4f} | f1 {metrics['f1_score']:.4f} "
              f"| roc-auc {metrics['roc_auc']:.4f}")
        print(f"  confusion matrix: {metrics['confusion_matrix']}")
        if delta is not None:
            print(f"  matches training metrics: {metrics['matches_stored_metrics']} (Δ={delta:.2e})")
        print("  classification report:")
        for label in ("FAKE", "REAL"):
            block = metrics["classification_report"][label]
            print(f"    {label:>4}: precision {block['precision']:.3f} "
                  f"recall {block['recall']:.3f} f1 {block['f1-score']:.3f} "
                  f"support {block['support']}")

        plot_confusion_matrix(metrics, key, config.IMAGES_DIR / f"confusion_matrix_{key}.png")
        stored_history = stored.get("history")
        if stored_history:
            plot_training_curves(stored_history, key, config.IMAGES_DIR / f"training_{key}.png")

    if len(results) > 1:
        plot_model_comparison(results, config.IMAGES_DIR / "model_comparison.png")
        plot_roc_curves(roc_curves, config.IMAGES_DIR / "roc_curves.png")

    rows = [
        {
            "Model": item["display_name"],
            "Accuracy": round(item["accuracy"], 4),
            "Precision": round(item["precision"], 4),
            "Recall": round(item["recall"], 4),
            "F1-score": round(item["f1_score"], 4),
            "ROC-AUC": round(item["roc_auc"] or 0, 4),
            "Parameters": item["parameters"],
            "ms/article": item["ms_per_article"],
        }
        for item in results.values()
    ]
    comparison = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "threshold": threshold,
        "test_samples": int(len(y_test)),
        "vocabulary_size": int(preprocessor.vocab_size),
        "max_len": int(preprocessor.max_len),
        "results": results,
        "best_model": max(results, key=lambda k: results[k]["f1_score"]),
        "comparison": rows,
    }
    dump_json(report, config.REPORTS_DIR / "evaluation_report.json")

    print("\n" + "=" * 74)
    print(" COMPARISON")
    print("=" * 74)
    print(comparison.to_string(index=False))
    print(f"\nReport : {config.REPORTS_DIR / 'evaluation_report.json'}")
    print(f"Charts : {config.IMAGES_DIR}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved fake-news models")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", choices=["lstm", "bilstm", "both"], default="both")
    args = parser.parse_args()
    chosen = None if args.model == "both" else [args.model]
    raise SystemExit(main(threshold=args.threshold, selected=chosen))
