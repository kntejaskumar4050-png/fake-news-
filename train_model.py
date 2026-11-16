"""
Train and compare the LSTM and BiLSTM fake-news classifiers.

Usage
-----
    python train_model.py                # train both models (default 8 epochs)
    python train_model.py --epochs 3     # quicker run
    python train_model.py --model lstm   # only one architecture

Outputs
-------
models/lstm_model.keras, models/bilstm_model.keras
models/tokenizer.pkl, models/preprocessor_config.json
models/metrics.json (test metrics, training history, config)
models/history/*.csv
static/images/training_*.png, confusion_matrix_*.png, model_comparison.png
reports/model_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Reduce noisy native logs before TensorFlow is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import config
from evaluation import (
    compute_metrics,
    dump_json,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_curves,
)
from preprocess import TextPreprocessor, load_clean_dataset, split_dataset


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seeds(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# Model architectures
# --------------------------------------------------------------------------- #
def build_lstm_model(vocab_size: int, max_len: int = config.MAX_LEN) -> keras.Model:
    """Embedding -> LSTM -> Dropout -> Dense -> sigmoid."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_len,), dtype="int32", name="tokens"),
            keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=config.EMBEDDING_DIM,
                name="embedding",
            ),
            keras.layers.SpatialDropout1D(config.SPATIAL_DROPOUT, name="embedding_dropout"),
            keras.layers.LSTM(
                config.LSTM_UNITS,
                dropout=0.2,
                recurrent_dropout=config.RECURRENT_DROPOUT,
                name="lstm",
            ),
            keras.layers.Dropout(config.DROPOUT, name="dropout"),
            keras.layers.Dense(config.DENSE_UNITS, activation="relu", name="dense"),
            keras.layers.Dropout(0.20, name="dropout_2"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="fake_news_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            clipnorm=config.GRADIENT_CLIPNORM,
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_bilstm_model(vocab_size: int, max_len: int = config.MAX_LEN) -> keras.Model:
    """Embedding -> Bidirectional(LSTM) -> Dropout -> Dense -> sigmoid."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_len,), dtype="int32", name="tokens"),
            keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=config.EMBEDDING_DIM,
                name="embedding",
            ),
            keras.layers.SpatialDropout1D(config.SPATIAL_DROPOUT, name="embedding_dropout"),
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    config.LSTM_UNITS,
                    dropout=0.2,
                    recurrent_dropout=config.RECURRENT_DROPOUT,
                ),
                name="bilstm",
            ),
            keras.layers.Dropout(config.DROPOUT, name="dropout"),
            keras.layers.Dense(config.DENSE_UNITS, activation="relu", name="dense"),
            keras.layers.Dropout(0.20, name="dropout_2"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="fake_news_bilstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            clipnorm=config.GRADIENT_CLIPNORM,
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


MODEL_BUILDERS = {"lstm": build_lstm_model, "bilstm": build_bilstm_model}
MODEL_PATHS = {"lstm": config.LSTM_MODEL_PATH, "bilstm": config.BILSTM_MODEL_PATH}


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def main(epochs: int = config.EPOCHS, selected: list[str] | None = None,
         batch_size: int = config.BATCH_SIZE) -> int:
    started = time.time()
    set_seeds()

    print("=" * 74)
    print(" FAKE NEWS DETECTION — MODEL TRAINING")
    print("=" * 74)
    print(f"TensorFlow {tf.__version__} | Python {os.sys.version.split()[0]}")

    df = load_clean_dataset()
    train_df, val_df, test_df = split_dataset(df)
    print(f"Dataset: {len(df):,} clean articles "
          f"(fake={int((df['label'] == 0).sum()):,}, real={int((df['label'] == 1).sum()):,})")
    print(f"Split  : train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Tokenizer is fitted on the TRAINING split only -> no leakage.
    preprocessor = TextPreprocessor().fit(train_df["clean"].tolist())
    preprocessor.save()
    vocab_size = preprocessor.vocab_size
    print(f"Tokenizer: vocabulary={vocab_size:,} | max_len={preprocessor.max_len}")

    X_train = preprocessor.transform(train_df["clean"])
    X_val = preprocessor.transform(val_df["clean"])
    X_test = preprocessor.transform(test_df["clean"])
    y_train = train_df["label"].to_numpy().astype("float32")
    y_val = val_df["label"].to_numpy().astype("float32")
    y_test = test_df["label"].to_numpy().astype("float32")

    print(f"Tensor shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    selected = selected or ["lstm", "bilstm"]
    results: dict[str, dict] = {}
    roc_data: dict[str, tuple] = {}
    overall_start = time.time()

    for key in selected:
        print("\n" + "-" * 74)
        print(f" Training {config.DISPLAY_NAMES[key]}")
        print("-" * 74)
        set_seeds()
        model = MODEL_BUILDERS[key](vocab_size, preprocessor.max_len)
        model.summary(print_fn=lambda line, **_: print("  " + line))

        checkpoint = MODEL_PATHS[key]
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=config.LR_PATIENCE,
                min_lr=1e-5,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(config.HISTORY_DIR / f"{key}_history.csv"), append=False
            ),
        ]

        model_start = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        train_seconds = time.time() - model_start
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        # The checkpoint callback saved the best val_loss weights; reload them so
        # evaluation is guaranteed to use exactly what is shipped.
        if Path(checkpoint).exists():
            model = keras.models.load_model(checkpoint)

        probabilities = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
        metrics = compute_metrics(y_test, probabilities)
        metrics.update(
            {
                "display_name": config.DISPLAY_NAMES[key],
                "model_file": checkpoint.name,
                "parameters": int(model.count_params()),
                "epochs_trained": len(history_dict["loss"]),
                "train_seconds": round(train_seconds, 1),
                "history": history_dict,
            }
        )
        results[key] = metrics
        roc_data[key] = (y_test, probabilities)

        print(f"\n{config.DISPLAY_NAMES[key]} test results")
        print(f"  accuracy : {metrics['accuracy']:.4f}")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall   : {metrics['recall']:.4f}")
        print(f"  f1-score : {metrics['f1_score']:.4f}")
        print(f"  roc-auc  : {metrics['roc_auc']:.4f}")
        print(f"  confusion: {metrics['confusion_matrix']}")
        print(f"  trained in {train_seconds:.1f}s ({metrics['epochs_trained']} epochs)")

        plot_training_curves(
            history_dict, key, config.IMAGES_DIR / f"training_{key}.png"
        )
        plot_confusion_matrix(
            metrics, key, config.IMAGES_DIR / f"confusion_matrix_{key}.png"
        )
        np.save(config.MODELS_DIR / f"test_probs_{key}.npy", probabilities)

    # ------------------------------------------------------------------ #
    # Comparison artefacts
    # ------------------------------------------------------------------ #
    comparison_keys = [k for k in ["lstm", "bilstm"] if k in results]
    best_model = max(results, key=lambda k: (results[k]["f1_score"], results[k]["accuracy"]))

    if len(comparison_keys) >= 2:
        plot_model_comparison(
            {k: results[k] for k in comparison_keys},
            config.IMAGES_DIR / "model_comparison.png",
        )
        try:
            from evaluation import plot_roc_curves
            plot_roc_curves({k: roc_data[k] for k in comparison_keys},
                            config.IMAGES_DIR / "roc_curves.png")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] ROC plot skipped: {exc}")

    comparison_rows = []
    for key in comparison_keys:
        item = results[key]
        comparison_rows.append(
            {
                "Model": config.DISPLAY_NAMES[key],
                "Accuracy": round(item["accuracy"], 4),
                "Precision": round(item["precision"], 4),
                "Recall": round(item["recall"], 4),
                "F1-score": round(item["f1_score"], 4),
                "ROC-AUC": round(item["roc_auc"] or 0, 4),
                "Parameters": item["parameters"],
                "Epochs": item["epochs_trained"],
                "Train time (s)": item["train_seconds"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "dataset": {
            "total_clean": int(len(df)),
            "fake": int((df["label"] == 0).sum()),
            "real": int((df["label"] == 1).sum()),
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
            "vocab_size": int(vocab_size),
            "max_len": int(preprocessor.max_len),
            "max_words": int(config.MAX_WORDS),
        },
        "training": {
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "embedding_dim": config.EMBEDDING_DIM,
            "lstm_units": config.LSTM_UNITS,
            "dropout": config.DROPOUT,
            "optimizer": "Adam",
            "learning_rate": config.LEARNING_RATE,
            "loss": "binary_crossentropy",
            "threshold": 0.5,
            "label_map": {"0": "FAKE", "1": "REAL"},
            "total_training_seconds": round(time.time() - overall_start, 1),
        },
        "models": results,
        "best_model": best_model,
        "comparison": comparison_rows,
    }
    dump_json(payload, config.METRICS_PATH)

    print("\n" + "=" * 74)
    print(" MODEL COMPARISON (held-out test set)")
    print("=" * 74)
    print(comparison_df.to_string(index=False))
    print(f"\nBest model by F1-score: {config.DISPLAY_NAMES[best_model]}")
    print(f"Metrics saved -> {config.METRICS_PATH}")
    print(f"Artifacts     -> {config.IMAGES_DIR}")
    print(f"Total time    : {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM and BiLSTM fake-news classifiers")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--model", choices=["lstm", "bilstm", "both"], default="both")
    args = parser.parse_args()
    chosen = None if args.model == "both" else [args.model]
    raise SystemExit(main(epochs=args.epochs, selected=chosen, batch_size=args.batch_size))
