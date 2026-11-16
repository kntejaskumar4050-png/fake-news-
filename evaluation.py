"""Shared evaluation helpers: metrics, plots and reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config

# Consistent academic-report visual style
NAVY = "#0b1f3a"
BLUE = "#1d4ed8"
LIGHT_BLUE = "#38bdf8"
RED = "#dc2626"
GREEN = "#059669"
GRID = "#e2e8f0"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelcolor": NAVY,
    "axes.edgecolor": GRID,
    "text.color": NAVY,
})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: Sequence[int], probabilities: Iterable[float],
                    threshold: float = 0.5) -> dict:
    """Compute the full metric suite used for the model comparison."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float).ravel()
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "samples": int(len(y_true)),
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["FAKE", "REAL"],
            output_dict=True, zero_division=0,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def plot_training_curves(history: dict, model_key: str, out_path: Path,
                         title: str | None = None) -> Path:
    """Loss + accuracy curves for one model, saved as a PNG."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    epochs = range(1, len(history.get("loss", [])) + 1)
    label = title or config.DISPLAY_NAMES.get(model_key, model_key)

    axes[0].plot(epochs, history["loss"], marker="o", lw=2, color=BLUE, label="Training loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], marker="s", lw=2,
                     color=RED, label="Validation loss")
    axes[0].set_title(f"{label} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(list(epochs))

    axes[1].plot(epochs, history.get("accuracy", []), marker="o", lw=2,
                 color=GREEN, label="Training accuracy")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], marker="s", lw=2,
                     color=LIGHT_BLUE, label="Validation accuracy")
    axes[1].set_title(f"{label} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(frameon=False)
    axes[1].set_xticks(list(epochs))

    fig.suptitle(f"{label} training history", fontsize=15, fontweight="bold", color=NAVY)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(metrics: dict, model_key: str, out_path: Path) -> Path:
    """Seaborn heatmap of the test-set confusion matrix."""
    matrix = np.array(metrics["confusion_matrix"])
    label = config.DISPLAY_NAMES.get(model_key, model_key)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(
        matrix, annot=True, fmt=",d", cmap="Blues", cbar=False,
        xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"],
        annot_kws={"size": 14, "weight": "bold"}, linewidths=1,
        linecolor="white", ax=ax,
    )
    ax.set_title(f"{label} — Confusion Matrix\nTest samples: {metrics['samples']:,}",
                 color=NAVY)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_model_comparison(results: dict[str, dict], out_path: Path) -> Path:
    """Grouped bar chart comparing LSTM and BiLSTM on the held-out test set."""
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    models = list(results.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(models), 1)
    colors = {"lstm": BLUE, "bilstm": LIGHT_BLUE}

    fig, ax = plt.subplots(figsize=(11, 5))
    for index, key in enumerate(models):
        values = [
            (results[key].get(metric) or 0.0) * 100 for metric in metric_keys
        ]
        bars = ax.bar(
            x + index * width - (len(models) - 1) * width / 2,
            values, width, label=config.DISPLAY_NAMES.get(key, key),
            color=colors.get(key, NAVY), edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=NAVY)
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(80, 101.5)
    ax.set_ylabel("Score (%)")
    ax.set_title("LSTM vs BiLSTM — held-out test set performance", color=NAVY)
    ax.legend(frameon=False, ncol=len(models), loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_roc_curves(curves: dict[str, tuple[Sequence[int], Sequence[float]]],
                    out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    colors = {"lstm": BLUE, "bilstm": LIGHT_BLUE}
    for key, (y_true, probs) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, lw=2.2, color=colors.get(key, NAVY),
                label=config.DISPLAY_NAMES.get(key, key))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — test set", color=NAVY)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def dump_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
