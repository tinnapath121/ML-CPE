import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path, title=None):
    """Accuracy and loss curves — the main tool for spotting overfitting.

    history: a Keras History object or its .history dict.
    """

    h = getattr(history, "history", history)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(h["accuracy"], label="train")
    axes[0].plot(h["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(h["loss"], label="train")
    axes[1].plot(h["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_epoch_comparison(rows, save_path):
    """Accuracy vs number of epochs, one line per CNN configuration.

    rows: list of dicts with keys config, epochs, val_acc, test_acc.
    """

    configs = list(dict.fromkeys(r["config"] for r in rows))
    epochs = sorted({r["epochs"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, key, title in (
        (axes[0], "val_acc", "Validation accuracy"),
        (axes[1], "test_acc", "Test accuracy"),
    ):
        for config in configs:
            points = sorted((r["epochs"], r[key]) for r in rows
                            if r["config"] == config)
            ax.plot(*zip(*points), marker="o", label=config)

        ax.set_xticks(epochs)
        ax.set_xlabel("Number of epochs")
        ax.set_title(title)
        ax.grid(alpha=0.3)

        # Accuracy cannot exceed 1.0, so do not let autoscale go past it
        ax.set_ylim(top=min(ax.get_ylim()[1], 1.01))

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_config_comparison(summary, save_path):
    """Bar chart: validation vs test accuracy of each configuration's best model.

    summary: list of dicts with keys config, best_val_acc, test_acc.
    """

    names = [s["config"] for s in summary]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_val = ax.bar(x - width / 2, [s["best_val_acc"] for s in summary],
                      width, label="validation")
    bars_test = ax.bar(x + width / 2, [s["test_acc"] for s in summary],
                       width, label="test")

    for bar in (*bars_val, *bars_test):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height() * 100:.1f}", ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("CNN configuration comparison", loc="left")

    # Zoom in on the useful range so small differences stay visible.
    # Headroom above 1.0 keeps the value labels clear of the title.
    low = min(min(s["best_val_acc"], s["test_acc"]) for s in summary)
    ax.set_ylim(max(0.0, low - 0.1), 1.08)
    ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0])
    # Legend above the axes, right-aligned, so it never covers a bar
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
              frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
