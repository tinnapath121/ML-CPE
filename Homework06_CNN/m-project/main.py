import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
from tensorflow import keras

from data_loader import load_data
from preprocessing import to_features, compute_norm_stats
from split_data import split_dataset
from cnn_model import train_model, predict_model, EpochSnapshot
from evaluate import (
    evaluate_model,
    plot_history,
    plot_epoch_comparison,
    plot_config_comparison,
)

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_path")   # data_path/<class>/*.jpg
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = None   # None = use all images
EPOCHS = 30            # longest run; shorter runs are read off at the checkpoints
BATCH_SIZE = 32
SEED = 42

# Epoch counts to compare (all read from one run of EPOCHS epochs)
EPOCH_CHECKPOINTS = [5, 10, 20, 30]

# CNN configurations to compare: number of conv layers and number of neurons.
# Everything else (augmentation, optimizer, seed, data) is identical.
CONFIGS = {
    "A_conv32_d64":          {"conv_filters": (32,),          "dense_units": 64},
    "B_conv32-64_d128":      {"conv_filters": (32, 64),       "dense_units": 128},
    "C_conv32-64-128_d128":  {"conv_filters": (32, 64, 128),  "dense_units": 128},  # instructor's original
    "D_conv64-128-256_d256": {"conv_filters": (64, 128, 256), "dense_units": 256},
}


def parse_args():
    parser = argparse.ArgumentParser(description="CNN config / epoch comparison")
    parser.add_argument("--data", default=DATA_PATH, help="dataset folder")
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS,
                        help="limit images per class (quick tests)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="longest training length")
    return parser.parse_args()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"Saved: {path}")


def main():
    args = parse_args()
    epochs = args.epochs

    # Checkpoints never exceed the longest run, and always include it
    checkpoints = sorted({e for e in EPOCH_CHECKPOINTS if e <= epochs} | {epochs})

    print("--" * 30)
    print("CNN Image Recognition: configuration & epoch comparison")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(args.data, IMG_SIZE, args.max_per_class)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels

    np.save(f"{OUTPUT_DIR}/features.npy", X)

    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 3b: Standardization statistics (training set only)
    print("\n[Step 3b] Standardizing inputs: z = (x - mean) / std")
    norm_stats = compute_norm_stats(X_train)
    with open(f"{OUTPUT_DIR}/norm_stats.json", "w") as f:
        json.dump(norm_stats, f)
    print(f"Channel mean (RGB): {[round(m, 2) for m in norm_stats['mean']]}")
    print(f"Channel std  (RGB): {[round(s, 2) for s in norm_stats['std']]}")

    # Step 4: Train every configuration
    print(f"\n[Step 4] Training {len(CONFIGS)} configurations "
          f"for {epochs} epochs (checkpoints: {checkpoints})...")

    epoch_rows = []   # one row per (config, epoch checkpoint)
    summary = []      # one row per config (best-validation model)
    best = None

    for name, cfg in CONFIGS.items():
        print("\n" + "=" * 60)
        print(f"Configuration {name}")
        print("=" * 60)

        # Same seed for every configuration -> fair comparison
        keras.utils.set_random_seed(SEED)

        snapshot = EpochSnapshot(name, checkpoints, X_test, y_test)

        start = time.time()
        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            OUTPUT_DIR, epochs, BATCH_SIZE,
            conv_filters=cfg["conv_filters"],
            dense_units=cfg["dense_units"],
            norm_stats=norm_stats,
            extra_callbacks=[snapshot],
            model_name=name,
        )
        seconds = time.time() - start

        # Step 5: Prediction + Step 6: Evaluation for this configuration
        predictions = predict_model(model, X_test)
        test_acc = evaluate_model(y_test, predictions, classes)

        plot_history(history, f"{OUTPUT_DIR}/history_{name}.png", title=name)

        val_curve = history.history["val_accuracy"]
        best_epoch = int(np.argmax(val_curve)) + 1
        best_val_acc = float(max(val_curve))

        epoch_rows.extend(snapshot.rows)
        summary.append({
            "config": name,
            "conv_layers": len(cfg["conv_filters"]),
            "conv_filters": "-".join(str(f) for f in cfg["conv_filters"]),
            "dense_units": cfg["dense_units"],
            "params": int(model.count_params()),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": float(test_acc),
            "train_seconds": round(seconds, 1),
        })

        # Select the best model by VALIDATION accuracy, never by test accuracy
        if best is None or best_val_acc > best["val_acc"]:
            best = {"name": name, "val_acc": best_val_acc,
                    "predictions": predictions, "history": history}

    # Step 7: Save comparison tables and plots
    print("\n[Step 7] Saving comparison results...")

    write_csv(f"{OUTPUT_DIR}/results.csv", epoch_rows)   # config x epochs
    write_csv(f"{OUTPUT_DIR}/summary.csv", summary)      # config -> best model
    plot_epoch_comparison(epoch_rows, f"{OUTPUT_DIR}/epoch_comparison.png")
    plot_config_comparison(summary, f"{OUTPUT_DIR}/config_comparison.png")

    # Step 8: Best model under the names test_cnn.py expects
    print(f"\n[Step 8] Best configuration by validation accuracy: {best['name']}")

    shutil.copy(f"{OUTPUT_DIR}/models/{best['name']}.keras",
                f"{OUTPUT_DIR}/cnn_model.keras")
    shutil.copy(f"{OUTPUT_DIR}/history_{best['name']}.json",
                f"{OUTPUT_DIR}/history.json")

    evaluate_model(y_test, best["predictions"], classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(best["history"], f"{OUTPUT_DIR}/training_history.png",
                 title=best["name"])

    # Final table
    print("\n" + "-" * 78)
    print(f"{'config':<24}{'params':>10}{'best ep':>9}{'val acc':>10}"
          f"{'test acc':>10}{'sec':>8}")
    print("-" * 78)
    for s in summary:
        print(f"{s['config']:<24}{s['params']:>10}{s['best_epoch']:>9}"
              f"{s['best_val_acc'] * 100:>9.2f}%{s['test_acc'] * 100:>9.2f}%"
              f"{s['train_seconds']:>8}")
    print("-" * 78)
    print("Run test_cnn.py to see predictions of the best model.")


if __name__ == "__main__":
    main()
