import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
IMG_SIZE = 100
N_SAMPLES = 4


def test_svm(n_samples=N_SAMPLES):

    # Load model and test set
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # Pick random images
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # Predict & Calculate Probability
    X_sample_scaled = scaler.transform(X_sample)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sample_scaled)
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1) * 100
    else:
        predictions = model.predict(X_sample_scaled)
        confidences = None

    # Show results in a 2x2 grid
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        # แสดงผลข้อความพร้อม % แบบรูปของอาจารย์
        if confidences is not None:
            title_text = f"Pred: {pred} ({int(confidences[i])}%)\nTrue: {true}"
        else:
            title_text = f"Pred: {pred}\nTrue: {true}"

        ax.set_title(title_text, color=color)

        print(f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
              f"{'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_svm()