import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import data_loader
import evaluate
from knn_tf import TFKNNClassifier

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def title(text):
    print("\n" + "--" * 30)
    print(text)


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load data")

    data = data_loader.load_data()

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    print(f"data of all : {data['n_rows']} rows")
    print(f"classes of predictions : {class_names}")
    print(
        f"split data    : train {len(y_train)} / validation {len(y_val)} / test {len(y_test)}"
    )

    title("STEP 2 : search k for finding the best value")

    # ปรับ k_values ให้เหมาะสมกับขนาดข้อมูลฝั่ง Train (มีประมาณ 10 ตัว)
    k_values = [1, 2, 3, 5, 7]
    scores = []

    for k in k_values:
        model = TFKNNClassifier(k=k)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        scores.append(acc)
        print(f"   k = {k:>2}  ->  validation accuracy = {acc:.4f}")

    best_k = k_values[int(np.argmax(scores))]
    print(f"\n>>> k value is best at {best_k}")

    evaluate.plot_k_curve(k_values, scores, OUT_DIR / "01_k_curve.png")

    title(
        f"STEP 3 : Training model with k = {best_k} and evaluate on test set"
    )

    model = TFKNNClassifier(k=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))
    print(f"Accuracy on test : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print("\nthe result show in class-wise report:\n")
    evaluate.print_report(y_test, y_pred, class_names)

    cm = evaluate.plot_confusion_matrix(
        y_test, y_pred, class_names, OUT_DIR / "02_confusion_matrix.png"
    )
    print("Confusion Matrix (rows = true labels, columns = predicted labels):")
    print(cm)

    title("STEP 4 : check code - compare with scikit-learn ALGORITHM")

    sk_model = KNeighborsClassifier(n_neighbors=best_k)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)

    print(f"KNN: result of our (TensorFlow) : {accuracy:.4f}")
    print(f"KNN: result of sklearn         : {np.mean(sk_pred == y_test):.4f}")
    print(
        f"KNN: matching predictions : {np.mean(sk_pred == y_pred) * 100:.1f} %"
    )

    title("STEP 5 : Is our model better than guessing? ")

    majority = np.bincount(y_train).argmax()
    baseline = float(np.mean(y_test == majority))

    print(
        f"Baseline (predict '{class_names[majority]}' every time) : {baseline:.4f}"
    )
    print(f"KNN: our model                                : {accuracy:.4f}")

    if accuracy >= baseline:
        print("\n[summary] KNN beats or equals baseline!")
    else:
        print("\n[summary] Model performs below baseline.")

    # ---------------------------------------------------------------------------
    title("save predictions to CSV file")

    evaluate.save_predictions(
        y_test, y_pred, class_names, OUT_DIR / "predictions.csv"
    )

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()