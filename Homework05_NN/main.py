# main.py
import json
import os
import numpy as np
import pandas as pd

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "animals")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 1000
BATCH_SIZE = 32

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data & Preprocess & Split
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    X = to_features(images)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, labels, TEST_SIZE, VAL_SIZE)

    # เซฟไฟล์ X_test, y_test และ classes สำหรับให้ test_nn.py ดึงไปใช้
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    # 2. กำหนดโครงสร้างการทดลอง
    configurations = {
        "Small (1 Layer)": [64],
        "Medium (2 Layers)": [128, 64],
        "Large (3 Layers)": [256, 128, 64]
    }
    epochs_to_test = [10, 20, 30]

    results = []

    print("\n" + "="*60)
    print(" Running LAB 6 Experiments: Configurations vs Epochs ")
    print("="*60)

    last_model = None
    last_history = None

    for config_name, layers_shape in configurations.items():
        for epochs in epochs_to_test:
            print(f"\n[Testing] Config: {config_name} | Epochs: {epochs}")

            # ใส่ output_dir=OUTPUT_DIR เพื่อเซฟ nn_model.keras ลงโฟลเดอร์ outputs
            model, history = train_model(
                X_train, y_train, X_val, y_val, len(classes),
                hidden_layers=layers_shape, output_dir=OUTPUT_DIR, epochs=epochs, batch_size=BATCH_SIZE
            )

            predictions = predict_model(model, X_test)
            acc = float((predictions == y_test).mean())

            results.append({
                "Configuration": config_name,
                "Hidden Layers": str(layers_shape),
                "Epochs": epochs,
                "Test Accuracy (%)": round(acc * 100, 2)
            })

            last_model = model
            last_history = history

    # 3. สร้างรูป Confusion Matrix และ Training History สำหรับโมเดลล่าสุด
    if last_model and last_history:
        predictions = predict_model(last_model, X_test)
        evaluate_model(y_test, predictions, classes, save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
        plot_history(last_history, f"{OUTPUT_DIR}/training_history.png")

    # 4. แสดงผลการเปรียบเทียบในรูปแบบตารางสรุป
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print(" EXPERIMENTAL RESULTS COMPARISON ")
    print("="*60)
    print(df_results.to_string(index=False))

    df_results.to_csv(os.path.join(OUTPUT_DIR, "experiment_results.csv"), index=False)
    print(f"\nSaved comparison results to {OUTPUT_DIR}/experiment_results.csv")

if __name__ == "__main__":
    main()