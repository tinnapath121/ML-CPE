"""Read CSV for drug-use-by-age

Convert text/symbols to number
Make Scaling for KNN
Split data: train / validation / test
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. วนลูปหาโฟลเดอร์หลัก (Homework03_KNN) อัตโนมัติ
project_root = Path(__file__).resolve()
while (
    project_root.name != "Homework03_KNN" and project_root != project_root.parent
):
    project_root = project_root.parent

CSV_PATH = project_root / "drug-use-by-age" / "drug-use-by-age.csv"


# ฟังก์ชันแปลงช่วงอายุเป็นกลุ่มคลาส (Target)
def get_age_group(age_str):
    age_str = str(age_str).strip()
    if age_str in ["12", "13", "14", "15", "16", "17"]:
        return "Teen"
    elif age_str in ["18", "19", "20", "21", "22-23", "24-25"]:
        return "Young Adult"
    else:
        return "Adult"


# ---------------------------------------------------------------------------
def load_data(test_size=0.2, seed=42):
    # Step 1: Read CSV & Validate existence
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"[Error] ไม่พบไฟล์ข้อมูลที่: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Step 2: Clean data (แปลงเครื่องหมาย '-' เป็น 0)
    df_clean = df.copy()
    for col in df_clean.columns:
        if col != "age":
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace("-", ""), errors="coerce"
            )
    df_clean = df_clean.fillna(0)

    # สร้าง Target Column
    df_clean["Age_Group"] = df_clean["age"].apply(get_age_group)
    TARGET = "Age_Group"

    # ดึงคอลัมน์ที่เป็น Feature (สถิติการใช้ยาตัวเลขทั้งหมด ยกเว้น age, n และ Age_Group)
    feature_cols = [
        c for c in df_clean.columns if c not in ["age", "n", "Age_Group"]
    ]

    X = df_clean[feature_cols].to_numpy(dtype="float32")

    class_names = sorted(df_clean[TARGET].unique())
    y = (
        df_clean[TARGET]
        .map({name: i for i, name in enumerate(class_names)})
        .to_numpy(dtype="int32")
    )

    # Step 3: Split data เป็น train / validation / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )

    # Step 4: Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "feature_names": feature_cols,
        "n_rows": len(df_clean),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("คลาส  :", data["class_names"])