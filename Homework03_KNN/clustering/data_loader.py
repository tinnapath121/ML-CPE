from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. วนลูปหาโฟลเดอร์หลัก (Homework03_KNN) อัตโนมัติ
project_root = Path(__file__).resolve()
while (
    project_root.name != "Homework03_KNN" and project_root != project_root.parent
):
    project_root = project_root.parent

# 2. ชี้ตำแหน่งไปยังโฟลเดอร์ drug-use-by-age
DATA_DIR = project_root / "drug-use-by-age"
CSV_PATH = DATA_DIR / "drug-use-by-age.csv"


# ---------------------------------------------------------------------------
def load_data():
    """โหลดข้อมูล drug-use-by-age และแปลงคอลัมน์ตัวเลขสำหรับทำ Clustering"""
    # ตรวจสอบว่าพบไฟล์จริงหรือไม่
    if not CSV_PATH.exists():
        # ถ้าไม่เจอตามชื่อ ให้ลองค้นหาไฟล์ .csv ใดก็ได้ในโฟลเดอร์ drug-use-by-age
        csv_files = list(DATA_DIR.glob("*.csv"))
        if csv_files:
            target_path = csv_files[0]
        else:
            raise FileNotFoundError(
                f"\n[Error] ไม่พบไฟล์ที่ตำแหน่ง:\n  {CSV_PATH}\n"
                f"กรุณาเช็กว่าโฟลเดอร์ {DATA_DIR} มีไฟล์ CSV อยู่หรือไม่"
            )
    else:
        target_path = CSV_PATH

    df = pd.read_csv(target_path)

    # คลีนข้อมูล: ไฟล์ drug-use-by-age มักมีค่า '-' ในช่องความถี่/สถิติ ให้แปลงเป็น NaN แล้วแทนด้วย 0
    df_clean = df.copy()
    for col in df_clean.columns:
        if col not in ["age"]:
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace("-", ""), errors="coerce"
            )

    df_clean = df_clean.fillna(0)

    # ดึงคอลัมน์ที่เป็นตัวเลขทั้งหมดยกเว้น 'age' และ 'n' (จำนวนประชากร) เพื่อใช้จัดกลุ่ม
    feature_cols = [
        col
        for col in df_clean.columns
        if col not in ["age", "n"]
        and df_clean[col].dtype in ["float64", "int64", "float32", "int32"]
    ]

    X_raw = df_clean[feature_cols].to_numpy(dtype="float32")
    X = StandardScaler().fit_transform(X_raw).astype("float32")

    return {"X": X, "X_raw": X_raw, "df": df, "features": feature_cols}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        data = load_data()
        print("โหลดข้อมูลสำเร็จจาก:", CSV_PATH)
        print("size data :", data["X"].shape)
        print("features :", data["features"][:5], "...")
    except Exception as e:
        print(e)