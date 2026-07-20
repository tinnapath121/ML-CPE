"""
Part 4: Feature Engineering
Dataset: student_sleep_mental_health_2026.csv

- Label Encoding
- One-Hot Encoding

วิธีใช้ (VS Code):
1. วางไฟล์ student_sleep_mental_health_2026.csv ไว้โฟลเดอร์เดียวกับสคริปต์นี้
   (หรือแก้ path ในตัวแปร CSV_PATH ด้านล่าง)
2. ติดตั้งไลบรารีที่ต้องใช้ (ครั้งเดียว):
   pip install pandas scikit-learn
3. รันสคริปต์นี้ใน VS Code (กด Run Python File หรือ python feature_engineering.py)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# 0) โหลดข้อมูล
# -------------------------------------------------
CSV_PATH = "student_sleep_mental_health_2026.csv"
df = pd.read_csv(CSV_PATH)

print("=== ข้อมูลเดิม (5 แถวแรก) ===")
print(df.head())
print("\n=== ชนิดข้อมูลแต่ละคอลัมน์ ===")
print(df.dtypes)

# -------------------------------------------------
# ตัวแปรที่เป็น Categorical ในชุดข้อมูลนี้
#   - gender            : Non-binary, Female, Male, Prefer not to say   (ไม่มีลำดับ -> เหมาะกับ One-Hot)
#   - education_level    : High School, Undergraduate, Graduate         (มีลำดับ    -> เหมาะกับ Label Encoding)
#   - uses_sleep_app     : True/False (bool) 
#   - feels_burned_out   : True/False (bool)  (เป็น target/label ก็ได้)
# -------------------------------------------------

df_encoded = df.copy()

# =================================================
# 1) LABEL ENCODING
# =================================================
# ใช้กับตัวแปรที่ "มีลำดับ" (ordinal) เช่น education_level
# High School < Undergraduate < Graduate

# 1.1 กำหนดลำดับเอง (แนะนำ เพราะควบคุมลำดับได้ถูกต้องตามความหมายจริง)
education_order = {
    "High School": 0,
    "Undergraduate": 1,
    "Graduate": 2
}
df_encoded["education_level_encoded"] = df_encoded["education_level"].map(education_order)

# 1.2 หรือใช้ LabelEncoder จาก sklearn (จะเรียงตามตัวอักษร ไม่ตามความหมาย ควรระวัง)
le = LabelEncoder()
df_encoded["education_level_label_sklearn"] = le.fit_transform(df_encoded["education_level"])
print("\nLabelEncoder classes (education_level):", list(le.classes_))

# Label Encoding สำหรับตัวแปร boolean (True/False -> 1/0)
df_encoded["uses_sleep_app_encoded"] = df_encoded["uses_sleep_app"].astype(int)
df_encoded["feels_burned_out_encoded"] = df_encoded["feels_burned_out"].astype(int)

print("\n=== หลัง Label Encoding ===")
print(df_encoded[["education_level", "education_level_encoded",
                   "education_level_label_sklearn",
                   "uses_sleep_app", "uses_sleep_app_encoded",
                   "feels_burned_out", "feels_burned_out_encoded"]].head())

# =================================================
# 2) ONE-HOT ENCODING
# =================================================
# ใช้กับตัวแปรที่ "ไม่มีลำดับ" (nominal) เช่น gender
# เพราะตัวเลขที่ได้จาก Label Encoding อาจทำให้โมเดลเข้าใจผิดว่ามีลำดับ/ระยะห่างระหว่างค่า

df_onehot = pd.get_dummies(
    df_encoded,
    columns=["gender"],
    prefix="gender",
    drop_first=False   # ถ้าอยากลดปัญหา multicollinearity ตั้งเป็น True
)

print("\n=== หลัง One-Hot Encoding (คอลัมน์ gender_...) ===")
gender_cols = [c for c in df_onehot.columns if c.startswith("gender_")]
print(df_onehot[["student_id"] + gender_cols].head())

# =================================================
# 3) รวมทุกอย่างเป็นชุดข้อมูลสุดท้ายสำหรับโมเดล (ตัวอย่าง)
# =================================================
final_columns = [
    "age",
    "gender_Female", "gender_Male", "gender_Non-binary", "gender_Prefer not to say",
    "education_level_encoded",
    "avg_sleep_hours", "screen_time_hours", "social_media_hours",
    "study_hours_per_day", "exercise_hours_per_week",
    "caffeine_drinks_per_day", "stress_level", "anxiety_score", "gpa",
    "uses_sleep_app_encoded",
    "feels_burned_out_encoded"  # เช่น ใช้เป็น target
]

df_final = df_onehot[final_columns]

print("\n=== Dataset สุดท้ายพร้อมใช้เทรนโมเดล ===")
print(df_final.head())
print("\nShape:", df_final.shape)

# -------------------------------------------------
# 4) บันทึกไฟล์ผลลัพธ์
# -------------------------------------------------
OUTPUT_PATH = "student_sleep_mental_health_2026_encoded.csv"
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\nบันทึกไฟล์ผลลัพธ์แล้วที่: {OUTPUT_PATH}")