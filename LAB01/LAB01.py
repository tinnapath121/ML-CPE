import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# LAB1: DATASET EXPLORATION
# Student Sleep & Mental Health Dataset
# ==========================================

print("=" * 80)
print(" " * 20 + "LAB1: DATASET EXPLORATION")
print("=" * 80)

# [1] LOAD DATASET
print("\n[1] LOAD DATASET")
print("-" * 80)
df = pd.read_csv('student_sleep_mental_health_2026.csv')
print(f"✓ Dataset loaded successfully!")
print(f"File name: student_sleep_mental_health_2026.csv")

# [2] DISPLAY SHAPE
print("\n[2] DISPLAY SHAPE")
print("-" * 80)
rows, cols = df.shape
print(f"Number of Rows (Samples):    {rows}")
print(f"Number of Columns (Features): {cols}")
print(f"Dataset Size:                {rows} x {cols}")

# [3] DISPLAY DATA TYPES
print("\n[3] DISPLAY DATA TYPES")
print("-" * 80)
print(df.dtypes)

# [4] DISPLAY SUMMARY STATISTICS
print("\n[4] DISPLAY SUMMARY STATISTICS")
print("-" * 80)
print(df.describe())

# [5] DISPLAY MISSING VALUES
print("\n[5] DISPLAY MISSING VALUES")
print("-" * 80)
missing = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing.to_string(index=False))
print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")

# [6] DISPLAY DUPLICATE RECORDS
print("\n[6] DISPLAY DUPLICATE RECORDS")
print("-" * 80)
duplicates = df.duplicated().sum()
print(f"Total Duplicate Rows: {duplicates}")
print(f"Duplicate Percentage: {(duplicates / len(df) * 100):.2f}%")

if duplicates > 0:
    print("\nDuplicate Rows Preview:")
    print(df[df.duplicated(keep=False)].head(10))

# [7] DISPLAY CLASS DISTRIBUTION
print("\n[7] DISPLAY CLASS DISTRIBUTION")
print("-" * 80)

# หา categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    for col in categorical_cols[:3]:  # แสดง 3 columns แรก
        print(f"\n📊 Column: '{col}'")
        print(df[col].value_counts())
        print(f"\nPercentage:")
        print((df[col].value_counts() / len(df) * 100).round(2))
        print("-" * 40)

# [8] FIRST FEW ROWS
print("\n[8] FIRST 5 ROWS")
print("-" * 80)
print(df.head())

# [9] LAST FEW ROWS
print("\n[9] LAST 5 ROWS")
print("-" * 80)
print(df.tail())

# [10] SUMMARY REPORT
print("\n" + "=" * 80)
print(" " * 30 + "SUMMARY REPORT")
print("=" * 80)
print(f"✓ Total Rows:           {rows}")
print(f"✓ Total Columns:        {cols}")
print(f"✓ Total Missing Values: {df.isnull().sum().sum()}")
print(f"✓ Total Duplicate Rows: {duplicates}")
quality_score = (1 - (df.isnull().sum().sum() + duplicates) / (rows * cols)) * 100
print(f"✓ Data Quality Score:   {quality_score:.2f}%")
print("=" * 80)

# CREATE VISUALIZATIONS
print("\n📊 Creating visualizations...")

fig = plt.figure(figsize=(15, 10))

# Plot 1: Data Overview
ax1 = plt.subplot(2, 3, 1)
info = {
    'Total Rows': rows,
    'Total Cols': cols,
    'Missing': df.isnull().sum().sum(),
    'Duplicates': duplicates
}
ax1.bar(info.keys(), info.values(), color=['green', 'blue', 'red', 'orange'], edgecolor='black', linewidth=2)
ax1.set_title('Dataset Overview', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate(info.values()):
    ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Plot 2: Missing Values
ax2 = plt.subplot(2, 3, 2)
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    missing_counts[missing_counts > 0].plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
    ax2.set_title('Missing Values by Column', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Count')
else:
    ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.set_title('Missing Values', fontsize=12, fontweight='bold')

# Plot 3: Data Types Distribution
ax3 = plt.subplot(2, 3, 3)
dtype_counts = df.dtypes.value_counts()
colors = plt.cm.Set3(range(len(dtype_counts)))
ax3.pie(dtype_counts, labels=dtype_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax3.set_title('Data Types Distribution', fontsize=12, fontweight='bold')

# Plot 4-6: Numeric columns distribution
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for idx, col in enumerate(numeric_cols[:3]):
    ax = plt.subplot(2, 3, 4 + idx)
    ax.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.suptitle('Student Sleep & Mental Health - Dataset Exploration', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('dataset_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Chart saved as 'dataset_exploration.png'")
plt.show()

print("\n✓ LAB1 Dataset Exploration Complete!")
print("=" * 80)