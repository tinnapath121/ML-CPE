import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# LAB3: DATA CLEANING
# Student Sleep & Mental Health Dataset
# ==========================================

print("=" * 90)
print(" " * 25 + "LAB3: DATA CLEANING")
print("=" * 90)

# [1] LOAD DATASET
print("\n[1] LOAD DATASET")
print("-" * 90)
df = pd.read_csv('student_sleep_mental_health_2026.csv')
print(f"✓ Original Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Create a copy for cleaning
df_cleaned = df.copy()

print(f"\n✓ Dataset Overview:")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"  Data types: {df.dtypes.value_counts().to_dict()}")

# ==========================================
# [2] MISSING VALUE HANDLING
# ==========================================

print("\n[2] MISSING VALUE HANDLING")
print("-" * 90)

# Check for missing values
missing_before = df_cleaned.isnull().sum()
print(f"\n✓ Missing Values BEFORE Cleaning:")
print(f"  Total missing values: {missing_before.sum()}")
print(missing_before[missing_before > 0].to_string() if missing_before.sum() > 0 else "  No missing values found!")

# Strategy for handling missing values
print(f"\n✓ Missing Value Handling Strategy:")
print(f"  - Numeric columns: Fill with MEDIAN (robust to outliers)")
print(f"  - Categorical columns: Fill with MODE (most frequent value)")

# Handle missing values for numeric columns
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df_cleaned[col].isnull().sum() > 0:
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)
        print(f"  ✓ {col}: Filled {missing_before[col]} missing values with median {median_val:.2f}")

# Handle missing values for categorical columns
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        mode_val = df_cleaned[col].mode()[0]
        df_cleaned[col].fillna(mode_val, inplace=True)
        print(f"  ✓ {col}: Filled {missing_before[col]} missing values with mode '{mode_val}'")

# Verify
missing_after = df_cleaned.isnull().sum().sum()
print(f"\n✓ Missing Values AFTER Cleaning: {missing_after}")

# ==========================================
# [3] DUPLICATE REMOVAL
# ==========================================

print("\n[3] DUPLICATE REMOVAL")
print("-" * 90)

# Check for duplicates
duplicates_before = df_cleaned.duplicated().sum()
print(f"\n✓ Duplicate Records BEFORE Cleaning: {duplicates_before}")

if duplicates_before > 0:
    print(f"\n  Showing first 5 duplicate rows:")
    print(df_cleaned[df_cleaned.duplicated(keep=False)].head(10).to_string())
    
    # Remove duplicates (keep first occurrence)
    df_cleaned = df_cleaned.drop_duplicates(keep='first')
    print(f"\n✓ Removed {duplicates_before} duplicate rows (keep='first')")
else:
    print(f"  No duplicate rows found!")

print(f"✓ Dataset shape after duplicate removal: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")

# ==========================================
# [4] INCORRECT DATA CORRECTION
# ==========================================

print("\n[4] INCORRECT DATA CORRECTION")
print("-" * 90)

print(f"\n✓ Data Quality Checks:")

# Check for negative values in columns that should be positive
for col in numeric_cols:
    if (df_cleaned[col] < 0).sum() > 0:
        print(f"  ⚠ {col}: Found {(df_cleaned[col] < 0).sum()} negative values")
        # Option 1: Convert negative to absolute value
        # Option 2: Replace with NaN then fill with median
        df_cleaned[col] = df_cleaned[col].abs()
        print(f"    → Fixed: Converted to absolute values")

# Check for outliers (using IQR method)
print(f"\n✓ Outlier Detection (IQR Method):")
outliers_detected = 0

for col in numeric_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_count = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
    
    if outlier_count > 0:
        print(f"  ⚠ {col}: Found {outlier_count} outliers")
        print(f"    Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        outliers_detected += outlier_count

print(f"  Total outliers detected: {outliers_detected}")

# Check for data consistency
print(f"\n✓ Data Consistency Checks:")
for col in numeric_cols:
    if (df_cleaned[col] > 100).sum() > 0:  # Example: values > 100
        print(f"  ⚠ {col}: Values exceed reasonable range")
        
if (df_cleaned[numeric_cols[0]] < 0).sum() == 0:
    print(f"  ✓ All numeric values are within reasonable ranges")

# ==========================================
# [5] DATA TYPE CONVERSION
# ==========================================

print("\n[5] DATA TYPE CONVERSION")
print("-" * 90)

print(f"\n✓ Original Data Types:")
print(df.dtypes.to_string())

print(f"\n✓ Data Type Conversion Strategy:")

# Convert appropriate columns to more efficient types
# Example: Convert integer columns that represent categories to category type
original_types = df_cleaned.dtypes.copy()

# Optimize memory: convert object to category where appropriate
for col in categorical_cols:
    n_unique = df_cleaned[col].nunique()
    print(f"\n  {col}:")
    print(f"    Original type: object")
    print(f"    Unique values: {n_unique}")
    
    if n_unique <= 50:  # If few unique values, convert to category
        df_cleaned[col] = df_cleaned[col].astype('category')
        print(f"    → Converted to: category (saves memory)")

# Convert float columns with no decimal to int (if appropriate)
for col in numeric_cols:
    if (df_cleaned[col] % 1 == 0).all():  # All values are whole numbers
        if col not in df_cleaned.columns[df_cleaned.dtypes == 'category']:
            df_cleaned[col] = df_cleaned[col].astype('int64')
            print(f"\n  {col}:")
            print(f"    Original type: float64")
            print(f"    → Converted to: int64")

print(f"\n✓ Updated Data Types:")
print(df_cleaned.dtypes.to_string())

print(f"\n✓ Memory Usage Comparison:")
print(f"  Before: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"  After:  {df_cleaned.memory_usage(deep=True).sum() / 1024:.2f} KB")
memory_saved = (df.memory_usage(deep=True).sum() - df_cleaned.memory_usage(deep=True).sum()) / 1024
if memory_saved > 0:
    print(f"  Saved:  {memory_saved:.2f} KB ({(memory_saved / (df.memory_usage(deep=True).sum() / 1024) * 100):.1f}%)")

# ==========================================
# [6] COMPARE MEAN vs MEDIAN
# ==========================================

print("\n[6] COMPARE MEAN vs MEDIAN")
print("-" * 90)

# Select numeric columns for comparison
numeric_cols_cleaned = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n✓ Statistical Comparison (Original vs Cleaned):")
print(f"\n{'Column':<25} {'Metric':<15} {'Before':<15} {'After':<15}")
print("-" * 70)

comparison_data = []

for col in numeric_cols_cleaned:
    if col in numeric_cols:  # Only compare numeric columns
        mean_before = df[col].mean()
        median_before = df[col].median()
        mean_after = df_cleaned[col].mean()
        median_after = df_cleaned[col].median()
        
        comparison_data.append({
            'Column': col,
            'Mean_Before': mean_before,
            'Median_Before': median_before,
            'Mean_After': mean_after,
            'Median_After': median_after,
            'Mean_Diff': abs(mean_after - mean_before),
            'Median_Diff': abs(median_after - median_before)
        })
        
        print(f"{col:<25} {'Mean':<15} {mean_before:<15.4f} {mean_after:<15.4f}")
        print(f"{'':<25} {'Median':<15} {median_before:<15.4f} {median_after:<15.4f}")
        print("-" * 70)

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)

print(f"\n✓ Mean vs Median Interpretation:")
print(f"  - If Mean ≈ Median: Distribution is SYMMETRIC (normal)")
print(f"  - If Mean > Median: Distribution is RIGHT-SKEWED")
print(f"  - If Mean < Median: Distribution is LEFT-SKEWED")

print(f"\n✓ Distribution Shape Analysis:")
for col in numeric_cols_cleaned:
    mean_val = df_cleaned[col].mean()
    median_val = df_cleaned[col].median()
    skewness = df_cleaned[col].skew()
    
    if abs(mean_val - median_val) < 0.01 * mean_val:
        shape = "SYMMETRIC (Normal)"
    elif mean_val > median_val:
        shape = "RIGHT-SKEWED (Positive skew)"
    else:
        shape = "LEFT-SKEWED (Negative skew)"
    
    print(f"\n  {col}:")
    print(f"    Mean:      {mean_val:.4f}")
    print(f"    Median:    {median_val:.4f}")
    print(f"    Skewness:  {skewness:.4f}")
    print(f"    Shape:     {shape}")

# ==========================================
# [7] VISUALIZATIONS
# ==========================================

print("\n[7] CREATING VISUALIZATIONS")
print("-" * 90)

# Create comparison visualizations
fig = plt.figure(figsize=(16, 10))
fig.suptitle('LAB3: Data Cleaning - Before vs After Comparison', fontsize=16, fontweight='bold')

# Plot Mean vs Median comparison
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(comparison_df))
width = 0.2

ax1.bar(x - width*1.5, comparison_df['Mean_Before'], width, label='Mean (Before)', alpha=0.8)
ax1.bar(x - width*0.5, comparison_df['Median_Before'], width, label='Median (Before)', alpha=0.8)
ax1.bar(x + width*0.5, comparison_df['Mean_After'], width, label='Mean (After)', alpha=0.8)
ax1.bar(x + width*1.5, comparison_df['Median_After'], width, label='Median (After)', alpha=0.8)

ax1.set_xlabel('Variables')
ax1.set_ylabel('Value')
ax1.set_title('Mean vs Median Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['Column'], rotation=45, ha='right')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot data quality improvement
ax2 = plt.subplot(2, 2, 2)
quality_metrics = [
    ('Missing Values', df.isnull().sum().sum(), df_cleaned.isnull().sum().sum()),
    ('Duplicates', df.duplicated().sum(), df_cleaned.duplicated().sum()),
]

metrics_names = [x[0] for x in quality_metrics]
before_vals = [x[1] for x in quality_metrics]
after_vals = [x[2] for x in quality_metrics]

x_pos = np.arange(len(metrics_names))
ax2.bar(x_pos - 0.2, before_vals, 0.4, label='Before', alpha=0.8, color='red')
ax2.bar(x_pos + 0.2, after_vals, 0.4, label='After', alpha=0.8, color='green')

ax2.set_ylabel('Count')
ax2.set_title('Data Quality Improvement', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_names)
ax2.legend()
ax2.grid(alpha=0.3)

# Plot mean vs median difference
ax3 = plt.subplot(2, 2, 3)
ax3.bar(comparison_df['Column'], comparison_df['Mean_Diff'], label='Mean Difference', alpha=0.7)
ax3.bar(comparison_df['Column'], comparison_df['Median_Diff'], label='Median Difference', alpha=0.7)
ax3.set_xlabel('Variables')
ax3.set_ylabel('Absolute Difference')
ax3.set_title('Mean vs Median - Difference After Cleaning', fontweight='bold')
ax3.set_xticklabels(comparison_df['Column'], rotation=45, ha='right')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot memory optimization
ax4 = plt.subplot(2, 2, 4)
memory_before = df.memory_usage(deep=True).sum() / 1024
memory_after = df_cleaned.memory_usage(deep=True).sum() / 1024
memory_saved = memory_before - memory_after

categories = ['Memory Usage', 'Memory Saved']
values = [memory_before, memory_saved]
colors = ['#FF6B6B', '#4ECDC4']

ax4.bar(categories, values, color=colors, alpha=0.7)
ax4.set_ylabel('Memory (KB)')
ax4.set_title('Memory Optimization', fontweight='bold')
ax4.grid(alpha=0.3, axis='y')

for i, v in enumerate(values):
    ax4.text(i, v + 0.5, f'{v:.2f} KB', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('data_cleaning_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comparison chart saved as 'data_cleaning_comparison.png'")
plt.show()

# ==========================================
# [8] SUMMARY REPORT
# ==========================================

print("\n" + "=" * 90)
print(" " * 32 + "SUMMARY REPORT")
print("=" * 90)

print(f"\n✓ BEFORE CLEANING:")
print(f"  Rows:                 {df.shape[0]}")
print(f"  Columns:              {df.shape[1]}")
print(f"  Missing values:       {df.isnull().sum().sum()}")
print(f"  Duplicate rows:       {df.duplicated().sum()}")
print(f"  Memory usage:         {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n✓ AFTER CLEANING:")
print(f"  Rows:                 {df_cleaned.shape[0]}")
print(f"  Columns:              {df_cleaned.shape[1]}")
print(f"  Missing values:       {df_cleaned.isnull().sum().sum()}")
print(f"  Duplicate rows:       {df_cleaned.duplicated().sum()}")
print(f"  Memory usage:         {df_cleaned.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n✓ IMPROVEMENTS:")
print(f"  Rows removed:         {df.shape[0] - df_cleaned.shape[0]}")
print(f"  Missing values fixed: {df.isnull().sum().sum()}")
print(f"  Duplicates removed:   {df.duplicated().sum()}")
print(f"  Memory saved:         {(df.memory_usage(deep=True).sum() - df_cleaned.memory_usage(deep=True).sum()) / 1024:.2f} KB")

# Save cleaned dataset
df_cleaned.to_csv('student_sleep_mental_health_2026_CLEANED.csv', index=False)
print(f"\n✓ Cleaned dataset saved as 'student_sleep_mental_health_2026_CLEANED.csv'")

print("\n" + "=" * 90)
print("✓ LAB3 Data Cleaning Complete!")
print("=" * 90)