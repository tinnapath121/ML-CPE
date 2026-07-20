# ML-CPE
# LAB3: Data Cleaning
## Student Sleep & Mental Health Analysis

---

## 📋 Objective

LAB3 aims to **clean and prepare data for analysis** by:

1. ✓ Missing Value Handling
2. ✓ Duplicate Removal
3. ✓ Incorrect Data Correction
4. ✓ Data Type Conversion
5. ✓ Compare Mean vs Median

---

## 📁 File Structure

```
LAB03/
├── LAB03.py                                      # Python script for data cleaning
├── student_sleep_mental_health_2026.csv          # Original dataset
├── student_sleep_mental_health_2026_CLEANED.csv  # Cleaned dataset (output)
├── README.md                                     # Project documentation (this file)
└── data_cleaning_comparison.png                  # Before/After comparison charts
```

---

## 🔧 Data Cleaning Tasks

### 1. **Missing Value Handling**

**Strategy:**
- **Numeric columns:** Fill with MEDIAN (robust to outliers)
- **Categorical columns:** Fill with MODE (most frequent value)

**Why these methods?**
- Median: Not affected by extreme values
- Mode: Preserves categorical distribution

**Methods in script:**
```python
df[col].fillna(df[col].median())  # Numeric
df[col].fillna(df[col].mode()[0])  # Categorical
```

---

### 2. **Duplicate Removal**

**Methods:**
- Detect rows with identical values
- Remove keeping first occurrence (`keep='first'`)
- Alternative: Keep last occurrence (`keep='last'`)

**Benefits:**
- Ensures each record is unique
- Prevents bias in analysis
- Reduces dataset size

---

### 3. **Incorrect Data Correction**

**Checks performed:**
- Negative values where they shouldn't exist
- Outlier detection (IQR method)
- Data consistency validation
- Range verification

**Outlier Detection (IQR Method):**
```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Where:
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
```

**Correction methods:**
- Convert to absolute values for negative numbers
- Flag or cap outliers at bounds
- Verify data consistency

---

### 4. **Data Type Conversion**

**Optimizations:**
- Convert object columns with ≤50 unique values to `category`
- Convert float columns with all whole numbers to `int64`
- Reduces memory usage
- Improves performance

**Benefits:**
- **Memory savings:** Category < object
- **Speed:** Smaller data types process faster
- **Storage:** Less disk space required

---

### 5. **Compare Mean vs Median**

**What they mean:**
- **Mean:** Average of all values (affected by outliers)
- **Median:** Middle value (robust to outliers)

**Interpretation:**
- **Mean ≈ Median:** Symmetric distribution (normal)
- **Mean > Median:** Right-skewed (positive skew)
- **Mean < Median:** Left-skewed (negative skew)

**When to use:**
- **Mean:** Normal distribution, no major outliers
- **Median:** Skewed data, presence of outliers

**Skewness values:**
- Skewness = 0: Perfect symmetry
- Skewness > 0: Right-skewed
- Skewness < 0: Left-skewed

---

## 🚀 How to Use

### Prerequisites

Python 3.7+ and required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

### Steps to Run

1. **Create LAB03 Folder**:
   ```
   C:\Users\ketar\Desktop\ML-CPE\LAB03\
   ```

2. **Copy Files**:
   - Copy `LAB03.py` to the folder
   - Copy `student_sleep_mental_health_2026.csv` to the folder

3. **Open Terminal/PowerShell**:
   ```powershell
   cd "C:\Users\ketar\Desktop\ML-CPE\LAB03"
   ```

4. **Run Python Script**:
   ```powershell
   python LAB03.py
   ```

5. **View Results**:
   - Comparison chart displays on screen
   - Cleaned dataset saved as CSV
   - All statistics printed to console

---

## 📊 Expected Output

### Console Output Includes:

#### [1] LOAD DATASET
- Original dataset shape
- Memory usage

#### [2] MISSING VALUE HANDLING
- Number of missing values before/after
- Filling method for each column
- Verification of fixes

#### [3] DUPLICATE REMOVAL
- Number of duplicates found
- Preview of duplicate rows
- Rows removed count

#### [4] INCORRECT DATA CORRECTION
- Negative values detected and fixed
- Outliers found (IQR method)
- Data consistency checks

#### [5] DATA TYPE CONVERSION
- Original vs new data types
- Memory usage comparison
- Percentage saved

#### [6] MEAN vs MEDIAN COMPARISON
- Mean before/after for each column
- Median before/after for each column
- Distribution shape analysis
- Skewness values

#### [7] VISUALIZATIONS
- Creates 4 comparison charts

#### [8] SUMMARY REPORT
- Before/after statistics
- Total improvements made
- Cleaned dataset saved

### Output Files:

1. **data_cleaning_comparison.png**
   - 4 charts showing:
     - Mean vs Median comparison
     - Data quality improvement
     - Mean/Median differences
     - Memory optimization

2. **student_sleep_mental_health_2026_CLEANED.csv**
   - Cleaned dataset ready for analysis
   - All issues fixed
   - Optimized data types

---

## 📈 Mean vs Median Illustration

### Example 1: Normal Distribution
```
Distribution:  ▁▂▃▄▅▆▅▄▃▂▁
Mean = Median  (same position)
Interpretation: SYMMETRIC
```

### Example 2: Right-Skewed Distribution
```
Distribution:  ▂▃▄▅▆▇▅▃▂▁___
Mean > Median  (mean pulled right)
Interpretation: Tail on right side
Example: Income, Reaction time
```

### Example 3: Left-Skewed Distribution
```
Distribution:  ___▁▂▃▄▅▆▇▆▅▃▂
Mean < Median  (mean pulled left)
Interpretation: Tail on left side
Example: Exam scores (high concentration)
```

---

## 🔍 Detailed Explanations

### Missing Value Handling

**Why fill missing values?**
- Some algorithms can't handle NaN
- Provides complete data for analysis
- Prevents loss of information

**Methods:**
1. **Deletion:** Remove rows/columns (loses data)
2. **Imputation (filling):**
   - Mean/Median: Use central tendency
   - Forward/Backward fill: Use neighboring values
   - Interpolation: Estimate from pattern
   - Machine Learning: Predict using other features

**This lab uses:** Median (numeric) + Mode (categorical)

---

### Duplicate Removal

**Why remove duplicates?**
- Prevents bias in statistical analysis
- Each record should be unique
- Reduces data redundancy
- Improves accuracy of results

**When to keep duplicates?**
- If they represent real-world occurrences
- In time-series data (same value at different times)

---

### Outlier Detection

**IQR Method (Interquartile Range):**
```
Formula: Outlier if x < Q1 - 1.5×IQR  OR  x > Q3 + 1.5×IQR

Example:
Q1 = 20, Q3 = 40
IQR = 40 - 20 = 20
Lower Bound = 20 - 30 = -10
Upper Bound = 40 + 30 = 70

Values < -10 or > 70 = Outliers
```

**Handling outliers:**
- Cap at boundaries
- Remove completely
- Use separate analysis
- Investigate (might be important)

---

### Data Type Optimization

**Memory Comparison:**

| Type | Size | Example |
|------|------|---------|
| object | ~50 bytes | "Category A" |
| category | ~8 bytes | "Category A" |
| float64 | 8 bytes | 3.14 |
| int64 | 8 bytes | 42 |
| int32 | 4 bytes | 42 |

**This lab converts:**
- High-cardinality object → category (if ≤50 unique values)
- float with no decimals → int64

---

## 📝 Code Overview

### Main Functions Used:

```python
# Missing values
df.isnull()
df.fillna()

# Duplicates
df.duplicated()
df.drop_duplicates()

# Statistics
df.mean()
df.median()
df.quantile()
df.skew()

# Outliers (IQR)
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

# Data types
df.astype('category')
df.dtypes
df.memory_usage()
```

---

## 🎯 Learning Outcomes

After completing LAB3, you will understand:

1. **Missing Data:** How to identify and handle it
2. **Data Quality:** Importance of clean data
3. **Outliers:** Detection and handling methods
4. **Data Types:** How to optimize for performance
5. **Mean vs Median:** When to use each
6. **Distributions:** How to interpret data shape
7. **Skewness:** Understanding data asymmetry
8. **Data Preparation:** Essential for good analysis

---

## 🚨 Troubleshooting

### ❌ Error: FileNotFoundError
**Problem:** CSV file not found
**Solution:** Ensure `student_sleep_mental_health_2026.csv` is in the same folder

### ❌ Error: ModuleNotFoundError
**Problem:** Libraries not installed
**Solution:**
```bash
pip install pandas numpy matplotlib seaborn
```

### ⚠ Warning: No missing values found
**Problem:** Dataset is already clean
**Solution:** This is good! Lab still demonstrates the process

### ⚠ Data doesn't change much
**Problem:** Original dataset is already fairly clean
**Solution:** Lab demonstrates best practices anyway

---

## 💡 Best Practices

✅ **DO:**
- Check data before and after cleaning
- Document all changes made
- Save original data before processing
- Verify results make sense
- Use median for skewed data
- Remove true duplicates

❌ **DON'T:**
- Delete too much data without reason
- Ignore outliers without investigation
- Use mean for highly skewed data
- Change data type without verification
- Assume missing data is random
- Overwrite original dataset

---

## 📚 References

- [Pandas fillna Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)
- [Pandas drop_duplicates Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)
- [IQR Outlier Detection](https://en.wikipedia.org/wiki/Interquartile_range)
- [Data Cleaning Best Practices](https://www.kdnuggets.com/2017/12/data-cleaning-100-hours.html)

---

---

**✓ LAB3 Data Cleaning - Complete**