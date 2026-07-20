# ML-CPE
# LAB2: Data Visualization
## Student Sleep & Mental Health Analysis

---

## 📋 Objective

LAB2 aims to **create visualizations to explore and understand data patterns** using Python, Pandas, Matplotlib, and Seaborn. The lab focuses on:

1. ✓ Histogram - Distribution of numeric variables
2. ✓ Correlation Heatmap - Relationships between variables

---

## 📁 File Structure

```
LAB02/
├── LAB02.py                              # Python script for data visualization
├── student_sleep_mental_health_2026.csv  # Dataset (same as LAB1)
├── README.md                             # Project documentation (this file)
├── histograms.png                        # Histograms of all numeric columns
├── correlation_heatmap.png               # Correlation matrix heatmap
└── visualization_combined.png            # Combined summary visualization
```

---

## 🎨 Visualizations Created

### 1. **Histograms** (histograms.png)
Shows the distribution of each numeric variable:
- **What it shows:** Frequency distribution of values
- **How to read:** 
  - X-axis: Value ranges
  - Y-axis: Frequency (count)
  - Red dashed line: Mean value
  - Green dashed line: Median value
- **Use:** Identify data shape, outliers, and skewness

### 2. **Correlation Heatmap** (correlation_heatmap.png)
Shows relationships between numeric variables:
- **What it shows:** Correlation coefficients between all numeric variables
- **How to read:**
  - Values range from -1 to 1
  - Red color: Strong positive correlation
  - Blue color: Strong negative correlation
  - White color: No correlation (0)
  - Number in cell: Exact correlation value
- **Use:** Identify which variables are related

### 3. **Combined Visualization** (visualization_combined.png)
Summary view combining histograms and heatmap

---

## 🚀 How to Use

### Prerequisites

Python 3.7+ and the following libraries must be installed:

```bash
pip install pandas numpy matplotlib seaborn
```

### Steps to Run

1. **Create LAB02 Folder**:
   ```
   C:\Users\ketar\Desktop\ML-CPE\LAB02\
   ```

2. **Copy Files**:
   - Copy `LAB02.py` to the folder
   - Copy `student_sleep_mental_health_2026.csv` to the folder (same as LAB1)

3. **Open Terminal/PowerShell** in the LAB02 folder:
   ```powershell
   cd "C:\Users\ketar\Desktop\ML-CPE\LAB02"
   ```

4. **Run Python Script**:
   ```powershell
   python LAB02.py
   ```

5. **View Results**:
   - Charts will display in windows
   - PNG files will be saved to the folder
   - Statistics will print to console

---

## 📊 Expected Output

### Console Output Contains:

#### [1] LOAD DATASET
- ✓ Dataset loaded successfully!
- Displays dataset shape

#### [2] DATA PREPROCESSING
- Lists all numeric columns
- Lists all categorical columns

#### [3] HISTOGRAMS
- Creates histogram for each numeric variable
- Shows mean and median lines
- Displays statistics:
  - Mean, Median, Std Dev
  - Min, Max
  - Skewness, Kurtosis

#### [4] CORRELATION HEATMAP
- Creates correlation matrix
- Displays all correlation values
- Shows strongest correlations (top 10)

#### [5] COMBINED VISUALIZATION
- Summary of top histograms
- Small version of heatmap

#### [6] SUMMARY REPORT
- Total counts of columns
- List of generated files

### Output Files:

1. **histograms.png**
   - Multiple histograms (one per numeric column)
   - Each with mean and median lines
   - Grid for readability

2. **correlation_heatmap.png**
   - Full correlation matrix
   - Color-coded from -1 (blue) to +1 (red)
   - Annotated with values

3. **visualization_combined.png**
   - 4-panel visualization
   - 3 sample histograms
   - 1 heatmap

---

## 📝 Code Overview

### Libraries Used:
```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical computing
import matplotlib.pyplot     # Data visualization
import seaborn as sns       # Statistical visualization
```

### Key Functions:

**For Histograms:**
```python
plt.hist()              # Create histogram
ax.axvline()            # Add mean/median lines
```

**For Heatmap:**
```python
df.corr()               # Calculate correlation matrix
sns.heatmap()           # Create correlation visualization
```

**Data Preprocessing:**
```python
df.select_dtypes()      # Filter numeric/categorical columns
```

---

## 🔍 Interpretation Guide

### Understanding Histograms

**Normal Distribution:**
- Bell-shaped curve
- Mean ≈ Median
- Skewness close to 0

**Right-Skewed (Positively Skewed):**
- Tail on the right
- Mean > Median
- Positive skewness value

**Left-Skewed (Negatively Skewed):**
- Tail on the left
- Mean < Median
- Negative skewness value

### Understanding Correlation Values

| Correlation | Relationship |
|-------------|--------------|
| 1.0 | Perfect positive correlation |
| 0.7 to 0.9 | Strong positive correlation |
| 0.3 to 0.7 | Moderate positive correlation |
| 0.0 to 0.3 | Weak positive correlation |
| 0.0 | No correlation |
| -0.3 to 0.0 | Weak negative correlation |
| -0.7 to -0.3 | Moderate negative correlation |
| -0.9 to -0.7 | Strong negative correlation |
| -1.0 | Perfect negative correlation |

---

## 🎯 Learning Outcomes

After completing LAB2, you will learn:

1. **Histogram Creation** - How to create and interpret histograms
2. **Distribution Analysis** - How to understand data distribution shapes
3. **Correlation Analysis** - How to calculate and interpret correlations
4. **Heatmap Visualization** - How to create and read correlation heatmaps
5. **Statistical Measures** - Understanding mean, median, skewness, kurtosis
6. **Data Relationships** - How to identify relationships between variables
7. **Matplotlib & Seaborn** - Advanced visualization techniques

---

## 🚨 Troubleshooting

### ❌ Error: FileNotFoundError
**Problem:** Dataset file not found
**Solution:** Check that `student_sleep_mental_health_2026.csv` is in the same folder as LAB02.py

### ❌ Error: ModuleNotFoundError
**Problem:** Seaborn or Matplotlib not installed
**Solution:** 
```bash
pip install pandas numpy matplotlib seaborn
```

### ❌ Error: Charts don't display
**Problem:** Matplotlib backend issue
**Solution:** Add this at the beginning:
```python
import matplotlib
matplotlib.use('TkAgg')
```

### ⚠ Warning: Fewer than 2 numeric columns
**Problem:** Not enough columns for correlation
**Solution:** Ensure dataset has numeric columns for analysis

---

## 💡 Tips for Interpretation

### Histograms:
- **Peak location** shows most common values
- **Width** shows range of variation
- **Shape** shows distribution type
- **Outliers** appear as isolated bars

### Correlation Heatmap:
- **Diagonal** always shows 1.0 (perfect correlation with itself)
- **Symmetric** - same values above and below diagonal
- **Dark red** = strong positive (increase together)
- **Dark blue** = strong negative (increase/decrease together)
- **Light colors** = weak or no relationship

---

## 📚 References

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

---

## 📊 Example Interpretation

If you see:
- **Strong positive correlation (0.8)** between Sleep Hours and Mental Health Score
  - Interpretation: More sleep is associated with better mental health

- **Right-skewed histogram** for Stress Levels
  - Interpretation: Most students have low stress, few have high stress

- **Weak correlation (0.1)** between two variables
  - Interpretation: These variables are independent

---



**✓ LAB2 Data Visualization - Complete**