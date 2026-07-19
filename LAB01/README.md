# ML-CPE
# LAB1: Dataset Exploration
## Student Sleep & Mental Health Analysis

---

## 📋 Objective

LAB1 aims to **explore and analyze the structure of a dataset** (Dataset Exploration) using Python and Pandas through the following steps:

1. ✓ Load Dataset
2. ✓ Display Shape
3. ✓ Display Data Types
4. ✓ Display Summary Statistics
5. ✓ Display Missing Values
6. ✓ Display Duplicate Records
7. ✓ Display Class Distribution

---

## 📁 File Structure

```
LAB01/
├── LAB01.py                              # Python script for dataset exploration
├── student_sleep_mental_health_2026.csv  # Main dataset file
├── README.md                             # Project documentation (this file)
└── dataset_exploration.png               # Charts generated from script execution
```

---

## 📊 Dataset Information

**File Name:** `student_sleep_mental_health_2026.csv`

**Type:** CSV (Comma Separated Values)

**Size:** 203 KB

**Content:** Data about:
- 🛌 Student Sleep Habits
- 🧠 Student Mental Health
- 📊 Stress Factors
- 📈 Academic Performance Impact

---

## 🚀 How to Use

### Prerequisites

Python 3.7+ and the following libraries must be installed:

```bash
pip install pandas numpy matplotlib seaborn
```

### Steps to Run

1. **Open Terminal/PowerShell** in the LAB01 folder:
   ```powershell
   cd "C:\Users\ketar\Desktop\ML-CPE\LAB01"
   ```

2. **Run Python Script**:
   ```powershell
   python LAB01.py
   ```

3. **Wait for Results**:
   - Text data will display in Console
   - Charts will be saved as `dataset_exploration.png`

---

## 📈 Expected Output

### Console Output Contains:

#### [1] LOAD DATASET
- ✓ Dataset loaded successfully!

#### [2] DISPLAY SHAPE
- Number of Rows (students)
- Number of Columns (features)
- Total Size

#### [3] DISPLAY DATA TYPES
- int64, float64, object, etc.

#### [4] DISPLAY SUMMARY STATISTICS
- Mean (average)
- Std (standard deviation)
- Min/Max (minimum/maximum values)

#### [5] DISPLAY MISSING VALUES
- Number of Missing Values in each Column
- Percentage of Missing Values

#### [6] DISPLAY DUPLICATE RECORDS
- Number of Duplicate Rows
- Percentage of Duplicates

#### [7] DISPLAY CLASS DISTRIBUTION
- Data Distribution
- Percentage of each class

#### [8-9] FIRST/LAST ROWS
- Display first 5 rows
- Display last 5 rows

#### [10] SUMMARY REPORT
- Overall summary report
- Data Quality Score

### Chart Output:
File `dataset_exploration.png` contains 6 charts:
1. 📊 **Dataset Overview** - Bar chart showing dataset size
2. 📉 **Missing Values** - Missing values per column
3. 🥧 **Data Types Distribution** - Pie chart of data types
4-6. 📈 **Numeric Columns Distribution** - Histograms of numerical data distribution

---

## 📝 Code Overview

### Libraries Used:
```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical computing
import matplotlib.pyplot     # Data visualization
import seaborn as sns       # Statistical data visualization
```

### Main Functions:
- `pd.read_csv()` - Read CSV file
- `df.shape` - View dataset size
- `df.dtypes` - View data types
- `df.describe()` - Basic statistics
- `df.isnull()` - Find missing values
- `df.duplicated()` - Find duplicate rows
- `df.value_counts()` - Count frequencies
- `plt.savefig()` - Save charts

---

## 🎯 Expected Results

- ✅ Python script runs without errors
- ✅ Console displays all 7 sections of dataset information
- ✅ Chart file `dataset_exploration.png` is created
- ✅ Data Quality Score is displayed

---

## 🔍 Learning Outcomes

After completing LAB1, you will learn:

1. **Data Loading** - How to read CSV files with Pandas
2. **Data Inspection** - How to check dataset structure
3. **Data Quality** - How to find missing values and duplicates
4. **Statistical Analysis** - How to calculate basic statistics
5. **Data Visualization** - How to create charts with Matplotlib
6. **Class Distribution** - How to analyze category distribution

---

## 🚨 Troubleshooting

### ❌ Error: FileNotFoundError
**Problem:** Dataset file not found
**Solution:** Check that `student_sleep_mental_health_2026.csv` is in the same folder as the script

### ❌ Error: ModuleNotFoundError
**Problem:** Library not installed
**Solution:** 
```bash
pip install pandas numpy matplotlib seaborn
```

### ❌ Error: CSV encoding issues
**Problem:** Characters display incorrectly
**Solution:** Change encoding to:
```python
df = pd.read_csv('student_sleep_mental_health_2026.csv', encoding='utf-8')
```

---

## 📞 Support

If you have questions or encounter issues:
- Check the error message carefully
- Review the console output in detail
- Try running the script again after installing all libraries

---

## 📚 References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---


**✓ LAB1 Dataset Exploration - Complete**
