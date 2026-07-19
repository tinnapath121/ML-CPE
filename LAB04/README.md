# ML-CPE
# LAB04 - Feature Engineering (Label Encoding & One-Hot Encoding)

## Dataset
`student_sleep_mental_health_2026.csv` — contains student lifestyle and mental health data (sleep, screen time, stress, anxiety, GPA, etc.), including categorical columns: `gender`, `education_level`, `uses_sleep_app`, `feels_burned_out`.

## Requirements
- Python 3.x
- pandas
- scikit-learn

Install dependencies:
```bash
pip install pandas scikit-learn
```

## Workflow

1. **Load the dataset**
   Read `student_sleep_mental_health_2026.csv` into a pandas DataFrame and inspect its structure (`.head()`, `.dtypes`) to identify categorical vs. numerical columns.

2. **Identify categorical features**
   - `education_level` → ordinal (has a natural order: High School < Undergraduate < Graduate)
   - `gender` → nominal (no natural order)
   - `uses_sleep_app`, `feels_burned_out` → boolean (True/False)

3. **Apply Label Encoding**
   Used for ordinal / boolean features, since converting them to integers preserves meaningful order or a simple binary meaning:
   - `education_level` → mapped manually to 0, 1, 2 (High School → Undergraduate → Graduate), and also encoded using `sklearn.preprocessing.LabelEncoder` for comparison.
   - `uses_sleep_app`, `feels_burned_out` → converted from True/False to 1/0.

4. **Apply One-Hot Encoding**
   Used for nominal features with no inherent order, to avoid implying a false ranking between categories:
   - `gender` → expanded into separate binary columns (`gender_Female`, `gender_Male`, `gender_Non-binary`, `gender_Prefer not to say`) using `pd.get_dummies()`.

5. **Build the final encoded dataset**
   Combine the numerical columns with the encoded categorical columns into a single DataFrame ready for machine learning models.

6. **Save the result**
   Export the final DataFrame to `student_sleep_mental_health_2026_encoded.csv`.

## How to Run
```bash
python LAB04.py
```

## Output
- `student_sleep_mental_health_2026_encoded.csv` — the fully encoded dataset, with all categorical variables converted to numeric form, ready for model training.