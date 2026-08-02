### Titanic Feature Engineering and Encoding

A compact project that performs **feature engineering** and **encoding** on the Titanic dataset (Part 4 of the lab). The repository contains a short script that creates new features, applies **label encoding** and **one‑hot encoding**, and writes a ready‑to‑use CSV for machine learning experiments.

---

### Files
| **Filename** | **Description** |
|---|---|
| `Titanic-Dataset.csv` | Original dataset (place in the same folder as the script). |
| `encode_and_save.py` | Script that creates features, encodes categorical columns, and saves the processed CSV. |
| `titanic_preprocessed_encoded.csv` | Output CSV produced by the script (feature engineered + encoded). |

---

### Quick start
1. Put `Titanic-Dataset.csv` in the same folder as `encode_and_save.py`.  
2. Install the required library:
```bash
python -m pip install --user pandas
```
3. Run the script:
```bash
python encode_and_save.py
```
4. Open `titanic_preprocessed_encoded.csv` — it is ready for model training.

---

### What the script does (high level)
- **Fill missing values** (simple strategy): `Age` and `Fare` → median; `Embarked` → mode.  
- **Create new features**: `FamilySize`, `HasCabin`, `Title` (extracted from `Name`).  
- **Label encode** the binary column `Sex` (`male` → `0`, `female` → `1`).  
- **One‑hot encode** nominal columns: `Embarked`, `Pclass`, `Title` (using `drop_first=True`).  
- **Drop** non‑essential columns before saving: `PassengerId`, `Name`, `Ticket`, `Cabin`.  
- **Save** the processed dataset as `titanic_preprocessed_encoded.csv`.

---

### Code walkthrough — which lines run what and why
Below are the key code blocks from `encode_and_save.py` with plain‑English explanations you can paste into the README or use as inline comments.

- **Load the CSV**
```python
df = pd.read_csv(input_csv)
```
**Purpose:** Read the original dataset into a pandas DataFrame so we can inspect and transform it.

- **Ensure numeric columns and fill missing values**
```python
df['Age'] = pd.to_numeric(df.get('Age'), errors='coerce').fillna(df['Age'].median())
df['Fare'] = pd.to_numeric(df.get('Fare'), errors='coerce').fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
```
**Purpose:** Convert `Age` and `Fare` to numeric (coerce invalid entries to `NaN`) and fill missing values with the median; fill `Embarked` with the most common port. This prevents errors later and preserves central tendency.

- **Create `FamilySize`**
```python
df['FamilySize'] = df.get('SibSp', 0).fillna(0).astype(int) + df.get('Parch', 0).fillna(0).astype(int) + 1
```
**Purpose:** Combine sibling/spouse and parent/child counts into a single feature that often correlates with survival behavior.

- **Create `HasCabin`**
```python
df['HasCabin'] = df['Cabin'].notnull().astype(int) if 'Cabin' in df.columns else 0
```
**Purpose:** A binary indicator that captures whether a passenger has cabin information; presence of a cabin can proxy for higher class or different boarding conditions.

- **Extract and normalize `Title`**
```python
df['Title'] = df['Name'].astype(str).str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
vc = df['Title'].value_counts()
rare = vc[vc < 10].index
df['Title'] = df['Title'].replace(list(rare), 'Rare')
```
**Purpose:** Titles (Mr, Mrs, Miss, etc.) capture social status and age hints. Normalizing variants and grouping rare titles reduces categorical cardinality and noise.

- **Label encode `Sex`**
```python
df['Sex'] = df['Sex'].map({'male':0, 'female':1}).fillna(0).astype(int)
```
**Purpose:** Convert the binary `Sex` column to numeric values so models can use it directly.

- **One‑hot encode nominal columns**
```python
to_dummy = [c for c in ['Embarked','Pclass','Title'] if c in df.columns]
if to_dummy:
    df = pd.get_dummies(df, columns=to_dummy, drop_first=True)
```
**Purpose:** Convert multi‑category columns into dummy variables. `drop_first=True` removes one column per group to avoid perfect multicollinearity for linear models.

- **Drop unneeded columns and save**
```python
for c in ['PassengerId','Name','Ticket','Cabin']:
    if c in df.columns:
        df = df.drop(columns=c)
df.to_csv(out_encoded, index=False)
```
**Purpose:** Remove identifiers and raw text fields that are not used by the model and write the final encoded dataset to disk.

---

### Design choices and rationale
- **Median for numeric missing values:** robust to outliers and simple to implement for lab work.  
- **Mode for `Embarked`:** preserves the most common boarding port rather than dropping rows.  
- **Title grouping:** reduces noise from many rare titles while preserving useful social cues.  
- **One‑hot for nominal features:** prevents models from interpreting categories as ordinal.  
- **Drop raw text columns:** `Name`, `Ticket`, and `Cabin` are removed to keep the CSV compact; keep them if you plan to engineer more features.

---

### Next steps (suggested)
- Train a baseline model (Logistic Regression or Random Forest) using the processed CSV.  
- Compare performance with and without `Title`, `HasCabin`, and `FamilySize`.  
- Try alternative missing‑value strategies (group median by `Pclass`/`Sex`, or model‑based imputation).  
- Add scaling if you use distance‑based models (KNN, SVM).

---

### License
This repository uses the **MIT License**. Adjust as required by your instructor or institution.

---

If you want, I can generate a ready‑to‑paste `README.md` file with this exact content formatted for your repo. Would you like that?