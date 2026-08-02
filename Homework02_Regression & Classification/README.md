### Titanic LABs Combined README

#### Project Summary
This repository contains a compact script that runs **LAB 1 Regression**, **LAB 2 Classification**, and **LAB 3 Model Comparison** on the Titanic dataset. The script performs minimal preprocessing, trains simple and multiple linear regression models to predict **Age**, trains a logistic regression model to predict **Sex**, and produces a short model comparison summary.

---

#### Files
| **Filename** | **Purpose** |
|---|---|
| `Titanic-Dataset.csv` | Original dataset; place in the same folder as the script |
| `combined_labs.py` | Single script that runs LAB1, LAB2, LAB3 and saves summary |
| `model_comparison_summary.csv` | Generated summary of model metrics (created when script runs) |

---

#### Quick Start
1. Install dependencies:
```bash
python -m pip install --user pandas numpy scikit-learn seaborn matplotlib
```
2. Put `Titanic-Dataset.csv` in the same folder as `combined_labs.py`.  
3. Run the script:
```bash
python combined_labs.py
```
4. Open `model_comparison_summary.csv` for a compact metrics table and review the generated plots.

---

#### What the Script Runs and Why
**Shared preprocessing**  
- Fill numeric missing values for **Age** and **Fare** with median.  
- Create **FamilySize** = `SibSp + Parch + 1`.  
- Create **HasCabin** indicator.  
- Extract **Title** from `Name` and group rare titles.  
- Label encode **Sex** and one‑hot encode `Pclass`, `Embarked`, `Title`.

**LAB 1 Regression**  
- **Simple Linear Regression**: fit `Fare` → `Age`. Purpose: show single feature regression and baseline performance.  
- **Multiple Linear Regression**: fit multiple features → `Age`. Purpose: show multivariate regression and improved fit.  
- **Outputs**: MAE, MSE, RMSE, R² for test set and a Predicted vs Actual scatter plot.

**LAB 2 Classification**  
- **Logistic Regression** to predict `Sex` using features such as `Age`, `Fare`, `FamilySize`, `HasCabin`, and dummies.  
- **Decision Boundary Demo**: 2D visualization using `Age` vs `Fare` to illustrate the classifier boundary.  
- **Outputs**: classification report, confusion matrix plot, accuracy/precision/recall/F1.

**LAB 3 Model Comparison**  
- Compare **Simple vs Multiple Linear Regression** on train and test metrics to reveal underfitting/overfitting.  
- Compare **Regression vs Classification** by summarizing RMSE and accuracy.  
- **Outputs**: `model_comparison_summary.csv` with train/test metrics for each model.

---

#### Outputs Produced
- **Plots**: Predicted vs Actual Age, Decision Boundary, Confusion Matrix (displayed when script runs).  
- **CSV**: `model_comparison_summary.csv` containing per‑model metrics for quick reporting.

---

#### Notes and Recommendations
- The script uses **simple, reproducible defaults** suitable for lab submission. For improved performance consider: feature scaling, cross validation, imputation by group, feature selection, or more advanced models.  
- If your instructor requires different targets or stricter preprocessing (for example dropping rows with missing Age), adjust the preprocessing block accordingly.  
- Keep `Titanic-Dataset.csv` private only if your institution requires it; the Titanic dataset is commonly public and safe to include in coursework unless told otherwise.

---

#### License
**MIT License** — change if your course requires a different license.

---

You can paste this README directly into your repository. If you want a shorter version, a version in Thai, or a README that includes example output screenshots, I can generate that next.