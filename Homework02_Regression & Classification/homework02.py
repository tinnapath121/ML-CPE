import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -----------------------
# Shared preprocessing
# -----------------------
df_raw = pd.read_csv(r'C:\Users\ketar\ML-CPE\Homework01_Data preprocessing\Titanic-Dataset.csv') #Titanic-Dataset.csv pushไปปลอดภัยไหมอาจารบอกด้วยพอดีคอมมันหาไฟล์ไม่เจอ


# Basic cleaning / feature engineering used by both tasks
df = df_raw.copy()
df['Age'] = pd.to_numeric(df.get('Age'), errors='coerce')
df['Fare'] = pd.to_numeric(df.get('Fare'), errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['FamilySize'] = df.get('SibSp', 0).fillna(0).astype(int) + df.get('Parch', 0).fillna(0).astype(int) + 1
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# Title extraction (optional)
df['Title'] = df['Name'].astype(str).str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
df['Title'] = df['Title'].where(df['Title'].map(df['Title'].value_counts()) >= 10, 'Rare')
# One-hot encode selected categorical columns
to_dummy = [c for c in ['Pclass','Embarked','Title'] if c in df.columns]
if to_dummy:
    df = pd.get_dummies(df, columns=to_dummy, drop_first=True)

# Helper for regression metrics
def reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# -----------------------
# LAB 1: Regression (Age prediction)
# -----------------------
print("\n=== LAB 1: Regression ===")

# Use rows that have Age (we filled Age above, but keep original approach)
df_reg = df.copy()  # Age is filled; if you prefer dropna, uncomment next line
# df_reg = df_reg.dropna(subset=['Age'])

reg_features = ['Fare', 'FamilySize', 'HasCabin', 'Sex']
reg_features += [c for c in df_reg.columns if c.startswith(('Pclass_','Embarked_','Title_'))]
X_reg = df_reg[reg_features].fillna(0)
y_reg = df_reg['Age']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Simple Linear Regression (Fare -> Age)
print("\n-- Simple Linear Regression (Fare -> Age) --")
if 'Fare' in X_reg.columns:
    lr_simple = LinearRegression()
    lr_simple.fit(Xr_train[['Fare']], yr_train)
    y_pred_simple = lr_simple.predict(Xr_test[['Fare']])
    metrics_simple = reg_metrics(yr_test, y_pred_simple)
    print("SimpleLR(Fare) metrics:", metrics_simple)
else:
    lr_simple = None
    y_pred_simple = None
    metrics_simple = None
    print("Feature 'Fare' not available for simple regression.")

# Multiple Linear Regression (multiple features -> Age)
print("\n-- Multiple Linear Regression --")
lr_multi = LinearRegression()
lr_multi.fit(Xr_train, yr_train)
y_pred_multi = lr_multi.predict(Xr_test)
metrics_multi = reg_metrics(yr_test, y_pred_multi)
print("MultipleLR metrics:", metrics_multi)

# Plot predicted vs actual for multiple LR
plt.figure(figsize=(6,5))
plt.scatter(yr_test, y_pred_multi, alpha=0.6)
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--')
plt.xlabel('Actual Age'); plt.ylabel('Predicted Age'); plt.title('Multiple LR: Predicted vs Actual Age')
plt.tight_layout(); plt.show()

# -----------------------
# LAB 2: Classification (Gender prediction)
# -----------------------
print("\n=== LAB 2: Classification ===")

clf_features = ['Age', 'Fare', 'FamilySize', 'HasCabin']
clf_features += [c for c in df.columns if c.startswith(('Pclass_','Embarked_','Title_'))]
X_clf = df[clf_features].fillna(0)
y_clf = df['Sex'].fillna(0).astype(int)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Decision boundary demo (Age vs Fare) if both features exist
if 'Age' in X_clf.columns and 'Fare' in X_clf.columns:
    print("\n-- Decision boundary demo (Age vs Fare) --")
    demo_X = df[['Age','Fare']].fillna(0)
    demo_y = df['Sex'].fillna(0).astype(int)
    dtr, dte, ytr, yte = train_test_split(demo_X, demo_y, test_size=0.2, random_state=42)
    dclf = LogisticRegression(max_iter=1000).fit(dtr, ytr)
    xx, yy = np.meshgrid(
        np.linspace(demo_X['Age'].min(), demo_X['Age'].max(), 200),
        np.linspace(demo_X['Fare'].min(), demo_X['Fare'].max(), 200)
    )
    Z = dclf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(7,5))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.scatter(dte['Age'], dte['Fare'], c=yte, cmap='coolwarm', edgecolor='k', s=20)
    plt.xlabel('Age'); plt.ylabel('Fare'); plt.title('Decision Boundary (Age vs Fare)')
    plt.tight_layout(); plt.show()

# Logistic Regression for Gender prediction
print("\n-- Logistic Regression (Gender Prediction) --")
clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

print("\nClassification report:")
print(classification_report(yc_test, yc_pred, zero_division=0))

cm = confusion_matrix(yc_test, yc_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Gender Prediction)')
plt.tight_layout(); plt.show()

# Compute classification metrics
acc = accuracy_score(yc_test, yc_pred)
prec = precision_score(yc_test, yc_pred, zero_division=0)
rec = recall_score(yc_test, yc_pred, zero_division=0)
f1 = f1_score(yc_test, yc_pred, zero_division=0)
print("Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(acc, prec, rec, f1))

# -----------------------
# LAB 3: Model Comparison
# -----------------------
print("\n=== LAB 3: Model Comparison ===")

# Simple LR train metrics (if available)
if lr_simple is not None:
    y_pred_simple_train = lr_simple.predict(Xr_train[['Fare']])
    metrics_simple_train = reg_metrics(yr_train, y_pred_simple_train)
    print("\nSimpleLR (train):", metrics_simple_train)
    print("SimpleLR (test) :", metrics_simple)
else:
    metrics_simple_train = None

# Multiple LR train metrics
y_pred_multi_train = lr_multi.predict(Xr_train)
metrics_multi_train = reg_metrics(yr_train, y_pred_multi_train)
print("\nMultipleLR (train):", metrics_multi_train)
print("MultipleLR (test) :", metrics_multi)

# Classification train vs test accuracy
yc_pred_train = clf.predict(Xc_train)
acc_train = accuracy_score(yc_train, yc_pred_train)
print("\nLogistic Regression (Gender) train acc: {:.3f}, test acc: {:.3f}".format(acc_train, acc))

# Prepare summary and save to CSV
summary_rows = []

if metrics_simple is not None:
    summary_rows.append({
        'model': 'SimpleLR(Fare)',
        'task': 'regression',
        'train_MAE': metrics_simple_train['MAE'] if metrics_simple_train else None,
        'train_RMSE': metrics_simple_train['RMSE'] if metrics_simple_train else None,
        'train_R2': metrics_simple_train['R2'] if metrics_simple_train else None,
        'test_MAE': metrics_simple['MAE'],
        'test_RMSE': metrics_simple['RMSE'],
        'test_R2': metrics_simple['R2']
    })

summary_rows.append({
    'model': 'MultipleLR',
    'task': 'regression',
    'train_MAE': metrics_multi_train['MAE'],
    'train_RMSE': metrics_multi_train['RMSE'],
    'train_R2': metrics_multi_train['R2'],
    'test_MAE': metrics_multi['MAE'],
    'test_RMSE': metrics_multi['RMSE'],
    'test_R2': metrics_multi['R2']
})

summary_rows.append({
    'model': 'LogisticRegression',
    'task': 'classification',
    'train_accuracy': acc_train,
    'test_accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1': f1
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('model_comparison_summary.csv', index=False)
print("\nSaved model comparison summary to model_comparison_summary.csv")
print("\nSummary:\n", summary_df)
