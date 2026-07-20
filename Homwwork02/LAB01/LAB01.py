"""
ML-LAB-01: Linear Regression - Age Prediction
Dataset: student_sleep_mental_health_2026.csv
สร้างโมเดล Simple & Multiple Linear Regression เพื่อทำนายอายุ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & EXPLORE DATA
# ============================================================
print("=" * 60)
print("LAB 1: Linear Regression - Age Prediction")
print("=" * 60)

# Load dataset
df = pd.read_csv('student_sleep_mental_health_2026.csv')

print("\n📊 Dataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# 2. DATA PREPARATION
# ============================================================
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Handle categorical variables (gender, education_level, etc.)
df_processed = df.copy()

# Convert boolean columns to int
df_processed['uses_sleep_app'] = df_processed['uses_sleep_app'].astype(int)
df_processed['feels_burned_out'] = df_processed['feels_burned_out'].astype(int)

# One-hot encoding for gender
gender_dummies = pd.get_dummies(df_processed['gender'], prefix='gender', drop_first=True)
df_processed = pd.concat([df_processed, gender_dummies], axis=1)

# One-hot encoding for education_level
education_dummies = pd.get_dummies(df_processed['education_level'], prefix='education', drop_first=True)
df_processed = pd.concat([df_processed, education_dummies], axis=1)

# Drop original categorical columns and student_id
df_processed = df_processed.drop(['student_id', 'gender', 'education_level'], axis=1)

print(f"\n✅ Processed data shape: {df_processed.shape}")
print(f"Columns: {list(df_processed.columns)}")

# ============================================================
# 3. DEFINE TARGET AND FEATURES
# ============================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

# Target variable
y = df_processed['age']

# All features except age
X_all = df_processed.drop('age', axis=1)

print(f"\n🎯 Target Variable: age")
print(f"📈 Number of Features: {X_all.shape[1]}")
print(f"Features: {list(X_all.columns)}")

# ============================================================
# 4. SPLIT DATA
# ============================================================
print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT (80-20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# ============================================================
# 5. SIMPLE LINEAR REGRESSION (1 Feature)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: SIMPLE LINEAR REGRESSION")
print("=" * 60)

# Using stress_level as single feature
X_train_simple = X_train[['stress_level']].values
X_test_simple = X_test[['stress_level']].values

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

y_pred_simple_train = model_simple.predict(X_train_simple)
y_pred_simple_test = model_simple.predict(X_test_simple)

# Metrics for Simple Linear Regression
mse_simple_train = mean_squared_error(y_train, y_pred_simple_train)
mse_simple_test = mean_squared_error(y_test, y_pred_simple_test)
rmse_simple_train = np.sqrt(mse_simple_train)
rmse_simple_test = np.sqrt(mse_simple_test)
mae_simple_test = mean_absolute_error(y_test, y_pred_simple_test)
r2_simple_train = r2_score(y_train, y_pred_simple_train)
r2_simple_test = r2_score(y_test, y_pred_simple_test)

print(f"\nModel: age = {model_simple.intercept_:.4f} + {model_simple.coef_[0]:.4f} × stress_level")
print(f"\n📊 Training Performance:")
print(f"   RMSE: {rmse_simple_train:.4f}")
print(f"   R² Score: {r2_simple_train:.4f}")
print(f"\n📊 Testing Performance:")
print(f"   RMSE: {rmse_simple_test:.4f}")
print(f"   MAE: {mae_simple_test:.4f}")
print(f"   R² Score: {r2_simple_test:.4f}")

# ============================================================
# 6. MULTIPLE LINEAR REGRESSION (All Features)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: MULTIPLE LINEAR REGRESSION")
print("=" * 60)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train)

y_pred_multiple_train = model_multiple.predict(X_train_scaled)
y_pred_multiple_test = model_multiple.predict(X_test_scaled)

# Metrics for Multiple Linear Regression
mse_multiple_train = mean_squared_error(y_train, y_pred_multiple_train)
mse_multiple_test = mean_squared_error(y_test, y_pred_multiple_test)
rmse_multiple_train = np.sqrt(mse_multiple_train)
rmse_multiple_test = np.sqrt(mse_multiple_test)
mae_multiple_test = mean_absolute_error(y_test, y_pred_multiple_test)
r2_multiple_train = r2_score(y_train, y_pred_multiple_train)
r2_multiple_test = r2_score(y_test, y_pred_multiple_test)

print(f"\n📊 Training Performance:")
print(f"   RMSE: {rmse_multiple_train:.4f}")
print(f"   R² Score: {r2_multiple_train:.4f}")
print(f"\n📊 Testing Performance:")
print(f"   RMSE: {rmse_multiple_test:.4f}")
print(f"   MAE: {mae_multiple_test:.4f}")
print(f"   R² Score: {r2_multiple_test:.4f}")

# Feature importance (coefficients)
print(f"\n🔍 Top 5 Most Important Features (by coefficient magnitude):")
feature_importance = pd.DataFrame({
    'Feature': X_all.columns,
    'Coefficient': model_multiple.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
print(feature_importance[['Feature', 'Coefficient']].head(10).to_string(index=False))

# ============================================================
# 7. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Simple LR', 'Multiple LR'],
    'Train RMSE': [rmse_simple_train, rmse_multiple_train],
    'Test RMSE': [rmse_simple_test, rmse_multiple_test],
    'Train R²': [r2_simple_train, r2_multiple_train],
    'Test R²': [r2_simple_test, r2_multiple_test]
})

print("\n" + comparison_df.to_string(index=False))

best_model = "Multiple LR" if r2_multiple_test > r2_simple_test else "Simple LR"
print(f"\n✨ Best Model: {best_model}")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS...")
print("=" * 60)

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LAB 1: Linear Regression - Age Prediction', fontsize=16, fontweight='bold')

# Plot 1: Simple Linear Regression
axes[0, 0].scatter(X_test_simple, y_test, alpha=0.5, label='Actual', s=30)
axes[0, 0].plot(X_test_simple, y_pred_simple_test, color='red', linewidth=2, label='Prediction')
axes[0, 0].set_xlabel('Stress Level')
axes[0, 0].set_ylabel('Age')
axes[0, 0].set_title(f'Simple Linear Regression (R² = {r2_simple_test:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Multiple Linear Regression - Actual vs Predicted
axes[0, 1].scatter(y_test, y_pred_multiple_test, alpha=0.5, s=30)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Age')
axes[0, 1].set_ylabel('Predicted Age')
axes[0, 1].set_title(f'Multiple LR: Actual vs Predicted (R² = {r2_multiple_test:.4f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals Distribution
residuals = y_test - y_pred_multiple_test
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution (Multiple LR)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Model Comparison
models = ['Simple LR', 'Multiple LR']
test_rmse = [rmse_simple_test, rmse_multiple_test]
colors = ['#FF6B6B', '#4ECDC4']
axes[1, 1].bar(models, test_rmse, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Test RMSE')
axes[1, 1].set_title('Model Comparison: Test RMSE')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(test_rmse):
    axes[1, 1].text(i, v + 0.1, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('lab1_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: lab1_results.png")
plt.show()

# ============================================================
# 9. PREDICTIONS EXAMPLE
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS (Multiple Linear Regression)")
print("=" * 60)

# Show some predictions
results_df = pd.DataFrame({
    'Actual Age': y_test.values[:10],
    'Predicted Age': y_pred_multiple_test[:10],
    'Error': y_test.values[:10] - y_pred_multiple_test[:10],
    'Error %': ((y_test.values[:10] - y_pred_multiple_test[:10]) / y_test.values[:10] * 100)
})

print("\n" + results_df.to_string(index=False))

print("\n" + "=" * 60)
print("✨ LAB 1 COMPLETED!")
print("=" * 60)