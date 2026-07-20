import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# LAB2: DATA VISUALIZATION
# Student Sleep & Mental Health Dataset
# ==========================================

print("=" * 80)
print(" " * 20 + "LAB2: DATA VISUALIZATION")
print("=" * 80)

# [1] LOAD DATASET
print("\n[1] LOAD DATASET")
print("-" * 80)
df = pd.read_csv('student_sleep_mental_health_2026.csv')
print(f"✓ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# [2] DATA PREPROCESSING
print("\n[2] DATA PREPROCESSING")
print("-" * 80)

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"✓ Found {len(numeric_cols)} numeric columns:")
for col in numeric_cols:
    print(f"   - {col}")

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n✓ Found {len(categorical_cols)} categorical columns:")
for col in categorical_cols:
    print(f"   - {col}")

# ==========================================
# [3] HISTOGRAM - Numeric Columns Distribution
# ==========================================

print("\n[3] CREATING HISTOGRAMS")
print("-" * 80)

# Create histograms for all numeric columns
fig_hist = plt.figure(figsize=(16, 12))
fig_hist.suptitle('Histograms - Distribution of Numeric Variables', 
                   fontsize=16, fontweight='bold', y=0.995)

n_cols = len(numeric_cols)
n_rows = (n_cols + 2) // 3  # Calculate number of rows needed (3 columns per row)

for idx, col in enumerate(numeric_cols, 1):
    ax = plt.subplot(n_rows, 3, idx)
    
    # Create histogram
    ax.hist(df[col].dropna(), bins=25, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add statistics
    mean = df[col].mean()
    median = df[col].median()
    
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    
    ax.set_title(f'{col}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
print("✓ Histograms saved as 'histograms.png'")
plt.show()

# Print histogram statistics
print("\nHistogram Statistics:")
print("-" * 80)
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Mean:     {df[col].mean():.4f}")
    print(f"  Median:   {df[col].median():.4f}")
    print(f"  Std Dev:  {df[col].std():.4f}")
    print(f"  Min:      {df[col].min():.4f}")
    print(f"  Max:      {df[col].max():.4f}")
    print(f"  Skewness: {df[col].skew():.4f}")
    print(f"  Kurtosis: {df[col].kurtosis():.4f}")

# ==========================================
# [4] CORRELATION HEATMAP
# ==========================================

print("\n[4] CREATING CORRELATION HEATMAP")
print("-" * 80)

# Calculate correlation matrix (only for numeric columns)
if len(numeric_cols) > 1:
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig_heatmap = plt.figure(figsize=(12, 10))
    
    sns.heatmap(correlation_matrix, 
                annot=True,                      # Show correlation values
                cmap='coolwarm',                  # Color scheme
                center=0,                        # Center at 0
                fmt='.2f',                       # Format: 2 decimal places
                square=True,                     # Make squares
                linewidths=0.5,                  # Add gridlines
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1)                # Scale from -1 to 1
    
    plt.title('Correlation Heatmap - Numeric Variables', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation Heatmap saved as 'correlation_heatmap.png'")
    plt.show()
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print("-" * 80)
    print(correlation_matrix.to_string())
    
    # Find strongest correlations
    print("\n\nStrongest Correlations (excluding diagonal):")
    print("-" * 80)
    
    # Get upper triangle of correlation matrix
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_pairs.append({
                'Variable 1': correlation_matrix.columns[i],
                'Variable 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })
    
    # Sort by absolute correlation value
    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df['Abs_Correlation'] = corr_pairs_df['Correlation'].abs()
    corr_pairs_df = corr_pairs_df.sort_values('Abs_Correlation', ascending=False)
    
    # Print top 10
    print(corr_pairs_df[['Variable 1', 'Variable 2', 'Correlation']].head(10).to_string(index=False))
    
else:
    print("⚠ Warning: Need at least 2 numeric columns for correlation analysis")

# ==========================================
# [5] COMBINED VISUALIZATION
# ==========================================

print("\n[5] CREATING COMBINED VISUALIZATION")
print("-" * 80)

# Create a figure with distribution and heatmap
fig_combined = plt.figure(figsize=(16, 6))

# Plot 1: Sample histograms (top 3 numeric columns)
for idx, col in enumerate(numeric_cols[:3], 1):
    ax = plt.subplot(1, 4, idx)
    ax.hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution: {col}', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)

# Plot 2: Correlation heatmap (smaller version)
if len(numeric_cols) > 1:
    ax4 = plt.subplot(1, 4, 4)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f', 
                square=True, 
                cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1,
                ax=ax4)
    ax4.set_title('Correlation Matrix', fontweight='bold')

fig_combined.suptitle('LAB2: Data Visualization Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_combined.png', dpi=300, bbox_inches='tight')
print("✓ Combined visualization saved as 'visualization_combined.png'")
plt.show()

# ==========================================
# [6] SUMMARY REPORT
# ==========================================

print("\n" + "=" * 80)
print(" " * 30 + "SUMMARY REPORT")
print("=" * 80)
print(f"\n✓ Total Numeric Columns:      {len(numeric_cols)}")
print(f"✓ Total Categorical Columns:  {len(categorical_cols)}")
print(f"✓ Total Rows:                 {df.shape[0]}")
print(f"✓ Total Columns:              {df.shape[1]}")

print(f"\n✓ Visualizations Created:")
print(f"  1. histograms.png              - Distribution of all numeric variables")
print(f"  2. correlation_heatmap.png     - Correlation between numeric variables")
print(f"  3. visualization_combined.png  - Summary of histograms and heatmap")

print("\n" + "=" * 80)
print("✓ LAB2 Data Visualization Complete!")
print("=" * 80)