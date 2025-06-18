import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_PATH = 'data/MachineLearningRating_v3.csv' 
OUTPUT_PLOTS_DIR = 'plots/eda/' 
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# --- Data Understanding & Loading ---
print("--- Data Loading & Initial Understanding ---")
try:
    
    df = pd.read_csv(DATA_PATH) 
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it's downloaded and placed correctly.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check the file's content and try specifying the correct delimiter (e.g., delimiter='|' for pipe-separated).")
    exit()

# --- Data Structure & Quality Assessment ---
print("\n--- Data Structure & Types ---")
print(df.info()) 

print("\n--- Missing Values Assessment ---")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percentage})
print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False))

# --- Convert 'TransactionDate' to datetime ---
# Assuming 'TransactionDate' is the column with date information based on your description
if 'TransactionDate' in df.columns:
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    print("\n'TransactionDate' converted to datetime format.")
else:
    print("\nWarning: 'TransactionDate' column not found. Date-based analysis might be affected.")

# ---  Data Summarization: Descriptive Statistics ---
print("\n--- Descriptive Statistics for Numerical Features ---")
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

financial_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured', 'CustomValueEstimate', 'CapitalOutstanding']
present_financial_cols = [col for col in financial_cols if col in df.columns]

if present_financial_cols:
    print(df[present_financial_cols].describe())
else:
    print("No relevant financial columns found for descriptive statistics.")

# --- Univariate Analysis ---
print("\n--- Univariate Analysis: Distributions ---")

# Histograms for numerical columns (select a few important ones)
for col in ['TotalPremium', 'TotalClaims', 'Rating', 'CustomValueEstimate', 'Cylinders', 'NumberOfDoors', 'Kilowatts']:
    if col in df.columns and df[col].dtype != object: 
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f'hist_{col}.png'))
        plt.close()
        print(f"Generated histogram for {col}.")

# Bar charts for key categorical columns
categorical_cols_to_plot = ['Province', 'VehicleType', 'Gender', 'MaritalStatus', 'Product', 'CoverCategory']
for col in categorical_cols_to_plot:
    if col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col].dropna(), order=df[col].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f'bar_{col}.png'))
        plt.close()
        print(f"Generated bar chart for {col}.")

# --- Bivariate or Multivariate Analysis ---
print("\n--- Bivariate/Multivariate Analysis ---")

# Overall Loss Ratio (TotalClaims / TotalPremium)
# Handle potential division by zero for TotalPremium
df['LossRatio'] = df.apply(lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] > 0 else 0, axis=1)
print(f"\nOverall Portfolio Loss Ratio: {df['LossRatio'].mean():.2f}")

# Loss Ratio by Province, VehicleType, Gender
if 'Province' in df.columns:
    print("\nLoss Ratio by Province:")
    print(df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False))
    plt.figure(figsize=(12, 7))
    sns.barplot(x='LossRatio', y='Province', data=df.groupby('Province')['LossRatio'].mean().reset_index().sort_values(by='LossRatio', ascending=False), palette='plasma')
    plt.title('Average Loss Ratio by Province')
    plt.xlabel('Average Loss Ratio')
    plt.ylabel('Province')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'loss_ratio_by_province.png'))
    plt.close()

if 'VehicleType' in df.columns:
    print("\nLoss Ratio by VehicleType:")
    print(df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False))
    plt.figure(figsize=(10, 6))
    sns.barplot(x='LossRatio', y='VehicleType', data=df.groupby('VehicleType')['LossRatio'].mean().reset_index().sort_values(by='LossRatio', ascending=False), palette='magma')
    plt.title('Average Loss Ratio by VehicleType')
    plt.xlabel('Average Loss Ratio')
    plt.ylabel('Vehicle Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'loss_ratio_by_vehicletype.png'))
    plt.close()

if 'Gender' in df.columns:
    print("\nLoss Ratio by Gender:")
    print(df.groupby('Gender')['LossRatio'].mean().sort_values(ascending=False))
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Gender', y='LossRatio', data=df.groupby('Gender')['LossRatio'].mean().reset_index(), palette='cividis')
    plt.title('Average Loss Ratio by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Loss Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'loss_ratio_by_gender.png'))
    plt.close()



if 'TransactionDate' in df.columns and 'PostalCode' in df.columns:
    monthly_data = df.set_index('TransactionDate').resample('M').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    ).reset_index()

    print("\nMonthly Trends (Overall):")
    print(monthly_data.head())

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='TransactionDate', y='TotalPremium', data=monthly_data, label='Total Premium', marker='o')
    sns.lineplot(x='TransactionDate', y='TotalClaims', data=monthly_data, label='Total Claims', marker='o')
    plt.title('Monthly Trends: Total Premium vs Total Claims')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'monthly_trends.png'))
    plt.close()
    print("Generated monthly trends plot.")

    # Correlation matrix for numerical columns (select relevant ones)
    correlation_cols = present_financial_cols + ['Rating', 'Cylinders', 'Kilowatts', 'NumberOfDoors']
    correlation_df = df[correlation_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Key Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'correlation_matrix.png'))
    plt.close()
    print("Generated correlation matrix.")

# --- 6. Outlier Detection ---
print("\n--- Outlier Detection using Box Plots ---")
for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
    if col in df.columns and df[col].dtype != object:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[col].dropna(), color='lightcoral')
        plt.title(f'Box Plot of {col} (Outlier Detection)')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f'boxplot_{col}.png'))
        plt.close()
        print(f"Generated box plot for {col}.")# Handle potential division by zero for TotalPremium


print("\n--- EDA Analysis Complete. Check 'plots/eda/' directory for generated plots. ---")