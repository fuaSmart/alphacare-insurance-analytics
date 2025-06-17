import pandas as pd
import numpy as np
import os
import shap # For model interpretability
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
DATA_PATH = 'data/MachineLearningRating_v3.csv' # Your data file
OUTPUT_MODELS_DIR = 'models/' # Directory to save trained models (optional for this task, but good practice)
OUTPUT_PLOTS_DIR = 'plots/modeling/' # Directory to save plots for modeling results

# Create output directories if they don't exist
if not os.path.exists(OUTPUT_MODELS_DIR):
    os.makedirs(OUTPUT_MODELS_DIR)
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# --- 1. Data Loading ---
print("--- Data Loading ---")
try:
    # Explicitly set low_memory=False to suppress DtypeWarning if it's about memory usage
    # This might help Pandas infer types more consistently for large files, though
    # the TypeError is about actual mixed types in certain columns.
    df = pd.read_csv(DATA_PATH, low_memory=False) # Added low_memory=False
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it's downloaded and placed correctly.")
    exit()

# --- 2. Data Preparation for Claim Severity Prediction ---
# Target: TotalClaims (only for policies with claims > 0)
# Evaluation: RMSE, R-squared

print("\n--- Data Preparation for Claim Severity Prediction ---")

# Filter for policies where claims occurred (TotalClaims > 0)
df_claims_only = df[df['TotalClaims'] > 0].copy()
print(f"Filtered dataset for Claim Severity: {df_claims_only.shape[0]} policies with claims.")

# Define Features (X) and Target (y) for Claim Severity Model
features = [
    'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 'Bank',
    'AccountType', 'MaritalStatus', 'Gender', 'Country', 'Province',
    'PostalCode', 'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType',
    'RegistrationYear', 'Make', 'Model', 'Cylinders', 'Cubiccapacity',
    'Kilowatts', 'Bodytype', 'NumberOfDoors', 'CustomValueEstimate',
    'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding', 'NewVehicle',
    'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet',
    'SumInsured', 'TermFrequency', 'ExcessSelected', 'CoverCategory',
    'CoverType', 'CoverGroup', 'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType'
]

# Ensure all selected features exist in the dataframe
selected_features = [f for f in features if f in df_claims_only.columns]
if len(selected_features) != len(features):
    missing_feats = set(features) - set(selected_features)
    print(f"Warning: The following features are missing from the dataset: {missing_feats}")
    # Remove missing features from selected_features list for consistent processing
    features = selected_features

X = df_claims_only[features].copy() # Ensure X is a copy to avoid SettingWithCopyWarning
y = df_claims_only['TotalClaims']

# --- Feature Engineering ---
# Create 'VehicleAge' from 'RegistrationYear' and 'TransactionDate'
if 'RegistrationYear' in X.columns and 'TransactionDate' in df_claims_only.columns:
    # Ensure TransactionDate is datetime before calculating max year
    if not pd.api.types.is_datetime64_any_dtype(df_claims_only['TransactionDate']):
        df_claims_only['TransactionDate'] = pd.to_datetime(df_claims_only['TransactionDate'], errors='coerce')
    current_year = df_claims_only['TransactionDate'].dt.year.max() # Use the latest year in dataset as 'current'
    X['VehicleAge'] = current_year - X['RegistrationYear']
    X = X.drop(columns=['RegistrationYear']) # Drop original year
    print("Engineered 'VehicleAge' feature.")
    # Remove 'RegistrationYear' from features and add 'VehicleAge'
    if 'RegistrationYear' in features:
        features.remove('RegistrationYear')
    features.append('VehicleAge')

# --- Identify numerical and categorical features ---
# Re-identify after feature engineering
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# --- CRITICAL FIX: Convert identified categorical columns to string type ---
# This ensures that OneHotEncoder receives uniform string data,
# preventing TypeError with mixed types (str, float, int).
print(f"\nConverting {len(categorical_features)} categorical columns to string type to handle mixed types...")
for col in categorical_features:
    # Fill NaN before converting to string, otherwise NaN becomes 'nan' string.
    # We will let SimpleImputer handle this. Just convert to string.
    X[col] = X[col].astype(str)
    # Check for 'nan' strings if SimpleImputer strategy is not 'most_frequent' later
    # X[col] = X[col].replace('nan', np.nan) # If you want to keep them as NaNs for imputer


print(f"Numerical Features (After FE): {numerical_features}")
print(f"Categorical Features (After FE and type conversion): {categorical_features}")


# --- Create Preprocessing Pipelines ---
# Numerical Imputer: Strategy='mean' for simplicity. Could also use 'median'.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Categorical Imputer & One-Hot Encoder
# Impute missing categorical values with the most frequent value (mode)
# One-hot encode to convert categorical to numerical format
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- 3. Model Building and Evaluation ---

# Function to train and evaluate a model
def train_and_evaluate_model(model, name, X_train, y_train, X_test, y_test, preprocessor):
    print(f"\n--- Training {name} ---")
    # Create a full pipeline that first preprocesses, then applies the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               (name, model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")

    return pipeline, rmse, r2, y_pred, y_pred # Added y_pred return for consistency

# --- Implement Models ---

# 1. Linear Regression
lr_model, lr_rmse, lr_r2, lr_preds_train, lr_preds_test = train_and_evaluate_model(
    LinearRegression(), 'LinearRegression', X_train, y_train, X_test, y_test, preprocessor
)

# 2. Decision Tree Regressor
dt_model, dt_rmse, dt_r2, dt_preds_train, dt_preds_test = train_and_evaluate_model(
    DecisionTreeRegressor(random_state=42), 'DecisionTreeRegressor', X_train, y_train, X_test, y_test, preprocessor
)

# 3. Random Forest Regressor
rf_model, rf_rmse, rf_r2, rf_preds_train, rf_preds_test = train_and_evaluate_model(
    RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1), 'RandomForestRegressor', X_train, y_train, X_test, y_test, preprocessor
)

# 4. XGBoost Regressor
xgb_model, xgb_rmse, xgb_r2, xgb_preds_train, xgb_preds_test = train_and_evaluate_model(
    XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, n_jobs=-1), 'XGBoostRegressor', X_train, y_train, X_test, y_test, preprocessor
)

# --- 4. Report Comparison Between Models ---
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse, xgb_rmse],
    'R-squared': [lr_r2, dt_r2, rf_r2, xgb_r2]
}).sort_values(by='RMSE')

print("\n--- Model Performance Comparison (Claim Severity Prediction) ---")
print(results.to_string(index=False)) # Use to_string to display full table

# Identify the best model based on RMSE
best_model_name = results.iloc[0]['Model']
print(f"\nBest performing model for Claim Severity: {best_model_name} (RMSE: {results.iloc[0]['RMSE']:.2f}, R-squared: {results.iloc[0]['R-squared']:.2f})")

# --- 5. Model Interpretability (SHAP) ---
# Using SHAP for the best performing tree-based model (e.g., XGBoost, if it performs best)
print("\n--- Model Interpretability (SHAP) for Best Performing Model ---")

# Best model for interpretability.
best_pipeline = None
if 'XGBoost' in best_model_name:
    best_pipeline = xgb_model
elif 'Random Forest' in best_model_name:
    best_pipeline = rf_model
elif 'Decision Tree' in best_model_name:
    best_pipeline = dt_model # Although not ideal for general use, it's tree-based

if best_pipeline:
    # Get the preprocessed training data for SHAP Explainer
    # Use the preprocessor from the best pipeline to transform X_train
    # It's important to keep the order and transformation consistent
    X_train_processed = best_pipeline.named_steps['preprocessor'].transform(X_train)

    # Get feature names after one-hot encoding
    # Handle remainder='passthrough' columns if they exist and are not in other transformers
    try:
        # Get names for numerical features (from numerical_transformer)
        processed_numerical_feature_names = numerical_features
        # Get names for one-hot encoded categorical features
        processed_categorical_feature_names = list(best_pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
                                                    .named_steps['onehot'].get_feature_names_out(categorical_features))
        feature_names = processed_numerical_feature_names + processed_categorical_feature_names

        # If remainder='passthrough' was used and there were remaining columns
        # (unlikely in this setup as all features are categorized)
        # you might need to identify and add them here.
        # For simplicity, assume all are covered by num or cat.

    except Exception as e:
        print(f"Error getting feature names after preprocessing for SHAP: {e}")
        # Fallback to generic names if feature name extraction fails
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]


    # Convert processed data back to a DataFrame with correct column names for SHAP
    # Sample a smaller subset for SHAP calculation if dataset is very large, for performance
    sample_size = min(1000, X_train_processed.shape[0])
    X_train_processed_sample_indices = np.random.choice(X_train_processed.shape[0], sample_size, replace=False)
    X_train_processed_sample = X_train_processed[X_train_processed_sample_indices, :]
    X_train_processed_df_sample = pd.DataFrame(X_train_processed_sample, columns=feature_names)


    # Create a SHAP explainer for the trained model (the model part of the pipeline)
    # Access the actual model object within the pipeline for TreeExplainer
    model_for_shap = best_pipeline.named_steps[best_pipeline.steps[-1][0]]
    explainer = shap.TreeExplainer(model_for_shap)

    # Calculate SHAP values for the sampled data
    shap_values = explainer.shap_values(X_train_processed_df_sample)

    # --- SHAP Summary Plot (Feature Importance) ---
    plt.figure(figsize=(10, 7)) # Adjusted figure size for better readability
    shap.summary_plot(shap_values, X_train_processed_df_sample, show=False)
    plt.title('SHAP Summary Plot: Feature Importance for Claim Severity')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'shap_summary_plot.png'))
    plt.close()
    print("Generated SHAP summary plot.")

    # --- SHAP Dependence Plot (Example for one important feature) ---
    # To pick a meaningful feature for dependence plot, you'd usually look at the summary plot first.
    # For now, let's try 'VehicleAge' or 'CustomValueEstimate' if they are in feature_names
    important_feature_for_dependence = None
    if 'VehicleAge' in feature_names:
        important_feature_for_dependence = 'VehicleAge'
    elif 'CustomValueEstimate' in feature_names:
        important_feature_for_dependence = 'CustomValueEstimate'
    elif feature_names: # Pick the first feature if others not found
        important_feature_for_dependence = feature_names[0]

    if important_feature_for_dependence:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(important_feature_for_dependence, shap_values, X_train_processed_df_sample, show=False)
        plt.title(f'SHAP Dependence Plot: {important_feature_for_dependence} vs. Claim Severity Impact')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f'shap_dependence_{important_feature_for_dependence}.png'))
        plt.close()
        print(f"Generated SHAP dependence plot for {important_feature_for_dependence}.")

    print("\nSHAP analysis completed. Check 'plots/modeling/' for plots.")
else:
    print("\nSHAP analysis skipped as no suitable tree-based model was selected or found.")


print("\n--- Task 4: Predictive Modeling Completed ---")