import pandas as pd
import numpy as np
import os
import shap
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
DATA_PATH = 'data/MachineLearningRating_v3.csv' 
OUTPUT_MODELS_DIR = 'models/' 
OUTPUT_PLOTS_DIR = 'plots/modeling/' 

# Create output directories if they don't exist
if not os.path.exists(OUTPUT_MODELS_DIR):
    os.makedirs(OUTPUT_MODELS_DIR)
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# --- 1. Data Loading ---
print("--- Data Loading ---")
try:

    df = pd.read_csv(DATA_PATH, low_memory=False) 
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it's downloaded and placed correctly.")
    exit()

# --- 2. Data Preparation for Claim Severity Prediction ---


print("\n--- Data Preparation for Claim Severity Prediction ---")


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


selected_features = [f for f in features if f in df_claims_only.columns]
if len(selected_features) != len(features):
    missing_feats = set(features) - set(selected_features)
    print(f"Warning: The following features are missing from the dataset: {missing_feats}")
   
    features = selected_features

X = df_claims_only[features].copy() 
y = df_claims_only['TotalClaims']

# --- Feature Engineering ---

if 'RegistrationYear' in X.columns and 'TransactionDate' in df_claims_only.columns:
    # Ensure TransactionDate is datetime before calculating max year
    if not pd.api.types.is_datetime64_any_dtype(df_claims_only['TransactionDate']):
        df_claims_only['TransactionDate'] = pd.to_datetime(df_claims_only['TransactionDate'], errors='coerce')
    current_year = df_claims_only['TransactionDate'].dt.year.max() 
    X['VehicleAge'] = current_year - X['RegistrationYear']
    X = X.drop(columns=['RegistrationYear']) 
    print("Engineered 'VehicleAge' feature.")

    if 'RegistrationYear' in features:
        features.remove('RegistrationYear')
    features.append('VehicleAge')

# --- Identify numerical and categorical features ---

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()


print(f"\nConverting {len(categorical_features)} categorical columns to string type to handle mixed types...")
for col in categorical_features:

    X[col] = X[col].astype(str)



print(f"Numerical Features (After FE): {numerical_features}")
print(f"Categorical Features (After FE and type conversion): {categorical_features}")


# --- Create Preprocessing Pipelines ---

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Categorical Imputer & One-Hot Encoder

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

    return pipeline, rmse, r2, y_pred, y_pred 

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
print(results.to_string(index=False)) 

# Identify the best model based on RMSE
best_model_name = results.iloc[0]['Model']
print(f"\nBest performing model for Claim Severity: {best_model_name} (RMSE: {results.iloc[0]['RMSE']:.2f}, R-squared: {results.iloc[0]['R-squared']:.2f})")

# --- 5. Model Interpretability (SHAP) ---
print("\n--- Model Interpretability (SHAP) for Best Performing Model ---")

# Best model for interpretability.
best_pipeline = None
if 'XGBoost' in best_model_name:
    best_pipeline = xgb_model
elif 'Random Forest' in best_model_name:
    best_pipeline = rf_model
elif 'Decision Tree' in best_model_name:
    best_pipeline = dt_model 
if best_pipeline:
    
    X_train_processed = best_pipeline.named_steps['preprocessor'].transform(X_train)

   
    try:
        # Get names for numerical features
        processed_numerical_feature_names = numerical_features
        # Get names for one-hot encoded categorical features
        processed_categorical_feature_names = list(best_pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
                                                    .named_steps['onehot'].get_feature_names_out(categorical_features))
        feature_names = processed_numerical_feature_names + processed_categorical_feature_names

       
    except Exception as e:
        print(f"Error getting feature names after preprocessing for SHAP: {e}")

        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]


    # Convert processed data back to a DataFrame with correct column names for SHAP

    sample_size = min(1000, X_train_processed.shape[0])
    X_train_processed_sample_indices = np.random.choice(X_train_processed.shape[0], sample_size, replace=False)
    X_train_processed_sample = X_train_processed[X_train_processed_sample_indices, :]
    X_train_processed_df_sample = pd.DataFrame(X_train_processed_sample, columns=feature_names)


    # Create a SHAP explainer for the trained model 
    model_for_shap = best_pipeline.named_steps[best_pipeline.steps[-1][0]]
    explainer = shap.TreeExplainer(model_for_shap)

    # Calculate SHAP values for the sampled data
    shap_values = explainer.shap_values(X_train_processed_df_sample)

    # --- SHAP Summary Plot ---
    plt.figure(figsize=(10, 7)) # Adjusted figure size for better readability
    shap.summary_plot(shap_values, X_train_processed_df_sample, show=False)
    plt.title('SHAP Summary Plot: Feature Importance for Claim Severity')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'shap_summary_plot.png'))
    plt.close()
    print("Generated SHAP summary plot.")

    # --- SHAP Dependence Plot ---
    
    important_feature_for_dependence = None
    if 'VehicleAge' in feature_names:
        important_feature_for_dependence = 'VehicleAge'
    elif 'CustomValueEstimate' in feature_names:
        important_feature_for_dependence = 'CustomValueEstimate'
    elif feature_names: 
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