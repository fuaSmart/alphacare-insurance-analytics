import pandas as pd
import numpy as np
from scipy import stats 
import statsmodels.api as sm 
import os

# --- Configuration ---
DATA_PATH = 'data/MachineLearningRating_v3.csv' # Your data file

# --- Load Data ---
print("--- Data Loading ---")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it's downloaded and placed correctly.")
    exit()
except Exception as e:
    print(f"Error loading or parsing data. Check if '{DATA_PATH}' is truly comma-separated: {e}")
    exit()

# --- Data Preprocessing for Metrics ---
print("\n--- Data Preprocessing for Metrics ---")

# Convert 'TransactionDate' to datetime
if 'TransactionDate' in df.columns:
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    print("'TransactionDate' converted to datetime format.")
else:
    print("Warning: 'TransactionDate' column not found. Date-based analysis might be affected.")

#  Claim Frequency indicator: 1 if claim occurred, 0 otherwise
df['ClaimOccurred'] = (df['TotalClaims'] > 0).astype(int)
print("Added 'ClaimOccurred' column.")

# Margin is defined as TotalPremium - TotalClaims
df['Margin'] = df['TotalPremium'] - df['TotalClaims']
print("Added 'Margin' column.")

print("\nSample of calculated metrics:")
print(df[['TotalClaims', 'TotalPremium', 'ClaimOccurred', 'Margin']].head())

# Claim Severity is the average amount of a claim, GIVEN that a claim occurred.
df_claims_only = df[df['ClaimOccurred'] == 1].copy()
print(f"\nDataFrame for Claim Severity (only policies with claims): {df_claims_only.shape[0]} rows.")
if df_claims_only.empty:
    print("Warning: No policies with claims found. Claim Severity tests will be skipped.")


# --- Statistical Testing: Rejecting or Accepting Null Hypotheses ---
print("\n" + "="*50)
print("--- Starting A/B Hypothesis Testing ---")
print("="*50)

# Significance level
ALPHA = 0.05

# Hypothesis H₀: There are no risk differences across provinces.
print("\n--- Hypothesis 1: Risk Differences Across Provinces ---")


# Fusing top 2 from value counts if they exist:
top_provinces = df['Province'].value_counts().index.tolist()
if len(top_provinces) >= 2:
    province1 = top_provinces[0]
    province2 = top_provinces[1]
else:
    print("Not enough unique provinces to perform comparison. Skipping.")
    province1, province2 = None, None

if province1 and province2:
    df_prov_subset = df[df['Province'].isin([province1, province2])].copy()
    df_claims_prov_subset = df_claims_only[df_claims_only['Province'].isin([province1, province2])].copy()

    print(f"\nComparing {province1} (N={len(df_prov_subset[df_prov_subset['Province']==province1])}) vs {province2} (N={len(df_prov_subset[df_prov_subset['Province']==province2])})")

    # Contingency table: Rows are Province, Columns are ClaimOccurred (0/1)
    contingency_table_freq = pd.crosstab(df_prov_subset['Province'], df_prov_subset['ClaimOccurred'])
    if not contingency_table_freq.empty and contingency_table_freq.shape == (2,2) and (contingency_table_freq.values > 0).all():
        chi2_stat_freq, p_value_freq, dof_freq, expected_freq = stats.chi2_contingency(contingency_table_freq)
        print(f"\nClaim Frequency ({province1} vs {province2}) - Chi-squared Test:")
        print(f"  Chi2 Stat: {chi2_stat_freq:.4f}")
        print(f"  P-value: {p_value_freq:.4f}")
        if p_value_freq < ALPHA:
            print(f"  Conclusion: Reject H0. There is a significant difference in Claim Frequency between {province1} and {province2}.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Claim Frequency between {province1} and {province2}.")
    else:
        print("  Skipping Claim Frequency test due to empty or invalid contingency table (e.g., all zeros in a cell).")


    # Filter for claims only in selected provinces
    claims_prov1 = df_claims_prov_subset[df_claims_prov_subset['Province'] == province1]['TotalClaims'].dropna()
    claims_prov2 = df_claims_prov_subset[df_claims_prov_subset['Province'] == province2]['TotalClaims'].dropna()

    if len(claims_prov1) > 1 and len(claims_prov2) > 1: # t-test requires at least 2 samples per group
        ttest_stat_severity, p_value_severity = stats.ttest_ind(claims_prov1, claims_prov2, equal_var=False) # Welch's t-test (handles unequal variances)
        print(f"\nClaim Severity ({province1} vs {province2}) - Independent Samples t-Test:")
        print(f"  Mean {province1} Claim: {claims_prov1.mean():.2f}")
        print(f"  Mean {province2} Claim: {claims_prov2.mean():.2f}")
        print(f"  T-statistic: {ttest_stat_severity:.4f}")
        print(f"  P-value: {p_value_severity:.4f}")
        if p_value_severity < ALPHA:
            print(f"  Conclusion: Reject H0. There is a significant difference in Claim Severity between {province1} and {province2}.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Claim Severity between {province1} and {province2}.")
    else:
        print(f"  Skipping Claim Severity test: Insufficient data for {province1} or {province2} with claims.")
else:
    print("  Cannot perform Hypothesis 1 tests due to insufficient unique provinces.")


# Similar to provinces, but with many more categories. We'll pick two representative zip codes or highly active ones for example.
print("\n--- Hypothesis 2: Risk Differences Between Zip Codes ---")

top_zip_codes = df['PostalCode'].value_counts().index.tolist()
if len(top_zip_codes) >= 2:
    zip_code1 = top_zip_codes[0] # Example: take the most frequent zip code
    zip_code2 = top_zip_codes[1] # Example: take the second most frequent
else:
    print("Not enough unique zip codes to perform comparison. Skipping.")
    zip_code1, zip_code2 = None, None

if zip_code1 and zip_code2:
    df_zip_subset = df[df['PostalCode'].isin([zip_code1, zip_code2])].copy()
    df_claims_zip_subset = df_claims_only[df_claims_only['PostalCode'].isin([zip_code1, zip_code2])].copy()

    print(f"\nComparing Zip Code {zip_code1} (N={len(df_zip_subset[df_zip_subset['PostalCode']==zip_code1])}) vs Zip Code {zip_code2} (N={len(df_zip_subset[df_zip_subset['PostalCode']==zip_code2])})")

    # Claim Frequency Test (Categorical: Chi-squared Test)
    contingency_table_zip_freq = pd.crosstab(df_zip_subset['PostalCode'], df_zip_subset['ClaimOccurred'])
    if not contingency_table_zip_freq.empty and contingency_table_zip_freq.shape == (2,2) and (contingency_table_zip_freq.values > 0).all():
        chi2_zip_freq, p_value_zip_freq, dof_zip_freq, expected_zip_freq = stats.chi2_contingency(contingency_table_zip_freq)
        print(f"\nClaim Frequency (Zip Code {zip_code1} vs {zip_code2}) - Chi-squared Test:")
        print(f"  Chi2 Stat: {chi2_zip_freq:.4f}")
        print(f"  P-value: {p_value_zip_freq:.4f}")
        if p_value_zip_freq < ALPHA:
            print(f"  Conclusion: Reject H0. Significant difference in Claim Frequency between Zip Code {zip_code1} and {zip_code2}.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Claim Frequency between Zip Code {zip_code1} and {zip_code2}.")
    else:
        print("  Skipping Claim Frequency test due to empty or invalid contingency table for zip codes.")


    # Claim Severity Test (Numerical: Independent Samples t-Test)
    claims_zip1 = df_claims_zip_subset[df_claims_zip_subset['PostalCode'] == zip_code1]['TotalClaims'].dropna()
    claims_zip2 = df_claims_zip_subset[df_claims_zip_subset['PostalCode'] == zip_code2]['TotalClaims'].dropna()

    if len(claims_zip1) > 1 and len(claims_zip2) > 1:
        ttest_zip_severity, p_value_zip_severity = stats.ttest_ind(claims_zip1, claims_zip2, equal_var=False)
        print(f"\nClaim Severity (Zip Code {zip_code1} vs {zip_code2}) - Independent Samples t-Test:")
        print(f"  Mean {zip_code1} Claim: {claims_zip1.mean():.2f}")
        print(f"  Mean {zip_code2} Claim: {claims_zip2.mean():.2f}")
        print(f"  T-statistic: {ttest_zip_severity:.4f}")
        print(f"  P-value: {p_value_zip_severity:.4f}")
        if p_value_zip_severity < ALPHA:
            print(f"  Conclusion: Reject H0. Significant difference in Claim Severity between Zip Code {zip_code1} and {zip_code2}.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Claim Severity between Zip Code {zip_code1} and {zip_code2}.")
    else:
        print(f"  Skipping Claim Severity test: Insufficient data for Zip Code {zip_code1} or {zip_code2} with claims.")
else:
    print("  Cannot perform Hypothesis 2 tests due to insufficient unique zip codes.")


# Hypothesis There are no significant margin (profit) differences between zip codes.
print("\n--- Hypothesis 3: Margin Differences Between Zip Codes ---")

if zip_code1 and zip_code2:
    # Margin Test (Numerical: Independent Samples t-Test)
    margin_zip1 = df_zip_subset[df_zip_subset['PostalCode'] == zip_code1]['Margin'].dropna()
    margin_zip2 = df_zip_subset[df_zip_subset['PostalCode'] == zip_code2]['Margin'].dropna()

    if len(margin_zip1) > 1 and len(margin_zip2) > 1:
        ttest_zip_margin, p_value_zip_margin = stats.ttest_ind(margin_zip1, margin_zip2, equal_var=False)
        print(f"\nMargin (Zip Code {zip_code1} vs {zip_code2}) - Independent Samples t-Test:")
        print(f"  Mean {zip_code1} Margin: {margin_zip1.mean():.2f}")
        print(f"  Mean {zip_code2} Margin: {margin_zip2.mean():.2f}")
        print(f"  T-statistic: {ttest_zip_margin:.4f}")
        print(f"  P-value: {p_value_zip_margin:.4f}")
        if p_value_zip_margin < ALPHA:
            print(f"  Conclusion: Reject H0. Significant difference in Margin between Zip Code {zip_code1} and {zip_code2}.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Margin between Zip Code {zip_code1} and {zip_code2}.")
    else:
        print(f"  Skipping Margin test: Insufficient data for Zip Code {zip_code1} or {zip_code2}.")
else:
    print("  Cannot perform Hypothesis 3 tests due to insufficient unique zip codes.")


# Hypothesis: There are no significant risk differences between Women and Men.
print("\n--- Hypothesis 4: Risk Differences Between Women and Men ---")

# Handle potential variations in gender values (e.g., 'F', 'M', or mixed case)
df['Gender_Cleaned'] = df['Gender'].replace({'Male ': 'Male', 'Female ':'Female', 'Not specified': np.nan}).str.strip() # Clean up common variations

gender1 = 'Male'
gender2 = 'Female'

df_gender_subset = df[df['Gender_Cleaned'].isin([gender1, gender2])].copy()
df_claims_gender_subset = df_claims_only[df_claims_only['Gender_Cleaned'].isin([gender1, gender2])].copy()


if df_gender_subset.empty:
    print("Warning: Data for Male or Female not found or empty after cleaning. Skipping gender comparison.")
else:
    print(f"\nComparing {gender1} (N={len(df_gender_subset[df_gender_subset['Gender_Cleaned']==gender1])}) vs {gender2} (N={len(df_gender_subset[df_gender_subset['Gender_Cleaned']==gender2])})")

    # Claim Frequency Test (Categorical: Chi-squared Test)
    contingency_table_gender_freq = pd.crosstab(df_gender_subset['Gender_Cleaned'], df_gender_subset['ClaimOccurred'])
    if not contingency_table_gender_freq.empty and contingency_table_gender_freq.shape == (2,2) and (contingency_table_gender_freq.values > 0).all():
        chi2_gender_freq, p_value_gender_freq, dof_gender_freq, expected_gender_freq = stats.chi2_contingency(contingency_table_gender_freq)
        print(f"\nClaim Frequency (Female vs Male) - Chi-squared Test:")
        print(f"  Chi2 Stat: {chi2_gender_freq:.4f}")
        print(f"  P-value: {p_value_gender_freq:.4f}")
        if p_value_gender_freq < ALPHA:
            print(f"  Conclusion: Reject H0. Significant difference in Claim Frequency between Women and Men.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Claim Frequency between Women and Men.")
    else:
        print("  Skipping Claim Frequency test due to empty or invalid contingency table for gender.")

    #  Claim Severity Test (Numerical
    claims_gender1 = df_claims_gender_subset[df_claims_gender_subset['Gender_Cleaned'] == gender1]['TotalClaims'].dropna()
    claims_gender2 = df_claims_gender_subset[df_claims_gender_subset['Gender_Cleaned'] == gender2]['TotalClaims'].dropna()

    if len(claims_gender1) > 1 and len(claims_gender2) > 1:
        ttest_gender_severity, p_value_gender_severity = stats.ttest_ind(claims_gender1, claims_gender2, equal_var=False)
        print(f"\nClaim Severity (Female vs Male) - Independent Samples t-Test:")
        print(f"  Mean {gender1} Claim: {claims_gender1.mean():.2f}")
        print(f"  Mean {gender2} Claim: {claims_gender2.mean():.2f}")
        print(f"  T-statistic: {ttest_gender_severity:.4f}")
        print(f"  P-value: {p_value_gender_severity:.4f}")
        if p_value_gender_severity < ALPHA:
            print(f"  Conclusion: Reject H0. Significant difference in Claim Severity between Women and Men.")
        else:
            print(f"  Conclusion: Fail to reject H0. No significant difference in Claim Severity between Women and Men.")
    else:
        print(f"  Skipping Claim Severity test: Insufficient data for {gender1} or {gender2} with claims.")