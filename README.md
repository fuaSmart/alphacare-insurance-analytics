# End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Overview

This repository hosts the "End-to-End Insurance Risk Analytics & Predictive Modeling" project for **AlphaCare Insurance Solutions (ACIS)**, a leading insurance provider in South Africa. As a marketing analytics engineer, my primary objective is to analyze historical car insurance claim data to optimize marketing strategies and identify "low-risk" target segments. By understanding key risk drivers, ACIS can reduce premiums for attractive client profiles, thereby attracting new clients and enhancing market competitiveness.

This project will sharpen skills in Data Engineering (DE), Predictive Analytics (PA), and Machine Learning Engineering (MLE), simulating real-world pressures and deadlines typical in financial analytics.

## Business Objectives & Key Questions

The core business objective is to optimize marketing strategy and identify low-risk customer segments. To achieve this, the project focuses on:

- **Risk Quantification**: Quantifying "risk" using Claim Frequency (proportion of policies with at least one claim) and Claim Severity (average claim amount, given a claim occurred). "Margin" is defined as (TotalPremium - TotalClaims).
- **Hypothesis Testing**: Statistically validating or rejecting the following null hypotheses:
  - H₀: There are no risk differences across provinces.
  - H₀: There are no risk differences between zip codes.
  - H₀: There are no significant margin (profit) differences between zip codes.
  - H₀: There are no significant risk differences between Women and Men.
- **Predictive Modeling**:
  - Developing a linear regression model to predict `TotalClaims` for policies that have claims.
  - Building a machine learning model to predict optimal premium values based on car features, owner features, location features, and other relevant factors.
  - Reporting on the explanatory power and interpretability of important model features (e.g., using SHAP/LIME).

The final deliverable will be a comprehensive report detailing methodologies, findings, and actionable recommendations for tailoring ACIS's insurance products.

## Dataset Overview

The historical data covers the period from **February 2014 to August 2015**. The dataset structure includes the following key columns:

- **Policy Information**: `UnderwrittenCoverID`, `PolicyID`, `TransactionDate`, `TransactionMonth`.
- **Client Information**: `IsVATRegistered`, `Citizenship`, `LegalType`, `Title`, `Language`, `Bank`, `AccountType`, `MaritalStatus`, `Gender`.
- **Client Location**: `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`.
- **Car Insured**: `ItemType`, `Mmcode`, `VehicleType`, `RegistrationYear`, `Make`, `Model`, `Cylinders`, `Cubiccapacity`, `Kilowatts`, `Bodytype`, `NumberOfDoors`, `VehicleIntroDate`, `CustomValueEstimate`, `AlarmImmobiliser`, `TrackingDevice`, `CapitalOutstanding`, `NewVehicle`, `WrittenOff`, `Rebuilt`, `Converted`, `CrossBorder`, `NumberOfVehiclesInFleet`.
- **Plan Information**: `SumInsured`, `TermFrequency`, `CalculatedPremiumPerTerm`, `ExcessSelected`, `CoverCategory`, `CoverType`, `CoverGroup`, `Section`, `Product`, `StatutoryClass`, `StatutoryRiskType`.
- **Payment & Claim**: `TotalPremium`, `TotalClaims`.

## Learning Outcomes

Throughout this project, the following learning outcomes will be emphasized:

- Understanding and extracting insights from complex data through various techniques, algorithms, statistical distributions, sampling, and visualization.
- Comprehending data structures and algorithms in EDA and machine learning pipelines.
- Writing modular and object-oriented Python code, including Python package building.
- Applying statistical models for A/B test analysis (logistic regression, chi-squared tests, t-tests).
- Designing and implementing robust A/B tests (sample size, control/test groups, success metrics).
- Managing and documenting versions of datasets and analysis results using DVC.

---

## 1: Project Planning - EDA & Stats

This task focused on establishing the foundational understanding of the car insurance claim dataset, assessing its quality, and uncovering initial patterns in risk and profitability through Exploratory Data Analysis (EDA).

### Methodology

#### 1. Data Acquisition and Initial Overview

The dataset was obtained as a pipe-separated values (.txt) file and subsequently converted to a standard `.csv` file for processing. It contains approximately 1.5 million rows of historical car insurance claim data. Initial inspection revealed a comprehensive set of features categorized across policy, client, location, vehicle, plan, and payment/claim information.

#### 2. Data Structure and Quality Assessment

A thorough assessment of data quality was performed to ensure reliability for subsequent analysis:

- **Data Types**: Columns were reviewed and converted to appropriate data types. Notably, the `TransactionDate` column was successfully converted to a datetime format for temporal analysis.
- **Missing Values**: Analysis of missing values showed varying levels of completeness. Critical financial columns like `TotalPremium` and `TotalClaims` were largely complete. However, several vehicle-related attributes (`Cylinders`, `Cubiccapacity`, `Kilowatts`, `Bodytype`, `NumberOfDoors`, `AlarmImmobiliser`, `TrackingDevice`) contained a significant percentage of missing data, which will require careful imputation or strategic handling in later modeling phases.
- **Outlier Detection**: Box plots generated for key numerical features such as `TotalPremium`, `TotalClaims`, and `CustomValueEstimate` revealed the presence of numerous outliers. These extreme values are significant and will be considered during data preprocessing for modeling, potentially requiring capping or transformation to prevent skewing analytical results.

#### 3. Data Summarization and Univariate Analysis

Descriptive statistics were calculated for numerical features to understand their central tendency and variability. Histograms were plotted to visualize the distributions of numerical columns (e.g., `TotalPremium`, `TotalClaims`, `CustomValueEstimate`), showing typical right-skewed distributions for financial data. Bar charts were used to illustrate the distributions of categorical variables (e.g., `Province`, `VehicleType`, `Gender`, `Product`), providing insights into the composition of the policy portfolio.

#### 4. Bivariate and Multivariate Analysis

Relationships and trends within the data were explored through bivariate and multivariate analysis:

- **Overall Loss Ratio**: The overall portfolio Loss Ratio (TotalClaims / TotalPremium) was computed. Policies with `TotalPremium` equal to zero were handled to prevent division by zero, assigning a loss ratio of 0 in such cases.
- **Segmented Loss Ratios**:
  - **By Province**: Significant variations in average loss ratio were observed across South African provinces. For example, Gauteng and KwaZulu-Natal generally exhibited higher loss ratios compared to Western Cape, suggesting regional risk differentiation.
  - **By Vehicle Type**: Distinct average loss ratio profiles were identified for different `VehicleType` categories (e.g., Sedans, SUVs, Bakkies), indicating that vehicle characteristics are likely important risk factors.
  - **By Gender**: Minor differences in average loss ratio were noted between genders, though formal hypothesis testing is required to confirm statistical significance.
- **Temporal Trends**: Monthly aggregated `TotalPremium` and `TotalClaims` were plotted over the 18-month period. The trends showed relative stability, indicating no major shifts in claims or premium volumes over time.
- **Correlations**: A heatmap of the correlation matrix for selected numerical features (e.g., `TotalPremium`, `TotalClaims`, `SumInsured`, `CustomValueEstimate`) was generated. This revealed expected positive correlations among financial values and helped identify features that might influence claims.

### Key Insights and Actionable Observations

The EDA phase yielded several critical insights for AlphaCare's business strategy:

- **Geographic Premium Adjustment**: The clear disparities in loss ratios across provinces (e.g., Gauteng vs. Western Cape) provide strong preliminary evidence that location is a significant risk differentiator. This insight suggests that geographically adjusted premiums could be a viable strategy to attract lower-risk clients in specific regions.
- **Vehicle-Specific Pricing**: Variations in loss ratios by `VehicleType` indicate that refining premium calculations based on specific vehicle characteristics (beyond just value) could optimize pricing and competitiveness.
- **Outlier Management**: The presence of high-value outliers in `TotalClaims` and `CustomValueEstimate` will necessitate robust data cleaning, transformation, or model-agnostic interpretation techniques to ensure that predictive models are not unduly influenced by extreme events.
- **Foundation for Hypothesis Testing**: The observed differences in loss ratios across various segments (province, vehicle type, gender) provide direct empirical backing for the null hypotheses to be tested in Task 3.

These insights were visually presented through a series of plots saved in the `plots/eda/` directory, including histograms, bar charts of distributions and loss ratios, line plots of temporal trends, and box plots for outlier detection.

---

## 2: Data Version Control (DVC)

This task focuses on establishing a reproducible and auditable data pipeline, a critical requirement in regulated industries like insurance. Data Version Control (DVC) is employed to manage and version our datasets, ensuring that any analysis or model result can be recreated with its exact corresponding data state.

### Methodology

#### 1. DVC Installation and Initialization

- DVC was installed using `pip` within the project's virtual environment.
- The `dvc init` command was executed in the repository root to initialize DVC, which created the necessary `.dvc/` directory and integrated with Git's `.gitignore` to prevent tracking of DVC's cache.

#### 2. Local Remote Storage Configuration

- A dedicated local directory, `~/dvc_storage/alphacare_data`, was created outside the Git repository to serve as the DVC remote storage.
- This directory was added as the default DVC remote using `dvc remote add -d localstorage ...`, establishing the location where DVC stores the actual data file versions.

#### 3. Data Tracking with DVC

- The primary dataset, `car_insurance.csv` (located in the `data/` subdirectory), was added to DVC tracking using `dvc add data/car_insurance.csv`.
- This action resulted in:
  - The actual `car_insurance.csv` file being managed by DVC's cache.
  - A small `data/car_insurance.csv.dvc` file being created, which contains metadata (like a content hash) that Git now tracks. This `.dvc` file acts as a pointer to the specific version of the data.
  - Automatic inclusion of `data/car_insurance.csv` in `.gitignore` to prevent Git from tracking the large data file directly.

#### 4. Data Versioning (Simulated)

- To demonstrate data versioning, a minor, non-impactful change was simulated in the `car_insurance.csv` file.
- Following the change, `dvc add data/car_insurance.csv` was run again, causing DVC to detect the modification, create a new data version in its cache, and update the `data/car_insurance.csv.dvc` file with the new hash. This process effectively creates a new, trackable version of the dataset.

#### 5. Pushing Data to Local Remote

- Finally, `dvc push` was executed to synchronize the DVC cache with the configured local remote storage (`~/dvc_storage/alphacare_data`). This command ensures that the actual data files corresponding to the `.dvc` files are stored in the designated remote location, making them available for others (or for later retrieval) independent of the Git repository.

This DVC setup ensures that our data inputs are rigorously version-controlled, allowing for full reproducibility of all analysis and model results, a critical requirement for auditing and regulatory compliance in the insurance domain.

## Task 3: A/B Hypothesis Testing

Task 3 involved statistically validating key hypotheses about car insurance risk drivers. This rigorous A/B testing approach is crucial for establishing data-backed foundations for AlphaCare's new segmentation and marketing strategies.

### 3.1 Key Metrics Defined

For this analysis, "risk" and "profitability" were quantified using the following metrics:

- **Claim Frequency**: Defined as the proportion of policies that incurred at least one claim. This is a binary metric (0 = no claim, 1 = claim occurred).
  $$ \text{Claim Frequency} = \frac{\text{Number of Policies with Claims}}{\text{Total Number of Policies}} $$
- **Claim Severity**: Defined as the average amount of a claim, calculated _only for policies where a claim actually occurred_ ($\text{TotalClaims} > 0$).
  $$ \text{Claim Severity} = \frac{\text{Total Claims Amount}}{\text{Number of Policies with Claims}} \quad (\text{for policies with } \text{TotalClaims} > 0) $$
- **Margin (Profit)**: Defined as the direct profitability per policy, calculated as the difference between the total premium received and the total claims paid.
  $$ \text{Margin} = \text{TotalPremium} - \text{TotalClaims} $$

### 3.2 Methodology: Statistical Testing

For each null hypothesis ($\text{H}_0$), a segmentation approach was applied to create comparison groups (analogous to A/B testing's control and test groups). Appropriate statistical tests were then conducted to evaluate the impact of the tested features on the defined KPIs. A significance level (alpha, $\alpha$) of 0.05 was consistently used.

- **For Categorical Outcomes (e.g., Claim Frequency)**: The **Chi-squared ($\chi^2$) test for independence** was employed. This test determines if there's a statistically significant association between two categorical variables (e.g., Province and Claim Occurred).

  - If $\text{p-value} < 0.05$: The null hypothesis is **rejected**, suggesting a statistically significant difference in claim frequency between the groups.
  - If $\text{p-value} \ge 0.05$: We **fail to reject** the null hypothesis, suggesting no statistically significant difference.

- **For Numerical Outcomes (e.g., Claim Severity, Margin)**: The **Independent Samples t-test (specifically Welch's t-test)** was used. This test compares the means of two independent groups to determine if they are significantly different. Welch's t-test is robust to unequal variances between groups, which is common in real-world data.
  - If $\text{p-value} < 0.05$: The null hypothesis is **rejected**, suggesting a statistically significant difference in the mean of the numerical metric (Claim Severity or Margin) between the groups.
  - If $\text{p-value} \ge 0.05$: We **fail to reject** the null hypothesis, suggesting no statistically significant difference.

The analysis involved comparing the most prevalent categories for features with more than two classes (e.g., selecting the top two provinces/zip codes by policy volume for comparison).

### 3.3 Test Execution and Results

The `hypothesis_testing.py` script was executed to perform the outlined statistical tests. Below are the findings for each null hypothesis:

#### Hypothesis 1: $H_0$: There are no risk differences across provinces.

- **Claim Frequency**:

  - **Result**: [**State your p-value here, e.g., p = 0.0001**]
  - **Conclusion**: [**Reject/Fail to Reject H0**]
  - **Business Recommendation**: [**Based on your script's output, interpret the result. E.g., "We reject the null hypothesis ($p < 0.01$). Our analysis shows that policies in [Province A, e.g., Gauteng] have a significantly higher claim frequency (X%) compared to policies in [Province B, e.g., Western Cape] (Y%). This strongly indicates that geographical location is a key risk differentiator. AlphaCare should consider implementing a regional risk adjustment factor in its premium pricing strategy to better reflect provincial risk profiles."**]

- **Claim Severity**:
  - **Result**: [**State your p-value here, e.g., p = 0.15**]
  - **Conclusion**: [**Reject/Fail to Reject H0**]
  - **Business Recommendation**: [**Based on your script's output, interpret the result. E.g., "We fail to reject the null hypothesis ($p > 0.05$). While there may be minor differences, the statistical evidence does not support a significant difference in the average claim amount (severity) between policies in [Province A] and [Province B]. Therefore, focusing solely on claim severity for provincial premium adjustments may not be justified based on this dataset."**]

#### Hypothesis 2: $H_0$: There are no risk differences between zip codes.

- **Claim Frequency**:

  - **Result**: [**State your p-value here**]
  - **Conclusion**: [**Reject/Fail to Reject H0**]
  - **Business Recommendation**: [**Interpret this. E.g., "We reject the null hypothesis ($p < 0.01$). Policies originating from Postal Code [Zip A] exhibit a significantly higher claim frequency compared to Postal Code [Zip B]. This highlights micro-geographical risk variations, suggesting opportunities for hyper-localized pricing strategies."**]

- **Claim Severity**:
  - **Result**: [**State your p-value here**]
  - **Conclusion**: [**Reject/Fail to Reject H0**]
  - **Business Recommendation**: [**Interpret this.**]

#### Hypothesis 3: $H_0$: There are no significant margin (profit) differences between zip codes.

- **Result**: [**State your p-value here**]
- **Conclusion**: [**Reject/Fail to Reject H0**]
- **Business Recommendation**: [**Interpret this. E.g., "We reject the null hypothesis ($p < 0.01$). The average profit margin per policy in Postal Code [Zip A] (e.g., R500) is significantly different from Postal Code [Zip B] (e.g., R350). This suggests that some zip codes are inherently more profitable than others, providing insights for targeted marketing and premium adjustments to optimize profitability."**]

#### Hypothesis 4: $H_0$: There are no significant risk differences between Women and Men.

- **Claim Frequency**:

  - **Result**: [**State your p-value here**]
  - **Conclusion**: [**Reject/Fail to Reject H0**]
  - **Business Recommendation**: [**Interpret this. E.g., "We fail to reject the null hypothesis ($p > 0.05$). The analysis does not provide statistically significant evidence of a difference in claim frequency between male and female policyholders. Therefore, current data does not support adjusting premiums based solely on gender for claim frequency."**]

- **Claim Severity**:
  - **Result**: [**State your p-value here**]
  - **Conclusion**: [**Reject/Fail to Reject H0**]
  - **Business Recommendation**: [**Interpret this.**]

### 3.4 Summary of Findings and Recommendations

[**Once all tests are run, provide a concise summary of the most impactful findings. Highlight which factors (province, zip code, gender) were found to be statistically significant risk/profit drivers and what specific actions AlphaCare should take based on these insights.**]
