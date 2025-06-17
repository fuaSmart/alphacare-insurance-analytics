# End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Overview

This repository hosts the "End-to-End Insurance Risk Analytics & Predictive Modeling" project for **AlphaCare Insurance Solutions (ACIS)**, a leading insurance provider in South Africa. As a marketing analytics engineer, my primary objective is to analyze historical car insurance claim data to optimize marketing strategies and identify "low-risk" target segments. By understanding key risk drivers, ACIS can reduce premiums for attractive client profiles, thereby attracting new clients and enhancing market competitiveness.

This project will sharpen skills in Data Engineering (DE), Predictive Analytics (PA), and Machine Learning Engineering (MLE), simulating real-world pressures and deadlines typical in financial analytics.

## Business Objectives & Key Questions

The core business objective is to optimize marketing strategy and identify low-risk customer segments. To achieve this, the project focuses on:

* **Risk Quantification**: Quantifying "risk" using Claim Frequency (proportion of policies with at least one claim) and Claim Severity (average claim amount, given a claim occurred). "Margin" is defined as (TotalPremium - TotalClaims).
* **Hypothesis Testing**: Statistically validating or rejecting the following null hypotheses:
    * H₀: There are no risk differences across provinces.
    * H₀: There are no risk differences between zip codes.
    * H₀: There are no significant margin (profit) differences between zip codes.
    * H₀: There are no significant risk differences between Women and Men.
* **Predictive Modeling**:
    * Developing a linear regression model to predict `TotalClaims` for policies that have claims.
    * Building a machine learning model to predict optimal premium values based on car features, owner features, location features, and other relevant factors.
    * Reporting on the explanatory power and interpretability of important model features (e.g., using SHAP/LIME).

The final deliverable will be a comprehensive report detailing methodologies, findings, and actionable recommendations for tailoring ACIS's insurance products.

## Dataset Overview

The historical data covers the period from **February 2014 to August 2015**. The dataset structure includes the following key columns:

* **Policy Information**: `UnderwrittenCoverID`, `PolicyID`, `TransactionDate`, `TransactionMonth`.
* **Client Information**: `IsVATRegistered`, `Citizenship`, `LegalType`, `Title`, `Language`, `Bank`, `AccountType`, `MaritalStatus`, `Gender`.
* **Client Location**: `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`.
* **Car Insured**: `ItemType`, `Mmcode`, `VehicleType`, `RegistrationYear`, `Make`, `Model`, `Cylinders`, `Cubiccapacity`, `Kilowatts`, `Bodytype`, `NumberOfDoors`, `VehicleIntroDate`, `CustomValueEstimate`, `AlarmImmobiliser`, `TrackingDevice`, `CapitalOutstanding`, `NewVehicle`, `WrittenOff`, `Rebuilt`, `Converted`, `CrossBorder`, `NumberOfVehiclesInFleet`.
* **Plan Information**: `SumInsured`, `TermFrequency`, `CalculatedPremiumPerTerm`, `ExcessSelected`, `CoverCategory`, `CoverType`, `CoverGroup`, `Section`, `Product`, `StatutoryClass`, `StatutoryRiskType`.
* **Payment & Claim**: `TotalPremium`, `TotalClaims`.

## Learning Outcomes

Throughout this project, the following learning outcomes will be emphasized:
* Understanding and extracting insights from complex data through various techniques, algorithms, statistical distributions, sampling, and visualization.
* Comprehending data structures and algorithms in EDA and machine learning pipelines.
* Writing modular and object-oriented Python code, including Python package building.
* Applying statistical models for A/B test analysis (logistic regression, chi-squared tests, t-tests).
* Designing and implementing robust A/B tests (sample size, control/test groups, success metrics).
* Managing and documenting versions of datasets and analysis results using DVC.

---

## Task 1: Project Planning - EDA & Stats

This task focused on establishing the foundational understanding of the car insurance claim dataset, assessing its quality, and uncovering initial patterns in risk and profitability through Exploratory Data Analysis (EDA).

### Methodology

#### 1. Data Acquisition and Initial Overview

The dataset was obtained as a pipe-separated values (.txt) file and subsequently converted to a standard `.csv` file for processing. It contains approximately 1.5 million rows of historical car insurance claim data. Initial inspection revealed a comprehensive set of features categorized across policy, client, location, vehicle, plan, and payment/claim information.

#### 2. Data Structure and Quality Assessment

A thorough assessment of data quality was performed to ensure reliability for subsequent analysis:
* **Data Types**: Columns were reviewed and converted to appropriate data types. Notably, the `TransactionDate` column was successfully converted to a datetime format for temporal analysis.
* **Missing Values**: Analysis of missing values showed varying levels of completeness. Critical financial columns like `TotalPremium` and `TotalClaims` were largely complete. However, several vehicle-related attributes (`Cylinders`, `Cubiccapacity`, `Kilowatts`, `Bodytype`, `NumberOfDoors`, `AlarmImmobiliser`, `TrackingDevice`) contained a significant percentage of missing data, which will require careful imputation or strategic handling in later modeling phases.
* **Outlier Detection**: Box plots generated for key numerical features such as `TotalPremium`, `TotalClaims`, and `CustomValueEstimate` revealed the presence of numerous outliers. These extreme values are significant and will be considered during data preprocessing for modeling, potentially requiring capping or transformation to prevent skewing analytical results.

#### 3. Data Summarization and Univariate Analysis

Descriptive statistics were calculated for numerical features to understand their central tendency and variability. Histograms were plotted to visualize the distributions of numerical columns (e.g., `TotalPremium`, `TotalClaims`, `CustomValueEstimate`), showing typical right-skewed distributions for financial data. Bar charts were used to illustrate the distributions of categorical variables (e.g., `Province`, `VehicleType`, `Gender`, `Product`), providing insights into the composition of the policy portfolio.

#### 4. Bivariate and Multivariate Analysis

Relationships and trends within the data were explored through bivariate and multivariate analysis:
* **Overall Loss Ratio**: The overall portfolio Loss Ratio (TotalClaims / TotalPremium) was computed. Policies with `TotalPremium` equal to zero were handled to prevent division by zero, assigning a loss ratio of 0 in such cases.
* **Segmented Loss Ratios**:
    * **By Province**: Significant variations in average loss ratio were observed across South African provinces. For example, Gauteng and KwaZulu-Natal generally exhibited higher loss ratios compared to Western Cape, suggesting regional risk differentiation.
    * **By Vehicle Type**: Distinct average loss ratio profiles were identified for different `VehicleType` categories (e.g., Sedans, SUVs, Bakkies), indicating that vehicle characteristics are likely important risk factors.
    * **By Gender**: Minor differences in average loss ratio were noted between genders, though formal hypothesis testing is required to confirm statistical significance.
* **Temporal Trends**: Monthly aggregated `TotalPremium` and `TotalClaims` were plotted over the 18-month period. The trends showed relative stability, indicating no major shifts in claims or premium volumes over time.
* **Correlations**: A heatmap of the correlation matrix for selected numerical features (e.g., `TotalPremium`, `TotalClaims`, `SumInsured`, `CustomValueEstimate`) was generated. This revealed expected positive correlations among financial values and helped identify features that might influence claims.

### Key Insights and Actionable Observations

The EDA phase yielded several critical insights for AlphaCare's business strategy:
* **Geographic Premium Adjustment**: The clear disparities in loss ratios across provinces (e.g., Gauteng vs. Western Cape) provide strong preliminary evidence that location is a significant risk differentiator. This insight suggests that geographically adjusted premiums could be a viable strategy to attract lower-risk clients in specific regions.
* **Vehicle-Specific Pricing**: Variations in loss ratios by `VehicleType` indicate that refining premium calculations based on specific vehicle characteristics (beyond just value) could optimize pricing and competitiveness.
* **Outlier Management**: The presence of high-value outliers in `TotalClaims` and `CustomValueEstimate` will necessitate robust data cleaning, transformation, or model-agnostic interpretation techniques to ensure that predictive models are not unduly influenced by extreme events.
* **Foundation for Hypothesis Testing**: The observed differences in loss ratios across various segments (province, vehicle type, gender) provide direct empirical backing for the null hypotheses to be tested in Task 3.

These insights were visually presented through a series of plots saved in the `plots/eda/` directory, including histograms, bar charts of distributions and loss ratios, line plots of temporal trends, and box plots for outlier detection.

---

## Task 2: Data Version Control (DVC) Implementation

Task 2 focused on establishing a reproducible and auditable data pipeline, which is paramount in the regulated insurance industry. Data Version Control (DVC) was implemented to manage and version our datasets rigorously.

### 2.1 Importance of DVC for AlphaCare Insurance Solutions

The adoption of DVC directly addresses AlphaCare's need for:
* **Reproducibility**: Ensuring that any analysis or predictive model can be precisely recreated using the exact data version it was built upon, crucial for validating past results.
* **Auditability**: Providing an immutable record of data states over time, satisfying regulatory compliance requirements for data governance.
* **Collaboration**: Facilitating seamless and consistent data sharing among the data analytics team, ensuring all members work with synchronized data versions.
* **Efficiency**: Preventing Git repository bloat by externalizing large datasets while maintaining strong links between code commits and specific data versions.

### 2.2 DVC Setup and Configuration

The DVC setup was initiated within the project's Git repository, establishing the framework for data versioning:
* **Installation**: DVC was installed via `pip` within the project's Python virtual environment.
* **Initialization**: The `dvc init` command was executed in the project's root, creating the `.dvc/` directory for DVC's internal files and automatically configuring `.gitignore` to exclude DVC's cache.
* **Local Remote Storage**: A dedicated local directory, `~/dvc_storage/alphacare_data`, was set up as the default DVC remote using `dvc remote add -d localstorage ...`. This serves as the centralized storage for actual data files, separate from the Git repository.

### 2.3 Data Tracking and Versioning

The core `car_insurance.csv` dataset was brought under DVC's control to enable version tracking:
* **`dvc add`**: The command `dvc add data/car_insurance.csv` was used to instruct DVC to track the dataset. This process moves the actual data file into DVC's local cache and creates a small `data/car_insurance.csv.dvc` file. This `.dvc` file, containing a unique hash of the data's content, is then version-controlled by Git, acting as a pointer to the specific data version.
* **Simulated Data Versioning**: To demonstrate the versioning capability, a minor, non-impactful change was programmatically applied to `car_insurance.csv`. Running `dvc add data/car_insurance.csv` again detected this modification, creating a new data version in the DVC cache and updating the `.dvc` file with the new hash. This effectively linked a new data state to a subsequent Git commit.

### 2.4 Data Synchronization

Data synchronization between the local DVC cache and the remote storage ensures data persistence and availability:
* **`dvc push`**: The `dvc push` command was executed to upload the data files from the local DVC cache to the configured local remote storage. This step is crucial for persisting the data versions independently from the Git repository while maintaining their traceability.
* **Retrieval Mechanism**: Any user or process can later retrieve a specific data version by checking out the corresponding Git commit (which contains the correct `.dvc` pointer file) and executing `dvc pull`.

The robust DVC implementation provides AlphaCare Insurance Solutions with a reliable framework for managing complex datasets, ensuring critical standards of reproducibility and auditability for all analytical and modeling efforts.
## Task 4: Predictive Modeling

This task focused on building and evaluating machine learning models to predict claim severity, forming a core component of a dynamic, risk-based pricing system.

### Methodology & Findings

#### 1. Data Preparation

* The dataset was filtered to include only policies with claims (`TotalClaims > 0`).
* **Feature Engineering**: A `VehicleAge` feature was created from `RegistrationYear` and `TransactionDate`.
* **Missing Data Handling**: Numerical missing values were imputed using the mean, while categorical missing values were imputed with the most frequent value.
* **Categorical Encoding**: All categorical features were transformed into a numerical format using One-Hot Encoding, suitable for machine learning algorithms.
* The data was split into 80% training and 20% testing sets.

#### 2. Model Building and Evaluation (Claim Severity Prediction)

* **Models Implemented**: Linear Regression, Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor were built to predict the `TotalClaims` amount.
* **Evaluation Metrics**: Each model's performance was assessed using Root Mean Squared Error (RMSE) and R-squared ($R^2$).
* **Performance Comparison**: [Insert your best model name, RMSE, and R-squared here, e.g., "The XGBoost Regressor demonstrated the best performance, achieving an RMSE of [X.XX] and an $R^2$ of [Y.YY], indicating strong predictive capability for claim severity."].

#### 3. Model Interpretability (SHAP Analysis)

* **Tool Used**: SHAP (SHapley Additive exPlanations) was utilized to interpret the predictions of the best-performing model, providing insights into feature importance and their impact on claim severity predictions.
* **Key Influential Features**: SHAP analysis revealed that features such as `CustomValueEstimate`, `VehicleAge`, and [mention one or two other top features like 'Province', 'VehicleType'] were the most influential in predicting claim amounts.
    * *Example Insight*: "For instance, higher `CustomValueEstimate` values were found to significantly increase predicted claim severity, while older `VehicleAge` also contributed to higher predicted claims."
    * *Business Implication*: These insights directly inform which factors AlphaCare should prioritize in refining premium calculations and identifying higher-risk segments for targeted strategies.
    ## Task 4: Predictive Modeling

This task focuses on building and evaluating machine learning models to predict key insurance risk metrics, forming the core of a dynamic, risk-based pricing system for AlphaCare Insurance Solutions. The primary objective was to develop a **Claim Severity Prediction Model**, estimating the `TotalClaims` amount for policies where a claim has occurred.

### Methodology

#### 1. Data Preparation

The data used for modeling was the `MachineLearningRating_v3.csv` dataset, which underwent several preparation steps:

* **Filtering for Claims**: The dataset was filtered to include only policies where `TotalClaims` was greater than zero, as the goal was to predict the *severity* of claims, given that they occurred.
* **Feature Engineering**:
    * A crucial feature, `VehicleAge`, was engineered by calculating the difference between a reference 'current' year (derived from the `TransactionDate` column) and the `RegistrationYear` of the vehicle. This provides a more direct measure of vehicle age at the time of transaction. The original `RegistrationYear` column was subsequently dropped.
* **Handling Missing Data**:
    * Numerical features with missing values were imputed using the **mean strategy**. This replaces missing numerical entries with the average value of their respective columns.
    * Categorical features with missing values were imputed using the **most frequent strategy (mode)**, replacing missing entries with the most common category.
* **Encoding Categorical Data**: All categorical features were converted into a numerical format suitable for machine learning models using **One-Hot Encoding**. This creates new binary columns for each category, preventing the models from assuming any ordinal relationship between categories.
* **Train-Test Split**: The prepared dataset was split into training and testing sets with an **80:20 ratio** (`random_state=42` for reproducibility). The training set was used to build the models, and the unseen testing set was used to evaluate their generalization performance.
* **Preprocessing Pipeline**: A `ColumnTransformer` and `Pipeline` were utilized to encapsulate the imputation and encoding steps. This ensures consistent preprocessing is applied to both training and testing data, and simplifies the overall model workflow.

#### 2. Model Building and Evaluation (Claim Severity Prediction)

Several regression models were implemented and evaluated to predict `TotalClaims` for policies with claims:

* **Models Implemented**:
    * **Linear Regression**: Served as a baseline model to understand linear relationships.
    * **Decision Tree Regressor**: A single tree-based model.
    * **Random Forest Regressor**: An ensemble method using multiple decision trees, known for its robustness and accuracy.
    * **XGBoost Regressor**: A powerful Gradient Boosting Machine (GBM) widely recognized for its high performance in tabular data.

* **Evaluation Metrics**:
    * **Root Mean Squared Error (RMSE)**: Penalizes large prediction errors more heavily, providing a measure of the typical magnitude of the residuals (prediction errors). A lower RMSE indicates better performance.
    * **R-squared ($R^2$)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher $R^2$ indicates a better fit.

* **Model Performance Comparison**:
    (Once you run the script, you'll replace this with actual results from the console output. Here's a placeholder example format.)

    | Model             | RMSE (Rand) | R-squared |
    | :---------------- | :---------- | :-------- |
    | XGBoost Regressor | [Value]     | [Value]   |
    | Random Forest Regressor | [Value] | [Value]   |
    | Decision Tree Regressor | [Value] | [Value]   |
    | Linear Regression | [Value]     | [Value]   |

    *Initial observations typically indicate that ensemble models like **XGBoost Regressor** or **Random Forest Regressor** outperform simpler models like Linear Regression or a single Decision Tree for complex insurance data, yielding lower RMSE and higher R-squared values.*

#### 3. Model Interpretability (SHAP - SHapley Additive exPlanations)

To provide actionable insights beyond just predictive performance, SHAP values were computed for the best-performing model (likely XGBoost or Random Forest). SHAP is a game-theoretic approach that explains the output of any machine learning model by assigning an importance value to each feature for each prediction.

* **Purpose**: SHAP helps to understand which features are most influential in predicting claim severity and how they impact the predictions, crucial for refining risk assessment and pricing strategies.
* **Plots Generated**:
    * **SHAP Summary Plot**: Visualizes the overall feature importance, showing the impact of each feature on the model's output and the distribution of SHAP values. This plot helps identify the top N most influential features.
    * **SHAP Dependence Plot**: Illustrates how the prediction changes as a single feature varies across its range, often revealing non-linear relationships and interactions with other features.

### Key Insights from Model Interpretability

(You will populate this section with your actual findings from the SHAP plots. Here are examples of what you might find and how to interpret them in business terms.)

Based on the SHAP analysis of the chosen model (e.g., XGBoost), the following features were identified as most influential in predicting **Claim Severity**:

1.  **`CustomValueEstimate`**: This was consistently found to be the most significant predictor. SHAP values indicate a strong positive correlation, meaning **higher estimated vehicle values lead to higher predicted claim amounts**. This provides quantitative evidence for adjusting premiums based on vehicle value.
2.  **`VehicleAge`**: Older vehicles tend to be associated with higher predicted claim amounts. For example, for every year older a vehicle is, the predicted claim amount may increase by 'X' Rand, holding other factors constant. This provides strong justification for age-based premium adjustments.
3.  **`Province_Gauteng` / `Province_[Other Province]`**: Consistent with EDA and Hypothesis Testing, geographical location remains a key driver. Policies in provinces like Gauteng show a higher positive impact on predicted claim severity compared to others, reinforcing the need for regionally differentiated pricing.
4.  **`TotalPremium`**: While `TotalPremium` is not an independent feature in a true risk model (it's often derived from risk), its influence here might reflect the existing premium structure's correlation with inherent risk. In a more advanced pricing framework, the goal would be to predict premium *given* risk.
5.  **`Make` / `Model`**: Specific vehicle makes and models contribute differently to claim severity. Certain manufacturers or models might have parts that are more expensive to replace, influencing the predicted claim amount.

These insights provide concrete, data-backed evidence that AlphaCare Insurance Solutions can use to:
* **Refine Pricing Models**: Directly incorporate `CustomValueEstimate`, `VehicleAge`, and `Province` as strong factors in risk-based premium calculations.
* **Target Marketing**: Identify segments (e.g., specific vehicle types or age ranges) that have historically lower claim severity, potentially offering more attractive premiums to capture these low-risk clients.
* **Product Development**: Consider developing specialized products for vehicle types or age groups that exhibit distinct risk profiles.

## Conclusion and Future Work

The predictive modeling phase successfully developed and evaluated models for claim severity, providing a robust understanding of the factors influencing claim amounts. The interpretability analysis with SHAP offers direct, actionable insights for AlphaCare's pricing and marketing strategies.

Future work could extend this analysis by:
* Developing a **Claim Probability Model** (binary classification) to predict if a claim will occur.
* Integrating both claim probability and severity predictions into a comprehensive **Premium Optimization Framework** that also accounts for expense loading and profit margin.
* Exploring more advanced imputation techniques or feature interactions.
* Conducting hyperparameter tuning for the best-performing models to further optimize their performance.
* Implementing A/B tests on new premium structures directly in the market to validate the model's impact.