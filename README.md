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
