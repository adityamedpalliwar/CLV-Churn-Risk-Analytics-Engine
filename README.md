# Customer Lifetime Value (CLV) and Churn Risk Analytics Engine

A beginner-level data science college project combining Machine Learning and Statistical Inference to analyze customer churn and customer lifetime value using the IBM Telco Customer Churn dataset.

---

## Project Structure

```text
CLV-Churn-Risk-Analytics-Engine/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original IBM Telco Customer Churn dataset (preserved)
├── telco_churn_cleaned.csv                # Cleaned & preprocessed dataset for modeling
├── load_data.py                           # Phase 1: Data loading and initial inspection
├── explore_data.py                        # Phase 2: Data understanding & statistical exploration
├── clean_data.py                          # Phase 3: Basic data cleaning and preprocessing
├── eda.py                                 # Phase 4: Exploratory Data Analysis & visual generation
├── churn_model.py                         # Phase 5: Baseline Churn Prediction (Logistic Regression)
├── main.py                                # Main pipeline orchestration script
├── plots/                                 # Generated EDA visualizations (7 figures)
├── requirements.txt                       # Project dependencies
├── .gitignore                             # Git ignore rules for clean repository
└── README.md                              # Project documentation and guide
```

---

## Getting Started

### 1. Prerequisites
- Python 3.8 or higher installed on your system.

### 2. Install Dependencies
Open a terminal in the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Dependencies: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`)*

---

## Running the Project

Run the entire end-to-end pipeline (Phases 1 through 5) via the main script:

```bash
python main.py
```

You can also run individual modules directly:
- **Phase 1 (Data Loading & Initial Inspection):**
  ```bash
  python load_data.py
  ```
- **Phase 2 (Data Understanding & Statistical Exploration):**
  ```bash
  python explore_data.py
  ```
- **Phase 3 (Basic Data Cleaning & Preprocessing):**
  ```bash
  python clean_data.py
  ```
- **Phase 4 (Exploratory Data Analysis):**
  ```bash
  python eda.py
  ```
- **Phase 5 (Baseline Churn Prediction Modeling):**
  ```bash
  python churn_model.py
  ```

---

## Project Phases

### Phase 1: Project Setup, Data Loading, and Basic Inspection
- Loads the dataset via relative path with clear `FileNotFoundError` handling.
- Displays basic shape, preview rows, column names, and data types with non-null counts.

### Phase 2: Basic Data Understanding & Statistical Exploration
- **Dataset Dimensions:** 7,043 rows, 21 columns, and 7,043 unique customer IDs.
- **Column Categorization:** 3 numerical and 18 categorical features.
- **Missing-Value Analysis:** Evaluates null counts and identifies implicit whitespace values in `TotalCharges`.
- **Target Analysis (Churn):** 73.46% retained vs. 26.54% churn rate.
- **Statistical Summary:** Computes descriptive statistics for numerical features.

### Phase 3: Basic Data Cleaning & Preprocessing
- **TotalCharges Conversion:** Coerced to numeric; 11 rows where `tenure == 0` imputed with `0.0` to preserve all customers.
- **Duplicate Records:** 0 duplicates found.
- **Feature Preparation:** `customerID` retained separately; `Churn` mapped to binary target (1 = Yes, 0 = No); numerical (4) and categorical (15) features isolated.
- **Dataset Export:** Saved `telco_churn_cleaned.csv` while preserving original CSV untouched.

### Phase 4: Exploratory Data Analysis (EDA)
- Generated 7 high-resolution visualizations in `plots/`:
  - `01_churn_distribution.png`: Overall Churn breakdown.
  - `02_churn_by_contract.png`: Month-to-month contracts have 42.71% churn rate.
  - `03_churn_by_tenure.png`: Churners leave early (median 10 months vs. 38 months).
  - `04_churn_by_monthly_charges.png`: Higher monthly charges correlate with higher churn.
  - `05_churn_by_payment_method.png`: Electronic check users have 45.29% churn rate.
  - `06_tenure_distribution.png`: Bimodal tenure distribution.
  - `07_monthly_charges_distribution.png`: Multimodal pricing distribution.

### Phase 5: Baseline Churn Prediction (Logistic Regression)
- **Data Splitting:** 80% train (5,634 samples) and 20% test (1,409 samples) with `random_state=42`.
- **Pipeline & Leakage Prevention:** Built with `ColumnTransformer` (`StandardScaler` for numerical attributes and `OneHotEncoder(drop='first')` for categorical attributes) within a Scikit-learn `Pipeline`.
- **Model:** Logistic Regression (`max_iter=1000`, `random_state=42`).
- **Performance Highlights:**
  - **Accuracy:** 82.11%
  - **Precision:** 68.62%
  - **Recall:** 59.79%
  - **F1-Score:** 63.90%
  - **ROC-AUC:** 0.8622
- **Risk Scoring:** Outputs calibrated churn probabilities and risk classifications (`HIGH RISK` vs. `LOW/MODERATE`) for customer retention prioritization.
