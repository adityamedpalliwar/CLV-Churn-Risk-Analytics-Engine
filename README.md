# Customer Lifetime Value (CLV) and Churn Risk Analytics Engine

A beginner-level data science college project combining Machine Learning and Statistical Inference to analyze customer churn and customer lifetime value using the IBM Telco Customer Churn dataset.

---

## Project Structure

```text
CLV-Churn-Risk-Analytics-Engine/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original IBM Telco Customer Churn dataset (preserved)
├── telco_churn_cleaned.csv                # Cleaned & preprocessed dataset for EDA/modeling
├── load_data.py                           # Phase 1: Data loading and initial inspection
├── explore_data.py                        # Phase 2: Data understanding & statistical exploration
├── clean_data.py                          # Phase 3: Basic data cleaning and preprocessing
├── eda.py                                 # Phase 4: Exploratory Data Analysis & visual generation
├── main.py                                # Main pipeline orchestration script
├── plots/                                 # Generated EDA visualizations
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_churn_by_tenure.png
│   ├── 04_churn_by_monthly_charges.png
│   ├── 05_churn_by_payment_method.png
│   ├── 06_tenure_distribution.png
│   └── 07_monthly_charges_distribution.png
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

*(Dependencies: `pandas`, `matplotlib`, `seaborn`)*

---

## Running the Project

Run the entire end-to-end pipeline (Phases 1 through 4) via the main script:

```bash
python main.py
```

You can also run modules individually:
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

---

## Project Phases

### Phase 1: Project Setup, Data Loading, and Basic Inspection
- Loads the dataset via relative path with clear `FileNotFoundError` handling.
- Displays basic shape, preview rows, column names, and data types with non-null counts.

### Phase 2: Basic Data Understanding & Statistical Exploration
- **1. Dataset Dimensions:** 7,043 rows, 21 columns, and 7,043 unique customer IDs.
- **2. Column Analysis & Categorization:** Classifies columns into numerical features (3) and categorical features (18).
- **3. Missing-Value Analysis:** Evaluates null counts for every column and highlights implicit missing values.
- **4. Duplicate Analysis:** Checks for duplicate rows across the dataset without altering data.
- **5. Target Analysis (Churn):** Computes class frequency counts, percentages, and the overall churn rate (26.54%).
- **6. Statistical Summary:** Computes descriptive statistics for numerical features.

### Phase 3: Basic Data Cleaning & Preprocessing
- **1. TotalCharges Conversion & Investigation:** Coerced to numeric; identified 11 rows where `tenure == 0` and `Churn == 'No'`; imputed with `0.0` preserving all customer records.
- **2. Missing Values Treatment:** Verified no other columns have missing values.
- **3. Duplicate Records Analysis:** 0 duplicate rows found; 0 removed.
- **4. Feature Preparation & Separation:** Retained `customerID` separately, encoded `Churn` into binary (1/0), and separated numerical (4) and categorical (15) features.
- **5. Dataset Export:** Exported `telco_churn_cleaned.csv` while keeping original dataset untouched.

### Phase 4: Exploratory Data Analysis (EDA)
- **1. Churn Distribution (`01_churn_distribution.png`):** Shows overall churn breakdown (73.46% retained vs. 26.54% churned).
- **2. Churn by Contract Type (`02_churn_by_contract.png`):** Compares churn rates across Month-to-month (42.71%), One-year (11.27%), and Two-year (2.83%) contracts.
- **3. Churn by Tenure (`03_churn_by_tenure.png`):** Boxplot showing churners leave early (median tenure of 10 months vs. 38 months for retained customers).
- **4. Churn by Monthly Charges (`04_churn_by_monthly_charges.png`):** Boxplot showing churners have higher monthly bills (median \$79.65 vs. \$64.43).
- **5. Churn by Payment Method (`05_churn_by_payment_method.png`):** Highlights high churn in Electronic Check users (45.29%) compared to automated payment methods (~15-17%).
- **6. Tenure Distribution (`06_tenure_distribution.png`):** Histogram/KDE displaying a bimodal customer base.
- **7. Monthly Charges Distribution (`07_monthly_charges_distribution.png`):** Histogram/KDE displaying distinct price tiers.
