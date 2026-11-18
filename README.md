# Customer Lifetime Value (CLV) and Churn Risk Analytics Engine

A beginner-level data science college project combining Machine Learning and Statistical Inference to analyze customer churn and customer lifetime value using the IBM Telco Customer Churn dataset.

---

## Project Structure

```text
CLV-Churn-Risk-Analytics-Engine/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Customer Churn dataset
├── load_data.py                           # Phase 1: Data loading and initial inspection
├── explore_data.py                        # Phase 2: Data understanding & statistical exploration
├── main.py                                # Main orchestration script
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

*(Currently requires `pandas`)*

---

## Running the Project

Run the entire pipeline (Phase 1 + Phase 2) via the main script:

```bash
python main.py
```

You can also run modules individually:
- To run data loading and initial inspection:
  ```bash
  python load_data.py
  ```
- To run data understanding and statistical exploration:
  ```bash
  python explore_data.py
  ```

---

## Project Phases

### Phase 1: Project Setup, Data Loading, and Basic Inspection
- Loads the dataset via relative path with clear `FileNotFoundError` handling.
- Displays basic shape, preview rows, column names, and data types with non-null counts.

### Phase 2: Basic Data Understanding & Statistical Exploration
- **1. Dataset Dimensions:** 7,043 rows, 21 columns, and 7,043 unique customer IDs.
- **2. Column Analysis & Categorization:** Classifies columns into numerical features (3) and categorical features (18).
- **3. Missing-Value Analysis:** Evaluates null counts for every column and highlights implicit missing values (such as blank spaces `' '` in `TotalCharges`).
- **4. Duplicate Analysis:** Checks for duplicate rows across the dataset without altering data.
- **5. Target Analysis (Churn):** Computes class frequency counts, percentages, and the overall churn rate (26.54%).
- **6. Statistical Summary:** Computes descriptive statistics (Mean, Median, Standard Deviation, Min, Max) for numerical features.
