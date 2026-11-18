# Customer Lifetime Value (CLV) and Churn Risk Analytics Engine

A beginner-level data science college project combining Machine Learning and Statistical Inference to analyze customer churn and customer lifetime value using the IBM Telco Customer Churn dataset.

---

## Phase 1: Project Setup, Data Loading, and Basic Inspection

This phase establishes the foundational project structure, loads the local dataset using pandas via a relative path with error handling, and displays initial structural properties of the dataset.

### Project Structure

```text
CLV-Churn-Risk-Analytics-Engine/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Customer Churn dataset
├── load_data.py                           # Data loading and inspection module
├── main.py                                # Main execution script
├── requirements.txt                       # Project dependencies for Phase 1
└── README.md                              # Project documentation and setup guide
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

*(For Phase 1, the only required external library is `pandas`)*

---

## Running the Project

Run the project by executing the main script:

```bash
python main.py
```

Alternatively, you can test the data loader module directly:

```bash
python load_data.py
```

---

## Phase 1 Inspection Outputs

When you run `main.py`, the following inspections will be performed and printed to the terminal:
1. **Dataset Shape**: Number of rows and columns (7,043 rows, 21 columns).
2. **First 5 Rows**: A preview of the first five customer records.
3. **Column Names**: The full list of feature names present in the dataset.
4. **Data Types & Non-Null Counts**: Column-by-column data types and null check via pandas `info()`.
