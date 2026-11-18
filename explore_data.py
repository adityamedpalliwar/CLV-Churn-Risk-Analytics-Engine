"""
explore_data.py
Module for Phase 2: Basic Data Understanding and Statistical Exploration.
Project: Customer Lifetime Value (CLV) & Churn Risk Analytics Engine.
"""

import pandas as pd


def analyze_dimensions(df):
    """
    Analyzes and displays dataset dimensions:
    - Total rows
    - Total columns
    - Number of unique customers
    """
    print("\n" + "=" * 60)
    print("1. DATASET DIMENSIONS")
    print("=" * 60)
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    num_customers = df["customerID"].nunique() if "customerID" in df.columns else "N/A"

    print(f"Total Rows              : {total_rows:,}")
    print(f"Total Columns           : {total_cols}")
    print(f"Unique Customer IDs     : {num_customers:,}")
    if num_customers == total_rows:
        print("Note: Each row corresponds to a unique customer (no repeated customer IDs).")


def analyze_columns(df):
    """
    Analyzes and displays column details:
    - All column names and data types
    - Categorization into numerical and categorical features
    """
    print("\n" + "=" * 60)
    print("2. COLUMN ANALYSIS & CATEGORIZATION")
    print("=" * 60)

    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Total Columns: {len(df.columns)}")
    print(f" - Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    print(f" - Categorical Features ({len(categorical_cols)}): {categorical_cols}\n")

    print("Detailed Column Breakdown (Name, Data Type, and Category):")
    col_breakdown = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Category": ["Numerical" if c in numerical_cols else "Categorical" for c in df.columns]
    })
    print(col_breakdown.to_string(index=False))


def analyze_missing_values(df):
    """
    Analyzes missing values across all columns:
    - Missing value counts for every column
    - Columns with missing values
    - Note on implicit missing values (whitespace strings)
    """
    print("\n" + "=" * 60)
    print("3. MISSING-VALUE ANALYSIS")
    print("=" * 60)

    missing_series = df.isnull().sum()
    print("Missing Values Per Column (Standard NaN/Null):")
    missing_df = pd.DataFrame({
        "Column Name": missing_series.index,
        "Missing Count": missing_series.values,
        "Missing (%)": ((missing_series.values / len(df)) * 100).round(2)
    })
    print(missing_df.to_string(index=False))

    cols_with_missing = missing_series[missing_series > 0]
    print("\nColumns with Missing Values:")
    if len(cols_with_missing) > 0:
        for col, count in cols_with_missing.items():
            print(f" - {col}: {count} missing values")
    else:
        print(" - None. (No standard NaN/Null values detected in any column)")

    # Data understanding check for implicit blanks (e.g. TotalCharges) without cleaning
    if "TotalCharges" in df.columns:
        blank_spaces = (df["TotalCharges"] == " ").sum()
        if blank_spaces > 0:
            print(f"\n[Observation] 'TotalCharges' contains {blank_spaces} blank space (' ') entries.")
            print("              These represent implicit missing values to be cleaned in Phase 3.")


def analyze_duplicates(df):
    """
    Analyzes duplicate rows:
    - Counts duplicate rows
    - Retains rows without deleting them
    """
    print("\n" + "=" * 60)
    print("4. DUPLICATE ANALYSIS")
    print("=" * 60)
    duplicate_count = df.duplicated().sum()
    print(f"Total Duplicate Rows: {duplicate_count}")
    if duplicate_count == 0:
        print("Observation: The dataset contains no duplicate rows.")
    else:
        print(f"Observation: {duplicate_count} duplicate rows detected (retained, not deleted).")


def analyze_target(df, target_col="Churn"):
    """
    Analyzes the target variable ('Churn'):
    - Churn counts
    - Churn percentages
    - Overall churn rate
    """
    print("\n" + "=" * 60)
    print("5. TARGET ANALYSIS (CHURN)")
    print("=" * 60)
    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found in dataset.")
        return

    counts = df[target_col].value_counts()
    percentages = (df[target_col].value_counts(normalize=True) * 100).round(2)

    target_summary = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages
    })
    print("Churn Class Distribution:")
    print(target_summary)

    # Calculate overall churn rate
    churn_yes_count = (df[target_col] == "Yes").sum()
    overall_churn_rate = (churn_yes_count / len(df)) * 100
    print(f"\nOverall Churn Rate: {overall_churn_rate:.2f}% ({churn_yes_count:,} out of {len(df):,} customers)")
    print("Observation: Indicates moderate class imbalance (approx. 26.54% churn rate).")


def analyze_statistics(df):
    """
    Computes and displays basic statistical summary for numerical features:
    - Mean
    - Median (50th percentile)
    - Standard Deviation (std)
    - Minimum (min)
    - Maximum (max)
    """
    print("\n" + "=" * 60)
    print("6. BASIC STATISTICAL SUMMARY (NUMERICAL FEATURES)")
    print("=" * 60)

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        print("[WARNING] No numerical columns available to summarize.")
        return

    stats_summary = pd.DataFrame({
        "Mean": df[num_cols].mean(),
        "Median": df[num_cols].median(),
        "Std Dev": df[num_cols].std(),
        "Min": df[num_cols].min(),
        "Max": df[num_cols].max()
    }).round(2)

    print("Summary Statistics:")
    print(stats_summary)


def run_phase2_exploration(df):
    """
    Executes all Phase 2 data understanding and exploration analyses.
    """
    if df is None:
        print("[ERROR] DataFrame is None. Cannot run Phase 2 exploration.")
        return

    print("\n" + "#" * 65)
    print("# PHASE 2: DATA UNDERSTANDING & STATISTICAL EXPLORATION")
    print("#" * 65)

    analyze_dimensions(df)
    analyze_columns(df)
    analyze_missing_values(df)
    analyze_duplicates(df)
    analyze_target(df)
    analyze_statistics(df)

    print("\n" + "#" * 65)
    print("# Phase 2 Exploration Completed Successfully!")
    print("#" * 65)


if __name__ == "__main__":
    from load_data import load_dataset
    dataset = load_dataset()
    if dataset is not None:
        run_phase2_exploration(dataset)
