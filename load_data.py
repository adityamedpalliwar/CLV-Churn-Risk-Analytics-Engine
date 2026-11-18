"""
load_data.py
Module for loading and inspecting the IBM Telco Customer Churn dataset.
Phase 1: Project Setup, Data Loading, and Basic Inspection.
"""

import pandas as pd


def load_dataset(file_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Loads the dataset from a relative CSV file path using pandas.
    Handles file-not-found errors and returns the DataFrame.
    
    Parameters:
        file_path (str): Relative path to the CSV file.
        
    Returns:
        pd.DataFrame or None: Loaded DataFrame, or None if file is not found.
    """
    try:
        # Load CSV using pandas with relative path
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset successfully loaded from: '{file_path}'")
        return df
    except FileNotFoundError:
        print(f"[ERROR] The file '{file_path}' was not found. Please ensure the CSV file is in the project folder.")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading the dataset: {e}")
        return None


def inspect_dataset(df):
    """
    Performs basic data inspection on the DataFrame:
    - Dataset shape (rows, columns)
    - First 5 rows
    - Column names
    - Data types and non-null counts
    
    Parameters:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    if df is None:
        print("[WARNING] No dataset available to inspect.")
        return

    # 1. Display dataset shape
    print("\n" + "=" * 50)
    print("1. DATASET SHAPE (Rows, Columns):")
    print("=" * 50)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 2. Display first 5 rows
    print("\n" + "=" * 50)
    print("2. FIRST 5 ROWS:")
    print("=" * 50)
    print(df.head())

    # 3. Display column names
    print("\n" + "=" * 50)
    print("3. COLUMN NAMES:")
    print("=" * 50)
    print(list(df.columns))

    # 4. Display data types and non-null counts
    print("\n" + "=" * 50)
    print("4. DATA TYPES & NON-NULL COUNTS:")
    print("=" * 50)
    df.info()


# Convenient aliases for function names
load_data = load_dataset
inspect_data = inspect_dataset


if __name__ == "__main__":
    # Allows load_data.py to be executed directly
    data = load_dataset()
    if data is not None:
        inspect_dataset(data)
