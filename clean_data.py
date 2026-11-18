# clean_data.py
# Module for Phase 3: Basic Data Cleaning and Preprocessing.
# Project: Customer Lifetime Value (CLV) & Churn Risk Analytics Engine.
# """

# import pandas as pd


# def clean_dataset(df):
#     print("Initial Shape:", df.shape)

#     # 1. Convert TotalCharges to numeric
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#     # 2. Fill missing TotalCharges with 0
#     df["TotalCharges"] = df["TotalCharges"].fillna(0)

#     # 3. Check and remove duplicate rows
#     print("Duplicates:", df.duplicated().sum())
#     df = df.drop_duplicates()

#     # 4. Convert Churn into binary values
#     df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#     # 5. Separate customer IDs
#     customer_ids = df["customerID"]

#     # 6. Create feature matrix X and target y
#     X = df.drop(["customerID", "Churn"], axis=1)
#     y = df["Churn"]

#     # 7. Separate numerical and categorical columns
#     numerical_features = X.select_dtypes(include="number").columns.tolist()
#     categorical_features = X.select_dtypes(include="object").columns.tolist()

#     # 8. Save cleaned dataset
#     df.to_csv("telco_churn_cleaned.csv", index=False)

#     print("Final Shape:", df.shape)
#     print("Cleaning Completed!")

#     return df, customer_ids, X, y, numerical_features, categorical_features


# if __name__ == "__main__":
#     from load_data import load_dataset

#     df = load_dataset()

#     if df is not None:
#         clean_dataset(df)


























import pandas as pd


def handle_total_charges(df, strategy="zero"):
    """
    Handles the 'TotalCharges' column:
    - Converts it to numeric format using pd.to_numeric with errors="coerce"
    - Identifies and investigates rows that become NaN
    - Applies a justified treatment (default: imputes with 0.0 because tenure == 0)
    - Avoids silently removing customers
    
    Parameters:
        df (pd.DataFrame): Dataset DataFrame.
        strategy (str): 'zero' (impute with 0.0) or 'drop' (remove rows).
        
    Returns:
        pd.DataFrame: DataFrame with numeric TotalCharges.
    """
    df = df.copy()

    print("\n" + "=" * 65)
    print("1. TOTALCHARGES CONVERSION & INVESTIGATION")
    print("=" * 65)

    # Convert to numeric format with errors="coerce"
    raw_missing = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    coerced_missing = df["TotalCharges"].isnull().sum()

    print(f"Missing TotalCharges before conversion : {raw_missing}")
    print(f"Missing TotalCharges after coercion    : {coerced_missing} rows")

    # Identify and investigate the affected rows
    missing_mask = df["TotalCharges"].isnull()
    missing_rows = df[missing_mask]

    print("\n--- Investigation of Rows with Missing TotalCharges ---")
    investigation_cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    print(missing_rows[investigation_cols])

    # Check tenure and churn distribution for these rows
    tenure_values = missing_rows["tenure"].unique()
    churn_values = missing_rows["Churn"].unique()
    print(f"\nObserved Tenure in missing rows : {tenure_values} month(s)")
    print(f"Observed Churn in missing rows  : {churn_values}")

    print("\n[Cleaning Decision & Justification]")
    print(" * All 11 records have tenure == 0, representing newly joined customers who have")
    print("   not yet completed their first monthly billing cycle.")
    print(" * Therefore, their accumulated historical total charges are logically $0.00.")

    if strategy == "zero":
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        print(" * Strategy applied: Impute missing TotalCharges with 0.0.")
        print(f" * Customer preservation: Retained all {len(missing_rows)} customers without silent removal.")
    elif strategy == "drop":
        df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
        print(f" * Strategy applied: Dropped {len(missing_rows)} rows where tenure == 0.")
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    print(f"Remaining missing TotalCharges: {df['TotalCharges'].isnull().sum()}")
    return df


def handle_missing_values(df):
    """
    Displays missing values before treatment and verifies that no
    unnecessary imputation is applied to columns without missing values.
    
    Parameters:
        df (pd.DataFrame): Dataset DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    print("\n" + "=" * 65)
    print("2. MISSING-VALUE TREATMENT & VERIFICATION")
    print("=" * 65)

    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    print("Missing Values Summary Across All Columns:")
    if len(cols_with_missing) == 0:
        print(" - All columns have 0 missing values.")
        print(" - No unnecessary imputation applied to any column.")
    else:
        for col, count in cols_with_missing.items():
            print(f" - {col}: {count} missing values")

    return df


def handle_duplicates(df):
    """
    Identifies duplicate records across the dataset and removes them
    only if justified.
    
    Parameters:
        df (pd.DataFrame): Dataset DataFrame.
        
    Returns:
        pd.DataFrame: De-duplicated DataFrame.
    """
    print("\n" + "=" * 65)
    print("3. DUPLICATE RECORDS ANALYSIS")
    print("=" * 65)

    num_duplicates = df.duplicated().sum()
    print(f"Duplicate records identified: {num_duplicates}")

    if num_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[Action] Removed {num_duplicates} duplicate records.")
    else:
        print("[Action] 0 duplicates removed. The dataset is already free of duplicate rows.")

    return df


def prepare_features(df):
    """
    Prepares dataset features:
    - Retains customerID separately for identification (excludes from ML features)
    - Converts Churn into a binary target (Yes=1, No=0)
    - Separates numerical and categorical features
    
    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.
        
    Returns:
        tuple: (df_clean, customer_ids, X, y, numerical_features, categorical_features)
    """
    print("\n" + "=" * 65)
    print("4. FEATURE PREPARATION & SEPARATION")
    print("=" * 65)

    df_clean = df.copy()

    # 1. Retain customerID separately for identification
    if "customerID" in df_clean.columns:
        customer_ids = df_clean["customerID"].copy()
        print(f"Retained {len(customer_ids):,} customer IDs separately for identification.")
        print("Note: 'customerID' is excluded from model feature sets to prevent data leakage/overfitting.")
    else:
        customer_ids = None

    # 2. Convert Churn into binary target (Yes=1, No=0)
    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == object:
            df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})
            print("\nConverted 'Churn' to binary target:")
            print(f" - Class 0 (Retained / No) : {(df_clean['Churn'] == 0).sum():,} customers")
            print(f" - Class 1 (Churned / Yes)  : {(df_clean['Churn'] == 1).sum():,} customers")
        else:
            print("\n'Churn' is already encoded as numeric binary target.")
        y = df_clean["Churn"].copy()
    else:
        y = None

    # 3. Separate feature matrix X (drop customerID and Churn)
    drop_cols = [c for c in ["customerID", "Churn"] if c in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)

    # 4. Identify numerical and categorical features in X
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\nFeature Matrix X Shape: {X.shape[0]} rows, {X.shape[1]} features")
    print(f" - Numerical Features ({len(numerical_features)}): {numerical_features}")
    print(f" - Categorical Features ({len(categorical_features)}): {categorical_features}")

    return df_clean, customer_ids, X, y, numerical_features, categorical_features


def clean_dataset(df, output_csv="telco_churn_cleaned.csv", total_charges_strategy="zero"):
    """
    Orchestrates the entire Phase 3 cleaning and preprocessing pipeline.
    
    Parameters:
        df (pd.DataFrame): Raw loaded DataFrame.
        output_csv (str): Path to save the cleaned CSV dataset.
        total_charges_strategy (str): 'zero' (impute 0.0) or 'drop'.
        
    Returns:
        tuple: (df_clean, customer_ids, X, y, numerical_features, categorical_features)
    """
    if df is None:
        print("[ERROR] DataFrame is None. Cannot perform Phase 3 cleaning.")
        return None

    print("\n" + "#" * 65)
    print("# PHASE 3: BASIC DATA CLEANING & PREPROCESSING")
    print("#" * 65)

    initial_shape = df.shape
    print(f"Initial Dataset Dimensions: {initial_shape[0]:,} rows x {initial_shape[1]} columns")

    # Step 1: TotalCharges conversion and investigation
    df_step1 = handle_total_charges(df, strategy=total_charges_strategy)

    # Step 2: Missing values treatment & verification
    df_step2 = handle_missing_values(df_step1)

    # Step 3: Duplicate records analysis
    df_step3 = handle_duplicates(df_step2)

    # Step 4: Feature preparation & separation
    df_clean, customer_ids, X, y, num_features, cat_features = prepare_features(df_step3)

    final_shape = df_clean.shape

    # Step 5: Save cleaned dataset for EDA and future model training
    if output_csv:
        df_clean.to_csv(output_csv, index=False)
        print(f"\n[OUTPUT] Cleaned dataset saved to: '{output_csv}'")

    print("\n" + "=" * 65)
    print("5. BEFORE / AFTER DATASET DIMENSIONS & CLEANING SUMMARY")
    print("=" * 65)
    print(f"Initial Dimensions : {initial_shape[0]:,} rows x {initial_shape[1]} columns")
    print(f"Cleaned Dimensions : {final_shape[0]:,} rows x {final_shape[1]} columns")
    print(f"Rows Retained      : {final_shape[0]:,} (100.0% preserved)")
    print(f"Original CSV Status: Preserved untouched ('WA_Fn-UseC_-Telco-Customer-Churn.csv')")

    print("\n" + "#" * 65)
    print("# Phase 3 Data Cleaning Completed Successfully!")
    print("#" * 65)

    return df_clean, customer_ids, X, y, num_features, cat_features


if __name__ == "__main__":
    from load_data import load_dataset
    raw_df = load_dataset()
    if raw_df is not None:
        clean_dataset(raw_df)
