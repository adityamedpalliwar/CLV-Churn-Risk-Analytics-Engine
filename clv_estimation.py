"""
clv_estimation.py
Module for Phase 6: Customer Lifetime Value (CLV) Estimation.
Project: Customer Lifetime Value (CLV) & Churn Risk Analytics Engine.
"""

import os
import pandas as pd


def estimate_customer_clv(df, pipeline, horizon_months=12):
    """
    Estimates Customer Lifetime Value (CLV) using a simple, transparent,
    and beginner-friendly formula based on churn probabilities.
    
    Formula:
        1. Retention_Probability = 1 - Churn_Probability
        2. Expected_Remaining_Months = horizon_months * Retention_Probability
        3. Estimated_Future_Revenue = MonthlyCharges * Expected_Remaining_Months
        4. Revenue_at_Risk = (MonthlyCharges * horizon_months) - Estimated_Future_Revenue
        5. Total_Estimated_CLV = TotalCharges (Historical) + Estimated_Future_Revenue
    
    Parameters:
        df (pd.DataFrame): Cleaned customer DataFrame.
        pipeline (Pipeline): Trained Logistic Regression pipeline from Phase 5.
        horizon_months (int): Future planning window in months (default: 12).
        
    Returns:
        pd.DataFrame: DataFrame containing customer CLV estimations.
    """
    df_clv = df.copy()

    # 1. Prepare features for prediction (exclude customerID and Churn if present)
    feature_cols = [c for c in df_clv.columns if c not in ["customerID", "Churn"]]
    X = df_clv[feature_cols]

    # 2. Extract churn probability from Logistic Regression model
    churn_probabilities = pipeline.predict_proba(X)[:, 1]

    # 3. Compute retention probability and expected remaining lifetime
    retention_probabilities = 1.0 - churn_probabilities
    expected_remaining_months = (horizon_months * retention_probabilities).round(1)

    # 4. Compute Estimated Future Revenue and Revenue at Risk
    estimated_future_rev = (df_clv["MonthlyCharges"] * expected_remaining_months).round(2)
    max_possible_future_rev = (df_clv["MonthlyCharges"] * horizon_months).round(2)
    revenue_at_risk = (max_possible_future_rev - estimated_future_rev).round(2)

    # 5. Compute Total Estimated CLV (Historical + Future)
    total_clv = (df_clv["TotalCharges"] + estimated_future_rev).round(2)

    # Attach computed metrics to DataFrame
    df_clv["Churn_Probability"] = churn_probabilities.round(4)
    df_clv["Retention_Probability"] = retention_probabilities.round(4)
    df_clv["Expected_Remaining_Months"] = expected_remaining_months
    df_clv["Estimated_Future_CLV"] = estimated_future_rev
    df_clv["Revenue_at_Risk"] = revenue_at_risk
    df_clv["Total_Estimated_CLV"] = total_clv

    return df_clv


def display_clv_summary(df_clv, horizon_months=12):
    """
    Displays summary statistics, cohort comparisons, and sample customer profiles
    using full, appropriate column names.
    
    Parameters:
        df_clv (pd.DataFrame): DataFrame with CLV metrics.
        horizon_months (int): Planning window in months.
    """
    print("\n" + "=" * 80)
    print("1. CLV METHODOLOGY, FORMULA & ASSUMPTIONS")
    print("=" * 80)
    print(f" * Planning Horizon (T)           : {horizon_months} months (configurable)")
    print(" * P(Retention)                   = 1 - P(Churn)")
    print(" * Expected Remaining Months      = T * P(Retention)")
    print(" * Estimated Future CLV           = MonthlyCharges * Expected Remaining Months")
    print(" * Revenue at Risk                = (MonthlyCharges * T) * P(Churn)")
    print(" * Total Estimated CLV            = Historical TotalCharges + Estimated Future CLV")
    print(" * Key Principle                  : Historical charges are kept strictly separate")
    print("                                    from estimated future revenue.")

    print("\n" + "=" * 80)
    print("2. OVERALL CLV METRICS SUMMARY (7,043 CUSTOMERS)")
    print("=" * 80)

    summary_rows = [
        {"Metric": "Tenure (months)", "Mean": f"{df_clv['tenure'].mean():.1f}", "Median": f"{df_clv['tenure'].median():.1f}", "Std Dev": f"{df_clv['tenure'].std():.1f}", "Min": f"{df_clv['tenure'].min():.1f}", "Max": f"{df_clv['tenure'].max():.1f}"},
        {"Metric": "MonthlyCharges", "Mean": f"${df_clv['MonthlyCharges'].mean():.2f}", "Median": f"${df_clv['MonthlyCharges'].median():.2f}", "Std Dev": f"${df_clv['MonthlyCharges'].std():.2f}", "Min": f"${df_clv['MonthlyCharges'].min():.2f}", "Max": f"${df_clv['MonthlyCharges'].max():.2f}"},
        {"Metric": "TotalCharges (Hist)", "Mean": f"${df_clv['TotalCharges'].mean():.2f}", "Median": f"${df_clv['TotalCharges'].median():.2f}", "Std Dev": f"${df_clv['TotalCharges'].std():.2f}", "Min": f"${df_clv['TotalCharges'].min():.2f}", "Max": f"${df_clv['TotalCharges'].max():.2f}"},
        {"Metric": "Expected Rem Months", "Mean": f"{df_clv['Expected_Remaining_Months'].mean():.1f}", "Median": f"{df_clv['Expected_Remaining_Months'].median():.1f}", "Std Dev": f"{df_clv['Expected_Remaining_Months'].std():.1f}", "Min": f"{df_clv['Expected_Remaining_Months'].min():.1f}", "Max": f"{df_clv['Expected_Remaining_Months'].max():.1f}"},
        {"Metric": "Estimated Future CLV", "Mean": f"${df_clv['Estimated_Future_CLV'].mean():.2f}", "Median": f"${df_clv['Estimated_Future_CLV'].median():.2f}", "Std Dev": f"${df_clv['Estimated_Future_CLV'].std():.2f}", "Min": f"${df_clv['Estimated_Future_CLV'].min():.2f}", "Max": f"${df_clv['Estimated_Future_CLV'].max():.2f}"},
        {"Metric": "Revenue at Risk", "Mean": f"${df_clv['Revenue_at_Risk'].mean():.2f}", "Median": f"${df_clv['Revenue_at_Risk'].median():.2f}", "Std Dev": f"${df_clv['Revenue_at_Risk'].std():.2f}", "Min": f"${df_clv['Revenue_at_Risk'].min():.2f}", "Max": f"${df_clv['Revenue_at_Risk'].max():.2f}"},
        {"Metric": "Total Estimated CLV", "Mean": f"${df_clv['Total_Estimated_CLV'].mean():.2f}", "Median": f"${df_clv['Total_Estimated_CLV'].median():.2f}", "Std Dev": f"${df_clv['Total_Estimated_CLV'].std():.2f}", "Min": f"${df_clv['Total_Estimated_CLV'].min():.2f}", "Max": f"${df_clv['Total_Estimated_CLV'].max():.2f}"}
    ]
    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("3. SAMPLE CUSTOMER CLV PROFILES (FIRST 8 CUSTOMERS)")
    print("=" * 80)

    sample = df_clv.head(8).copy()
    sample_full = pd.DataFrame({
        "CustomerID": sample["customerID"],
        "Tenure": [f"{int(t)} mos" for t in sample["tenure"]],
        "MonthlyCharges": [f"${m:.2f}" for m in sample["MonthlyCharges"]],
        "TotalCharges": [f"${t:.2f}" for t in sample["TotalCharges"]],
        "Churn Probability": [f"{p*100:.1f}%" for p in sample["Churn_Probability"]],
        "Expected Remaining Months": [f"{r:.1f} mos" for r in sample["Expected_Remaining_Months"]],
        "Estimated Future CLV": [f"${f:.2f}" for f in sample["Estimated_Future_CLV"]],
        "Total Estimated CLV": [f"${tot:.2f}" for tot in sample["Total_Estimated_CLV"]]
    })
    print(sample_full.to_string(index=False))

    print("\n" + "=" * 80)
    print("4. STRATEGIC COMPARISON: HIGH-RISK vs LOW-RISK COHORTS")
    print("=" * 80)

    high_risk = df_clv[df_clv["Churn_Probability"] >= 0.50]
    low_risk = df_clv[df_clv["Churn_Probability"] < 0.50]

    cohort_comparison = pd.DataFrame([
        {
            "Cohort": "High Churn Risk (P >= 50%)",
            "Customer Count": f"{len(high_risk):,}",
            "Avg MonthlyCharges": f"${high_risk['MonthlyCharges'].mean():.2f}",
            "Avg Remaining Months": f"{high_risk['Expected_Remaining_Months'].mean():.1f} mos",
            "Avg Future CLV": f"${high_risk['Estimated_Future_CLV'].mean():.2f}",
            "Avg Revenue at Risk": f"${high_risk['Revenue_at_Risk'].mean():.2f}"
        },
        {
            "Cohort": "Low/Moderate Risk (P < 50%)",
            "Customer Count": f"{len(low_risk):,}",
            "Avg MonthlyCharges": f"${low_risk['MonthlyCharges'].mean():.2f}",
            "Avg Remaining Months": f"{low_risk['Expected_Remaining_Months'].mean():.1f} mos",
            "Avg Future CLV": f"${low_risk['Estimated_Future_CLV'].mean():.2f}",
            "Avg Revenue at Risk": f"${low_risk['Revenue_at_Risk'].mean():.2f}"
        }
    ])
    print(cohort_comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("5. HOW CHURN PREDICTION INFORMS CLV ESTIMATION")
    print("=" * 80)
    print(" 1. Risk-Adjusted Projections:")
    print("    Without churn probability, a business naively forecasts MonthlyCharges * 12.")
    print("    Multiplying by P(Retention) discounts expected revenue for fragile accounts,")
    print("    preventing revenue over-estimation.")
    print(" 2. Targeted Retention & ROI:")
    print("    Identifying 'Revenue at Risk' allows teams to prioritize retention interventions")
    print("    on high-value accounts with imminent risk, maximizing retention campaign ROI.")

    print("\n" + "=" * 80)
    print("6. ASSUMPTIONS & LIMITATIONS")
    print("=" * 80)
    print(" * Assumption 1 (Constant Monthly Spend): Assumes monthly charges stay constant")
    print("   across the horizon without immediate plan upgrades or downgrades.")
    print(" * Assumption 2 (Linear Risk Discount): Uses P(Retention) as a direct scaling")
    print("   factor for expected active customer months.")
    print(" * Limitation 1 (Gross Revenue vs. Net Profit): Estimates gross revenue collected,")
    print("   not net profit (infrastructure and customer service costs are omitted).")
    print(" * Limitation 2 (Estimated, Not Guaranteed): Represents an expected value to guide")
    print("   business prioritization, not a guaranteed financial audit.")


def run_phase6_clv(df, pipeline=None, horizon_months=12):
    """
    Executes the Phase 6 Customer Lifetime Value (CLV) estimation pipeline.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset DataFrame.
        pipeline (Pipeline): Trained model pipeline. If None, trains a baseline pipeline.
        horizon_months (int): Future planning window (default: 12 months).
        
    Returns:
        pd.DataFrame: DataFrame augmented with CLV metrics.
    """
    if df is None:
        print("[ERROR] DataFrame is None. Cannot run Phase 6 CLV estimation.")
        return None

    print("\n" + "#" * 80)
    print("# PHASE 6: CUSTOMER LIFETIME VALUE (CLV) ESTIMATION")
    print("#" * 80)

    # Train a baseline pipeline if not provided
    if pipeline is None:
        print("No pre-trained pipeline passed. Training baseline Logistic Regression model...")
        from churn_model import split_features_target, build_pipeline
        from sklearn.model_selection import train_test_split

        X, y, customer_ids, num_cols, cat_cols = split_features_target(df)
        pipeline = build_pipeline(num_cols, cat_cols)
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y, customer_ids, test_size=0.2, random_state=42
        )
        pipeline.fit(X_train, y_train)

    # Estimate CLV
    df_clv = estimate_customer_clv(df, pipeline, horizon_months=horizon_months)

    # Display summaries
    display_clv_summary(df_clv, horizon_months=horizon_months)

    print("\n" + "#" * 80)
    print("# Phase 6 CLV Estimation Completed Successfully!")
    print("#" * 80)

    return df_clv


if __name__ == "__main__":
    cleaned_csv_path = "telco_churn_cleaned.csv"
    if os.path.exists(cleaned_csv_path):
        print(f"Loading cleaned dataset from '{cleaned_csv_path}'...")
        cleaned_df = pd.read_csv(cleaned_csv_path)
    else:
        from load_data import load_dataset
        from clean_data import clean_dataset
        print("Cleaned CSV not found. Running Phase 1 & 3 first...")
        raw_df = load_dataset()
        cleaned_df, _, _, _, _, _ = clean_dataset(raw_df)

    run_phase6_clv(cleaned_df, horizon_months=12)
