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

    print("\n" + "=" * 65)
    print("1. CLV METHODOLOGY, FORMULA & ASSUMPTIONS")
    print("=" * 65)
    print(" * Concept: Simplified Estimated Future Revenue (not actual net profit).")
    print(f" * Planning Horizon (T): {horizon_months} months (easily adjustable).")
    print(" * Mathematical Formulation:")
    print("     - P(Retention) = 1 - P(Churn)")
    print("     - Expected Remaining Months = T * P(Retention)")
    print("     - Estimated Future Revenue  = MonthlyCharges * Expected Remaining Months")
    print("     - Revenue at Risk           = (MonthlyCharges * T) * P(Churn)")
    print("     - Total Lifetime Value      = Historical TotalCharges + Estimated Future Revenue")
    print(" * Separation: Historical revenue (TotalCharges) is kept strictly separate")
    print("   from future estimated revenue.")
    print(" * Transparency: No black-box probabilistic models or hidden discount rates.")

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
    Displays summary statistics, cohort comparisons, and sample customer profiles.
    
    Parameters:
        df_clv (pd.DataFrame): DataFrame with CLV metrics.
        horizon_months (int): Planning window in months.
    """
    print("\n" + "=" * 65)
    print("2. OVERALL CLV METRICS SUMMARY (ENTIRE CUSTOMER BASE)")
    print("=" * 65)

    summary_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Expected_Remaining_Months",
        "Estimated_Future_CLV",
        "Revenue_at_Risk",
        "Total_Estimated_CLV"
    ]
    summary_df = df_clv[summary_cols].describe().round(2)
    # Transpose for easier horizontal reading
    print(summary_df.loc[["mean", "std", "min", "50%", "max"]].T)

    print("\n" + "=" * 65)
    print("3. SAMPLE CUSTOMER CLV PROFILES (FIRST 8 CUSTOMERS)")
    print("=" * 65)

    display_cols = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn_Probability",
        "Expected_Remaining_Months",
        "Estimated_Future_CLV",
        "Total_Estimated_CLV"
    ]
    sample_df = df_clv[display_cols].head(8).copy()
    sample_df["MonthlyCharges"] = sample_df["MonthlyCharges"].map("${:.2f}".format)
    sample_df["TotalCharges"] = sample_df["TotalCharges"].map("${:.2f}".format)
    sample_df["Churn_Probability"] = (sample_df["Churn_Probability"] * 100).map("{:.1f}%".format)
    sample_df["Expected_Remaining_Months"] = sample_df["Expected_Remaining_Months"].map("{:.1f} mos".format)
    sample_df["Estimated_Future_CLV"] = sample_df["Estimated_Future_CLV"].map("${:.2f}".format)
    sample_df["Total_Estimated_CLV"] = sample_df["Total_Estimated_CLV"].map("${:.2f}".format)
    print(sample_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("4. STRATEGIC COMPARISON: HIGH-RISK vs LOW-RISK COHORTS")
    print("=" * 65)

    high_risk = df_clv[df_clv["Churn_Probability"] >= 0.50]
    low_risk = df_clv[df_clv["Churn_Probability"] < 0.50]

    cohort_comparison = pd.DataFrame([
        {
            "Cohort": "High Churn Risk (P >= 50%)",
            "Customers": f"{len(high_risk):,}",
            "Avg Monthly Charge": f"${high_risk['MonthlyCharges'].mean():.2f}",
            "Avg Remaining Months": f"{high_risk['Expected_Remaining_Months'].mean():.1f} mos",
            "Avg Future CLV": f"${high_risk['Estimated_Future_CLV'].mean():.2f}",
            "Avg Revenue at Risk": f"${high_risk['Revenue_at_Risk'].mean():.2f}"
        },
        {
            "Cohort": "Low/Moderate Risk (P < 50%)",
            "Customers": f"{len(low_risk):,}",
            "Avg Monthly Charge": f"${low_risk['MonthlyCharges'].mean():.2f}",
            "Avg Remaining Months": f"{low_risk['Expected_Remaining_Months'].mean():.1f} mos",
            "Avg Future CLV": f"${low_risk['Estimated_Future_CLV'].mean():.2f}",
            "Avg Revenue at Risk": f"${low_risk['Revenue_at_Risk'].mean():.2f}"
        }
    ])
    print(cohort_comparison.to_string(index=False))

    print("\n" + "=" * 65)
    print("5. HOW CHURN PREDICTION INFORMS CLV ESTIMATION")
    print("=" * 65)
    print(" 1. Risk-Adjusted Revenue: Without churn prediction, companies naively assume all")
    print("    customers stay for the full planning window (MonthlyCharges * 12). Incorporating")
    print("    churn probability prevents overestimating future revenue from fragile accounts.")
    print(" 2. Proactive Retention Prioritization: By identifying 'Revenue at Risk', managers")
    print("    can focus retention interventions on high-value customers who have a high churn probability,")
    print("    maximizing the return on retention marketing spend.")

    print("\n" + "=" * 65)
    print("6. ASSUMPTIONS & LIMITATIONS")
    print("=" * 65)
    print(" * Assumption 1 (Constant Monthly Spend): Assumes monthly charges remain steady over the horizon")
    print("   without account upgrades, downgrades, or price revisions.")
    print(" * Assumption 2 (Linear Risk Discount): Assumes churn probability linearly scales expected active months.")
    print(" * Limitation 1 (Revenue vs. Profit): This calculates gross expected revenue, not net profit")
    print("   (operational overhead, server infrastructure, and customer acquisition costs are omitted).")
    print(" * Limitation 2 (No Future Certainty): This is an empirical estimate to guide decision-making,")
    print("   not a guaranteed financial forecast.")


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

    print("\n" + "#" * 65)
    print("# PHASE 6: CUSTOMER LIFETIME VALUE (CLV) ESTIMATION")
    print("#" * 65)

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

    print("\n" + "#" * 65)
    print("# Phase 6 CLV Estimation Completed Successfully!")
    print("#" * 65)

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
