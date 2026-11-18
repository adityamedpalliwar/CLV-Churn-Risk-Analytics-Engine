"""
customer_risk_analysis.py
Module for Phase 7: Customer-Level Risk and Value Analysis.
Project: Customer Lifetime Value (CLV) & Churn Risk Analytics Engine.
"""

import os
import pandas as pd

# =============================================================================
# CLEAR & EDITABLE THRESHOLDS
# Customize these thresholds based on business requirements.
# =============================================================================
DEFAULT_PROB_HIGH = 0.50     # Churn probability >= 50% -> High Risk
DEFAULT_PROB_MEDIUM = 0.25   # Churn probability >= 25% (and < 50%) -> Medium Risk
DEFAULT_VALUE_HIGH = 300.00  # Estimated Future CLV >= $300.00 -> High Value


def analyze_customer_risk_and_value(
    df,
    pipeline=None,
    prob_high=DEFAULT_PROB_HIGH,
    prob_med=DEFAULT_PROB_MEDIUM,
    val_high=DEFAULT_VALUE_HIGH,
    horizon_months=12,
    output_csv="customer_risk_value_analysis.csv"
):
    """
    Performs customer-level risk and value analysis.
    
    Includes for each customer:
    - CustomerID
    - Tenure
    - MonthlyCharges
    - TotalCharges (historical)
    - Predicted churn probability
    - Estimated future CLV
    - Revenue at Risk = Churn Probability * Estimated Future CLV
    - Customer Priority: High Risk, Medium Risk, Low Risk
    
    Parameters:
        df (pd.DataFrame): Customer DataFrame.
        pipeline (Pipeline): Trained Logistic Regression pipeline.
        prob_high (float): Threshold for High Risk (default: 0.50).
        prob_med (float): Threshold for Medium Risk (default: 0.25).
        val_high (float): Threshold for High Value (default: 300.00).
        horizon_months (int): Future planning window in months (default: 12).
        output_csv (str): Path to export customer-level results.
        
    Returns:
        pd.DataFrame: Customer-level risk and value analysis DataFrame.
    """
    df_analysis = df.copy()

    # 1. If churn probability and future CLV are not yet present, compute them
    if "Churn_Probability" not in df_analysis.columns or "Estimated_Future_CLV" not in df_analysis.columns:
        if pipeline is None:
            from churn_model import split_features_target, build_pipeline
            from sklearn.model_selection import train_test_split
            X, y, customer_ids, num_cols, cat_cols = split_features_target(df_analysis)
            pipeline = build_pipeline(num_cols, cat_cols)
            X_train, X_test, y_train, y_test, _, _ = train_test_split(
                X, y, customer_ids, test_size=0.2, random_state=42
            )
            pipeline.fit(X_train, y_train)

        from clv_estimation import estimate_customer_clv
        df_analysis = estimate_customer_clv(df_analysis, pipeline, horizon_months=horizon_months)

    # 2. Phase 7 Specific Formula: Revenue at Risk = Churn Probability * Estimated Future CLV
    df_analysis["Revenue_at_Risk"] = (
        df_analysis["Churn_Probability"] * df_analysis["Estimated_Future_CLV"]
    ).round(2)

    # 3. Simple Priority Classification (High Risk, Medium Risk, Low Risk)
    def classify_priority(prob):
        if prob >= prob_high:
            return "High Risk"
        elif prob >= prob_med:
            return "Medium Risk"
        else:
            return "Low Risk"

    df_analysis["Customer_Priority"] = df_analysis["Churn_Probability"].apply(classify_priority)

    # 4. Value Classification
    df_analysis["Value_Tier"] = df_analysis["Estimated_Future_CLV"].apply(
        lambda v: "High Value" if v >= val_high else "Standard Value"
    )

    # Flag intersection: High Churn Risk AND High Estimated Value
    df_analysis["High_Risk_High_Value"] = (
        (df_analysis["Customer_Priority"] == "High Risk") &
        (df_analysis["Value_Tier"] == "High Value")
    )

    # 5. Export customer-level results table if requested
    if output_csv:
        export_cols = [
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn_Probability",
            "Estimated_Future_CLV",
            "Revenue_at_Risk",
            "Customer_Priority",
            "Value_Tier",
            "High_Risk_High_Value"
        ]
        df_analysis[export_cols].to_csv(output_csv, index=False)

    return df_analysis


def display_phase7_reports(df_analysis, prob_high=DEFAULT_PROB_HIGH, val_high=DEFAULT_VALUE_HIGH):
    """
    Displays the four required Phase 7 reports using full, appropriate column names
    aligned with the CLV estimator tables.
    """
    print("\n" + "=" * 80)
    print("1. CUSTOMER-LEVEL RESULTS TABLE (FIRST 8 ACCOUNTS)")
    print("=" * 80)

    sample = df_analysis.head(8).copy()
    table_1 = pd.DataFrame({
        "CustomerID": sample["customerID"],
        "Tenure": [f"{int(t)} mos" for t in sample["tenure"]],
        "MonthlyCharges": [f"${m:.2f}" for m in sample["MonthlyCharges"]],
        "TotalCharges": [f"${t:.2f}" for t in sample["TotalCharges"]],
        "Churn Probability": [f"{p*100:.1f}%" for p in sample["Churn_Probability"]],
        "Estimated Future CLV": [f"${f:.2f}" for f in sample["Estimated_Future_CLV"]],
        "Revenue at Risk": [f"${r:.2f}" for r in sample["Revenue_at_Risk"]],
        "Customer Priority": sample["Customer_Priority"]
    })
    print(table_1.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. HIGH-RISK CUSTOMERS LIST (TOP 8 BY REVENUE AT RISK)")
    print("=" * 80)
    print(f" * Filter Criteria: Churn Probability >= {prob_high*100:.0f}%")
    high_risk_all = df_analysis[df_analysis["Customer_Priority"] == "High Risk"].sort_values(
        by="Revenue_at_Risk", ascending=False
    )
    print(f" * Total High-Risk Customers Identified: {len(high_risk_all):,}\n")

    hr_sample = high_risk_all.head(8).copy()
    table_2 = pd.DataFrame({
        "CustomerID": hr_sample["customerID"],
        "Tenure": [f"{int(t)} mos" for t in hr_sample["tenure"]],
        "MonthlyCharges": [f"${m:.2f}" for m in hr_sample["MonthlyCharges"]],
        "TotalCharges": [f"${t:.2f}" for t in hr_sample["TotalCharges"]],
        "Churn Probability": [f"{p*100:.1f}%" for p in hr_sample["Churn_Probability"]],
        "Estimated Future CLV": [f"${f:.2f}" for f in hr_sample["Estimated_Future_CLV"]],
        "Revenue at Risk": [f"${r:.2f}" for r in hr_sample["Revenue_at_Risk"]],
        "Customer Priority": hr_sample["Customer_Priority"]
    })
    print(table_2.to_string(index=False))

    print("\n" + "=" * 80)
    print("3. CUSTOMERS WITH HIGH CHURN RISK AND HIGH ESTIMATED VALUE (TOP 8)")
    print("=" * 80)
    print(f" * Filter Criteria: Churn Probability >= {prob_high*100:.0f}% AND Estimated Future CLV >= ${val_high:.2f}")
    high_risk_high_val = df_analysis[df_analysis["High_Risk_High_Value"]].sort_values(
        by="Revenue_at_Risk", ascending=False
    )
    print(f" * Total Priority Accounts Identified: {len(high_risk_high_val):,} (Top Retention Targets)\n")

    hrhv_sample = high_risk_high_val.head(8).copy()
    table_3 = pd.DataFrame({
        "CustomerID": hrhv_sample["customerID"],
        "Tenure": [f"{int(t)} mos" for t in hrhv_sample["tenure"]],
        "MonthlyCharges": [f"${m:.2f}" for m in hrhv_sample["MonthlyCharges"]],
        "TotalCharges": [f"${t:.2f}" for t in hrhv_sample["TotalCharges"]],
        "Churn Probability": [f"{p*100:.1f}%" for p in hrhv_sample["Churn_Probability"]],
        "Estimated Future CLV": [f"${f:.2f}" for f in hrhv_sample["Estimated_Future_CLV"]],
        "Revenue at Risk": [f"${r:.2f}" for r in hrhv_sample["Revenue_at_Risk"]],
        "Customer Priority": hrhv_sample["Customer_Priority"]
    })
    print(table_3.to_string(index=False))

    print("\n" + "=" * 80)
    print("4. SUMMARY STATISTICS: CUSTOMER RISK AND VALUE")
    print("=" * 80)

    summary_data = []
    tiers = ["High Risk", "Medium Risk", "Low Risk"]
    for tier in tiers:
        sub = df_analysis[df_analysis["Customer_Priority"] == tier]
        summary_data.append({
            "Priority": tier,
            "Customer Count": f"{len(sub):,}",
            "Share of Base": f"{(len(sub)/len(df_analysis))*100:.1f}%",
            "Avg Churn Probability": f"{sub['Churn_Probability'].mean()*100:.1f}%",
            "Avg MonthlyCharges": f"${sub['MonthlyCharges'].mean():.2f}",
            "Avg Future CLV": f"${sub['Estimated_Future_CLV'].mean():.2f}",
            "Avg Revenue at Risk": f"${sub['Revenue_at_Risk'].mean():.2f}",
            "Total Revenue at Risk": f"${sub['Revenue_at_Risk'].sum():,.2f}"
        })

    # Add the high-risk & high-value sub-cohort
    sub_hrhv = df_analysis[df_analysis["High_Risk_High_Value"]]
    summary_data.append({
        "Priority": " -> High Risk & High Value",
        "Customer Count": f"{len(sub_hrhv):,}",
        "Share of Base": f"{(len(sub_hrhv)/len(df_analysis))*100:.1f}%",
        "Avg Churn Probability": f"{sub_hrhv['Churn_Probability'].mean()*100:.1f}%",
        "Avg MonthlyCharges": f"${sub_hrhv['MonthlyCharges'].mean():.2f}",
        "Avg Future CLV": f"${sub_hrhv['Estimated_Future_CLV'].mean():.2f}",
        "Avg Revenue at Risk": f"${sub_hrhv['Revenue_at_Risk'].mean():.2f}",
        "Total Revenue at Risk": f"${sub_hrhv['Revenue_at_Risk'].sum():,.2f}"
    })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


def run_phase7_analysis(
    df,
    pipeline=None,
    prob_high=DEFAULT_PROB_HIGH,
    prob_med=DEFAULT_PROB_MEDIUM,
    val_high=DEFAULT_VALUE_HIGH,
    horizon_months=12
):
    """
    Executes the complete Phase 7 customer-level risk and value analysis.
    """
    if df is None:
        print("[ERROR] DataFrame is None. Cannot run Phase 7 analysis.")
        return None

    print("\n" + "#" * 80)
    print("# PHASE 7: CUSTOMER-LEVEL RISK AND VALUE ANALYSIS")
    print("#" * 80)
    print(" * Formula: Revenue at Risk = Churn Probability * Estimated Future CLV")
    print(f" * Editable Thresholds: High Risk >= {prob_high*100:.0f}%, Medium Risk >= {prob_med*100:.0f}%, High Value >= ${val_high:.2f}")

    df_analysis = analyze_customer_risk_and_value(
        df,
        pipeline=pipeline,
        prob_high=prob_high,
        prob_med=prob_med,
        val_high=val_high,
        horizon_months=horizon_months
    )

    display_phase7_reports(df_analysis, prob_high=prob_high, val_high=val_high)

    print("\n[OUTPUT] Customer-level results saved to: 'customer_risk_value_analysis.csv'")
    print("\n" + "#" * 80)
    print("# Phase 7 Risk & Value Analysis Completed Successfully!")
    print("#" * 80)

    return df_analysis


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

    run_phase7_analysis(cleaned_df)
