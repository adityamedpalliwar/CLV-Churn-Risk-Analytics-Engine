"""
eda.py
Module for Phase 4: Exploratory Data Analysis (EDA).
Project: Customer Lifetime Value (CLV) & Churn Risk Analytics Engine.
"""

import os
import matplotlib
# Use headless non-interactive backend to prevent GUI thread conflicts and terminal echo glitches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set global seaborn styling
sns.set_theme(style="whitegrid")
PALETTE = {"No (Retained)": "#2b7bba", "Yes (Churned)": "#e6550d"}


def prepare_eda_data(df):
    """
    Prepares DataFrame for EDA by ensuring readable Churn labels.
    """
    df_eda = df.copy()
    if "Churn" in df_eda.columns:
        if df_eda["Churn"].dtype in [int, "int64"]:
            df_eda["Churn_Label"] = df_eda["Churn"].map({0: "No (Retained)", 1: "Yes (Churned)"})
        else:
            df_eda["Churn_Label"] = df_eda["Churn"].map({"No": "No (Retained)", "Yes": "Yes (Churned)"})
    return df_eda


def plot_churn_distribution(df, output_dir="plots"):
    """
    Visualization 1: Churn distribution (Yes vs No)
    """
    print("\n" + "=" * 65)
    print("1. CHURN DISTRIBUTION (YES vs NO)")
    print("=" * 65)

    counts = df["Churn_Label"].value_counts()
    percentages = (df["Churn_Label"].value_counts(normalize=True) * 100).round(2)
    stats_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})
    print(stats_df)

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x="Churn_Label",
        hue="Churn_Label",
        palette=PALETTE,
        legend=False,
        order=["No (Retained)", "Yes (Churned)"]
    )
    plt.title("Overall Customer Churn Distribution", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Churn Status", fontsize=12, labelpad=8)
    plt.ylabel("Number of Customers", fontsize=12, labelpad=8)

    # Annotate counts and percentages inside bars
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    file_path = os.path.join(output_dir, "01_churn_distribution.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def plot_churn_by_contract(df, output_dir="plots"):
    """
    Visualization 2: Churn by contract type
    """
    print("\n" + "=" * 65)
    print("2. CHURN BY CONTRACT TYPE")
    print("=" * 65)

    contract_stats = df.groupby("Contract")["Churn"].agg(
        Total="count",
        Churned="sum",
        Churn_Rate=lambda x: (x.mean() * 100)
    ).round(2)
    contract_stats["Retention_Rate"] = (100 - contract_stats["Churn_Rate"]).round(2)
    print(contract_stats)

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="Contract",
        hue="Churn_Label",
        palette=PALETTE,
        order=["Month-to-month", "One year", "Two year"]
    )
    plt.title("Customer Churn by Contract Type", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Contract Type", fontsize=12, labelpad=8)
    plt.ylabel("Number of Customers", fontsize=12, labelpad=8)
    plt.legend(title="Churn Status", fontsize=10)

    # Bar annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height):,}",
                        (p.get_x() + p.get_width() / 2., height + 40),
                        ha="center", va="bottom", fontsize=9)

    file_path = os.path.join(output_dir, "02_churn_by_contract.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def plot_churn_by_tenure(df, output_dir="plots"):
    """
    Visualization 3: Churn by tenure
    """
    print("\n" + "=" * 65)
    print("3. CHURN BY TENURE")
    print("=" * 65)

    tenure_stats = df.groupby("Churn_Label")["tenure"].describe().round(2)
    print(tenure_stats[["count", "mean", "std", "50%", "min", "max"]])

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn_Label",
        y="tenure",
        hue="Churn_Label",
        palette=PALETTE,
        legend=False,
        order=["No (Retained)", "Yes (Churned)"],
        width=0.45
    )
    plt.title("Tenure Comparison: Retained vs Churned Customers", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Churn Status", fontsize=12, labelpad=8)
    plt.ylabel("Tenure (Months)", fontsize=12, labelpad=8)

    # Annotate medians on boxplot
    medians = df.groupby("Churn_Label")["tenure"].median()
    for idx, (label, median_val) in enumerate(medians.items()):
        plt.text(idx, median_val + 1.5, f"Median: {median_val:.0f} mos",
                 ha="center", fontweight="bold", color="black", fontsize=10)

    file_path = os.path.join(output_dir, "03_churn_by_tenure.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def plot_churn_by_monthly_charges(df, output_dir="plots"):
    """
    Visualization 4: Churn by MonthlyCharges
    """
    print("\n" + "=" * 65)
    print("4. CHURN BY MONTHLY CHARGES")
    print("=" * 65)

    mc_stats = df.groupby("Churn_Label")["MonthlyCharges"].describe().round(2)
    print(mc_stats[["count", "mean", "std", "50%", "min", "max"]])

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn_Label",
        y="MonthlyCharges",
        hue="Churn_Label",
        palette=PALETTE,
        legend=False,
        order=["No (Retained)", "Yes (Churned)"],
        width=0.45
    )
    plt.title("Monthly Charges Comparison: Retained vs Churned Customers", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Churn Status", fontsize=12, labelpad=8)
    plt.ylabel("Monthly Charges ($)", fontsize=12, labelpad=8)

    # Annotate medians on boxplot
    medians = df.groupby("Churn_Label")["MonthlyCharges"].median()
    for idx, (label, median_val) in enumerate(medians.items()):
        plt.text(idx, median_val + 2, f"Median: ${median_val:.2f}",
                 ha="center", fontweight="bold", color="black", fontsize=10)

    file_path = os.path.join(output_dir, "04_churn_by_monthly_charges.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def plot_churn_by_payment_method(df, output_dir="plots"):
    """
    Visualization 5: Churn by PaymentMethod
    """
    print("\n" + "=" * 65)
    print("5. CHURN BY PAYMENT METHOD")
    print("=" * 65)

    pm_stats = df.groupby("PaymentMethod")["Churn"].agg(
        Total="count",
        Churned="sum",
        Churn_Rate=lambda x: (x.mean() * 100)
    ).round(2).sort_values(by="Churn_Rate", ascending=False)
    print(pm_stats)

    plt.figure(figsize=(10, 5.5))
    sns.countplot(
        data=df,
        y="PaymentMethod",
        hue="Churn_Label",
        palette=PALETTE,
        order=pm_stats.index
    )
    plt.title("Customer Churn by Payment Method", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Number of Customers", fontsize=12, labelpad=8)
    plt.ylabel("Payment Method", fontsize=12, labelpad=8)
    plt.legend(title="Churn Status", loc="lower right", fontsize=10)

    file_path = os.path.join(output_dir, "05_churn_by_payment_method.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def plot_tenure_distribution(df, output_dir="plots"):
    """
    Visualization 6: Tenure distribution
    """
    print("\n" + "=" * 65)
    print("6. TENURE DISTRIBUTION")
    print("=" * 65)

    # Formatted horizontal summary table
    s = df["tenure"]
    stats_df = pd.DataFrame([{
        "Count": int(s.count()),
        "Mean": round(s.mean(), 2),
        "Median": round(s.median(), 2),
        "Std Dev": round(s.std(), 2),
        "Min": round(s.min(), 2),
        "25%": round(s.quantile(0.25), 2),
        "75%": round(s.quantile(0.75), 2),
        "Max": round(s.max(), 2)
    }])
    print(stats_df.to_string(index=False))

    mean_val = s.mean()
    median_val = s.median()

    plt.figure(figsize=(9, 5))
    sns.histplot(s, kde=True, bins=30, color="#2b7bba", edgecolor="white")
    plt.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f} mos")
    plt.axvline(median_val, color="#7570b3", linestyle="-", linewidth=2, label=f"Median: {median_val:.1f} mos")

    plt.title("Overall Distribution of Customer Tenure", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Tenure (Months)", fontsize=12, labelpad=8)
    plt.ylabel("Number of Customers", fontsize=12, labelpad=8)
    plt.legend(fontsize=10)

    file_path = os.path.join(output_dir, "06_tenure_distribution.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def plot_monthly_charges_distribution(df, output_dir="plots"):
    """
    Visualization 7: MonthlyCharges distribution
    """
    print("\n" + "=" * 65)
    print("7. MONTHLY CHARGES DISTRIBUTION")
    print("=" * 65)

    # Formatted horizontal summary table
    s = df["MonthlyCharges"]
    stats_df = pd.DataFrame([{
        "Count": int(s.count()),
        "Mean ($)": round(s.mean(), 2),
        "Median ($)": round(s.median(), 2),
        "Std Dev ($)": round(s.std(), 2),
        "Min ($)": round(s.min(), 2),
        "25% ($)": round(s.quantile(0.25), 2),
        "75% ($)": round(s.quantile(0.75), 2),
        "Max ($)": round(s.max(), 2)
    }])
    print(stats_df.to_string(index=False))

    mean_val = s.mean()
    median_val = s.median()

    plt.figure(figsize=(9, 5))
    sns.histplot(s, kde=True, bins=30, color="#1b9e77", edgecolor="white")
    plt.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: ${mean_val:.2f}")
    plt.axvline(median_val, color="#7570b3", linestyle="-", linewidth=2, label=f"Median: ${median_val:.2f}")

    plt.title("Overall Distribution of Monthly Charges", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Monthly Charges ($)", fontsize=12, labelpad=8)
    plt.ylabel("Number of Customers", fontsize=12, labelpad=8)
    plt.legend(fontsize=10)

    file_path = os.path.join(output_dir, "07_monthly_charges_distribution.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plot Saved] {file_path}")


def run_phase4_eda(df, output_dir="plots"):
    """
    Executes all Phase 4 EDA visualizations and statistical analyses.
    """
    if df is None:
        print("[ERROR] DataFrame is None. Cannot perform Phase 4 EDA.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 65)
    print("# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("#" * 65)

    df_eda = prepare_eda_data(df)

    plot_churn_distribution(df_eda, output_dir)
    plot_churn_by_contract(df_eda, output_dir)
    plot_churn_by_tenure(df_eda, output_dir)
    plot_churn_by_monthly_charges(df_eda, output_dir)
    plot_churn_by_payment_method(df_eda, output_dir)
    plot_tenure_distribution(df_eda, output_dir)
    plot_monthly_charges_distribution(df_eda, output_dir)

    print("\n" + "#" * 65)
    print(f"# Phase 4 EDA Completed Successfully! All 7 plots saved to: '{output_dir}/'")
    print("#" * 65)


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

    run_phase4_eda(cleaned_df)
