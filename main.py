"""
main.py
Main entry point for the Customer Lifetime Value (CLV)
and Churn Risk Analytics Engine project.
"""

from load_data import load_dataset, inspect_dataset
from explore_data import run_phase2_exploration


def main():
    print("==================================================================")
    print(" Customer Lifetime Value (CLV) & Churn Risk Analytics Engine")
    print("==================================================================\n")

    # Step 1: Phase 1 - Load dataset and perform initial inspection
    print(">>> [PHASE 1] Loading Dataset...")
    df = load_dataset()

    if df is None:
        print("\n[FAILED] Unable to load dataset. Please verify the CSV file location.")
        return

    inspect_dataset(df)

    # Step 2: Phase 2 - Run basic data understanding & statistical exploration
    run_phase2_exploration(df)


if __name__ == "__main__":
    main()
