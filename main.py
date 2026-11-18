"""
main.py
Main entry point for Phase 1 of the Customer Lifetime Value (CLV)
and Churn Risk Analytics Engine project.
"""

from load_data import load_dataset, inspect_dataset


def main():
    print("==================================================================")
    print(" Customer Lifetime Value (CLV) & Churn Risk Analytics Engine")
    print(" Phase 1: Basic Data Loading and Inspection")
    print("==================================================================\n")

    # Step 1: Call the data-loading functionality
    df = load_dataset()

    # Step 2: Run basic data inspection if loading was successful
    if df is not None:
        inspect_dataset(df)
        print("\n[INFO] Phase 1 data loading and inspection completed successfully!")
    else:
        print("\n[FAILED] Unable to load dataset. Please check the file path and try again.")


if __name__ == "__main__":
    main()
