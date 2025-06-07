from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the credit scoring dataset.

    Steps:
    1. Remove unnecessary index column
    2. Fill missing values with median
    3. Remove extreme outliers in DebtRatio
    4. Remove extreme values in RevolvingUtilizationOfUnsecuredLines
    5. Cap high values in delinquency columns
    6. Split features and target

    Args:
        df: Raw input DataFrame

    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the target series
    """
    # 1. Remove unnecessary index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2. Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # 3. Remove extreme outliers in DebtRatio
    df = df[df["DebtRatio"] <= 3489.025]

    # 4. Remove extreme values in RevolvingUtilizationOfUnsecuredLines
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 10]

    # 5. Cap high values in delinquency columns
    cap_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in cap_cols:
        df.loc[df[col] > 90, col] = 18

    # 6. Split features and target
    y = df["SeriousDlqin2yrs"]
    X = df.drop(columns=["SeriousDlqin2yrs"])

    return X, y


def main():
    """Main function to run preprocessing pipeline."""
    # Define paths
    data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df = load_data(data_dir / "cs-training.csv")
    X, y = preprocess_data(df)

    # Save processed data
    X.to_csv(processed_dir / "X_processed.csv", index=False)
    y.to_csv(processed_dir / "y_processed.csv", index=False)

    print("Data preprocessing completed successfully!")
    print(f"Processed data saved to {processed_dir}")


if __name__ == "__main__":
    main()
