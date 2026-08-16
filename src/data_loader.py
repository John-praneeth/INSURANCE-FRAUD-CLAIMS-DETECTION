"""
Data loading and initial dataset cleaning module.
"""

import os
import pandas as pd
from typing import Tuple, Optional


def load_raw_data(data_path: str = "data/raw/insurance_claims.csv") -> pd.DataFrame:
    """
    Loads the raw insurance claims CSV dataset and performs initial sanitization.

    Parameters
    ----------
    data_path : str
        Path to the insurance_claims.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned initial dataframe with invalid trailing columns dropped.
    """
    possible_paths = [
        data_path,
        os.path.join("..", data_path),
        "data/raw/insurance_claims.csv",
        "../data/raw/insurance_claims.csv",
        "insurance_claims.csv",
        "../insurance_claims.csv",
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break

    if found_path is None:
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_csv(found_path)

    # Remove unneeded empty columns like '_c39' if present
    if "_c39" in df.columns:
        df = df.drop(columns=["_c39"])

    # Drop fully null columns if any
    df = df.dropna(how="all", axis=1)

    return df


def prepare_target(df: pd.DataFrame, target_col: str = "fraud_reported") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates features and binary-encoded target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column ('fraud_reported').

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature DataFrame (X) and Target Series (y) where 1=Fraud (Y), 0=Legitimate (N).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns.")

    X = df.drop(columns=[target_col]).copy()
    
    # Target conversion: Y -> 1, N -> 0
    y_raw = df[target_col].astype(str).str.strip().str.upper()
    y = y_raw.map({"Y": 1, "N": 0})
    
    if y.isnull().any():
        raise ValueError("Target mapping produced null values. Expected values 'Y' or 'N'.")

    return X, y
