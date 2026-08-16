"""
Data tests for raw dataset loading and target validation.
"""

import os
import pandas as pd
import pytest
from src.data_loader import load_raw_data, prepare_target


def test_data_loader_file_exists():
    assert os.path.exists("data/raw/insurance_claims.csv") or os.path.exists("insurance_claims.csv")


def test_load_raw_data_shape():
    df = load_raw_data("data/raw/insurance_claims.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "_c39" not in df.columns


def test_prepare_target():
    df = load_raw_data("data/raw/insurance_claims.csv")
    X, y = prepare_target(df, target_col="fraud_reported")
    
    assert "fraud_reported" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert (y == 1).sum() == 247
    assert (y == 0).sum() == 753
