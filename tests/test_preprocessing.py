"""
Preprocessing and feature engineering unit tests.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_loader import load_raw_data, prepare_target
from src.feature_engineering import create_engineered_features
from src.preprocessing import drop_uninformative_columns, get_feature_types, create_preprocessor_pipeline


def test_feature_engineering():
    df = load_raw_data("data/raw/insurance_claims.csv")
    X, _ = prepare_target(df)
    
    X_fe = create_engineered_features(X)
    assert "policy_tenure_days" in X_fe.columns
    assert "vehicle_age_at_incident" in X_fe.columns
    assert "claim_to_premium_ratio" in X_fe.columns
    assert "policy_bind_date" not in X_fe.columns


def test_drop_uninformative_columns():
    df = load_raw_data("data/raw/insurance_claims.csv")
    X, _ = prepare_target(df)
    X_clean = drop_uninformative_columns(X)
    
    assert "policy_number" not in X_clean.columns
    assert "insured_zip" not in X_clean.columns
    assert "incident_location" not in X_clean.columns


def test_preprocessor_pipeline():
    df = load_raw_data("data/raw/insurance_claims.csv")
    X, _ = prepare_target(df)
    X_clean = drop_uninformative_columns(create_engineered_features(X))
    
    num_cols, cat_cols = get_feature_types(X_clean)
    preprocessor = create_preprocessor_pipeline(num_cols, cat_cols)
    X_trans = preprocessor.fit_transform(X_clean)
    
    assert isinstance(X_trans, np.ndarray)
    assert not np.isnan(X_trans).any()
