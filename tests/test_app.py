"""
Streamlit app helper and predictor integration tests.
"""

import pandas as pd
import pytest
from src.predict import FraudPredictor


def test_batch_dataframe_prediction():
    predictor = FraudPredictor()
    test_df = pd.read_csv("data/processed/test.csv").head(5)
    X_test = test_df.drop(columns=["fraud_reported"])
    
    res_df = predictor.predict_dataframe(X_test)
    assert "fraud_probability" in res_df.columns
    assert "risk_level" in res_df.columns
    assert len(res_df) == 5
