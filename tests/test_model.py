"""
Unit tests for model inference, thresholds, and artifact integrity.
"""

import os
import joblib
import pandas as pd
import pytest
from src.predict import FraudPredictor


def test_model_artifacts_exist():
    assert os.path.exists("models/best_model.joblib")
    assert os.path.exists("models/preprocessing_pipeline.joblib")
    assert os.path.exists("models/threshold_config.json")


def test_predictor_single_claim():
    predictor = FraudPredictor()
    
    sample_claim = {
        "months_as_customer": 128,
        "age": 34,
        "policy_number": 999999,
        "policy_bind_date": "2014-05-15",
        "policy_state": "OH",
        "policy_csl": "250/500",
        "policy_deductable": 1000,
        "policy_annual_premium": 1250.0,
        "umbrella_limit": 0,
        "insured_zip": 43081,
        "insured_sex": "MALE",
        "insured_education_level": "MD",
        "insured_occupation": "craft-repair",
        "insured_hobbies": "chess",
        "insured_relationship": "husband",
        "capital-gains": 0,
        "capital-loss": 0,
        "incident_date": "2015-02-17",
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Side Collision",
        "incident_severity": "Major Damage",
        "authorities_contacted": "Police",
        "incident_state": "NY",
        "incident_city": "Columbus",
        "incident_location": "9999 Sample St",
        "incident_hour_of_the_day": 3,
        "number_of_vehicles_involved": 1,
        "property_damage": "YES",
        "bodily_injuries": 2,
        "witnesses": 1,
        "police_report_available": "NO",
        "total_claim_amount": 75000,
        "injury_claim": 15000,
        "property_claim": 15000,
        "vehicle_claim": 45000,
        "auto_make": "Dodge",
        "auto_model": "RAM",
        "auto_year": 2007,
    }
    
    res = predictor.predict_single_claim(sample_claim)
    assert "fraud_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(res["recommended_action"], str)
