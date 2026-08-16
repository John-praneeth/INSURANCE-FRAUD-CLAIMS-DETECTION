"""
Inference module for processing individual or batch insurance claim predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

from src.feature_engineering import create_engineered_features
from src.preprocessing import drop_uninformative_columns


class FraudPredictor:
    """Production predictor class loading saved model artifact and threshold configs."""

    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        config_path: str = "models/threshold_config.json"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

        self.model_pipeline = joblib.load(model_path)
        
        self.threshold = 0.40
        self.low_cutoff = 0.30
        self.high_cutoff = 0.60

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                self.threshold = cfg.get("optimal_threshold", self.threshold)
                self.low_cutoff = cfg.get("low_risk_cutoff", self.low_cutoff)
                self.high_cutoff = cfg.get("high_risk_cutoff", self.high_cutoff)

    def predict_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a raw input DataFrame and returns probabilities, risk tiers, and recommendations.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw claim records matching the dataset schema.

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame augmented with prediction results.
        """
        df_fe = create_engineered_features(df_raw)
        df_clean = drop_uninformative_columns(df_fe)

        # Predict probability for class 1
        probs = self.model_pipeline.predict_proba(df_clean)[:, 1]

        results = []
        for p in probs:
            p_val = float(p)
            if p_val < self.low_cutoff:
                risk = "LOW"
                rec = "Normal processing, subject to standard controls."
                flag = "Legitimate"
            elif p_val < self.high_cutoff:
                risk = "MEDIUM"
                rec = "Require additional verification before payout."
                flag = "Potentially Suspicious"
            else:
                risk = "HIGH"
                rec = "Flag for immediate manual investigation."
                flag = "Suspicious / Potentially Fraudulent"

            results.append({
                "fraud_probability": round(p_val, 4),
                "fraud_probability_percent": f"{p_val * 100:.1f}%",
                "risk_level": risk,
                "prediction": flag,
                "recommended_action": rec
            })

        res_df = pd.DataFrame(results)
        return pd.concat([df_raw.reset_index(drop=True), res_df], axis=1)

    def predict_single_claim(self, claim_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts fraud risk for a single claim dictionary input.

        Parameters
        ----------
        claim_dict : Dict[str, Any]
            Dictionary containing claim field attributes.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing prediction probability, risk category, and recommendation.
        """
        df_single = pd.DataFrame([claim_dict])
        res_df = self.predict_dataframe(df_single)
        row = res_df.iloc[0]

        return {
            "fraud_probability": float(row["fraud_probability"]),
            "fraud_probability_percent": str(row["fraud_probability_percent"]),
            "risk_level": str(row["risk_level"]),
            "prediction": str(row["prediction"]),
            "recommended_action": str(row["recommended_action"])
        }
