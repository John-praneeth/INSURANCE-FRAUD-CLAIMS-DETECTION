"""
Feature engineering module for domain-specific ratio calculation and date processing.
"""

import pandas as pd
import numpy as np


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from policy and incident dates, computes
    domain ratios, and handles domain transformation without introducing data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input features DataFrame.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with engineered features added and raw date columns removed.
    """
    df_out = df.copy()

    # Convert date columns to datetime objects
    if "policy_bind_date" in df_out.columns:
        df_out["policy_bind_date"] = pd.to_datetime(df_out["policy_bind_date"], errors="coerce")
    if "incident_date" in df_out.columns:
        df_out["incident_date"] = pd.to_datetime(df_out["incident_date"], errors="coerce")

    # Date calculations
    if "policy_bind_date" in df_out.columns and "incident_date" in df_out.columns:
        df_out["policy_tenure_days"] = (df_out["incident_date"] - df_out["policy_bind_date"]).dt.days
        df_out["policy_tenure_days"] = df_out["policy_tenure_days"].fillna(df_out["months_as_customer"] * 30)

    if "incident_date" in df_out.columns:
        df_out["incident_year"] = df_out["incident_date"].dt.year
        df_out["incident_month"] = df_out["incident_date"].dt.month
        df_out["incident_day"] = df_out["incident_date"].dt.day
        df_out["incident_day_of_week"] = df_out["incident_date"].dt.dayofweek
        # Drop raw date objects
        df_out = df_out.drop(columns=["incident_date"])

    if "policy_bind_date" in df_out.columns:
        df_out["policy_bind_year"] = df_out["policy_bind_date"].dt.year
        df_out = df_out.drop(columns=["policy_bind_date"])

    # Vehicle age at incident
    if "incident_year" in df_out.columns and "auto_year" in df_out.columns:
        df_out["vehicle_age_at_incident"] = df_out["incident_year"] - df_out["auto_year"]
    elif "auto_year" in df_out.columns:
        df_out["vehicle_age_at_incident"] = 2015 - df_out["auto_year"]

    # Claim component ratios
    if "total_claim_amount" in df_out.columns:
        total = np.maximum(df_out["total_claim_amount"].values, 1.0)
        
        if "policy_annual_premium" in df_out.columns:
            premium = np.maximum(df_out["policy_annual_premium"].values, 1.0)
            df_out["claim_to_premium_ratio"] = total / premium

        if "injury_claim" in df_out.columns:
            df_out["injury_claim_ratio"] = df_out["injury_claim"] / total

        if "property_claim" in df_out.columns:
            df_out["property_claim_ratio"] = df_out["property_claim"] / total

        if "vehicle_claim" in df_out.columns:
            df_out["vehicle_claim_ratio"] = df_out["vehicle_claim"] / total

    # Handle missing string representations '?' by converting them to explicit string 'MISSING'
    for col in df_out.select_dtypes(include=["object", "string"]).columns:
        df_out[col] = df_out[col].replace("?", "MISSING").fillna("MISSING")

    return df_out
