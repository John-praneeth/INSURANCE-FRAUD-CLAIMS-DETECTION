"""
Preprocessing pipeline creation and column transformation module.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Uninformative identifier columns that cause overfitting or leakage
IDENTIFIER_COLUMNS = ["policy_number", "insured_zip", "incident_location"]


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for custom feature engineering."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from src.feature_engineering import create_engineered_features
        return create_engineered_features(X)


def drop_uninformative_columns(df: pd.DataFrame, drop_cols: List[str] = IDENTIFIER_COLUMNS) -> pd.DataFrame:
    """
    Drops identifier columns that do not contribute to predictive modeling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.
    drop_cols : List[str]
        List of column names to drop.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns removed.
    """
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical column names from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.

    Returns
    -------
    Tuple[List[str], List[str]]
        Numerical column names list, Categorical column names list.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def create_preprocessor_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Creates a ColumnTransformer pipeline for standard scaling of numerical variables
    and one-hot encoding of categorical variables.

    Parameters
    ----------
    num_cols : List[str]
        List of numerical feature names.
    cat_cols : List[str]
        List of categorical feature names.

    Returns
    -------
    ColumnTransformer
        Fitted or un-fitted scikit-learn ColumnTransformer object.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor
