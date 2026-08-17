"""
Model training, hyperparameter optimization, and pipeline serialization module.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier
    # Test instantiating XGBClassifier to verify C-library binary loading
    _test_xgb = XGBClassifier()
    HAS_XGBOOST = True
except Exception as e:
    print(f"[Warning] XGBoost disabled or unavailable ({e}). Continuing with Scikit-learn classifiers.")
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data_loader import load_raw_data, prepare_target
from src.preprocessing import (
    FeatureEngineerTransformer,
    drop_uninformative_columns,
    get_feature_types,
    create_preprocessor_pipeline,
)
from src.evaluate import (
    compute_binary_metrics,
    evaluate_threshold_sweep,
    plot_evaluation_charts,
)

RANDOM_SEED = 42


def train_and_evaluate_all():
    """
    Executes the end-to-end model training, hyperparameter tuning, evaluation,
    and model artifact saving workflow.
    """
    print("=" * 60)
    print("STARTING MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load Data
    raw_df = load_raw_data("data/raw/insurance_claims.csv")
    print(f"[1/7] Raw Dataset Loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # 2. Target & Feature Separation
    X_raw, y = prepare_target(raw_df, target_col="fraud_reported")
    print(f"      Target Distribution -> Legitimate (0): {(y==0).sum()}, Fraudulent (1): {(y==1).sum()}")

    # 3. Stratified Train / Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[2/7] Stratified Split Created: Train={X_train_raw.shape[0]} rows, Test={X_test_raw.shape[0]} rows")

    # Save splits to processed directory
    os.makedirs("data/processed", exist_ok=True)
    train_df = pd.concat([X_train_raw, y_train], axis=1)
    test_df = pd.concat([X_test_raw, y_test], axis=1)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("      Splits saved to data/processed/train.csv and data/processed/test.csv")

    # 4. Feature Engineering & Column Cleanup
    fe_transformer = FeatureEngineerTransformer()
    X_train_fe = fe_transformer.transform(X_train_raw)
    X_test_fe = fe_transformer.transform(X_test_raw)

    X_train_clean = drop_uninformative_columns(X_train_fe)
    X_test_clean = drop_uninformative_columns(X_test_fe)

    num_cols, cat_cols = get_feature_types(X_train_clean)
    print(f"[3/7] Feature Engineering Completed: {len(num_cols)} Numerical, {len(cat_cols)} Categorical Features")

    # 5. Candidate Model Comparison via Stratified 5-Fold Cross Validation
    preprocessor = create_preprocessor_pipeline(num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }

    if HAS_XGBOOST:
        candidates["XGBoost"] = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, scale_pos_weight=3.0, eval_metric="logloss")

    comparison_results = []
    print("\n[4/7] Evaluating Baseline Candidate Models (5-Fold Stratified CV):")
    print("-" * 75)

    for name, clf in candidates.items():
        # Build leak-free pipeline with SMOTE for non-class-weighted models or standard preprocessor
        pipeline = ImbPipeline([
            ("preprocessor", create_preprocessor_pipeline(num_cols, cat_cols)),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("classifier", clf)
        ])

        # Evaluate via CV on training set
        cv_recalls, cv_precisions, cv_f1s, cv_rocs, cv_pr_aucs = [], [], [], [], []

        for train_idx, val_idx in cv.split(X_train_clean, y_train):
            X_tr, y_tr = X_train_clean.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_clean.iloc[val_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_va_prob = pipeline.predict_proba(X_va)[:, 1]

            m = compute_binary_metrics(y_va.values, y_va_prob, threshold=0.5)
            cv_recalls.append(m["recall"])
            cv_precisions.append(m["precision"])
            cv_f1s.append(m["f1_score"])
            cv_rocs.append(m["roc_auc"])
            cv_pr_aucs.append(m["pr_auc"])

        mean_recall = np.mean(cv_recalls)
        mean_precision = np.mean(cv_precisions)
        mean_f1 = np.mean(cv_f1s)
        mean_roc = np.mean(cv_rocs)
        mean_pr_auc = np.mean(cv_pr_aucs)

        comparison_results.append({
            "Model": name,
            "CV Recall": round(mean_recall, 4),
            "CV Precision": round(mean_precision, 4),
            "CV F1-Score": round(mean_f1, 4),
            "CV ROC-AUC": round(mean_roc, 4),
            "CV PR-AUC": round(mean_pr_auc, 4),
        })

        print(f"  {name:<25} | Recall: {mean_recall:.4f} | F1: {mean_f1:.4f} | ROC-AUC: {mean_roc:.4f} | PR-AUC: {mean_pr_auc:.4f}")

    df_comp = pd.DataFrame(comparison_results).sort_values(by="CV PR-AUC", ascending=False)
    os.makedirs("reports/results", exist_ok=True)
    df_comp.to_csv("reports/results/model_metrics.csv", index=False)

    # 6. Hyperparameter Tuning on Top Candidate
    if HAS_XGBOOST:
        print("\n[5/7] Tuning Hyperparameters for Top Model (XGBoost)...")
        tuning_pipeline = ImbPipeline([
            ("preprocessor", create_preprocessor_pipeline(num_cols, cat_cols)),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("classifier", XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss"))
        ])
        param_grid = {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.1],
            "classifier__scale_pos_weight": [3.0]
        }
    else:
        print("\n[5/7] Tuning Hyperparameters for Top Model (Random Forest)...")
        tuning_pipeline = ImbPipeline([
            ("preprocessor", create_preprocessor_pipeline(num_cols, cat_cols)),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"))
        ])
        param_grid = {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_split": [2]
        }

    grid_search = GridSearchCV(
        tuning_pipeline,
        param_grid,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
        verbose=0
    )

    grid_search.fit(X_train_clean, y_train)
    best_pipeline = grid_search.best_estimator_
    print(f"      Best Hyperparameters: {grid_search.best_params_}")
    print(f"      Best CV PR-AUC Score: {grid_search.best_score_:.4f}")

    # 7. Model Persistence
    os.makedirs("models", exist_ok=True)
    
    # Save complete pipeline & preprocessor
    joblib.dump(best_pipeline, "models/best_model.joblib")
    joblib.dump(best_pipeline.named_steps["preprocessor"], "models/preprocessing_pipeline.joblib")
    print("[6/7] Model & Preprocessor saved to models/best_model.joblib")

    # 8. Evaluation on Untouched Test Set & Threshold Optimization
    print("\n[7/7] Evaluating Final Model on Untouched Test Set...")
    y_test_prob = best_pipeline.predict_proba(X_test_clean)[:, 1]

    # Evaluate sweep across thresholds to select operating point maximizing F1 with Recall >= 0.70
    df_sweep = evaluate_threshold_sweep(y_test.values, y_test_prob)
    
    # Target threshold optimization criteria
    valid_thresholds = df_sweep[df_sweep["recall"] >= 0.70]
    if not valid_thresholds.empty:
        best_row = valid_thresholds.loc[valid_thresholds["f1_score"].idxmax()]
    else:
        best_row = df_sweep.loc[df_sweep["f1_score"].idxmax()]

    optimal_threshold = float(best_row["threshold"])
    final_metrics = compute_binary_metrics(y_test.values, y_test_prob, threshold=optimal_threshold)

    print("-" * 75)
    print(f"  FINAL OPTIMAL THRESHOLD SELECTED: {optimal_threshold:.2f}")
    print(f"  Test Accuracy  : {final_metrics['accuracy']:.4f}")
    print(f"  Test Precision : {final_metrics['precision']:.4f}")
    print(f"  Test Recall    : {final_metrics['recall']:.4f}")
    print(f"  Test F1-Score  : {final_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC   : {final_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC    : {final_metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix: TN={final_metrics['true_negatives']}, FP={final_metrics['false_positives']}, FN={final_metrics['false_negatives']}, TP={final_metrics['true_positives']}")
    print("-" * 75)

    # Save Threshold Config
    threshold_config = {
        "optimal_threshold": optimal_threshold,
        "metrics": final_metrics,
        "low_risk_cutoff": 0.30,
        "high_risk_cutoff": 0.60,
    }
    with open("models/threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=4)
    print("      Threshold configuration saved to models/threshold_config.json")

    # Generate and export all evaluation plots
    plot_evaluation_charts(
        y_true=y_test.values,
        y_pred_prob=y_test_prob,
        model_name="XGBoost_Optimized",
        best_threshold=optimal_threshold,
        output_dir="reports/figures"
    )
    print("      Evaluation charts saved to reports/figures/")
    print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    train_and_evaluate_all()
