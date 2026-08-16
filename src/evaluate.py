"""
Model evaluation, threshold tuning, and visualization reporting module.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, roc_curve
)


def compute_binary_metrics(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification evaluation metrics for a given decision threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1).
    y_pred_prob : np.ndarray
        Predicted probabilities for target class 1.
    threshold : float
        Decision probability cutoff.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing calculated metrics and confusion matrix breakdown.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate Precision-Recall AUC
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_pts, precision_pts)
    
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics


def evaluate_threshold_sweep(
    y_true: np.ndarray, y_pred_prob: np.ndarray, thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Evaluates classification performance across multiple probability thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True ground truth binary target values.
    y_pred_prob : np.ndarray
        Predicted probabilities.
    thresholds : np.ndarray
        Array of threshold values to test. Defaults to 0.10 to 0.90 in increments of 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing metrics across evaluated thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.05)

    records = []
    for th in thresholds:
        m = compute_binary_metrics(y_true, y_pred_prob, threshold=th)
        records.append(m)

    df_sweep = pd.DataFrame(records)
    return df_sweep


def plot_evaluation_charts(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    model_name: str = "Best_Model",
    best_threshold: float = 0.5,
    output_dir: str = "reports/figures"
):
    """
    Generates and saves ROC curve, PR curve, Confusion Matrix, and Threshold Sweep plots.

    Parameters
    ----------
    y_true : np.ndarray
        True binary target values.
    y_pred_prob : np.ndarray
        Predicted probabilities.
    model_name : str
        Name of model for figure titles.
    best_threshold : float
        Selected decision threshold.
    output_dir : str
        Directory where figures will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"]
    )
    plt.title(f"Confusion Matrix - {model_name} (Th={best_threshold:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"), dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_pts, precision_pts)

    plt.figure(figsize=(7, 6))
    plt.plot(recall_pts, precision_pts, color="#e05d06", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_pr_curve.png"), dpi=300)
    plt.close()

    # 4. Threshold Sweep Plot
    df_sweep = evaluate_threshold_sweep(y_true, y_pred_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(df_sweep["threshold"], df_sweep["precision"], marker="o", label="Precision", color="green")
    plt.plot(df_sweep["threshold"], df_sweep["recall"], marker="s", label="Recall", color="red")
    plt.plot(df_sweep["threshold"], df_sweep["f1_score"], marker="^", label="F1-Score", color="blue")
    plt.axvline(x=best_threshold, color="black", linestyle="--", label=f"Optimal Th ({best_threshold:.2f})")
    plt.xlabel("Decision Probability Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs. Performance Metrics")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_threshold_sweep.png"), dpi=300)
    plt.close()
