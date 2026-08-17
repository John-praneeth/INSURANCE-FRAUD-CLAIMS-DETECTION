"""
Script to generate all EDA and model charts for the Academic Internship Report.
"""

import os
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, prepare_target
from src.feature_engineering import create_engineered_features
from src.preprocessing import drop_uninformative_columns

output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Helvetica', 'font.size': 10})

df = load_raw_data("data/raw/insurance_claims.csv")

# 1. Target Distribution Pie Chart
plt.figure(figsize=(6, 5))
fraud_counts = df["fraud_reported"].value_counts()
colors = ["#2563eb", "#dc2626"]
plt.pie(
    fraud_counts,
    labels=[f"Legitimate ({fraud_counts['N']} - {fraud_counts['N']/len(df)*100:.1f}%)", 
            f"Fraudulent ({fraud_counts['Y']} - {fraud_counts['Y']/len(df)*100:.1f}%)"],
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    explode=(0, 0.08),
    shadow=True,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)
plt.title("Overall Claim Distribution (Legitimate vs Fraudulent)", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_target_distribution.png"), dpi=300)
plt.close()

# 2. Incident Severity vs Fraud Rate
plt.figure(figsize=(7, 5))
ax = sns.countplot(
    data=df,
    x="incident_severity",
    hue="fraud_reported",
    palette={"N": "#2563eb", "Y": "#dc2626"},
    order=["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
)
plt.title("Incident Severity vs Fraud Reported", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Incident Severity Level", fontweight='bold')
plt.ylabel("Number of Claims", fontweight='bold')
plt.legend(title="Fraud Reported", labels=["No (Legitimate)", "Yes (Fraud)"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_incident_severity.png"), dpi=300)
plt.close()

# 3. Insured Hobbies vs Fraud Rate (Top 10 Hobbies)
plt.figure(figsize=(9, 5))
top_hobbies = df["insured_hobbies"].value_counts().head(8).index
df_hobbies = df[df["insured_hobbies"].isin(top_hobbies)]
ax = sns.countplot(
    data=df_hobbies,
    x="insured_hobbies",
    hue="fraud_reported",
    palette={"N": "#2563eb", "Y": "#dc2626"},
    order=top_hobbies
)
plt.title("Insured Hobbies vs Fraud Reported (Top Categories)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Insured Hobbies", fontweight='bold')
plt.ylabel("Claim Count", fontweight='bold')
plt.xticks(rotation=20)
plt.legend(title="Fraud Reported", labels=["No (Legitimate)", "Yes (Fraud)"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_insured_hobbies.png"), dpi=300)
plt.close()

# 4. Total Claim Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="total_claim_amount",
    hue="fraud_reported",
    palette={"N": "#2563eb", "Y": "#dc2626"},
    kde=True,
    bins=30,
    element="step"
)
plt.title("Total Claim Amount Distribution by Fraud Status", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Total Claim Amount ($)", fontweight='bold')
plt.ylabel("Frequency", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_claim_amount_distribution.png"), dpi=300)
plt.close()

# 5. Top 10 Feature Importances from Trained Model
model_pipeline = joblib.load("models/best_model.joblib")
classifier = model_pipeline.named_steps["classifier"]
preprocessor = model_pipeline.named_steps["preprocessor"]

# Get feature names from preprocessor
try:
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(9, 5))
    clean_labels = [col.replace("num__", "").replace("cat__", "").replace("remainder__", "") for col in feat_imp.index]
    sns.barplot(x=feat_imp.values, y=clean_labels, palette="viridis")
    plt.title("Top 10 Features Driving XGBoost Fraud Risk Predictions", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Relative Feature Importance (Weight)", fontweight='bold')
    plt.ylabel("Predictor Features", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_top_feature_importance.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Feature importance plot skipped: {e}")

print("All report figures generated successfully in reports/figures/!")
