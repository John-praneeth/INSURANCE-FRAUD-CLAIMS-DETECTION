"""
Script to programmatically generate and execute all 6 required Jupyter Notebooks.
"""

import os
import json
import nbformat as nbf


def make_cell_code(source: str):
    return nbf.v4.new_code_cell(source)


def make_cell_md(source: str):
    return nbf.v4.new_markdown_cell(source)


def build_nb_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("# Notebook 01: Data Understanding & Initial Quality Audit\n\n**Project Title**: Insurance Fraud Claims Detection Engine\n\n### Objective:\nPerform an empirical audit of the Kaggle `insurance_claims.csv` dataset, analyzing shape, column data types, missing value encoding, duplicate records, target distribution, and potential data leakage fields."),
        make_cell_code("""import os
import pandas as pd
import numpy as np

# Load raw dataset from source of truth
data_path = '../data/raw/insurance_claims.csv' if os.path.exists('../data/raw/insurance_claims.csv') else 'data/raw/insurance_claims.csv'
df = pd.read_csv(data_path)

print("1. Dataset Shape:", df.shape)
print("2. Column Names (Total", len(df.columns), "):")
print(df.columns.tolist())
"""),
        make_cell_code("""# 3. Data Types Summary
print("--- Data Types ---")
print(df.dtypes.value_counts())
print("\\n--- Detailed Dtypes ---")
print(df.dtypes)
"""),
        make_cell_code("""# 4. Missing Values Analysis
print("--- Standard NaN Counts ---")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])

print("\\n--- Question Mark '?' String Missing Values ---")
for col in df.columns:
    q_count = (df[col] == '?').sum()
    if q_count > 0:
        print(f"  {col}: {q_count} records ({q_count / len(df) * 100:.2f}%)")
"""),
        make_cell_code("""# 5. Target Variable Distribution
print("--- Target Distribution (fraud_reported) ---")
print(df['fraud_reported'].value_counts())
print("\\n--- Target Percentage ---")
print(df['fraud_reported'].value_counts(normalize=True) * 100)
"""),
        make_cell_code("""# 6. Duplicate Rows & Identifiers Audit
print("Duplicate Rows:", df.duplicated().sum())
print("Unique Policy Numbers:", df['policy_number'].nunique())
print("Unique Insured Zips:", df['insured_zip'].nunique())
print("Unique Incident Locations:", df['incident_location'].nunique())
""")
    ]
    return nb


def build_nb_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("# Notebook 02: Exploratory Data Analysis (EDA)\n\nVisualizing demographic, policy, vehicle, and incident distributions against fraud outcomes (`fraud_reported`)."),
        make_cell_code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

data_path = '../data/raw/insurance_claims.csv' if os.path.exists('../data/raw/insurance_claims.csv') else 'data/raw/insurance_claims.csv'
df = pd.read_csv(data_path)
df['fraud_binary'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
"""),
        make_cell_code("""# 1. Target Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='fraud_reported', palette=['#2b5c8f', '#e05d06'])
plt.title('Fraud Reported Count Distribution (Target)')
plt.ylabel('Count')
plt.xlabel('Fraud Reported')
plt.show()
"""),
        make_cell_code("""# 2. Fraud Rate by Incident Severity
plt.figure(figsize=(8, 4))
severity_df = df.groupby('incident_severity')['fraud_binary'].agg(['count', 'mean']).reset_index()
severity_df.columns = ['incident_severity', 'total_claims', 'fraud_rate']

sns.barplot(data=severity_df, x='incident_severity', y='fraud_rate', palette='Oranges_r')
plt.title('Fraud Rate by Incident Severity')
plt.ylabel('Fraud Rate')
plt.show()
print(severity_df)
"""),
        make_cell_code("""# 3. Numerical Features Distribution: Total Claim Amount by Fraud
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='fraud_reported', y='total_claim_amount', palette=['#2b5c8f', '#e05d06'])
plt.title('Total Claim Amount Distribution by Fraud Status')
plt.show()
"""),
        make_cell_code("""# 4. Correlation Matrix of Numerical Attributes
num_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(num_df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Numerical Feature Correlation Matrix')
plt.show()
""")
    ]
    return nb


def build_nb_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("# Notebook 03: Feature Engineering & Preprocessing Pipeline Design\n\nDesigning feature engineering transformers and leak-free `ColumnTransformer` pipelines."),
        make_cell_code("""import os, sys
sys.path.append('..')
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, prepare_target
from src.feature_engineering import create_engineered_features
from src.preprocessing import drop_uninformative_columns, get_feature_types, create_preprocessor_pipeline

data_path = '../data/raw/insurance_claims.csv' if os.path.exists('../data/raw/insurance_claims.csv') else 'data/raw/insurance_claims.csv'
df = load_raw_data(data_path)
X, y = prepare_target(df)
print("Base Features Shape:", X.shape)
"""),
        make_cell_code("""# Apply Feature Engineering
X_fe = create_engineered_features(X)
X_clean = drop_uninformative_columns(X_fe)

print("Engineered Features Shape:", X_clean.shape)
print("Engineered Columns Added:")
print([c for c in X_clean.columns if c not in X.columns])
"""),
        make_cell_code("""# Fit Preprocessor
num_cols, cat_cols = get_feature_types(X_clean)
preprocessor = create_preprocessor_pipeline(num_cols, cat_cols)
X_trans = preprocessor.fit_transform(X_clean)

print("Transformed Matrix Shape:", X_trans.shape)
print("No NaNs in output matrix:", not np.isnan(X_trans).any())
""")
    ]
    return nb


def build_nb_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("# Notebook 04: Baseline Model Benchmarking & Hyperparameter Tuning\n\nTraining candidate models using 5-fold Stratified CV with SMOTE for imbalance handling, followed by XGBoost hyperparameter tuning."),
        make_cell_code("""import os, sys
sys.path.append('..')
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, prepare_target
from src.train import train_and_evaluate_all

# Run complete reproducible training process
train_and_evaluate_all()
"""),
        make_cell_code("""# View Model Comparison Results Table
df_res = pd.read_csv('../reports/results/model_metrics.csv' if os.path.exists('../reports/results/model_metrics.csv') else 'reports/results/model_metrics.csv')
print("--- Baseline Model Benchmark Table ---")
print(df_res.to_string(index=False))
""")
    ]
    return nb


def build_nb_05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("# Notebook 05: Final Model Evaluation & Operating Threshold Sweep\n\nEvaluating the tuned model on the untouched test set across multiple probability decision thresholds."),
        make_cell_code("""import os, sys, json
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

config_path = '../models/threshold_config.json' if os.path.exists('../models/threshold_config.json') else 'models/threshold_config.json'
with open(config_path) as f:
    cfg = json.load(f)

print("=== FINAL MODEL PERFORMANCE SUMMARY ===")
print("Optimal Threshold:", cfg['optimal_threshold'])
for k, v in cfg['metrics'].items():
    print(f"  {k:<20}: {v}")
"""),
        make_cell_code("""# Display Saved Evaluation Plots
from IPython.display import Image, display

fig_dir = '../reports/figures' if os.path.exists('../reports/figures') else 'reports/figures'
for fig_name in ['XGBoost_Optimized_confusion_matrix.png', 'XGBoost_Optimized_roc_curve.png', 'XGBoost_Optimized_pr_curve.png', 'XGBoost_Optimized_threshold_sweep.png']:
    fig_path = os.path.join(fig_dir, fig_name)
    if os.path.exists(fig_path):
        print(f"Rendering {fig_name}:")
        display(Image(filename=fig_path))
""")
    ]
    return nb


def build_nb_06():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("# Notebook 06: Model Explainability & SHAP Analysis\n\nUsing SHAP (SHapley Additive exPlanations) to explain global feature importance and individual claim risk predictions."),
        make_cell_code("""import os, sys, joblib
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

model_path = '../models/best_model.joblib' if os.path.exists('../models/best_model.joblib') else 'models/best_model.joblib'
pipeline = joblib.load(model_path)

test_path = '../data/processed/test.csv' if os.path.exists('../data/processed/test.csv') else 'data/processed/test.csv'
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=['fraud_reported'])
y_test = test_df['fraud_reported']

from src.feature_engineering import create_engineered_features
from src.preprocessing import drop_uninformative_columns

X_test_clean = drop_uninformative_columns(create_engineered_features(X_test))
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

X_test_trans = preprocessor.transform(X_test_clean)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_test_trans)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

print("SHAP Explainer initialized successfully! Shape:", np.shape(shap_vals))
"""),
        make_cell_code("""# Global Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals, X_test_trans, feature_names=feature_names, plot_type="bar", show=False)
plt.title("Top Global Features Influencing Insurance Fraud Risk")
plt.tight_layout()
plt.show()
""")
    ]
    return nb


def generate_all():
    os.makedirs("notebooks", exist_ok=True)
    nb_builders = [
        ("notebooks/01_data_understanding.ipynb", build_nb_01),
        ("notebooks/02_eda.ipynb", build_nb_02),
        ("notebooks/03_preprocessing.ipynb", build_nb_03),
        ("notebooks/04_model_training.ipynb", build_nb_04),
        ("notebooks/05_model_evaluation.ipynb", build_nb_05),
        ("notebooks/06_explainability.ipynb", build_nb_06),
    ]

    for path, builder in nb_builders:
        nb = builder()
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generated {path}")


if __name__ == "__main__":
    generate_all()
