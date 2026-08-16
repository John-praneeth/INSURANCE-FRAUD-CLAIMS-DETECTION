# Insurance Fraud Claims Detection Engine 🛡️

An end-to-end Machine Learning classification and risk-screening engine designed to analyze automobile insurance claims and flag suspicious records for human investigation.

---

## 1. Project Title
**Insurance Fraud Claims Detection Engine: Machine Learning Risk Screening Framework**

## 2. Problem Statement
Automobile insurance fraud causes billions of dollars in annual losses across the insurance industry. Fraudulent claims range from staged accidents and exaggerated property damage to falsified injury reports. Manual review of every submitted claim is cost-prohibitive and inefficient, while blanket automated approvals expose insurers to severe financial leakage. 

## 3. Objective
Build a supervised machine learning decision-support system that predicts whether an automobile insurance claim is:
* **Legitimate (Class 0)**
* **Suspicious / Potentially Fraudulent (Class 1)**

The system prioritizes suspicious claims for human claims adjusters while providing configurable risk levels (**LOW**, **MEDIUM**, **HIGH**) and clear feature-level recommendations.

> **Important Operational Disclaimer**: This system operates as a **fraud-risk screening tool**, NOT a system that conclusively declares a customer fraudulent. Predictions serve as decision support requiring human review.

---

## 4. Dataset
* **Source**: Kaggle — Auto Insurance Claims Data (`buntyshah/auto-insurance-claims-data`)
* **Primary Source File**: `insurance_claims.csv`
* **Dataset Shape**: 1,000 automobile insurance claim records, 40 attributes.
* **Target Column**: `fraud_reported` ('Y' = Fraudulent, 'N' = Legitimate)

## 5. Dataset Features
The dataset contains 39 predictor features categorized into:
* **Customer Demographics**: `age`, `months_as_customer`, `insured_sex`, `insured_education_level`, `insured_occupation`, `insured_hobbies`, `insured_relationship`, `capital-gains`, `capital-loss`.
* **Policy Details**: `policy_number`, `policy_bind_date`, `policy_state`, `policy_csl`, `policy_deductable`, `policy_annual_premium`, `umbrella_limit`, `insured_zip`.
* **Incident Metrics**: `incident_date`, `incident_type`, `collision_type`, `incident_severity`, `authorities_contacted`, `incident_state`, `incident_city`, `incident_location`, `incident_hour_of_the_day`, `number_of_vehicles_involved`, `property_damage`, `bodily_injuries`, `witnesses`, `police_report_available`.
* **Claim Amounts & Vehicle Attributes**: `total_claim_amount`, `injury_claim`, `property_claim`, `vehicle_claim`, `auto_make`, `auto_model`, `auto_year`.

---

## 6. Methodology
1. **Data Ingestion & Integrity Audit**: Validation of columns, data types, missing value representations (`'?'`), and dropping uninformative trailing columns (`_c39`).
2. **Stratified Partitioning**: 80/20 train/test split with stratification on `fraud_reported`.
3. **Domain Feature Engineering**: Time-lapse extraction (`policy_tenure_days`, `vehicle_age_at_incident`, `incident_day_of_week`) and claim component ratio calculations.
4. **Leak-Free Preprocessing**: Custom `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
5. **Class Imbalance Management**: Evaluating SMOTE oversampling vs Class Weighting within cross-validation folds.
6. **Model Benchmarking & Tuning**: Evaluating Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, and XGBoost via 5-Fold Stratified Cross-Validation.
7. **Threshold Sweep & Optimization**: Evaluating operating decision thresholds (0.10 to 0.90) to maximize Target Class Recall while maintaining Precision.

---

## 7. Technology Stack
* **Core**: Python 3.11+
* **Data Processing & Analytics**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn, Plotly
* **Machine Learning**: Scikit-Learn, Imbalanced-Learn (`imblearn`), XGBoost
* **Model Serialization**: Joblib
* **Web Deployment**: Streamlit
* **Model Explainability**: SHAP (SHapley Additive exPlanations)
* **Testing**: Pytest

---

## 8. Architecture & Pipeline Workflow

```
[Raw CSV: insurance_claims.csv] 
              │
              ▼
    [data_loader.py] ─── Target Conversion ('Y'->1, 'N'->0)
              │
              ▼
 [Stratified Train / Test Split] (80% Train / 20% Test)
              │
              ▼
 [feature_engineering.py] ── Date decomposition & ratio metrics
              │
              ▼
  [preprocessing.py] ────── Leak-free ColumnTransformer & Imputer
              │
              ▼
     [train.py] ────────── 5-Fold Stratified CV + SMOTE + XGBoost Tuning
              │
              ▼
    [evaluate.py] ──────── Threshold Optimization (0.45 Operating Point)
              │
              ▼
[Saved Artifacts: models/] ── best_model.joblib & preprocessor
              │
              ▼
  [app/streamlit_app.py] ── Interactive Executive Dashboard & Risk Screening Portal
```

---

## 9. Exploratory Data Analysis (EDA) Highlights
* **Target Distribution**: 753 Legitimate claims (75.3%) vs 247 Fraudulent claims (24.7%).
* **Key Risk Features**:
  * **Incident Severity**: Claims with `Major Damage` demonstrated a significantly higher fraud rate (~60%) compared to `Trivial Damage` (<10%).
  * **Insured Hobbies**: Hobbies like `chess` and `cross-fit` exhibited statistically higher fraud proportions in this dataset.
  * **Claim Ratios**: Fraudulent claims showed higher `total_claim_amount` relative to `policy_annual_premium`.

## 10. Data Preprocessing
* Missing values encoded as `'?'` in `collision_type`, `property_damage`, and `police_report_available` were converted into explicit `'MISSING'` categorical tokens.
* Numerical missing values were imputed using median strategies.
* Categorical features were encoded using `OneHotEncoder(handle_unknown='ignore')`.

## 11. Feature Engineering
* `policy_tenure_days`: Difference between `incident_date` and `policy_bind_date`.
* `vehicle_age_at_incident`: `incident_year` minus `auto_year`.
* `claim_to_premium_ratio`: `total_claim_amount` / (`policy_annual_premium` + 1.0).
* Component ratios: `injury_claim_ratio`, `property_claim_ratio`, `vehicle_claim_ratio`.

---

## 12. Model Benchmark Comparison

| Model Algorithm | CV Recall | CV Precision | CV F1-Score | CV ROC-AUC | CV PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.7226** | **0.6550** | **0.6862** | **0.8474** | **0.6629** |
| Logistic Regression | 0.7424 | 0.6220 | 0.6767 | 0.8657 | 0.6676 |
| Gradient Boosting | 0.7372 | 0.6621 | 0.6977 | 0.8642 | 0.6464 |
| Decision Tree | 0.5508 | 0.5488 | 0.5498 | 0.7007 | 0.6086 |
| Random Forest | 0.4247 | 0.6012 | 0.4972 | 0.8520 | 0.6055 |
| K-Nearest Neighbors | 0.8786 | 0.2671 | 0.4091 | 0.5616 | 0.4133 |

## 13. Evaluation Metrics on Untouched Test Set (200 Claims)

* **Optimal Threshold**: `0.45`
* **Accuracy**: `84.00%`
* **Precision**: `64.91%`
* **Recall (Target Class 1)**: `75.51%` (Catches 37 out of 49 fraudulent claims in test set)
* **F1-Score**: `69.81%`
* **ROC-AUC**: `84.12%`
* **PR-AUC**: `59.89%`
* **Confusion Matrix**: `TN = 131`, `FP = 20`, `FN = 12`, `TP = 37`

---

## 14. Final Model Selection
**XGBoost Classifier with SMOTE & Class Weight Scaling** was selected as the production model due to its superior **PR-AUC** performance, strong probability calibration, and robustness across threshold sweeps.

## 15. Model Explainability (SHAP)
Using `SHAP (SHapley Additive exPlanations)` TreeExplainer:
1. Top global risk drivers include `incident_severity_Major Damage`, `total_claim_amount`, `insured_hobbies_chess`, `vehicle_claim`, and `policy_annual_premium`.
2. SHAP waterfall explanations provide individual claim transparency for adjusters.

---

## 16. Streamlit Web Application
Features:
1. **Executive Dashboard**: KPI Cards, Target Distribution Pie Chart, Confusion Matrix Heatmap.
2. **Claim Risk Screening Portal**: Interactive form for single-claim screening, risk meter (LOW/MEDIUM/HIGH), and action recommendations.
3. **Model Analytics & Explainability**: Metrics summary table and evaluation plots.

---

## 17. Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/insurance-fraud-detection.git
cd insurance-fraud-detection

# 2. Create and activate Python 3.11+ virtual environment
python3 -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows (Command Prompt / PowerShell):
# .venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 18. How to Run

```bash
# Run unit tests to verify system integrity
pytest tests/

# Execute end-to-end model training, tuning, and artifact generation
python src/train.py

# Launch the interactive Streamlit Web Application
streamlit run app/streamlit_app.py
```

---

## 19. Project Structure

```
insurance-fraud-detection/
├── data/
│   ├── raw/
│   │   └── insurance_claims.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── best_model.joblib
│   ├── preprocessing_pipeline.joblib
│   └── threshold_config.json
├── reports/
│   ├── figures/
│   └── results/
│       └── model_metrics.csv
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_app.py
├── requirements.txt
├── README.md
├── ACADEMIC_INTERNSHIP_REPORT.md
└── .gitignore
```

---

## 20. Limitations
* Dataset size is relatively small (1,000 total claims).
* Synthetic categorical associations (such as `insured_hobbies`) are dataset-specific artifacts that require validation against real enterprise data.

## 21. Future Enhancements
* Incorporating text NLP processing on police report unstructured narrative texts.
* Integrating real-time API endpoints (FastAPI / Docker deployment).
* Implementing automated ML monitoring for concept and data drift.

## 22. Ethical Considerations
* **Fairness & Bias**: Gender (`insured_sex`) and demographic variables must be monitored to ensure non-discriminatory screening.
* **Human-in-the-Loop**: Automated decisions must never deny payouts directly; human adjusters make final claim determinations.
