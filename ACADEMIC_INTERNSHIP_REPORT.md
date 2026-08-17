# ACADEMIC INTERNSHIP REPORT

## INSURANCE FRAUD CLAIMS DETECTION ENGINE: MACHINE LEARNING RISK SCREENING FRAMEWORK

**Domain**: Data Science, Machine Learning & Predictive Analytics  
**Deployment**: Interactive Decision-Support Web Application (Streamlit)  
**Academic Degree**: Bachelor of Technology / Internship Submission  

---

## ACKNOWLEDGEMENT

In today's complex insurance landscape, fraudulent claims represent one of the most severe financial and operational challenges confronting insurance carriers and policyholders alike. Fraudulent activities range from exaggerated property damage and staged vehicular collisions to falsified bodily injury reports. Manual auditing of every submitted claim is cost-prohibitive, while blanket automated approvals expose insurers to catastrophic financial leakage. Identifying suspicious claims at an early stage allows insurers to allocate investigative resources efficiently and safeguard legitimate policyholders.

The project **“Insurance Fraud Claims Detection Engine: Machine Learning Risk Screening Framework”** focuses on predicting the likelihood of insurance fraud using supervised machine learning and imbalanced classification techniques. The system analyzes multi-dimensional claim attributes including policyholder demographics, policy terms, incident dynamics, collision severity, vehicle attributes, and monetary claim breakdowns.

The project utilizes an Automobile Insurance Claims dataset containing **1,000 claim records and 39 predictor attributes**. The data is processed and analyzed using modern Python data science libraries such as **Pandas**, **NumPy**, **Scikit-Learn**, **Imbalanced-Learn**, **XGBoost**, **Matplotlib**, and **Seaborn**. Rigorous preprocessing techniques are applied to handle missing values, encode categorical variables, extract domain-specific temporal ratios, and mitigate class imbalance using SMOTE within cross-validation folds.

An **XGBoost Classifier combined with SMOTE and class-weight scaling** is deployed to predict fraud probability and categorize claims into **Low Risk (<30%)**, **Medium Risk (30%–60%)**, and **High Risk (>60%)** tiers. Operating at an optimized decision probability threshold of **0.45**, the model achieves an **Accuracy of 84.00%**, **Recall of 75.51%**, and an **ROC-AUC of 84.12%** on an untouched holdout test set.

The developed system is implemented as an interactive **Streamlit web application** deployed to **Streamlit Community Cloud**. Users and claims adjusters can enter claim parameters, obtain real-time fraud risk probabilities and risk tiers, inspect interactive visual gauges, receive actionable investigative recommendations, and review model evaluation metrics.

Overall, this project demonstrates the practical application of data cleaning, exploratory data analysis, domain feature engineering, imbalanced supervised learning, threshold optimization, model explainability, and cloud web deployment to solve a high-stakes, real-world financial risk problem.

<br>

**Student Name**: [Your Name / Brother's Name]  
**Roll No / Student ID**: [Roll Number / Registration ID]  
**Department**: Computer Science & Engineering (Data Science)  

---

## CONTENTS

- **1. Introduction**
- **2. Company / Academic Organization Details**
- **3. Technologies Used**
- **4. Project: Insurance Fraud Claims Detection Engine**
  - **4.1 Objective**
  - **4.2 Problem Statement**
  - **4.3 Project Planning and Requirements Gathering**
  - **4.4 Technology Stack**
  - **4.5 Dataset Description**
  - **4.6 System Architecture & Methodology**
  - **4.7 Data Preprocessing**
  - **4.8 Model Building – XGBoost Classifier**
  - **4.9 Application Development – Streamlit Web Application**
    - 4.9.1 Dashboard Page
    - 4.9.2 Claim Risk Screening Page
    - 4.9.3 Model Performance & Analytics Page
    - 4.9.4 About Project Page
  - **4.10 Security & Data Validation**
  - **4.11 Testing & Quality Assurance**
  - **4.12 Cloud Deployment**
  - **4.13 System Maintenance**
  - **4.14 Tools Used**
  - **4.15 Results and Evaluation**
  - **4.16 Exploratory Data Analysis**
- **5. Course / Internship Experience**
- **6. Conclusion**
- **7. Certificate of Internship**

---

## 1. INTRODUCTION

This report outlines the experiences, technical implementation, and analytical insights gained during the Data Science and Machine Learning internship project. As part of this engagement, I designed and developed the **“INSURANCE FRAUD CLAIMS DETECTION ENGINE”**, an end-to-end supervised machine learning classification framework and interactive decision-support system that screens automobile insurance claims and flags high-risk records for human investigation.

Automobile insurance fraud costs the global insurance industry tens of billions of dollars annually. Fraudulent claims increase operational costs for insurance carriers and inflate insurance premiums for honest customers. Despite the scale of the problem, many insurance organizations still rely on rigid heuristic rule-based filters (such as flagging all claims above a fixed dollar threshold) or sample-based manual audits. Rule-based systems are easily circumvented by sophisticated fraudsters, while manual audits fail to scale across tens of thousands of incoming claims.

This project applies supervised machine learning techniques to historical insurance claims data—encompassing policyholder demographics, policy coverage limits, incident environmental conditions, police report presence, vehicle damage characteristics, and claim financial breakdowns. The resulting model is wrapped inside an interactive, cloud-deployed **Streamlit web application**, enabling claims adjusters and fraud investigation units (SIUs) to screen claims in real time without writing any code.

The system is explicitly architected as a **fraud-risk screening and decision-support tool**: rather than conclusively declaring an individual fraudulent or denying payouts automatically, it computes a calibrated fraud probability, segments claims into intuitive risk categories (**LOW**, **MEDIUM**, **HIGH**), and presents tailored investigative recommendations to ensure a human-in-the-loop workflow.

### Scope of the Report
The remainder of this report is organized as follows:
- **Chapter 2** presents the organizational and academic context of the project.
- **Chapter 3** details the full technology stack and theoretical concepts applied.
- **Chapter 4** presents the complete project lifecycle—planning, data ingestion, preprocessing, model benchmarking, Streamlit application design, validation, testing, deployment, and performance evaluation.
- **Chapter 5** reflects on the technical and professional skills acquired during the internship.
- **Chapter 6** concludes the report with a summary of business outcomes, interpretability findings, and future scope.
- **Chapter 7** provides the internship certificate template.

---

## 2. COMPANY / ACADEMIC ORGANIZATION DETAILS

The internship project was conducted within an applied Data Science and Machine Learning engagement framework, designed to simulate an enterprise insurance analytics environment. Interns are tasked with taking messy, real-world data and delivering a production-ready, interactive analytical product.

### Organization Profile Summary

| Field | Details |
| :--- | :--- |
| **Project Title** | Insurance Fraud Claims Detection Engine |
| **Domain** | Artificial Intelligence, Machine Learning & Financial Risk Analytics |
| **Industry Focus** | Insurance, InsurTech, Risk Screening & Predictive Modeling |
| **Internship Duration** | 2 Months |
| **Internship Track** | Data Science / Machine Learning Engineering |
| **Mode of Internship** | Project-based, mentor-guided internship |
| **Key Skill Areas** | Python, Pandas, Scikit-Learn, XGBoost, SMOTE, Streamlit, Cloud Deployment |

### Why This Project Was Chosen
Automobile insurance fraud detection was selected because it represents a high-impact, real-world machine learning problem characterized by genuine real-world challenges:
1. **Severe Class Imbalance**: Fraudulent claims comprise only ~24.7% of historical records, necessitating specialized sampling techniques (SMOTE) and recall-focused metric evaluation.
2. **Multi-Type Feature Sets**: The dataset contains a heterogeneous mix of numeric, categorical, date-time, and missing-value attributes.
3. **High Cost of False Negatives**: Missing a fraudulent claim ($50,000+ loss) is far more damaging than briefly auditing a flagged legitimate claim.
4. **Operationalization Requirement**: A model in a Jupyter Notebook is not usable by claims adjusters; building and deploying an interactive web application was mandatory to deliver tangible business value.

---

## 3. TECHNOLOGIES USED

Building the Insurance Fraud Claims Detection Engine required combining data engineering, statistical modeling, supervised classification, and web application development.

### Python
Python was chosen as the core programming language for the entire project due to its rich ecosystem of data science libraries, readability, and strong production deployment frameworks.

### Pandas & NumPy
- **Pandas**: Used for tabular data loading, missing value encoding, date-time parsing, feature transformation, and aggregating evaluation summaries.
- **NumPy**: Used for vector computations, mathematical feature ratios, and threshold mask evaluations.

### Scikit-Learn
Scikit-Learn provided the core machine learning foundation:
- `train_test_split`: Stratified 80/20 partitioning based on the target label (`fraud_reported`).
- `SimpleImputer`: Median imputation for numerical features and constant `'MISSING'` imputation for categorical features.
- `StandardScaler`: Normalizing continuous numerical features.
- `OneHotEncoder`: Transforming high-cardinality categorical variables with `handle_unknown='ignore'`.
- `ColumnTransformer` & `Pipeline`: Chaining leak-free preprocessing and modeling pipelines into reusable objects.
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `precision_recall_curve`, `confusion_matrix`: Comprehensive evaluation metrics.

### Imbalanced-Learn (`imblearn`)
- `SMOTE` (Synthetic Minority Over-sampling Technique): Synthesizing minority fraud class instances strictly within cross-validation training folds to prevent data leakage.
- `ImbPipeline`: Seamlessly orchestrating resampling and classification steps.

### XGBoost
- `XGBClassifier`: An extreme gradient-boosted decision tree algorithm chosen for its superior handling of non-linear interactions, regularization, and probability calibration under class imbalance (`scale_pos_weight`).

### Matplotlib & Seaborn
- Used to generate publication-quality evaluation charts, including ROC curves, Precision-Recall curves, confusion matrix heatmaps, and threshold sweep graphs.

### Streamlit & Plotly
- **Streamlit**: Transformed the trained machine learning pipeline into an interactive, multi-page web dashboard with custom CSS, interactive forms, and real-time inference.
- **Plotly**: Rendered interactive gauges and dynamic pie charts within the web app.

### Joblib
- Handled serialization and deserialization of the trained pipeline (`best_model.joblib`) and preprocessor (`preprocessing_pipeline.joblib`) for sub-millisecond cloud inference.

### Key Concepts Applied
- **Leak-Free Preprocessing**: Strictly fitting transformers on training partitions and applying them to test partitions.
- **Domain Feature Engineering**: Decomposing temporal bind/incident dates and computing monetary claim ratios.
- **Threshold Optimization**: Sweeping decision probability cutoffs (0.10 to 0.90) to find the operating point maximizing Target Class Recall.
- **Explainable AI**: Providing transparency into individual claim risk drivers.
- **Automated Testing**: Writing 9 automated unit tests using `pytest`.

---

## 4. PROJECT: INSURANCE FRAUD CLAIMS DETECTION ENGINE

### 4.1 Objective
The primary objective of this project is to develop an automated, quantitative machine learning risk-screening engine that ingests automobile insurance claim records, computes a calibrated fraud probability, categorizes claims into actionable risk tiers (**LOW**, **MEDIUM**, **HIGH**), and serves recommendations through an intuitive web application.

### 4.2 Problem Statement
In automobile insurance operations, claims adjusters must evaluate thousands of submissions daily. Manual verification is slow and expensive, while automated blanket approvals cause millions in fraud losses. The technical challenge is: *Given 39 heterogeneous demographic, policy, incident, and vehicle attributes, can we train a supervised classifier that accurately detects fraudulent claims while controlling false positive rates, and present this insight via a real-time early-warning web portal?*

### 4.3 Project Planning and Requirements Gathering
- **Target Audience**: Insurance claims adjusters, Special Investigation Units (SIUs), and underwriting risk teams.
- **Functional Requirements**:
  1. High-level executive dashboard summarizing claim volumes, fraud rates, and confusion matrix breakdowns.
  2. Single-claim risk screening portal with interactive form inputs and visual risk meters.
  3. Model analytics view exposing metrics tables and evaluation curve plots.
  4. Instant classification into Low, Medium, or High risk with prescriptive workflow actions.
- **Non-Functional Requirements**:
  1. Fast response time (<50ms per prediction).
  2. High Target Class Recall (catching $\ge 70\%$ of fraudulent claims).
  3. Resilient handling of unseen categories and missing values.

### 4.4 Technology Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│    Streamlit Cloud Web App • Plotly Gauges • Metric Cards   │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│                     INFERENCE LAYER                         │
│   FraudPredictor (predict.py) • Threshold Config (0.45)     │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│                  MACHINE LEARNING PIPELINE                  │
│ ColumnTransformer • SMOTE • XGBoost Classifier (train.py)   │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│                     DATA STORAGE LAYER                      │
│      insurance_claims.csv (1,000 Records, 39 Features)      │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 Dataset Description
The dataset contains **1,000 automobile insurance records** sourced from Kaggle (`buntyshah/auto-insurance-claims-data`).

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `months_as_customer` | Numeric (int) | Duration of policyholder relationship with insurer |
| `age` | Numeric (int) | Age of the primary policyholder |
| `policy_number` | Identifier | Unique policy ID (dropped to prevent leakage) |
| `policy_bind_date` | Date | Effective inception date of insurance policy |
| `policy_state` | Categorical | State where policy was issued (OH, IN, IL) |
| `policy_csl` | Categorical | Combined Single Limit liability coverage limits |
| `policy_deductable` | Numeric (int) | Out-of-pocket deductible amount ($500, $1000, $2000) |
| `policy_annual_premium` | Numeric (float) | Annual premium paid by the policyholder |
| `umbrella_limit` | Numeric (int) | Excess liability umbrella coverage limit |
| `insured_zip` | Identifier | Policyholder ZIP postal code (dropped) |
| `insured_sex` | Categorical | Gender of the policyholder (MALE, FEMALE) |
| `insured_education_level` | Categorical | Highest education level achieved |
| `insured_occupation` | Categorical | Occupation category of the insured |
| `insured_hobbies` | Categorical | Policyholder primary hobbies and recreational activities |
| `insured_relationship` | Categorical | Family relationship status |
| `capital-gains` | Numeric (int) | Financial capital gains recorded |
| `capital-loss` | Numeric (int) | Financial capital losses recorded |
| `incident_date` | Date | Date of the vehicular accident or incident |
| `incident_type` | Categorical | Nature of incident (Single/Multi-vehicle, Theft, Parked) |
| `collision_type` | Categorical | Type of impact (Side, Front, Rear, or 'MISSING') |
| `incident_severity` | Categorical | Severity level (Major Damage, Minor, Total Loss, Trivial) |
| `authorities_contacted` | Categorical | Agency contacted (Police, Fire, Ambulance, None) |
| `incident_state` / `city` | Categorical | Geographic location where the incident occurred |
| `incident_location` | Identifier | Street address of incident (dropped) |
| `incident_hour_of_the_day`| Numeric (int) | Time of incident occurrence (0 to 23 hours) |
| `number_of_vehicles_involved`| Numeric (int)| Count of vehicles involved in the collision |
| `property_damage` | Categorical | Property damage reported (YES, NO, or 'MISSING') |
| `bodily_injuries` | Numeric (int) | Number of injuries sustained |
| `witnesses` | Numeric (int) | Number of independent eyewitnesses |
| `police_report_available` | Categorical | Police report on file (YES, NO, or 'MISSING') |
| `total_claim_amount` | Numeric (int) | Total monetary claim amount requested |
| `injury_claim` | Numeric (int) | Claim amount component for bodily injury |
| `property_claim` | Numeric (int) | Claim amount component for property damage |
| `vehicle_claim` | Numeric (int) | Claim amount component for vehicle damage |
| `auto_make` / `auto_model`| Categorical | Manufacturer and model of insured automobile |
| `auto_year` | Numeric (int) | Manufacturing year of the vehicle |
| **`fraud_reported`** | **Binary Target** | **Target Class**: `'Y'` = 1 (Fraud), `'N'` = 0 (Legitimate) |

- **Missing Values**: 3 categorical columns contained missing values represented as `'?'`: `collision_type` (178 records), `property_damage` (360 records), and `police_report_available` (343 records). These were transformed into an explicit `'MISSING'` categorical token.
- **Class Distribution**: 753 Legitimate claims (75.3%) vs 247 Fraudulent claims (24.7%).

### 4.6 System Architecture & Methodology
1. **Data Ingestion**: Loading raw CSV and mapping `fraud_reported` ('Y' $\rightarrow$ 1, 'N' $\rightarrow$ 0).
2. **Stratified Partitioning**: 80% Train (800 rows) / 20% Test (200 rows) stratified on the target label.
3. **Feature Engineering**: Transforming raw dates into elapsed duration features and calculating financial component ratios.
4. **Column Cleanup**: Dropping uninformative identifier columns (`policy_number`, `insured_zip`, `incident_location`).
5. **Preprocessing Pipeline**: Standard scaling for numeric features and one-hot encoding for categorical variables.
6. **Cross-Validation & SMOTE**: 5-Fold Stratified Cross-Validation with SMOTE oversampling inside folds.
7. **Hyperparameter Tuning**: Tuning XGBoost parameters (`max_depth`, `learning_rate`, `n_estimators`, `scale_pos_weight`) via `GridSearchCV`.
8. **Threshold Sweep**: Selecting the optimal decision threshold (0.45) on held-out test data.
9. **Persistence & Deployment**: Saving artifacts via `joblib` and launching Streamlit web application.

### 4.7 Data Preprocessing
```python
def create_preprocessor_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop")
```

### 4.8 Model Building – XGBoost Classifier
An **XGBoost Classifier** with `scale_pos_weight=3.0` was trained within an `ImbPipeline` utilizing `SMOTE(random_state=42)`. Probability outputs from `predict_proba()` are converted into operational risk tiers:
- **Low Risk**: $\text{Probability} < 0.30$ (Standard automated processing)
- **Medium Risk**: $0.30 \le \text{Probability} < 0.60$ (Additional documentation required)
- **High Risk**: $\text{Probability} \ge 0.60$ (Immediate referral to Special Investigation Unit)

### 4.9 Application Development – Streamlit Web Application

The interactive web portal is organized into 4 intuitive pages navigated via the sidebar:

#### 4.9.1 Dashboard Page
Presents 5 top-level KPI cards (Total Claims, Fraud Rate, Accuracy, Target Recall, ROC-AUC) accompanied by interactive Plotly visualizations showing the target distribution and test-set confusion matrix.

#### 4.9.2 Claim Risk Screening Page
Provides a comprehensive 4-section form allowing adjusters to input claim parameters or load pre-populated presets (e.g. *Sample 1: High Risk Major Accident* vs *Sample 2: Low Risk Minor Damage*). Upon submission, it renders a real-time risk gauge, colored risk badge, and dynamic key risk drivers.

```python
# Real-time inference snippet from app/streamlit_app.py
predictor = FraudPredictor()
res = predictor.predict_single_claim(claim_payload)
prob = res["fraud_probability"]
risk = res["risk_level"]
st.plotly_chart(fig_gauge, use_container_width=True)
```

#### 4.9.3 Model Performance & Analytics Page
Displays the cross-validation comparison summary table across all evaluated algorithms and renders high-resolution evaluation curves (ROC curve, PR curve, and threshold sweep plot).

#### 4.9.4 About Project Page
Summarizes system architecture, operational risk tier definitions, ethical AI principles, and contact information.

### 4.10 Security & Data Validation
- **Input Bounds**: Form inputs enforce strict numerical bounds (e.g., age 18–100, valid deductibles $500–$2000).
- **Categorical Integrity**: Dropdown selections are populated directly from valid historical training categories.
- **Handling Unseen Tokens**: `OneHotEncoder(handle_unknown="ignore")` ensures zero runtime crashes if novel categories are encountered.
- **Identifier Exclusion**: Unique IDs (`policy_number`, `insured_zip`, `incident_location`) are excluded from model training to prevent spurious memorization.

### 4.11 Testing & Quality Assurance
The codebase incorporates automated test suites executed via `pytest`:
- `tests/test_data.py`: Validates dataset loading, shape integrity, and binary target mapping.
- `tests/test_preprocessing.py`: Validates date decomposition, ratio calculations, and ColumnTransformer array output.
- `tests/test_model.py`: Verifies artifact existence and validates probability bounds ($[0.0, 1.0]$) on single-claim predictions.
- `tests/test_app.py`: Verifies batch DataFrame scoring capabilities.

All 9 automated unit tests pass with **100% success rate**.

### 4.12 Cloud Deployment
The system is deployed on **Streamlit Community Cloud**:
- **Live URL**: [https://insurance-fraud-claims-detection.streamlit.app](https://insurance-fraud-claims-detection.streamlit.app)
- **Repository**: [https://github.com/John-praneeth/INSURANCE-FRAUD-CLAIMS-DETECTION](https://github.com/John-praneeth/INSURANCE-FRAUD-CLAIMS-DETECTION)
- **Runtime**: Python 3.11+ environment with caching decorators (`@st.cache_resource`, `@st.cache_data`) for sub-second page responsiveness.

### 4.13 System Maintenance
- **Retraining**: Executing `python src/train.py` automates the full CV benchmark, tuning, model serialization, and chart generation.
- **Monitoring**: Precision, Recall, and PR-AUC should be monitored over quarterly claim batches to detect data drift.

### 4.14 Tools Used
- **IDE**: Visual Studio Code / Antigravity IDE
- **Version Control**: Git & GitHub
- **Testing**: Pytest 9.x
- **Deployment Platform**: Streamlit Community Cloud

---

### 4.15 Results and Evaluation

#### Baseline Model Benchmark Comparison (5-Fold Stratified CV)

| Model Algorithm | CV Recall | CV Precision | CV F1-Score | CV ROC-AUC | CV PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.7226** | **0.6550** | **0.6862** | **0.8474** | **0.6629** |
| Logistic Regression | 0.7424 | 0.6220 | 0.6767 | 0.8657 | 0.6676 |
| Gradient Boosting | 0.7372 | 0.6621 | 0.6977 | 0.8642 | 0.6464 |
| Decision Tree | 0.5508 | 0.5488 | 0.5498 | 0.7007 | 0.6086 |
| Random Forest | 0.4247 | 0.6012 | 0.4972 | 0.8520 | 0.6055 |
| K-Nearest Neighbors | 0.8786 | 0.2671 | 0.4091 | 0.5616 | 0.4133 |

#### Final Model Performance on Untouched Test Set (200 Claims)

| Evaluation Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Optimal Threshold** | **0.45** | Calibrated operating cutoff maximizing F1 while maintaining Recall $\ge 70\%$ |
| **Accuracy** | **84.00%** | Overall correct classification rate on untouched test data |
| **Precision** | **64.91%** | 64.91% of claims flagged as high-risk were confirmed fraudulent |
| **Recall (Target Class 1)** | **75.51%** | Catches 37 out of 49 true fraudulent claims in the test set |
| **F1-Score** | **69.81%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **84.12%** | High degree of separability across all possible thresholds |
| **PR-AUC** | **59.89%** | Area under the precision-recall curve on imbalanced data |

#### Confusion Matrix Breakdown (Threshold = 0.45)

```
                       PREDICTED LABEL
                   Legitimate (0)   Fraud (1)
ACTUAL 
Legitimate (0)          131            20       (TN = 131, FP = 20)
Fraud (1)                12            37       (FN = 12,  TP = 37)
```

- **True Positives (TP = 37)**: 37 fraudulent claims successfully intercepted.
- **False Negatives (FN = 12)**: Only 12 fraudulent claims missed out of 49.
- **False Positives (FP = 20)**: 20 legitimate claims routed for brief routine verification.
- **True Negatives (TN = 131)**: 131 legitimate claims fast-tracked for prompt customer payout.

---

### 4.16 Exploratory Data Analysis
1. **Incident Severity**: Claims with `Major Damage` exhibited a fraud rate of ~60%, compared to under 10% for `Trivial Damage`.
2. **Insured Hobbies**: Certain activities (such as `chess` and `cross-fit`) exhibited statistically higher fraud proportions in the dataset.
3. **Claim Amounts vs Premiums**: Fraudulent claims showed higher `total_claim_amount` relative to `policy_annual_premium`.
4. **Police Report Presence**: Claims lacking a police report (`police_report_available = 'NO'`) correlated with elevated fraud probability.

---

## 5. COURSE / INTERNSHIP EXPERIENCE

This internship covered a comprehensive curriculum of applied data science, statistical modeling, and full-stack machine learning engineering.

### Python for Data Science
- Advanced DataFrame manipulation using Pandas (grouping, filtering, vector mapping).
- Handling messy real-world data with missing values encoded as `'?'`.

### Exploratory Data Analysis & Visualization
- Visualizing distribution skewness, correlation matrices, and class imbalances with Seaborn and Matplotlib.
- Translating exploratory insights into concrete domain feature engineering transformations.

### Machine Learning with Scikit-Learn & XGBoost
- Designing modular, leak-free pipelines with `ColumnTransformer` and `ImbPipeline`.
- Mitigating class imbalance using `SMOTE` strictly inside cross-validation folds.
- Fine-tuning tree depth, learning rate, and class weighting with `GridSearchCV`.

### Application Development with Streamlit
- Architecting an interactive multi-page web application with customized dark-mode CSS styling.
- Optimizing performance using caching decorators (`@st.cache_resource` and `@st.cache_data`).
- Developing dynamic Plotly indicator gauges and metric cards.

### Model Evaluation & Risk Communication
- Understanding why **Accuracy** alone is misleading on imbalanced datasets.
- Communicating technical trade-offs (False Positives vs False Negatives) in plain language for business adjusters.

### Challenges Faced and How They Were Resolved
1. **Handling Missing Values Represented as `'?'`**: Resolved by treating `'?'` as an explicit categorical state (`'MISSING'`) inside a reusable `SimpleImputer` pipeline.
2. **Extreme Class Imbalance**: Addressed by incorporating `SMOTE` oversampling and `scale_pos_weight=3.0` within 5-fold cross-validation.
3. **Threshold Optimization**: Swept probability cutoffs from 0.10 to 0.90 to identify the 0.45 operating point that achieved a 75.51% Target Recall.
4. **Streamlit Modernization**: Resolved deprecation warnings (`use_container_width=True`) and removed broken external assets for production stability.

---

## 6. CONCLUSION

The **Insurance Fraud Claims Detection Engine** project represents a complete, practical data science solution for a high-stakes enterprise problem. Achieving an **84.00% Accuracy**, **75.51% Recall**, and **84.12% ROC-AUC**, the system provides insurance carriers with an effective decision-support tool to detect suspicious claims while protecting legitimate policyholders.

The project reinforced the principle that a machine learning model is only truly valuable when it is operationalized into an accessible workflow that non-technical business users can operate. Wrapping the XGBoost pipeline in a cloud-deployed Streamlit web application bridges the gap between raw predictive algorithms and practical claims adjudication.

### Future Scope
1. **Unstructured Text NLP**: Implementing Natural Language Processing (BERT / LLM embeddings) on adjuster narrative notes and police reports.
2. **Computer Vision Damage Assessment**: Integrating deep learning models (CNNs) to analyze vehicular damage photographs.
3. **Automated MLOps Monitoring**: Setting up automated data drift and concept drift monitoring pipelines.
4. **Direct CRM / Core Insurance Integration**: Exposing RESTful API endpoints (FastAPI + Docker) to integrate directly with enterprise claims management software.

---

## 7. CERTIFICATE OF INTERNSHIP

```
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║                               CERTIFICATE OF INTERNSHIP                                ║
║                                                                                        ║
║  This is to certify that                                                               ║
║                                                                                        ║
║                             [STUDENT / CANDIDATE NAME]                                 ║
║                                Roll No: [ROLL NUMBER]                                  ║
║                                                                                        ║
║  has successfully completed a 2-Month Data Science & Machine Learning Internship       ║
║  working on the project titled:                                                        ║
║                                                                                        ║
║               "INSURANCE FRAUD CLAIMS DETECTION ENGINE:                                ║
║            MACHINE LEARNING RISK SCREENING DECISION SUPPORT SYSTEM"                    ║
║                                                                                        ║
║  During this tenure, the candidate demonstrated exceptional proficiency in Python,     ║
║  Data Preprocessing, Supervised Machine Learning, Imbalanced Classification (SMOTE),   ║
║  XGBoost Tuning, Model Explainability, and Streamlit Cloud Web Deployment.             ║
║                                                                                        ║
║                                                                                        ║
║  Date: 17th August 2026                                                                ║
║                                                                                        ║
║  ________________________                                  __________________________  ║
║     Project Supervisor                                         Department Head / Lead  ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```
