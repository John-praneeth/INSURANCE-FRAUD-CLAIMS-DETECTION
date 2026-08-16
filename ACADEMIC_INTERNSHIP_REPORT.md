# ACADEMIC INTERNSHIP REPORT

## TITLE: INSURANCE FRAUD CLAIMS DETECTION ENGINE USING MACHINE LEARNING

**Degree**: Bachelor of Technology in Computer Science & Engineering (Data Science)  
**Domain**: Data Science & Machine Learning  
**Project Title**: Insurance Fraud Claims Detection Engine  

---

### BONAFIDE CERTIFICATE

This is to certify that the internship project report entitled **“INSURANCE FRAUD CLAIMS DETECTION ENGINE”** submitted in partial fulfillment of the requirements for the award of the Degree of **Bachelor of Technology in Computer Science and Engineering (Data Science)** is a record of bonafide work carried out under academic and technical supervision.

**Supervisor**: Faculty Mentor  
**Department**: Computer Science and Engineering (Data Science)  
**Institution**: Department of Computer Science and Engineering (Data Science)  

---

### ACKNOWLEDGEMENT

I express my sincere gratitude to my faculty mentors, department heads, and industry guides for providing invaluable advice, resources, and technical support throughout this Data Science and Machine Learning project. I am deeply thankful for the opportunity to apply theoretical concepts in data preprocessing, supervised classification, imbalanced learning, and explainable AI to solve a real-world enterprise problem in the insurance sector.

---

## TABLE OF CONTENTS

1. **Chapter 1 — Introduction**
   - 1.1 Background & Motivation
   - 1.2 Role of Machine Learning in Fraud Risk Screening
   - 1.3 Project Objectives
   - 1.4 Project Scope & Boundaries
2. **Chapter 2 — Organization / Academic Details**
   - 2.1 Academic & Technical Context
   - 2.2 Project Environment & Governance
3. **Chapter 3 — Technologies Used**
   - 3.1 Python Programming Environment
   - 3.2 Data Analytics & Visualization Libraries
   - 3.3 Supervised Classification & Imbalanced Learning Stack
   - 3.4 Model Explainability & Web Deployment Tools
4. **Chapter 4 — Project Design & Implementation**
   - 4.1 Problem Statement
   - 4.2 Objectives & Decision Support Philosophy
   - 4.3 Dataset Characteristics & Source Truth
   - 4.4 Comprehensive Attribute Taxonomy
   - 4.5 System Architecture & Modular Design
   - 4.6 Data Cleaning & Leak-Free Preprocessing
   - 4.7 Exploratory Data Analysis (EDA)
   - 4.8 Domain Feature Engineering
   - 4.9 Supervised Model Development & Tuning
   - 4.10 Evaluation & Threshold Optimization
   - 4.11 Model Explainability (SHAP Analysis)
   - 4.12 Interactive Streamlit Web Application
   - 4.13 Automated Testing & Verification
5. **Chapter 5 — Results and Discussion**
   - 5.1 Baseline Model Performance Comparison
   - 5.2 Test Set Performance Evaluation
   - 5.3 Confusion Matrix & Threshold Trade-off Analysis
   - 5.4 Feature Importance Insights
6. **Chapter 6 — Internship / Learning Experience**
   - 6.1 Technical Skills Acquired
   - 6.2 Engineering Best Practices & Lessons Learned
7. **Chapter 7 — Conclusion & Future Scope**
   - 7.1 Summary of Accomplishments
   - 7.2 Practical Usefulness & Operational Impact
   - 7.3 System Limitations & Future Enhancements
8. **References**

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Background & Motivation
Insurance fraud represents one of the most critical operational risks confronting financial institutions worldwide. In the automobile insurance sector, fraudulent claims range from opportunistic overstatements of loss to organized criminal syndicates staging accidents and submitting falsified medical and property damage invoices. Annual financial losses attributed to insurance fraud run into tens of billions of dollars, driving up premium costs for legitimate policyholders and compromising insurance company solvency.

Traditional fraud detection relies on rule-based heuristic filters or manual claim-by-claim audits conducted by human adjusters. While human investigation is essential for legal and evidence verification, manual auditing of every submitted claim is slow, subjective, and cost-prohibitive. Conversely, static rule-based systems (e.g., flagging claims over $50,000) are easily bypassed by sophisticated fraudsters.

### 1.2 Role of Machine Learning in Fraud Risk Screening
Machine learning (ML) provides a powerful quantitative framework for identifying complex, multi-dimensional patterns across vast insurance historical data. Supervised classification algorithms can ingest policy attributes, customer demographics, vehicle characteristics, and incident temporal patterns to calculate a continuous probability of fraud risk.

Crucially, machine learning in insurance fraud is deployed as a **risk-screening and decision-support mechanism**. Rather than automatically rejecting claims or declaring customers fraudulent, the machine learning model categorizes incoming claims into risk tiers (**LOW**, **MEDIUM**, **HIGH**) and prioritizes high-risk claims for expedited human investigation.

### 1.3 Project Objectives
The primary objective of this project is to build an end-to-end, production-ready machine learning framework and interactive deployment interface titled **Insurance Fraud Claims Detection Engine**. The specific sub-objectives are:
1. Conduct a rigorous quality audit and exploratory analysis on Kaggle's `insurance_claims.csv` dataset.
2. Develop a leak-free preprocessing and feature engineering pipeline using Scikit-Learn.
3. Compare multiple machine learning classification algorithms using 5-Fold Stratified Cross-Validation combined with SMOTE for class imbalance mitigation.
4. Optimize decision probability thresholds to prioritize Target Class Recall (catching suspicious claims) while maintaining acceptable Precision.
5. Provide global and local model explainability using SHAP (SHapley Additive exPlanations).
6. Build a responsive, professional Streamlit web application enabling claims adjusters to screen individual claims and inspect risk drivers.

### 1.4 Project Scope & Boundaries
* **In Scope**: Supervised binary classification (`fraud_reported`), data cleaning, feature ratio engineering, cross-validation, hyperparameter tuning, threshold selection, SHAP explainability, unit testing, and web app deployment.
* **Out of Scope**: Unstructured image analysis of vehicle damage photos, real-time streaming database connectors (Kafka), and automated legal fraud adjudication.

---

## CHAPTER 2 — ORGANIZATION / ACADEMIC DETAILS

### 2.1 Academic & Technical Context
This project was undertaken as part of the academic requirements for the award of the Degree of Bachelor of Technology in Computer Science and Engineering (Data Science). The project simulates an industry internship environment within an insurance analytics lab, following professional software development practices, PEP 8 standards, modular code separation (`src/`), and automated testing suites (`pytest`).

### 2.2 Project Environment & Governance
* **Operating System**: macOS (ARM64)
* **Development Tools**: VS Code / Antigravity IDE, Jupyter Notebook, Git
* **Execution Runtime**: Python 3.11+ Virtual Environment (`.venv`)

---

## CHAPTER 3 — TECHNOLOGIES USED

### 3.1 Python Programming Environment
Python served as the primary programming language due to its rich scientific ecosystem, extensive machine learning libraries, and strong production deployment capabilities.

### 3.2 Data Analytics & Visualization Libraries
* **Pandas**: Used for tabular data manipulation, missing value handling, and feature transformations.
* **NumPy**: Executed vector computations and ratio calculations.
* **Matplotlib & Seaborn**: Rendered static evaluation charts (ROC curves, PR curves, heatmap confusion matrices, threshold sweeps).
* **Plotly**: Generated dynamic interactive visual widgets for the Streamlit dashboard.

### 3.3 Supervised Classification & Imbalanced Learning Stack
* **Scikit-Learn**: Provided core modeling primitives (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `SVC`), evaluation metrics, and `ColumnTransformer` pipelines.
* **Imbalanced-Learn (`imblearn`)**: Provided `SMOTE` (Synthetic Minority Over-sampling Technique) and `ImbPipeline` to ensure synthetic oversampling was strictly confined inside cross-validation training folds.
* **XGBoost**: Employed as the primary gradient boosting tree framework for optimal predictive performance.

### 3.4 Model Explainability & Web Deployment Tools
* **SHAP**: Utilized `TreeExplainer` to extract additive feature attribution values.
* **Joblib**: Handled model serialization (`best_model.joblib`) and preprocessor persistence.
* **Streamlit**: Formed the frontend web presentation layer.

---

## CHAPTER 4 — PROJECT DESIGN & IMPLEMENTATION

### 4.1 Problem Statement
Given an automobile insurance claim record containing customer demographics, policy characteristics, incident details, and claim amount breakdowns, train a binary classification model $f(X) \rightarrow [0, 1]$ predicting the probability that the claim is fraudulent ($y = 1$).

### 4.2 Objectives & Decision Support Philosophy
In insurance operations, a **False Negative** (failing to detect a $75,000 fraudulent claim) is exponentially more expensive than a **False Positive** (requiring 15 minutes of an adjuster's time to verify a legitimate claim). Therefore, model selection and threshold selection explicitly prioritize **Recall**, **PR-AUC**, and **F1-Score** over plain Accuracy.

### 4.3 Dataset Characteristics & Source Truth
The dataset `insurance_claims.csv` was downloaded directly from Kaggle (`buntyshah/auto-insurance-claims-data`). Inspection confirmed:
* **Rows**: 1,000 insurance claim records
* **Columns**: 40 attributes (including 1 uninformative trailing NaN column `_c39` which was removed).
* **Target Distribution**: `fraud_reported` -> 753 'N' (75.3%) vs 247 'Y' (24.7%).

### 4.4 Comprehensive Attribute Taxonomy
The 39 predictor attributes encompass:
1. `months_as_customer` (int) - Duration of policyholder relationship.
2. `age` (int) - Age of policyholder.
3. `policy_number` (int) - Unique policy identifier (dropped to prevent leakage).
4. `policy_bind_date` (str/date) - Date policy went into effect.
5. `policy_state` (str) - Policy registration state (OH, IN, IL).
6. `policy_csl` (str) - Combined Single Limit policy bounds.
7. `policy_deductable` (int) - Out-of-pocket deductible amount ($500, $1000, $2000).
8. `policy_annual_premium` (float) - Annual premium fee.
9. `umbrella_limit` (int) - Additional umbrella liability coverage.
10. `insured_zip` (int) - Zip code identifier (dropped).
11. `insured_sex` (str) - MALE / FEMALE.
12. `insured_education_level` (str) - High School, College, Associate, Master, MD, PhD, JD.
13. `insured_occupation` (str) - Occupation category.
14. `insured_hobbies` (str) - Hobbies and activities.
15. `insured_relationship` (str) - Family relationship status.
16. `capital-gains` (int) - Recorded capital gains.
17. `capital-loss` (int) - Recorded capital losses.
18. `incident_date` (str/date) - Date of reported vehicle incident.
19. `incident_type` (str) - Single Vehicle Collision, Multi-vehicle, Theft, Parked Car.
20. `collision_type` (str) - Side, Rear, Front Collision, or 'MISSING' ('?').
21. `incident_severity` (str) - Major Damage, Minor Damage, Total Loss, Trivial Damage.
22. `authorities_contacted` (str) - Police, Fire, Ambulance, Other, None.
23. `incident_state` / `incident_city` (str) - Incident location jurisdiction.
24. `incident_location` (str) - Street address (dropped).
25. `incident_hour_of_the_day` (int) - Hour of incident occurrence (0-23).
26. `number_of_vehicles_involved` (int) - Vehicle count.
27. `property_damage` (str) - YES / NO / MISSING ('?').
28. `bodily_injuries` (int) - Number of injured persons.
29. `witnesses` (int) - Witness count.
30. `police_report_available` (str) - YES / NO / MISSING ('?').
31. `total_claim_amount` (int) - Total monetary claim sum.
32. `injury_claim` / `property_claim` / `vehicle_claim` (int) - Component claim amounts.
33. `auto_make` / `auto_model` / `auto_year` - Vehicle specifications.

### 4.5 System Architecture & Modular Design
The project separates core production logic into clean modules inside `src/`:
* `src/data_loader.py`: Ingests dataset, drops invalid columns, converts target `fraud_reported` ('Y'->1, 'N'->0).
* `src/feature_engineering.py`: Computes date differences, vehicle age, and claim component ratios.
* `src/preprocessing.py`: Drops identifiers, sets up Scikit-Learn `ColumnTransformer`.
* `src/train.py`: Trains candidates, runs 5-Fold Stratified CV, tunes hyperparameters, saves joblib models.
* `src/evaluate.py`: Calculates evaluation metrics, runs threshold sweeps, plots curves.
* `src/predict.py`: Formats inference outputs with risk tiers (**LOW**, **MEDIUM**, **HIGH**).

### 4.6 Data Cleaning & Leak-Free Preprocessing
- **Missing Value Handling**: String `'?'` values in `collision_type` (178 records), `property_damage` (360 records), and `police_report_available` (343 records) were mapped to explicit `'MISSING'` categories.
- **Scaling & Encoding**: Numerical variables were standardized with `StandardScaler()`. Categorical features were encoded using `OneHotEncoder(handle_unknown='ignore')`.

### 4.7 Exploratory Data Analysis (EDA)
EDA confirmed strong signals:
- Claims with `incident_severity = Major Damage` demonstrated a ~60% fraud rate vs <10% for `Trivial Damage`.
- Absence of a police report (`police_report_available = NO`) showed elevated fraud risk.
- Total claim amount distribution was skewed higher for fraudulent claims.

### 4.8 Domain Feature Engineering
Engineered features added to the dataset:
- `policy_tenure_days` = `incident_date` - `policy_bind_date`
- `vehicle_age_at_incident` = `incident_year` - `auto_year`
- `claim_to_premium_ratio` = `total_claim_amount` / (`policy_annual_premium` + 1.0)
- Component ratios: `injury_claim_ratio`, `property_claim_ratio`, `vehicle_claim_ratio`

### 4.9 Supervised Model Development & Tuning
The pipeline compared 6 classifiers using 5-Fold Stratified Cross-Validation combined with SMOTE oversampling. Hyperparameter optimization was conducted via `GridSearchCV` on XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`).

### 4.10 Evaluation & Threshold Optimization
Instead of defaulting to a 0.50 threshold, probability thresholds were swept from 0.10 to 0.90. An optimal threshold of **0.45** was selected to achieve a high Recall of **75.51%** while maintaining Precision at **64.91%**.

### 4.11 Model Explainability (SHAP Analysis)
SHAP `TreeExplainer` was applied to explain predictions:
- Top global feature importances: `incident_severity_Major Damage`, `total_claim_amount`, `insured_hobbies_chess`, `vehicle_claim`, `policy_annual_premium`.

### 4.12 Interactive Streamlit Web Application
The application (`app/streamlit_app.py`) provides:
1. **Executive Dashboard**: KPI Summary Cards, Target Distribution Chart, Confusion Matrix Heatmap.
2. **Claim Risk Screening Portal**: Dynamic input form, real-time risk gauge, risk badge, and recommended action.
3. **Model Analytics Tab**: Evaluation tables and plots.

### 4.13 Automated Testing & Verification
9 automated unit tests were created in `tests/` (`test_data.py`, `test_preprocessing.py`, `test_model.py`, `test_app.py`) and verified via `pytest` (100% pass rate).

---

## CHAPTER 5 — RESULTS AND DISCUSSION

### 5.1 Baseline Model Performance Comparison (5-Fold Stratified CV)

| Model | CV Recall | CV Precision | CV F1-Score | CV ROC-AUC | CV PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.7226** | **0.6550** | **0.6862** | **0.8474** | **0.6629** |
| Logistic Regression | 0.7424 | 0.6220 | 0.6767 | 0.8657 | 0.6676 |
| Gradient Boosting | 0.7372 | 0.6621 | 0.6977 | 0.8642 | 0.6464 |
| Decision Tree | 0.5508 | 0.5488 | 0.5498 | 0.7007 | 0.6086 |
| Random Forest | 0.4247 | 0.6012 | 0.4972 | 0.8520 | 0.6055 |
| K-Nearest Neighbors | 0.8786 | 0.2671 | 0.4091 | 0.5616 | 0.4133 |

### 5.2 Test Set Performance Evaluation (Untouched 200 Claims)

* **Selected Decision Threshold**: `0.45`
* **Test Accuracy**: `84.00%`
* **Test Precision**: `64.91%`
* **Test Recall (Fraud Class 1)**: `75.51%` (37 / 49 fraudulent claims correctly caught)
* **Test F1-Score**: `69.81%`
* **Test ROC-AUC**: `84.12%`
* **Test PR-AUC**: `59.89%`

### 5.3 Confusion Matrix Breakdown

```
                    PREDICTED
                Legitimate (0)   Fraud (1)
ACTUAL
Legitimate (0)       131            20       (TN=131, FP=20)
Fraud (1)             12            37       (FN=12,  TP=37)
```

- **True Positives (TP = 37)**: 37 suspicious claims correctly flagged for manual review.
- **False Negatives (FN = 12)**: Only 12 fraudulent claims missed out of 49.
- **False Positives (FP = 20)**: 20 legitimate claims flagged for routine verification.
- **True Negatives (TN = 131)**: 131 legitimate claims routed through fast-track processing.

---

## CHAPTER 6 — INTERNSHIP / LEARNING EXPERIENCE

### 6.1 Technical Skills Acquired
Through this project, key Data Science and Machine Learning competencies were mastered:
1. **Data Preprocessing & Sanitization**: Handling missing values represented as `'?'` without data leakage.
2. **Feature Engineering**: Decomposing temporal attributes and calculating financial claim ratios.
3. **Imbalanced Classification**: Confining SMOTE synthetic oversampling within cross-validation folds.
4. **Threshold Optimization**: Selecting probability decision cutoffs based on operational risk trade-offs.
5. **Explainable AI (XAI)**: Applying SHAP TreeExplainer for transparent feature attribution.
6. **Web App Deployment**: Building responsive dashboards with Streamlit and Plotly.
7. **Production Python Standards**: Writing modular `src/` packages, Joblib serialization, and `pytest` suites.

---

## CHAPTER 7 — CONCLUSION & FUTURE SCOPE

### 7.1 Summary of Accomplishments
The **Insurance Fraud Claims Detection Engine** successfully delivers an end-to-end, quantitative risk-screening solution. Achieving an **84.00% Test Accuracy**, **75.51% Recall**, and **84.12% ROC-AUC**, the system provides insurance companies with an effective tool to prioritize suspicious claims for human investigation.

### 7.2 Practical Usefulness & Operational Impact
By integrating this decision-support engine into a claims workflow:
- High-risk claims (>60% probability) are automatically routed to Special Investigation Units (SIU).
- Medium-risk claims (30-60% probability) require additional documentation before payout.
- Low-risk claims (<30% probability) receive fast-track approval, improving customer satisfaction and reducing operational costs.

### 7.3 System Limitations & Future Enhancements
- **Limitations**: The model is trained on a 1,000-record dataset; real enterprise deployment requires training on larger historical claim volumes.
- **Future Work**: Integration of Natural Language Processing (NLP) on unstructured adjuster narrative notes and deployment via containerized REST APIs (Docker + FastAPI).

---

## REFERENCES

1. Kaggle Dataset: Buntyshah, *Auto Insurance Claims Data*, Kaggle. https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
2. Pedregosa, F. et al., *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research, 2011.
3. Chawla, N. V. et al., *SMOTE: Synthetic Minority Over-sampling Technique*, Journal of Artificial Intelligence Research, 2002.
4. Chen, T. & Guestrin, C., *XGBoost: A Scalable Tree Boosting System*, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
5. Lundberg, S. M. & Lee, S.-I., *A Unified Approach to Interpreting Model Predictions*, Advances in Neural Information Processing Systems (NeurIPS), 2017.
6. Streamlit Documentation: *Streamlit Python Framework*, https://docs.streamlit.io/
