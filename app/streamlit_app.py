"""
Streamlit Web Application - Insurance Fraud Claims Detection Engine
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config with custom title and wide layout
st.set_page_config(
    page_title="Insurance Fraud Claims Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and professional styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2640 0%, #111827 100%);
        border: 1px solid #2e3a59;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #3b82f6;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-high {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-medium {
        background-color: #78350f;
        color: #fde047;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-low {
        background-color: #14532d;
        color: #86efac;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #1f2937;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        border-radius: 4px;
        font-size: 13px;
        color: #d1d5db;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Loads saved model pipeline, preprocessor, and threshold configuration."""
    model_path = "models/best_model.joblib"
    config_path = "models/threshold_config.json"

    if not os.path.exists(model_path):
        # Check parent dir
        model_path = "../models/best_model.joblib"
        config_path = "../models/threshold_config.json"

    pipeline = joblib.load(model_path)
    
    config = {
        "optimal_threshold": 0.45,
        "low_risk_cutoff": 0.30,
        "high_risk_cutoff": 0.60,
        "metrics": {"accuracy": 0.84, "precision": 0.65, "recall": 0.75, "roc_auc": 0.84}
    }
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    return pipeline, config


# Load artifacts
try:
    pipeline, config = load_model_artifacts()
    threshold = config.get("optimal_threshold", 0.45)
    metrics = config.get("metrics", {})
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <div style="font-size: 48px; line-height: 1;">🛡️</div>
        <h2 style="margin: 8px 0 0 0; color: #60a5fa; font-size: 20px; font-weight: 700;">Fraud Guard AI</h2>
        <span style="color: #94a3b8; font-size: 12px; letter-spacing: 0.5px;">RISK SCREENING ENGINE</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Executive Dashboard", "🔎 Claim Risk Screening", "📈 Model Analytics & Explainability", "ℹ️ About System"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Operational Config")
st.sidebar.info(f"**Operating Threshold**: `{threshold:.2f}`\n\n**Low Risk Cutoff**: `< 0.30`\n\n**High Risk Cutoff**: `> 0.60`")
st.sidebar.caption("Insurance Fraud Claims Detection Engine v1.0.0")


# PAGE 1: EXECUTIVE DASHBOARD
if page == "📊 Executive Dashboard":
    st.title("🛡️ Insurance Fraud Claims Detection Engine")
    st.markdown("##### AI-Powered Decision Support System for Automobile Insurance Risk Screening")
    
    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <b>Decision Support Notice</b>: This engine screens automobile insurance claims for fraud risk probability. Predictions serve as prioritization guidance for human claims investigators and do not conclusively declare customer fraud.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Total Claims Analyzed</div><div class="metric-value">1,000</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Fraud Rate</div><div class="metric-value">24.7%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">{metrics.get("accuracy", 0.84)*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Target Recall</div><div class="metric-value">{metrics.get("recall", 0.75)*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">ROC-AUC Score</div><div class="metric-value">{metrics.get("roc_auc", 0.84):.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts Section
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Target Distribution (Legitimate vs Fraudulent)")
        fig_pie = px.pie(
            names=["Legitimate (75.3%)", "Fraudulent (24.7%)"],
            values=[753, 247],
            color_discrete_sequence=["#3b82f6", "#ef4444"],
            hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ffffff")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Model Confusion Matrix Breakdown (Test Set)")
        tn = metrics.get("true_negatives", 131)
        fp = metrics.get("false_positives", 20)
        fn = metrics.get("false_negatives", 12)
        tp = metrics.get("true_positives", 37)
        
        cm_data = pd.DataFrame([
            {"Actual": "Legitimate (0)", "Predicted": "Legitimate (0)", "Count": tn},
            {"Actual": "Legitimate (0)", "Predicted": "Fraud (1)", "Count": fp},
            {"Actual": "Fraud (1)", "Predicted": "Legitimate (0)", "Count": fn},
            {"Actual": "Fraud (1)", "Predicted": "Fraud (1)", "Count": tp},
        ])
        fig_cm = px.density_heatmap(
            cm_data, x="Predicted", y="Actual", z="Count", text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ffffff")
        st.plotly_chart(fig_cm, use_container_width=True)


# PAGE 2: CLAIM RISK SCREENING PORTAL
elif page == "🔎 Claim Risk Screening":
    st.title("🔎 Claim Risk Screening Portal")
    st.markdown("Enter automobile insurance claim parameters below to run real-time risk screening.")

    # Preset Sample Selector
    preset = st.selectbox(
        "Load Sample Preset Claim:",
        ["Custom Input", "Sample 1: High Risk Major Accident", "Sample 2: Low Risk Single Vehicle Damage"]
    )

    default_vals = {
        "months_as_customer": 128,
        "age": 34,
        "policy_state": "OH",
        "policy_csl": "250/500",
        "policy_deductable": 1000,
        "policy_annual_premium": 1250.0,
        "umbrella_limit": 0,
        "insured_sex": "MALE",
        "insured_education_level": "MD",
        "insured_occupation": "craft-repair",
        "insured_hobbies": "chess",
        "insured_relationship": "husband",
        "capital_gains": 0,
        "capital_loss": 0,
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Side Collision",
        "incident_severity": "Major Damage",
        "authorities_contacted": "Police",
        "incident_state": "NY",
        "incident_city": "Columbus",
        "incident_hour_of_the_day": 3,
        "number_of_vehicles_involved": 1,
        "property_damage": "YES",
        "bodily_injuries": 2,
        "witnesses": 1,
        "police_report_available": "NO",
        "total_claim_amount": 75000,
        "injury_claim": 15000,
        "property_claim": 15000,
        "vehicle_claim": 45000,
        "auto_make": "Dodge",
        "auto_model": "RAM",
        "auto_year": 2007,
        "policy_bind_date": "2014-05-15",
        "incident_date": "2015-02-17"
    }

    if preset == "Sample 1: High Risk Major Accident":
        default_vals.update({
            "incident_severity": "Major Damage",
            "total_claim_amount": 88000,
            "insured_hobbies": "chess",
            "police_report_available": "NO",
            "incident_hour_of_the_day": 3
        })
    elif preset == "Sample 2: Low Risk Single Vehicle Damage":
        default_vals.update({
            "incident_severity": "Minor Damage",
            "total_claim_amount": 5400,
            "insured_hobbies": "reading",
            "police_report_available": "YES",
            "incident_hour_of_the_day": 14
        })

    with st.form("claim_form"):
        st.subheader("1. Policyholder & Customer Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            months = st.number_input("Months as Customer", 0, 600, default_vals["months_as_customer"])
            age = st.number_input("Age", 18, 100, default_vals["age"])
            sex = st.selectbox("Insured Sex", ["MALE", "FEMALE"], index=0 if default_vals["insured_sex"] == "MALE" else 1)
        with col2:
            edu = st.selectbox("Education Level", ["MD", "PhD", "Associate", "College", "High School", "JD", "Master"], index=0)
            occ = st.selectbox("Occupation", ["craft-repair", "machine-op-insp", "prof-specialty", "sales", "tech-support", "exec-managerial", "other-service"], index=0)
            hobby = st.selectbox("Hobbies", ["chess", "cross-fit", "reading", "hiking", "yachting", "paintball", "golf", "camping"], index=0)
        with col3:
            rel = st.selectbox("Relationship", ["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family"], index=0)
            gains = st.number_input("Capital Gains ($)", 0, 100000, default_vals["capital_gains"])
            loss = st.number_input("Capital Loss ($)", 0, 100000, default_vals["capital_loss"])

        st.subheader("2. Policy Details")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            state = st.selectbox("Policy State", ["OH", "IN", "IL"], index=0)
            csl = st.selectbox("Policy CSL", ["250/500", "100/300", "500/1000"], index=0)
        with pcol2:
            deductible = st.select_slider("Deductible ($)", [500, 1000, 2000], value=default_vals["policy_deductable"])
            premium = st.number_input("Annual Premium ($)", 500.0, 3000.0, float(default_vals["policy_annual_premium"]))
        with pcol3:
            umbrella = st.number_input("Umbrella Limit ($)", 0, 10000000, default_vals["umbrella_limit"])
            bind_date = st.date_input("Policy Bind Date", pd.to_datetime(default_vals["policy_bind_date"]))

        st.subheader("3. Incident & Claim Information")
        icol1, icol2, icol3 = st.columns(3)
        with icol1:
            inc_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"], index=0)
            col_type = st.selectbox("Collision Type", ["Side Collision", "Rear Collision", "Front Collision", "MISSING"], index=0)
            severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"], index=0)
        with icol2:
            authorities = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"], index=0)
            inc_hour = st.slider("Incident Hour (0-23)", 0, 23, default_vals["incident_hour_of_the_day"])
            vehicles = st.number_input("Vehicles Involved", 1, 10, default_vals["number_of_vehicles_involved"])
        with icol3:
            prop_dmg = st.selectbox("Property Damage", ["YES", "NO", "MISSING"], index=0)
            injuries = st.number_input("Bodily Injuries", 0, 10, default_vals["bodily_injuries"])
            witnesses = st.number_input("Witnesses Count", 0, 10, default_vals["witnesses"])
            police_rep = st.selectbox("Police Report Available", ["YES", "NO", "MISSING"], index=1 if default_vals["police_report_available"] == "NO" else 0)
            inc_date = st.date_input("Incident Date", pd.to_datetime(default_vals["incident_date"]))

        st.subheader("4. Claim Amounts & Vehicle Details")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            total_claim = st.number_input("Total Claim Amount ($)", 0, 200000, default_vals["total_claim_amount"])
            injury_claim = st.number_input("Injury Claim ($)", 0, 100000, default_vals["injury_claim"])
            property_claim = st.number_input("Property Claim ($)", 0, 100000, default_vals["property_claim"])
            vehicle_claim = st.number_input("Vehicle Claim ($)", 0, 150000, default_vals["vehicle_claim"])
        with vcol2:
            make = st.selectbox("Auto Make", ["Dodge", "Ford", "Chevrolet", "Toyota", "Honda", "BMW", "Audi", "Nissan", "Subaru", "Jeep"], index=0)
            model = st.text_input("Auto Model", default_vals["auto_model"])
            year = st.number_input("Auto Year", 1995, 2025, default_vals["auto_year"])

        submitted = st.form_submit_button("🚨 Analyze Claim Risk")

    if submitted:
        from src.predict import FraudPredictor
        predictor = FraudPredictor()

        claim_payload = {
            "months_as_customer": months,
            "age": age,
            "policy_number": 999999,
            "policy_bind_date": str(bind_date),
            "policy_state": state,
            "policy_csl": csl,
            "policy_deductable": deductible,
            "policy_annual_premium": premium,
            "umbrella_limit": umbrella,
            "insured_zip": 43081,
            "insured_sex": sex,
            "insured_education_level": edu,
            "insured_occupation": occ,
            "insured_hobbies": hobby,
            "insured_relationship": rel,
            "capital-gains": gains,
            "capital-loss": loss,
            "incident_date": str(inc_date),
            "incident_type": inc_type,
            "collision_type": col_type,
            "incident_severity": severity,
            "authorities_contacted": authorities,
            "incident_state": "NY",
            "incident_city": "Columbus",
            "incident_location": "9999 Sample St",
            "incident_hour_of_the_day": inc_hour,
            "number_of_vehicles_involved": vehicles,
            "property_damage": prop_dmg,
            "bodily_injuries": injuries,
            "witnesses": witnesses,
            "police_report_available": police_rep,
            "total_claim_amount": total_claim,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_make": make,
            "auto_model": model,
            "auto_year": year,
        }

        res = predictor.predict_single_claim(claim_payload)
        prob = res["fraud_probability"]
        risk = res["risk_level"]

        st.markdown("### Risk Analysis Output")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if prob > 0.6 else ("#eab308" if prob > 0.3 else "#22c55e")},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(234, 179, 8, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff", height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            if risk == "HIGH":
                st.markdown(f'<div class="risk-high">RISK LEVEL: HIGH ({prob*100:.1f}%)</div>', unsafe_allow_html=True)
            elif risk == "MEDIUM":
                st.markdown(f'<div class="risk-medium">RISK LEVEL: MEDIUM ({prob*100:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">RISK LEVEL: LOW ({prob*100:.1f}%)</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Prediction Status**: `{res['prediction']}`")
            st.markdown(f"**Recommended Workflow Action**:\n> 📋 **{res['recommended_action']}**")
            
            st.markdown("##### Key Risk Drivers Observed:")
            factors = []
            if severity == "Major Damage":
                factors.append("• **Incident Severity**: Major Damage claims demonstrate high historical fraud rate.")
            if hobby in ["chess", "cross-fit"]:
                factors.append(f"• **Insured Hobby**: '{hobby}' shows statistically elevated fraud correlation in dataset.")
            if total_claim > 60000:
                factors.append(f"• **High Total Claim Amount**: ${total_claim:,.2f} is in the upper quartile of claim values.")
            if police_rep == "NO":
                factors.append("• **Police Report Missing**: Absence of official police report raises risk profile.")

            if not factors:
                factors.append("• Claim attributes fall within normal operational baselines.")

            for f in factors:
                st.markdown(f)


# PAGE 3: MODEL ANALYTICS & EXPLAINABILITY
elif page == "📈 Model Analytics & Explainability":
    st.title("📈 Model Analytics & Global Explainability")
    
    st.subheader("Model Evaluation Summary Table")
    df_metrics = pd.read_csv("reports/results/model_metrics.csv" if os.path.exists("reports/results/model_metrics.csv") else "../reports/results/model_metrics.csv")
    st.dataframe(df_metrics, use_container_width=True)

    st.subheader("Evaluation Visualizations")
    tab1, tab2, tab3 = st.columns(3)
    fig_dir = "reports/figures" if os.path.exists("reports/figures") else "../reports/figures"

    if os.path.exists(f"{fig_dir}/XGBoost_Optimized_roc_curve.png"):
        tab1.image(f"{fig_dir}/XGBoost_Optimized_roc_curve.png", caption="ROC Curve", use_container_width=True)
    if os.path.exists(f"{fig_dir}/XGBoost_Optimized_pr_curve.png"):
        tab2.image(f"{fig_dir}/XGBoost_Optimized_pr_curve.png", caption="Precision-Recall Curve", use_container_width=True)
    if os.path.exists(f"{fig_dir}/XGBoost_Optimized_threshold_sweep.png"):
        tab3.image(f"{fig_dir}/XGBoost_Optimized_threshold_sweep.png", caption="Threshold Performance Sweep", use_container_width=True)


# PAGE 4: ABOUT SYSTEM
elif page == "ℹ️ About System":
    st.title("ℹ️ Project Architecture & Methodology")
    st.markdown("""
    ### Project Overview
    The **Insurance Fraud Claims Detection Engine** is an end-to-end machine learning system built for screening automobile insurance claims.
    
    ### Pipeline Workflow
    1. **Data Ingestion**: Standardized loader with target conversion (`fraud_reported` → 1/0).
    2. **Feature Engineering**: Decomposes dates into policy tenure and incident temporal features, while computing claim-to-premium ratios.
    3. **Preprocessing**: Leak-free `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
    4. **Imbalance Handling**: Integrated `SMOTE` oversampling within cross-validation folds.
    5. **Tuned Model**: XGBoost / Gradient Boosting Classifier optimized for **Precision-Recall AUC**.
    6. **Threshold Optimization**: Decision probability threshold set to `0.45` to maximize recall while controlling false positive rates.
    """)
