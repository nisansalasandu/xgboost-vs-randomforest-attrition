"""
pages/01_predictor.py
Live Attrition Predictor — enter employee details, get predictions from both models.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
from utils.predictor import predict, get_top_features, load_models, load_feature_columns

st.set_page_config(page_title="Live Predictor", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:1.8rem; font-weight:700; color:#1f4e79; }
    .risk-high   { background:#f8d7da; color:#721c24; padding:8px 20px;
                   border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .risk-medium { background:#fff3cd; color:#856404; padding:8px 20px;
                   border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .risk-low    { background:#d4edda; color:#155724; padding:8px 20px;
                   border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .model-card  { background:#f8f9fa; border-radius:12px; padding:1.2rem;
                   border:1px solid #dee2e6; text-align:center; }
    .prob-number { font-size:2.8rem; font-weight:800; margin:8px 0; }
    .rf-color    { color:#1f4e79; }
    .xgb-color   { color:#8b4513; }
    .rec-box     { background:#e8f4f8; border-left:4px solid #1f4e79;
                   padding:1rem; border-radius:0 8px 8px 0; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔮 Live Attrition Predictor</p>',
            unsafe_allow_html=True)
st.markdown("Enter employee details to get attrition risk predictions from **both models simultaneously**.")

# ── Check models are loaded ────────────────────────────────────────────────────
rf_pipeline, xgb_pipeline, err = load_models()
if err:
    st.error(f"Models not found: {err}\n\nPlease run Notebooks 02 and 03 first.")
    st.stop()

feature_columns = load_feature_columns()
if feature_columns is None:
    st.error("Feature columns not found. Please run Notebook 01 first.")
    st.stop()

st.success(f"Both models loaded successfully. Using {len(feature_columns)} features.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("### Employee Details")
st.caption("Fill in the employee information below. Remaining features use dataset median values.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Info**")
        age      = st.slider("Age", 18, 60, 35)
        gender   = st.selectbox("Gender", ["Male", "Female"])
        marital  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance = st.slider("Distance from Home (km)", 1, 29, 7)
        education= st.selectbox("Education Level",
                                 [1, 2, 3, 4, 5],
                                 format_func=lambda x: {1:'Below College',2:'College',
                                                          3:'Bachelor',4:'Master',
                                                          5:'Doctor'}[x])

    with col2:
        st.markdown("**Job Details**")
        dept       = st.selectbox("Department",
                                   ["Research & Development", "Sales", "Human Resources"])
        job_role   = st.selectbox("Job Role",
                                   ['Sales Executive', 'Research Scientist',
                                    'Laboratory Technician', 'Manufacturing Director',
                                    'Healthcare Representative', 'Manager',
                                    'Sales Representative', 'Research Director',
                                    'Human Resources'])
        job_level  = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        overtime   = st.selectbox("OverTime", ["No", "Yes"])
        travel     = st.selectbox("Business Travel",
                                   ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    with col3:
        st.markdown("**Compensation & Satisfaction**")
        income     = st.slider("Monthly Income ($)", 1000, 20000, 6500, step=100)
        job_sat    = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_sat    = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        wlb        = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        stock      = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        yrs_co     = st.slider("Years at Company", 0, 40, 5)
        total_yrs  = st.slider("Total Working Years", 0, 40, 10)

    submitted = st.form_submit_button("🔮 Predict Attrition Risk",
                                       use_container_width=True,
                                       type="primary")

# ── Run prediction ─────────────────────────────────────────────────────────────
if submitted:
    user_inputs = {
        'Age'                     : age,
        'Gender'                  : gender,
        'MaritalStatus'           : marital,
        'DistanceFromHome'        : distance,
        'Education'               : education,
        'Department'              : dept,
        'JobRole'                 : job_role,
        'JobLevel'                : job_level,
        'OverTime'                : overtime,
        'BusinessTravel'          : travel,
        'MonthlyIncome'           : income,
        'JobSatisfaction'         : job_sat,
        'EnvironmentSatisfaction' : env_sat,
        'WorkLifeBalance'         : wlb,
        'StockOptionLevel'        : stock,
        'YearsAtCompany'          : yrs_co,
        'TotalWorkingYears'       : total_yrs,
    }

    with st.spinner("Running predictions..."):
        result = predict(user_inputs)

    if result['error']:
        st.error(f"Prediction error: {result['error']}")
    else:
        st.divider()
        st.markdown("## Prediction Results")

        # ── Side-by-side model results ─────────────────────────────────────────
        col_rf, col_xgb = st.columns(2)

        with col_rf:
            rf_pct = round(result['rf_proba'] * 100, 1)
            st.markdown(f"""
            <div class="model-card">
                <div style="font-size:1rem;font-weight:600;color:#1f4e79;margin-bottom:4px">
                    Random Forest
                </div>
                <div class="prob-number rf-color">{rf_pct}%</div>
                <div style="font-size:0.85rem;color:#666;margin-bottom:10px">
                    Attrition probability
                </div>
                <span class="{result['rf_css']}">{result['rf_label']} Risk</span>
            </div>
            """, unsafe_allow_html=True)

        with col_xgb:
            xgb_pct = round(result['xgb_proba'] * 100, 1)
            st.markdown(f"""
            <div class="model-card">
                <div style="font-size:1rem;font-weight:600;color:#8b4513;margin-bottom:4px">
                    XGBoost
                </div>
                <div class="prob-number xgb-color">{xgb_pct}%</div>
                <div style="font-size:0.85rem;color:#666;margin-bottom:10px">
                    Attrition probability
                </div>
                <span class="{result['xgb_css']}">{result['xgb_label']} Risk</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Agreement / disagreement banner ───────────────────────────────────
        st.markdown("")
        rf_pred  = 1 if result['rf_proba']  >= 0.5 else 0
        xgb_pred = 1 if result['xgb_proba'] >= 0.5 else 0

        if rf_pred == xgb_pred:
            label = "At Risk" if rf_pred == 1 else "Not at Risk"
            st.success(f"✅ Both models agree — **{label}** (default threshold 0.5)")
        else:
            st.warning("⚠️ Models disagree at default threshold (0.5). Consider reviewing with lower threshold.")

        # ── Progress bars ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Probability Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Random Forest**")
            st.progress(result['rf_proba'])
            st.caption(f"{rf_pct}% attrition probability")
        with col2:
            st.markdown("**XGBoost**")
            st.progress(result['xgb_proba'])
            st.caption(f"{xgb_pct}% attrition probability")

        # ── Top features ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Top Contributing Features")
        st.caption("These are the features each model considers most important globally (from training).")

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Random Forest — Top 5 (Gini)**")
            rf_imp = get_top_features('rf', 5)
            if not rf_imp.empty:
                for feat, val in rf_imp.items():
                    clean = feat.replace('_', ' ').title()
                    st.metric(label=clean, value=f"{val:.4f}")
            else:
                st.info("Feature importance file not found.")

        with fc2:
            st.markdown("**XGBoost — Top 5 (Gain)**")
            xgb_imp = get_top_features('xgb', 5)
            if not xgb_imp.empty:
                for feat, val in xgb_imp.items():
                    clean = feat.replace('_', ' ').title()
                    st.metric(label=clean, value=f"{val:.4f}")
            else:
                st.info("Feature importance file not found.")

        # ── HR Recommendation ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### HR Recommendation")

        avg_proba = (result['rf_proba'] + result['xgb_proba']) / 2

        if avg_proba >= 0.60:
            rec_icon  = "🔴"
            rec_title = "High Risk — Immediate Action Recommended"
            rec_text  = (
                "This employee shows a high probability of attrition. "
                "Consider scheduling a one-on-one meeting to understand concerns. "
                "Review compensation, promotion timeline, and workload. "
                "OverTime and low job satisfaction are common drivers in this range."
            )
        elif avg_proba >= 0.35:
            rec_icon  = "🟡"
            rec_title = "Medium Risk — Monitor Closely"
            rec_text  = (
                "This employee shows moderate attrition risk. "
                "Regular check-ins and engagement surveys are recommended. "
                "Ensure work-life balance and career development opportunities are visible. "
                "Consider recognition programs if performance ratings are high."
            )
        else:
            rec_icon  = "🟢"
            rec_title = "Low Risk — Continue Engagement"
            rec_text  = (
                "This employee appears engaged and stable. "
                "Continue standard engagement practices. "
                "Ensure career growth opportunities remain available to maintain satisfaction. "
                "Revisit quarterly as tenure and role changes can shift risk."
            )

        st.markdown(f"""
        <div class="rec-box">
            <strong>{rec_icon} {rec_title}</strong><br><br>
            {rec_text}<br><br>
            <small style="color:#666">
                Average model probability: <strong>{round(avg_proba*100,1)}%</strong> &nbsp;|&nbsp;
                RF: {rf_pct}% &nbsp;|&nbsp; XGBoost: {xgb_pct}%
            </small>
        </div>
        """, unsafe_allow_html=True)

        # ── Input summary expander ─────────────────────────────────────────────
        with st.expander("View full input data sent to models"):
            st.dataframe(result['input_df'].T.rename(columns={0: 'Value'}),
                         use_container_width=True)
