"""
pages/03_about.py
About the project — background, methodology, team, dataset, GitHub link.
"""
import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
    .main-header  { font-size:1.8rem; font-weight:700; color:#1f4e79; }
    .section-head { font-size:1.2rem; font-weight:600; color:#1f4e79; margin-top:1.5rem; }
    .team-card    { background:#f0f4f8; border-radius:10px; padding:1rem;
                    text-align:center; border-top:4px solid #1f4e79; }
    .method-box   { background:#f8f9fa; border-left:4px solid #1f4e79;
                    padding:1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; }
    .step-badge   { background:#1f4e79; color:white; border-radius:50%;
                    width:28px; height:28px; display:inline-flex;
                    align-items:center; justify-content:center;
                    font-weight:700; font-size:0.85rem; margin-right:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">About This Project</p>', unsafe_allow_html=True)

# ── Project overview ───────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Project Overview</p>', unsafe_allow_html=True)
st.markdown("""
This is an **Advanced Machine Learning internship project** that builds, tunes, and compares
two powerful ensemble classifiers — **XGBoost** and **Random Forest** — to predict employee
attrition using the IBM HR Analytics dataset.

The goal is not just to predict attrition, but to understand *how* and *why* each model makes
its predictions differently, and which model is better suited for HR decision support.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Problem Type**\n\nBinary classification — predict whether an employee will leave (Yes) or stay (No).")
with col2:
    st.info("**Business Goal**\n\nHelp HR teams identify at-risk employees early so proactive retention actions can be taken.")
with col3:
    st.info("**Research Question**\n\nWhich model — XGBoost or Random Forest — performs better on imbalanced HR attrition data?")

st.divider()

# ── Dataset ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Dataset</p>', unsafe_allow_html=True)

d1, d2 = st.columns([1, 2])
with d1:
    st.markdown("""
    **IBM HR Analytics Employee Attrition & Performance**

    - Source: Kaggle (IBM data scientists)
    - Type: Synthetic, based on real HR patterns
    - Rows: 1,470 employees
    - Columns: 35 original features
    - Target: `Attrition` (Yes = 237, No = 1233)
    - Class imbalance: **84% No / 16% Yes**
    """)
with d2:
    import pandas as pd
    feature_groups = pd.DataFrame({
        'Category'    : ['Demographics', 'Work Factors', 'Satisfaction Scores', 'Compensation'],
        'Examples'    : [
            'Age, Gender, MaritalStatus, DistanceFromHome',
            'Department, JobRole, JobLevel, OverTime, BusinessTravel',
            'JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance',
            'MonthlyIncome, StockOptionLevel, PercentSalaryHike'
        ],
        'Feature Count': [6, 10, 6, 6],
    })
    st.dataframe(feature_groups, use_container_width=True, hide_index=True)

st.divider()

# ── Methodology ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Methodology</p>', unsafe_allow_html=True)

steps = [
    ("EDA & Visualisation",
     "Explored distributions, correlations, attrition rates by category. "
     "Generated 9 figures covering all feature groups."),
    ("Data Cleaning & Encoding",
     "Dropped zero-variance columns (EmployeeCount, StandardHours, Over18, EmployeeNumber). "
     "Binary encoding for Gender & OverTime. One-hot encoding for 5 categorical columns. "
     "Target: Yes->1, No->0."),
    ("Class Imbalance — SMOTE",
     "Applied SMOTE (Synthetic Minority Oversampling) on training data only "
     "to balance the 84:16 class ratio to 50:50."),
    ("Leakage-free Cross-Validation",
     "Used ImbPipeline from imbalanced-learn to wrap SMOTE inside each CV fold. "
     "This prevents synthetic SMOTE samples from appearing in validation folds, "
     "which would inflate CV scores unrealistically."),
    ("Hyperparameter Tuning",
     "RandomizedSearchCV with 80 iterations, 5-fold stratified CV, optimising ROC-AUC. "
     "Both models used identical settings for a fair comparison."),
    ("Evaluation",
     "Evaluated on the same held-out 20% test set (294 samples). "
     "Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC. "
     "Threshold optimisation to maximise F1 for each model."),
]

for i, (title, desc) in enumerate(steps, 1):
    st.markdown(f"""
    <div class="method-box">
        <span class="step-badge">{i}</span>
        <strong>{title}</strong><br>
        <span style="color:#555;font-size:0.9rem">{desc}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Key findings ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Key Findings</p>', unsafe_allow_html=True)

st.markdown("""
- Both models benefit significantly from threshold tuning — lowering the threshold from 0.5 improves recall (catching more at-risk employees) at the cost of some precision.
- Features most consistently important in both models: **OverTime**, **MonthlyIncome**, **StockOptionLevel**, **Age**, **JobSatisfaction**, **YearsAtCompany**.
- SMOTE + ImbPipeline is critical — without it, cross-validation scores are artificially inflated (0.97 AUC vs real 0.73 AUC on test data).
- XGBoost's boosting mechanism tends to assign high gain importance to a smaller set of features, while Random Forest spreads importance more broadly across features.
""")

st.divider()

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Tech Stack</p>', unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("**ML & Data**")
    st.markdown("scikit-learn\nxgboost\nimbalanced-learn\npandas\nnumpy")
with t2:
    st.markdown("**Visualisation**")
    st.markdown("matplotlib\nseaborn\nplotly")
with t3:
    st.markdown("**App**")
    st.markdown("streamlit\njoblib")
with t4:
    st.markdown("**Dev Tools**")
    st.markdown("Jupyter Notebooks\nGitHub\nPython 3.9+")

st.divider()

# ── Team ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Team</p>', unsafe_allow_html=True)

team = [
    ("Team Leader - Nisansala",     "Project coordination & Streamlit app"),
    ("Member 2 - Thinuri",        "EDA & visualisations"),
    ("Member 3 - Chandima",        "Data cleaning & preprocessing"),
    ("Member 4 - Ravindu",        "Random Forest model & tuning"),
    ("Member 5 - Hirushi",        "XGBoost model & tuning"),
    ("Member 6 - Disni",        "Evaluation metrics & plots"),
    ("Member 7 - Bhanuka",        "Feature importance & SHAP, Code Inspection"),
    ("Member 8 - Dasun",        "Final report & comparison notebook"),
]

team_cols = st.columns(4)
for i, (name, role) in enumerate(team):
    with team_cols[i % 4]:
        st.markdown(f"""
        <div class="team-card">
            <div style="font-size:1.5rem">👤</div>
            <div style="font-weight:600;font-size:0.9rem;margin:4px 0">{name}</div>
            <div style="font-size:0.75rem;color:#666">{role}</div>
        </div>
        <br>
        """, unsafe_allow_html=True)

st.divider()

# ── Links ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Links</p>', unsafe_allow_html=True)
st.markdown("""
- **GitHub Repository:** [xgboost-vs-randomforest-attrition](https://github.com/nisansalasandu/xgboost-vs-randomforest-attrition)
- **Dataset:** [IBM HR Analytics on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Duration:** 2 weeks (10 days) — Data Science Internship Project
""")
