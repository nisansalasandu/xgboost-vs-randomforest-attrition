import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border-left: 4px solid #1f4e79;
    }
    .winner-badge {
        background: #d4edda;
        color: #155724;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .risk-high   { background:#f8d7da; color:#721c24; padding:6px 16px;
                   border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-medium { background:#fff3cd; color:#856404; padding:6px 16px;
                   border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-low    { background:#d4edda; color:#155724; padding:6px 16px;
                   border-radius:20px; font-weight:700; font-size:1rem; }
    .section-divider { border-top: 2px solid #e0e0e0; margin: 1.5rem 0; }
    div[data-testid="stSidebarNav"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar branding ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Attrition Predictor")
    st.markdown("**XGBoost vs Random Forest**")
    st.markdown("IBM HR Analytics · Internship Project")
    st.divider()
    st.markdown("#### Navigate")
    st.page_link("app.py",               label="🏠  Home",            icon=None)
    st.page_link("pages/01_predictor.py", label="🔮  Live Predictor",  icon=None)
    st.page_link("pages/02_comparison.py",label="📈  Model Comparison",icon=None)
    st.page_link("pages/03_about.py",     label="ℹ️  About",           icon=None)
    st.divider()
    st.caption("Run Notebooks 01–04 before launching this app.")

# ── Home page content ──────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Employee Attrition Risk Predictor</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comparative study: XGBoost vs Random Forest · IBM HR Analytics Dataset</p>',
            unsafe_allow_html=True)

st.markdown("### What this app does")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🔮 Live Prediction**\n\nEnter employee details and get attrition risk predictions from both models simultaneously.")
with col2:
    st.info("**📈 Model Comparison**\n\nSide-by-side metrics, ROC curves, feature importances and confusion matrices.")
with col3:
    st.info("**📋 Full Transparency**\n\nSee exactly which features drive each prediction and how the models differ.")

st.divider()

# ── Load and show summary metrics ─────────────────────────────────────────────
import pandas as pd
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

st.markdown("### Model Performance Summary")
st.caption("Evaluated on the same held-out test set (294 samples, 16% attrition rate)")

try:
    rf_metrics  = pd.read_csv(os.path.join(MODELS_DIR, 'rf_metrics.csv'),  index_col=0)
    xgb_metrics = pd.read_csv(os.path.join(MODELS_DIR, 'xgb_metrics.csv'), index_col=0)

    metrics_df = pd.DataFrame({
        'Random Forest' : rf_metrics.iloc[0],
        'XGBoost'       : xgb_metrics.iloc[0],
    }).round(4)

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    cols = st.columns(len(metric_names))

    for col, metric in zip(cols, metric_names):
        rf_val  = metrics_df.loc[metric, 'Random Forest']
        xgb_val = metrics_df.loc[metric, 'XGBoost']
        winner  = 'RF' if rf_val > xgb_val else ('XGB' if xgb_val > rf_val else 'Tie')
        w_color = '#1f4e79' if winner == 'RF' else '#8b4513'
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#888;margin-bottom:4px">{metric}</div>
                <div style="font-size:1rem;font-weight:600;color:#1f4e79">RF: {rf_val:.3f}</div>
                <div style="font-size:1rem;font-weight:600;color:#8b4513">XGB: {xgb_val:.3f}</div>
                <div style="margin-top:6px;font-size:0.7rem;font-weight:600;
                            color:{w_color}">Winner: {winner}</div>
            </div>
            """, unsafe_allow_html=True)

except FileNotFoundError:
    st.warning("Metrics files not found. Please run Notebooks 02 and 03 first to generate `models/rf_metrics.csv` and `models/xgb_metrics.csv`.")

st.divider()

# ── Dataset overview ───────────────────────────────────────────────────────────
st.markdown("### Dataset at a Glance")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Total Employees", "1,470")
d2.metric("Features Used",   "44")
d3.metric("Attrition Rate",  "16.1%")
d4.metric("Train Samples",   "1,176")
d5.metric("Test Samples",    "294")

st.divider()
st.markdown("### Get started")
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/01_predictor.py",
                 label="🔮  Go to Live Predictor",
                 use_container_width=True)
with c2:
    st.page_link("pages/02_comparison.py",
                 label="📈  Go to Model Comparison",
                 use_container_width=True)
