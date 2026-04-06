"""
pages/02_comparison.py
Model Comparison — metrics, ROC curves, feature importances, confusion matrices.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

from utils.charts import (
    roc_overlay_chart, pr_overlay_chart,
    confusion_matrix_chart, feature_importance_chart,
    score_distribution_chart
)

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:1.8rem; font-weight:700; color:#1f4e79; }
    .winner-cell-rf  { background:#d4edda !important; color:#155724; font-weight:700; }
    .winner-cell-xgb { background:#cce5ff !important; color:#004085; font-weight:700; }
    .section-title   { font-size:1.2rem; font-weight:600; color:#1f4e79;
                       margin:1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Model Comparison</p>', unsafe_allow_html=True)
st.markdown("Full side-by-side evaluation of **Random Forest vs XGBoost** on the held-out test set.")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# ── Load data and models ───────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    try:
        rf_pipe  = joblib.load(os.path.join(MODELS_DIR, 'random_forest_tuned.joblib'))
        xgb_pipe = joblib.load(os.path.join(MODELS_DIR, 'xgboost_tuned.joblib'))
        X_test   = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
        y_test   = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).squeeze()
        rf_imp   = pd.read_csv(os.path.join(MODELS_DIR, 'rf_feature_importance.csv'),
                               index_col=0).squeeze()
        xgb_imp  = pd.read_csv(os.path.join(MODELS_DIR, 'xgb_feature_importance.csv'),
                               index_col=0).squeeze()
        return rf_pipe, xgb_pipe, X_test, y_test, rf_imp, xgb_imp, None
    except FileNotFoundError as e:
        return None, None, None, None, None, None, str(e)

rf_pipe, xgb_pipe, X_test, y_test, rf_imp, xgb_imp, err = load_all()

if err:
    st.error(f"Required file not found: {err}\n\nPlease run all notebooks first.")
    st.stop()

# ── Generate predictions ───────────────────────────────────────────────────────
y_pred_rf   = rf_pipe.predict(X_test)
y_proba_rf  = rf_pipe.predict_proba(X_test)[:, 1]
y_pred_xgb  = xgb_pipe.predict(X_test)
y_proba_xgb = xgb_pipe.predict_proba(X_test)[:, 1]

# ── Metrics ────────────────────────────────────────────────────────────────────
def get_metrics(y_pred, y_proba):
    return {
        'Accuracy'  : accuracy_score(y_test, y_pred),
        'Precision' : precision_score(y_test, y_pred, zero_division=0),
        'Recall'    : recall_score(y_test, y_pred, zero_division=0),
        'F1-Score'  : f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC'   : roc_auc_score(y_test, y_proba),
        'PR-AUC'    : average_precision_score(y_test, y_proba),
    }

rf_metrics  = get_metrics(y_pred_rf,  y_proba_rf)
xgb_metrics = get_metrics(y_pred_xgb, y_proba_xgb)

metrics_df = pd.DataFrame({
    'Random Forest': rf_metrics,
    'XGBoost'      : xgb_metrics,
}).round(4)

# ── Section 1 — Metrics Table ─────────────────────────────────────────────────
st.markdown('<p class="section-title">1. Performance Metrics</p>',
            unsafe_allow_html=True)
st.caption("Default decision threshold = 0.5")

# Styled display
display_rows = []
for metric in metrics_df.index:
    rf_val  = metrics_df.loc[metric, 'Random Forest']
    xgb_val = metrics_df.loc[metric, 'XGBoost']
    if rf_val > xgb_val:
        winner = 'Random Forest'
    elif xgb_val > rf_val:
        winner = 'XGBoost'
    else:
        winner = 'Tie'
    display_rows.append({
        'Metric'        : metric,
        'Random Forest' : rf_val,
        'XGBoost'       : xgb_val,
        'Winner'        : winner,
        'Difference'    : round(abs(rf_val - xgb_val), 4),
    })

display_df = pd.DataFrame(display_rows).set_index('Metric')
st.dataframe(display_df, use_container_width=True, height=260)

# ── Metric cards ──────────────────────────────────────────────────────────────
metric_cols = st.columns(6)
for col, (metric, row) in zip(metric_cols, display_df.iterrows()):
    rf_v  = row['Random Forest']
    xgb_v = row['XGBoost']
    delta = round(xgb_v - rf_v, 4)
    with col:
        st.metric(label=metric,
                  value=f"RF: {rf_v:.3f}",
                  delta=f"XGB: {xgb_v:.3f}  (Δ{delta:+.4f})")

# ── Optimal threshold comparison ──────────────────────────────────────────────
with st.expander("Show metrics at optimal thresholds"):
    thresholds = np.linspace(0.01, 0.99, 200)

    def best_threshold(y_true, y_proba):
        f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
               for t in thresholds]
        idx = np.argmax(f1s)
        return thresholds[idx], f1s[idx]

    rf_thr,  rf_f1  = best_threshold(y_test, y_proba_rf)
    xgb_thr, xgb_f1 = best_threshold(y_test, y_proba_xgb)

    y_rf_opt  = (y_proba_rf  >= rf_thr).astype(int)
    y_xgb_opt = (y_proba_xgb >= xgb_thr).astype(int)

    opt_df = pd.DataFrame({
        f'RF (thr={rf_thr:.2f})'  : get_metrics(y_rf_opt,  y_proba_rf),
        f'XGB (thr={xgb_thr:.2f})': get_metrics(y_xgb_opt, y_proba_xgb),
    }).round(4)
    st.dataframe(opt_df, use_container_width=True)

st.divider()

# ── Section 2 — ROC & PR Curves ───────────────────────────────────────────────
st.markdown('<p class="section-title">2. ROC & Precision-Recall Curves</p>',
            unsafe_allow_html=True)

curve_col1, curve_col2 = st.columns(2)
with curve_col1:
    fig_roc = roc_overlay_chart(y_test, y_proba_rf, y_proba_xgb)
    st.pyplot(fig_roc)
    plt.close()
with curve_col2:
    fig_pr = pr_overlay_chart(y_test, y_proba_rf, y_proba_xgb)
    st.pyplot(fig_pr)
    plt.close()

st.divider()

# ── Section 3 — Confusion Matrices ────────────────────────────────────────────
st.markdown('<p class="section-title">3. Confusion Matrices</p>',
            unsafe_allow_html=True)

thr_option = st.radio("Threshold", ["Default (0.5)", "Optimal (F1-maximising)"],
                      horizontal=True)

cm_col1, cm_col2 = st.columns(2)
if thr_option == "Default (0.5)":
    with cm_col1:
        fig = confusion_matrix_chart(y_test, y_pred_rf,  "Random Forest", "Blues")
        st.pyplot(fig); plt.close()
    with cm_col2:
        fig = confusion_matrix_chart(y_test, y_pred_xgb, "XGBoost", "Oranges")
        st.pyplot(fig); plt.close()
else:
    with cm_col1:
        fig = confusion_matrix_chart(y_test, y_rf_opt,
                                     f"Random Forest (thr={rf_thr:.2f})", "Blues")
        st.pyplot(fig); plt.close()
    with cm_col2:
        fig = confusion_matrix_chart(y_test, y_xgb_opt,
                                     f"XGBoost (thr={xgb_thr:.2f})", "Oranges")
        st.pyplot(fig); plt.close()

st.divider()

# ── Section 4 — Feature Importance ────────────────────────────────────────────
st.markdown('<p class="section-title">4. Feature Importance Comparison</p>',
            unsafe_allow_html=True)
st.caption("Random Forest uses Gini importance. XGBoost uses Gain importance.")

top_n = st.slider("Number of features to show", 5, 20, 15)

fi_col1, fi_col2 = st.columns(2)
with fi_col1:
    fig = feature_importance_chart(rf_imp.sort_values(ascending=False),
                                   "Random Forest — Gini Importance",
                                   "steelblue", top_n)
    st.pyplot(fig); plt.close()
with fi_col2:
    fig = feature_importance_chart(xgb_imp.sort_values(ascending=False),
                                   "XGBoost — Gain Importance",
                                   "darkorange", top_n)
    st.pyplot(fig); plt.close()

# Common top features
st.markdown("**Features in top 15 of BOTH models:**")
rf_top_set  = set(rf_imp.head(15).index)
xgb_top_set = set(xgb_imp.head(15).index)
common = sorted(rf_top_set & xgb_top_set)
if common:
    st.success(f"Shared top features ({len(common)}): {',  '.join(common)}")
else:
    st.info("No features in common in top 15.")

st.divider()

# ── Section 5 — Score Distributions ──────────────────────────────────────────
st.markdown('<p class="section-title">5. Predicted Probability Distributions</p>',
            unsafe_allow_html=True)
st.caption("How well separated are the model scores for each class?")
fig = score_distribution_chart(y_test, y_proba_rf, y_proba_xgb)
st.pyplot(fig); plt.close()

st.divider()

# ── Section 6 — Prediction Agreement ─────────────────────────────────────────
st.markdown('<p class="section-title">6. Prediction Agreement Analysis</p>',
            unsafe_allow_html=True)

both_correct   = ((y_pred_rf == y_test.values) & (y_pred_xgb == y_test.values)).sum()
rf_only_right  = ((y_pred_rf == y_test.values) & (y_pred_xgb != y_test.values)).sum()
xgb_only_right = ((y_pred_rf != y_test.values) & (y_pred_xgb == y_test.values)).sum()
both_wrong     = ((y_pred_rf != y_test.values) & (y_pred_xgb != y_test.values)).sum()
total = len(y_test)

a1, a2, a3, a4 = st.columns(4)
a1.metric("Both Correct",     f"{both_correct}",    f"{both_correct/total*100:.1f}%")
a2.metric("RF Only Correct",  f"{rf_only_right}",   f"{rf_only_right/total*100:.1f}%")
a3.metric("XGB Only Correct", f"{xgb_only_right}",  f"{xgb_only_right/total*100:.1f}%")
a4.metric("Both Wrong",       f"{both_wrong}",       f"{both_wrong/total*100:.1f}%")

st.caption(f"Agreement rate: {(both_correct+both_wrong)/total*100:.1f}%  |  "
           f"Total test samples: {total}")
