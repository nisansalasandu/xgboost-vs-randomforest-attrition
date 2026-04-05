"""
utils/charts.py
Reusable chart/plot functions used across pages.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score


def gauge_chart(probability: float, model_name: str, color: str) -> go.Figure:
    """Plotly gauge showing attrition probability 0–100%."""
    pct = round(probability * 100, 1)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = pct,
        title = {'text': model_name, 'font': {'size': 16}},
        delta = {'reference': 50, 'increasing': {'color': '#e74c3c'},
                 'decreasing': {'color': '#27ae60'}},
        number= {'suffix': '%', 'font': {'size': 28}},
        gauge = {
            'axis'     : {'range': [0, 100], 'tickwidth': 1},
            'bar'      : {'color': color, 'thickness': 0.25},
            'bgcolor'  : 'white',
            'borderwidth': 1,
            'bordercolor': '#ccc',
            'steps'    : [
                {'range': [0, 35],  'color': '#d4edda'},
                {'range': [35, 60], 'color': '#fff3cd'},
                {'range': [60, 100],'color': '#f8d7da'},
            ],
            'threshold': {
                'line' : {'color': '#333', 'width': 3},
                'thickness': 0.75,
                'value': 50
            },
        }
    ))
    fig.update_layout(
        height=260, margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def roc_overlay_chart(y_test, y_proba_rf, y_proba_xgb) -> plt.Figure:
    """Matplotlib ROC overlay — both models on same axes."""
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_proba_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
    auc_rf  = roc_auc_score(y_test, y_proba_rf)
    auc_xgb = roc_auc_score(y_test, y_proba_xgb)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_rf,  tpr_rf,  lw=2, color='steelblue',
            label=f'Random Forest (AUC={auc_rf:.3f})')
    ax.plot(fpr_xgb, tpr_xgb, lw=2, color='saddlebrown',
            label=f'XGBoost (AUC={auc_xgb:.3f})')
    ax.plot([0,1],[0,1], 'k:', lw=1, label='Random classifier')
    ax.fill_between(fpr_rf,  tpr_rf,  alpha=0.08, color='steelblue')
    ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.08, color='saddlebrown')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — RF vs XGBoost', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def pr_overlay_chart(y_test, y_proba_rf, y_proba_xgb) -> plt.Figure:
    """Matplotlib Precision-Recall overlay."""
    from sklearn.metrics import average_precision_score
    prec_rf,  rec_rf,  _ = precision_recall_curve(y_test, y_proba_rf)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
    ap_rf  = average_precision_score(y_test, y_proba_rf)
    ap_xgb = average_precision_score(y_test, y_proba_xgb)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_rf,  prec_rf,  lw=2, color='steelblue',
            label=f'Random Forest (AP={ap_rf:.3f})')
    ax.plot(rec_xgb, prec_xgb, lw=2, color='saddlebrown',
            label=f'XGBoost (AP={ap_xgb:.3f})')
    ax.axhline(baseline, color='grey', lw=1, linestyle='--',
               label=f'Baseline (AP={baseline:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def confusion_matrix_chart(y_test, y_pred, title: str, cmap: str) -> plt.Figure:
    """Matplotlib confusion matrix heatmap."""
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['No Attrition', 'Attrition'],
                yticklabels=['No Attrition', 'Attrition'],
                linewidths=0.5)
    ax.set_title(f'{title}\nTP={tp}  FP={fp}  FN={fn}  TN={tn}',
                 fontweight='bold', fontsize=10)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    return fig


def feature_importance_chart(feat_imp: pd.Series, title: str,
                              color: str, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    data = feat_imp.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    data.plot(kind='barh', ax=ax, color=color, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def score_distribution_chart(y_test, y_proba_rf, y_proba_xgb) -> plt.Figure:
    """Predicted probability distributions for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, y_proba, title, color in [
        (axes[0], y_proba_rf,  'Random Forest', 'steelblue'),
        (axes[1], y_proba_xgb, 'XGBoost',       'saddlebrown'),
    ]:
        ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6,
                color='green', label='No Attrition', density=True)
        ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6,
                color='red', label='Attrition', density=True)
        ax.axvline(0.5, color='black', lw=1.5, linestyle='--', label='Thr=0.5')
        ax.set_title(f'Score Distribution — {title}', fontweight='bold')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
