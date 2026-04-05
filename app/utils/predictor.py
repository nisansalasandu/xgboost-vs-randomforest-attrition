"""
utils/predictor.py
Shared prediction logic — loads both pipelines and prepares input data.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# ── Column reference from Notebook 01 output ──────────────────────────────────
FEATURE_COLUMNS = None  # loaded lazily from X_train.csv

@st.cache_resource
def load_models():
    """Load both trained pipelines. Cached so they load only once."""
    rf_path  = os.path.join(MODELS_DIR, 'random_forest_tuned.joblib')
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_tuned.joblib')
    try:
        rf_pipeline  = joblib.load(rf_path)
        xgb_pipeline = joblib.load(xgb_path)
        return rf_pipeline, xgb_pipeline, None
    except FileNotFoundError as e:
        return None, None, str(e)

@st.cache_data
def load_feature_columns():
    """Load the exact feature columns used during training."""
    path = os.path.join(PROCESSED_DIR, 'X_train.csv')
    try:
        df = pd.read_csv(path, nrows=1)
        return list(df.columns)
    except FileNotFoundError:
        return None

def build_input_row(user_inputs: dict, feature_columns: list) -> pd.DataFrame:
    """
    Convert user form inputs into a single-row DataFrame matching
    the exact 44-column format used in training.

    Steps:
    1. Start with all feature columns set to their median/default values
    2. Apply the binary-encoded columns (Gender, OverTime)
    3. Apply the one-hot encoded columns
    4. Overwrite with user-supplied numerical values
    """
    # Default row — zeros for one-hot columns, medians for numeric
    row = {col: 0 for col in feature_columns}

    # ── Numerical features — set directly ─────────────────────────────────────
    numerical_map = {
        'Age'                     : user_inputs.get('Age', 36),
        'DailyRate'               : user_inputs.get('DailyRate', 800),
        'DistanceFromHome'        : user_inputs.get('DistanceFromHome', 7),
        'Education'               : user_inputs.get('Education', 3),
        'EnvironmentSatisfaction' : user_inputs.get('EnvironmentSatisfaction', 3),
        'HourlyRate'              : user_inputs.get('HourlyRate', 65),
        'JobInvolvement'          : user_inputs.get('JobInvolvement', 3),
        'JobLevel'                : user_inputs.get('JobLevel', 2),
        'JobSatisfaction'         : user_inputs.get('JobSatisfaction', 3),
        'MonthlyIncome'           : user_inputs.get('MonthlyIncome', 6500),
        'MonthlyRate'             : user_inputs.get('MonthlyRate', 14000),
        'NumCompaniesWorked'      : user_inputs.get('NumCompaniesWorked', 2),
        'PercentSalaryHike'       : user_inputs.get('PercentSalaryHike', 14),
        'PerformanceRating'       : user_inputs.get('PerformanceRating', 3),
        'RelationshipSatisfaction': user_inputs.get('RelationshipSatisfaction', 3),
        'StockOptionLevel'        : user_inputs.get('StockOptionLevel', 1),
        'TotalWorkingYears'       : user_inputs.get('TotalWorkingYears', 10),
        'TrainingTimesLastYear'   : user_inputs.get('TrainingTimesLastYear', 3),
        'WorkLifeBalance'         : user_inputs.get('WorkLifeBalance', 3),
        'YearsAtCompany'          : user_inputs.get('YearsAtCompany', 5),
        'YearsInCurrentRole'      : user_inputs.get('YearsInCurrentRole', 3),
        'YearsSinceLastPromotion' : user_inputs.get('YearsSinceLastPromotion', 1),
        'YearsWithCurrManager'    : user_inputs.get('YearsWithCurrManager', 3),
    }
    for k, v in numerical_map.items():
        if k in row:
            row[k] = v

    # ── Binary encoded ─────────────────────────────────────────────────────────
    row['Gender']   = 1 if user_inputs.get('Gender', 'Male') == 'Male' else 0
    row['OverTime'] = 1 if user_inputs.get('OverTime', 'No') == 'Yes' else 0

    # ── One-hot encoded — BusinessTravel ──────────────────────────────────────
    bt = user_inputs.get('BusinessTravel', 'Travel_Rarely')
    if 'BusinessTravel_Travel_Frequently' in row:
        row['BusinessTravel_Travel_Frequently'] = 1 if bt == 'Travel_Frequently' else 0
    if 'BusinessTravel_Travel_Rarely' in row:
        row['BusinessTravel_Travel_Rarely'] = 1 if bt == 'Travel_Rarely' else 0

    # ── One-hot encoded — Department ──────────────────────────────────────────
    dept = user_inputs.get('Department', 'Research & Development')
    if 'Department_Research & Development' in row:
        row['Department_Research & Development'] = 1 if dept == 'Research & Development' else 0
    if 'Department_Sales' in row:
        row['Department_Sales'] = 1 if dept == 'Sales' else 0

    # ── One-hot encoded — EducationField ──────────────────────────────────────
    ef = user_inputs.get('EducationField', 'Life Sciences')
    for field in ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']:
        col = f'EducationField_{field}'
        if col in row:
            row[col] = 1 if ef == field else 0

    # ── One-hot encoded — JobRole ──────────────────────────────────────────────
    jr = user_inputs.get('JobRole', 'Sales Executive')
    for role in ['Human Resources', 'Laboratory Technician', 'Manager',
                 'Manufacturing Director', 'Research Director', 'Research Scientist',
                 'Sales Executive', 'Sales Representative']:
        col = f'JobRole_{role}'
        if col in row:
            row[col] = 1 if jr == role else 0

    # ── One-hot encoded — MaritalStatus ───────────────────────────────────────
    ms = user_inputs.get('MaritalStatus', 'Married')
    if 'MaritalStatus_Married' in row:
        row['MaritalStatus_Married'] = 1 if ms == 'Married' else 0
    if 'MaritalStatus_Single' in row:
        row['MaritalStatus_Single'] = 1 if ms == 'Single' else 0

    return pd.DataFrame([row])[feature_columns]


def predict(user_inputs: dict):
    """
    Run prediction with both models.
    Returns dict with probabilities, labels, and risk levels.
    """
    rf_pipeline, xgb_pipeline, error = load_models()
    if error:
        return {'error': error}

    feature_columns = load_feature_columns()
    if feature_columns is None:
        return {'error': 'Could not load feature columns from data/processed/X_train.csv'}

    input_df = build_input_row(user_inputs, feature_columns)

    rf_proba  = rf_pipeline.predict_proba(input_df)[0][1]
    xgb_proba = xgb_pipeline.predict_proba(input_df)[0][1]

    def risk_label(p):
        if p >= 0.60: return 'High',   'risk-high'
        if p >= 0.35: return 'Medium', 'risk-medium'
        return 'Low', 'risk-low'

    rf_label,  rf_css  = risk_label(rf_proba)
    xgb_label, xgb_css = risk_label(xgb_proba)

    return {
        'rf_proba'   : rf_proba,
        'xgb_proba'  : xgb_proba,
        'rf_label'   : rf_label,
        'xgb_label'  : xgb_label,
        'rf_css'     : rf_css,
        'xgb_css'    : xgb_css,
        'input_df'   : input_df,
        'error'      : None,
    }


def get_top_features(model_name: str, n: int = 5) -> pd.Series:
    """Load top N feature importances for a given model."""
    filename = 'rf_feature_importance.csv' if model_name == 'rf' else 'xgb_feature_importance.csv'
    path = os.path.join(MODELS_DIR, filename)
    try:
        imp = pd.read_csv(path, index_col=0).squeeze()
        return imp.sort_values(ascending=False).head(n)
    except FileNotFoundError:
        return pd.Series(dtype=float)
