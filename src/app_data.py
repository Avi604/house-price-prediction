"""
Loads the data, model and results once and keeps them in memory (cache),
so the app stays fast when the user moves between pages.
"""
import json

import joblib
import pandas as pd
import streamlit as st

from src.data_prep import PROJECT_DIR, clean_data

MODELS_DIR = PROJECT_DIR / "models"


@st.cache_data
def load_data():
    return clean_data()


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "house_price_model.joblib")


@st.cache_data
def load_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_test_predictions():
    return pd.read_csv(MODELS_DIR / "test_predictions.csv")


def format_price(lakhs):
    """78.5 -> '₹78.5 lakh', 150 -> '₹1.50 crore'."""
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} crore"
    return f"₹{lakhs:.1f} lakh"
