"""
Streamlit app for House Price Prediction & Analysis (Bengaluru).

Run from the project folder:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/analysis.py", title="Data Analysis", icon="📊"),
    st.Page("views/model.py", title="Model", icon="🤖"),
    st.Page("views/predict.py", title="Predict Price", icon="🔮"),
]

with st.sidebar:
    st.markdown("### 🏠 House Price Predictor")
    st.caption("Bengaluru • Python for Data Science mini project")

st.navigation(pages).run()

st.sidebar.markdown("---")
st.sidebar.caption("Made by Avi Shah & Anushka Patel")
