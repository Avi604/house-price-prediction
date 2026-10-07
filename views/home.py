import streamlit as st

from src.app_data import load_data, load_metrics

df = load_data()
metrics = load_metrics()
test = metrics["test"]

st.title("🏠 Bengaluru House Price Prediction & Analysis")
st.markdown(
    "Predict the price of a house in **Bengaluru** from its location, size and features, "
    "and explore what makes houses expensive. Built with **Python, pandas, scikit-learn and Streamlit**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Houses analysed", f"{metrics['houses']:,}")
c2.metric("Locations", metrics["locations"])
c3.metric("Model accuracy (R²)", f"{test['R2']:.2f}")
c4.metric("Typical error", f"{test['Median % error']:.0f}%")

st.divider()

left, right = st.columns([3, 2])
with left:
    st.subheader("How the project works")
    st.markdown(f"""
1. **Data:** 13,320 house listings from Bengaluru (*Bengaluru House price data*, Kaggle).
2. **Cleaning:** removed duplicates, converted text like "2 BHK" and "1133 - 1384 sq ft" into numbers,
   and removed unrealistic listings, leaving **{metrics['houses']:,} houses**.
3. **Analysis:** charts showing how area, location, BHK and other features affect price.
4. **Model:** compared 4 machine learning models with 5-fold cross-validation.
   The best was **{metrics['best_model']}**.
5. **Prediction:** enter any house's details and get its predicted price with a likely range.
""")
with right:
    st.subheader("Explore the app")
    st.page_link("views/analysis.py", label="See the data analysis", icon="📊")
    st.page_link("views/model.py", label="Compare the models", icon="🤖")
    st.page_link("views/predict.py", label="Predict a house price", icon="🔮")

    st.subheader("Quick facts")
    st.markdown(f"""
- Median house price: **₹{df['price'].median():.0f} lakh**
- Median size: **{df['total_sqft'].median():,.0f} sq ft**
- Median rate: **₹{df['price_per_sqft'].median():,.0f} per sq ft**
""")

st.divider()
st.subheader("Sample of the cleaned data")
st.dataframe(
    df[["location", "area_type", "total_sqft", "bhk", "bath", "balcony", "ready_to_move", "price"]].head(10),
    hide_index=True, width="stretch",
)
