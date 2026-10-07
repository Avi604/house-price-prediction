import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_data import format_price, load_data, load_metrics, load_model
from src.model import predict_price

df = load_data()
model = load_model()
low, high = load_metrics()["range_factors"]

st.title("🔮 Predict a House Price")
st.markdown("Enter the details of a house in Bengaluru to get its predicted price.")

locations = sorted(l for l in df["location"].unique() if l != "Other") + ["Other"]
area_types = sorted(df["area_type"].unique())

with st.form("house"):
    c1, c2 = st.columns(2)
    location = c1.selectbox("📍 Location", locations, index=locations.index("Whitefield"),
                            help="Locations with 10 or fewer houses in the data are grouped as 'Other'")
    area_type = c2.selectbox("🏗️ Area type", area_types, index=area_types.index("Super built-up Area"))

    c3, c4, c5 = st.columns(3)
    total_sqft = c3.number_input("📐 Total area (sq ft)", min_value=300, max_value=10000, value=1200, step=50)
    bhk = c4.number_input("🛏️ BHK (bedrooms)", min_value=1, max_value=10, value=2)
    bath = c5.number_input("🛁 Bathrooms", min_value=1, max_value=12, value=2)

    c6, c7 = st.columns(2)
    balcony = c6.slider("🌇 Balconies", 0, 3, 1)
    ready = c7.radio("🔑 Availability", ["Ready to move", "Under construction"], horizontal=True)

    submitted = st.form_submit_button("Predict price", type="primary", width="stretch")

if not submitted:
    st.info("Fill in the details and press **Predict price**.")
    st.stop()

# same rules we used to clean the data
problems = []
if total_sqft / bhk < 300:
    problems.append(f"{total_sqft} sq ft is very small for {bhk} BHK (less than 300 sq ft per bedroom).")
if bath > bhk + 2:
    problems.append(f"{bath} bathrooms is unusual for {bhk} BHK (more than BHK + 2).")
if problems:
    for p in problems:
        st.warning(p)
    st.caption("Houses like this were removed from the data as unrealistic, so the prediction may be unreliable.")

house = dict(location=location, total_sqft=total_sqft, bhk=bhk, bath=bath, balcony=balcony,
             area_type=area_type, ready_to_move=1 if ready == "Ready to move" else 0)
price = predict_price(model, **house)

st.divider()
st.markdown(f"<h2 style='text-align:center'>Predicted price: {format_price(price)}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center'>Likely range: <b>{format_price(price * low)}</b> to "
            f"<b>{format_price(price * high)}</b> &nbsp;(80% of real prices fall in this range)</p>",
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Rate per sq ft", f"₹{price * 100000 / total_sqft:,.0f}")
similar = df[(df["location"] == location) & (df["bhk"] == bhk)]
if len(similar) >= 3:
    m2.metric(f"Median {bhk} BHK in {location}", format_price(similar["price"].median()),
              help=f"Based on {len(similar)} houses in the data")
    m3.metric(f"Typical size there", f"{similar['total_sqft'].median():,.0f} sq ft")
else:
    m2.metric("Similar houses in data", len(similar))
    m3.caption("Too few similar houses in this location to compare.")

# ---------- what if ----------
st.subheader("What if…?")
st.caption("How the predicted price changes if one thing about this house is different.")
changes = [
    ("10% bigger area", dict(total_sqft=min(int(total_sqft * 1.1), 10000))),
    ("One more bathroom", dict(bath=bath + 1)),
    ("Plot instead of flat", dict(area_type="Plot Area")),
    ("Under construction" if ready == "Ready to move" else "Ready to move",
     dict(ready_to_move=0 if ready == "Ready to move" else 1)),
]
rows = []
for label, change in changes:
    new = predict_price(model, **{**house, **change})
    rows.append({"Change": label, "New price": format_price(new), "Difference": f"{(new / price - 1) * 100:+.1f}%"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("The same house in other locations")
compare = ["Electronic City Phase II", "Whitefield", "Hebbal", "Indira Nagar", "Rajaji Nagar", location]
others = []
for loc in dict.fromkeys(compare):
    p = predict_price(model, **{**house, "location": loc})
    others.append({"Location": loc, "Price (₹ lakhs)": round(p, 1), "this": loc == location})
others = pd.DataFrame(others).sort_values("Price (₹ lakhs)")
st.plotly_chart(px.bar(others, x="Price (₹ lakhs)", y="Location", orientation="h", color="this",
                       color_discrete_map={True: "indianred", False: "steelblue"}, text="Price (₹ lakhs)")
                .update_layout(showlegend=False), width="stretch")
