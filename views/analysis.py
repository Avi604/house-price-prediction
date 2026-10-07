import numpy as np
import plotly.express as px
import streamlit as st

from src.app_data import load_data

df = load_data()

st.title("📊 Data Analysis")
st.markdown("How do area, location, BHK and other features affect house prices in Bengaluru?")

# ---------- filters ----------
with st.expander("Filter the houses", expanded=False):
    f1, f2 = st.columns(2)
    bhk_range = f1.slider("BHK", 1, int(df["bhk"].max()), (1, 6))
    area_types = f2.multiselect("Area type", sorted(df["area_type"].unique()), default=sorted(df["area_type"].unique()))
data = df[df["bhk"].between(*bhk_range) & df["area_type"].isin(area_types)]
st.caption(f"Showing **{len(data):,}** of {len(df):,} houses")

if data.empty:
    st.warning("No houses match these filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["💰 Price", "📍 Location", "🛏️ BHK & Area", "🔗 Correlation"])

# ---------- price ----------
with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(data, x="price", nbins=60, title="Price distribution (₹ lakhs)",
                                 labels={"price": "Price (₹ lakhs)"}), width="stretch")
    c2.plotly_chart(px.histogram(x=np.log(data["price"]), nbins=60, title="Log of price",
                                 labels={"x": "log(price)"}, color_discrete_sequence=["darkorange"]),
                    width="stretch")
    st.info(f"Price is **right-skewed** (skewness {data['price'].skew():.1f}): most houses cost ₹40–100 lakhs, "
            f"a few cost crores. The log of price is much more balanced (skewness {np.log(data['price']).skew():.1f}), "
            "which is why the model predicts log(price).")

    fig = px.scatter(data, render_mode="svg", x="total_sqft", y="price", color="bhk", hover_data=["location", "bath"],
                     opacity=0.6, title="Area vs Price",
                     labels={"total_sqft": "Total area (sq ft)", "price": "Price (₹ lakhs)", "bhk": "BHK"})
    st.plotly_chart(fig, width="stretch")
    st.info(f"**Bigger houses cost more.** Correlation between area and price: "
            f"**{data[['total_sqft', 'price']].corr().iloc[0, 1]:.2f}**. Hover over a dot to see its location.")

# ---------- location ----------
with tab2:
    named = data[data["location"] != "Other"]
    loc_stats = (named.groupby("location")
                 .agg(houses=("price", "size"), median_price=("price", "median"),
                      price_per_sqft=("price_per_sqft", "mean"))
                 .sort_values("price_per_sqft", ascending=False))
    loc_stats = loc_stats[loc_stats["houses"] >= 5]

    c1, c2 = st.columns(2)
    top = loc_stats.head(10).reset_index()
    c1.plotly_chart(px.bar(top, x="price_per_sqft", y="location", orientation="h",
                           title="10 most expensive locations (₹ per sq ft)", color_discrete_sequence=["indianred"],
                           labels={"price_per_sqft": "₹ per sq ft", "location": ""})
                    .update_yaxes(autorange="reversed"), width="stretch")
    bottom = loc_stats.tail(10).sort_values("price_per_sqft").reset_index()
    c2.plotly_chart(px.bar(bottom, x="price_per_sqft", y="location", orientation="h",
                           title="10 cheapest locations (₹ per sq ft)", color_discrete_sequence=["seagreen"],
                           labels={"price_per_sqft": "₹ per sq ft", "location": ""})
                    .update_yaxes(autorange="reversed"), width="stretch")
    if len(loc_stats) >= 2:
        st.info(f"**Location changes the price a lot:** {loc_stats.index[0]} costs about "
                f"₹{loc_stats['price_per_sqft'].iloc[0]:,.0f} per sq ft, roughly "
                f"**{loc_stats['price_per_sqft'].iloc[0] / loc_stats['price_per_sqft'].iloc[-1]:.0f}×** "
                f"{loc_stats.index[-1]} (₹{loc_stats['price_per_sqft'].iloc[-1]:,.0f}).")

    st.subheader("Explore a location")
    choice = st.selectbox("Choose a location", loc_stats.sort_index().index)
    one = named[named["location"] == choice]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Houses", len(one))
    m2.metric("Median price", f"₹{one['price'].median():.0f} lakh")
    m3.metric("Rate per sq ft", f"₹{one['price_per_sqft'].mean():,.0f}",
              f"{(one['price_per_sqft'].mean() / data['price_per_sqft'].mean() - 1) * 100:+.0f}% vs city average")
    m4.metric("Most common", f"{one['bhk'].mode()[0]} BHK")
    st.plotly_chart(px.box(one, x="bhk", y="price", points="all", title=f"Prices in {choice} by BHK",
                           labels={"bhk": "BHK", "price": "Price (₹ lakhs)"}), width="stretch")

# ---------- BHK & area type ----------
with tab3:
    common = data[data["bhk"] <= 6]
    c1, c2 = st.columns(2)
    counts = common["bhk"].value_counts().sort_index().reset_index()
    c1.plotly_chart(px.bar(counts, x="bhk", y="count", title="Number of houses by BHK",
                           labels={"bhk": "BHK", "count": "Houses"}), width="stretch")
    c2.plotly_chart(px.box(common, x="bhk", y="price", points=False, title="Price by BHK (₹ lakhs)",
                           labels={"bhk": "BHK", "price": "Price (₹ lakhs)"}), width="stretch")
    med = common.groupby("bhk")["price"].median()
    st.info("**2 and 3 BHK houses are the most common.** Median price rises with BHK: "
            + ", ".join(f"{b} BHK ₹{p:.0f} lakh" for b, p in med.head(4).items()) + ".")

    c3, c4 = st.columns(2)
    at = data.groupby("area_type")["price_per_sqft"].median().reset_index()
    c3.plotly_chart(px.bar(at, x="area_type", y="price_per_sqft", title="Median ₹ per sq ft by area type",
                           color_discrete_sequence=["mediumpurple"],
                           labels={"area_type": "", "price_per_sqft": "₹ per sq ft"}), width="stretch")
    rt = (data.assign(status=data["ready_to_move"].map({1: "Ready to move", 0: "Under construction"}))
          .groupby("status")["price_per_sqft"].median().reset_index())
    c4.plotly_chart(px.bar(rt, x="status", y="price_per_sqft", title="Median ₹ per sq ft: ready vs under construction",
                           color_discrete_sequence=["goldenrod"], labels={"status": "", "price_per_sqft": "₹ per sq ft"}),
                    width="stretch")
    st.info("**Plots cost more per sq ft** than flats because the buyer also gets the land. "
            "Ready-to-move and under-construction houses cost almost the same per sq ft.")

# ---------- correlation ----------
with tab4:
    cols = ["price", "total_sqft", "bhk", "bath", "balcony", "ready_to_move", "price_per_sqft"]
    corr = data[cols].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation between numeric features", aspect="auto"),
                    width="stretch")
    st.info("**Area** has the strongest link with price, followed by bathrooms and BHK. "
            "Balcony and ready-to-move status have almost no effect.")
