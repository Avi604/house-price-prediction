import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_data import load_metrics, load_model, load_test_predictions
from src.model import linear_effects

metrics = load_metrics()
test = metrics["test"]
results = pd.DataFrame(metrics["comparison"])

st.title("🤖 Model Comparison & Evaluation")
st.markdown(
    "We trained **4 models** to predict the price from location, area type, total area, BHK, bathrooms, "
    "balconies and ready-to-move status, and compared them using **5-fold cross-validation**: "
    "each model is trained 5 times, each time tested on a different fifth of the data."
)

# ---------- comparison ----------
st.subheader("1. Comparing the models")
show = results.rename(columns={"R2": "R² (higher is better)", "R2 std": "R² variation",
                               "MAE (lakhs)": "Average error (₹ lakhs)", "RMSE (lakhs)": "RMSE (₹ lakhs)"})
st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "Model"})
             .highlight_min(subset=["Average error (₹ lakhs)"], color="#d4edda"),
             hide_index=True, width="stretch")
c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(results, x="R2", y="Model", orientation="h", title="R² score (higher is better)",
                       range_x=[0.7, 0.86]).update_yaxes(autorange="reversed"), width="stretch")
c2.plotly_chart(px.bar(results, x="MAE (lakhs)", y="Model", orientation="h",
                       title="Average error in ₹ lakhs (lower is better)", color_discrete_sequence=["indianred"])
                .update_yaxes(autorange="reversed"), width="stretch")
st.success(f"**Chosen model: {metrics['best_model']}.** It has the lowest average error, and its R² is practically "
           "tied with Gradient Boosting. It is also the simplest model and its results are easy to explain.")

with st.expander("Why does Linear Regression use log(price) and log(area)?"):
    st.markdown("""
Price grows **in proportion** to area. On the log scale, this relationship becomes a straight line,
which a linear model can learn well. Our first try, using log(price) with plain area, gave a **negative R²**,
because large houses got exponentially huge predictions. Using log(area) as well fixed this.
Tree models (Random Forest, Gradient Boosting) can learn curved patterns themselves, so they use the plain values.
""")

# ---------- test results ----------
st.subheader("2. Testing on unseen houses")
st.markdown(f"20% of houses (**{metrics['test_size']:,}**) were kept aside and never shown to the model during training.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² score", f"{test['R2']:.3f}", help="Share of price differences the model explains")
m2.metric("Average error", f"₹{test['MAE (lakhs)']:.1f} lakh")
m3.metric("Typical error", f"{test['Median % error']:.1f}%", help="Median of |predicted - actual| / actual")
m4.metric("RMSE", f"₹{test['RMSE (lakhs)']:.1f} lakh", help="Punishes big mistakes more")

tp = load_test_predictions()
c1, c2 = st.columns(2)
fig = px.scatter(tp, render_mode="svg", x="actual", y="predicted", opacity=0.45, log_x=True, log_y=True,
                 hover_data=["location", "total_sqft", "bhk"], title="Actual vs predicted price (log scale)",
                 labels={"actual": "Actual price (₹ lakhs)", "predicted": "Predicted price (₹ lakhs)"})
fig.add_trace(go.Scatter(x=[10, 2000], y=[10, 2000], mode="lines", name="perfect prediction",
                         line=dict(color="red")))
c1.plotly_chart(fig, width="stretch")
err = ((tp["predicted"] - tp["actual"]) / tp["actual"] * 100).clip(-100, 100)
c2.plotly_chart(px.histogram(x=err, nbins=60, title="Prediction error (%)", color_discrete_sequence=["mediumpurple"],
                             labels={"x": "(predicted − actual) ÷ actual × 100"}).add_vline(x=0, line_color="red"),
                width="stretch")
low, high = metrics["range_factors"]
st.info(f"Points follow the red line from cheap to very expensive houses, and errors are centred on 0%. "
        f"For 80% of test houses, the real price was between **{low:.2f}×** and **{high:.2f}×** of the prediction. "
        "The Predict page uses this as the likely price range.")

# ---------- what drives price ----------
st.subheader("3. What drives the price?")
effects = linear_effects(load_model())
area_coef = effects.loc[effects["feature"] == "log(total_sqft)", "coef"].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("10% more area", f"+{(1.10 ** area_coef - 1) * 100:.1f}% price")
coef = effects.set_index("feature")["coef"]
plot_vs_flat = (np.exp(coef["area type: Plot Area"] - coef["area type: Super built-up Area"]) - 1) * 100
bath = effects.loc[effects["feature"] == "bath", "price_change_%"].iloc[0]
c2.metric("Plot instead of a flat", f"{plot_vs_flat:+.0f}% price", help="Compared with a super built-up flat of the same size")
c3.metric("One more bathroom", f"{bath:+.1f}% price")

locs = effects[effects["feature"].str.startswith("location") & (effects["feature"] != "location: Other")]
locs = locs.sort_values("price_change_%")
both = pd.concat([locs.head(8), locs.tail(8)]).assign(location=lambda d: d["feature"].str.replace("location: ", ""))
both["effect"] = np.where(both["price_change_%"] < 0, "cheaper", "pricier")
st.plotly_chart(px.bar(both, x="price_change_%", y="location", orientation="h", color="effect",
                       color_discrete_map={"cheaper": "seagreen", "pricier": "indianred"},
                       title="Effect of location on price (% vs an average location)", height=520,
                       labels={"price_change_%": "% change in price", "location": ""}),
                width="stretch")
st.info("**Location is the biggest factor:** the most expensive areas cost about 3× an average location, "
        "while the cheapest are about 45% below it. Once area is fixed, more BHK slightly *lowers* the price "
        "(same space split into smaller rooms).")
