"""
Model building, comparison and prediction for the house price project.
Used by train.py, the notebook and the Streamlit app.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

CATEGORICAL = ["location", "area_type"]
NUMERIC = ["total_sqft", "bhk", "bath", "balcony", "ready_to_move"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"            # in lakhs


def _one_hot():
    # turns each location / area type into its own 0-1 column
    return OneHotEncoder(handle_unknown="ignore")


def _log_target(model):
    # the model learns log(price); predictions are converted back with exp()
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)


def build_models():
    """
    The four models we compare. Each one is a Pipeline: data preparation + model.

    Linear models use log(price) and log(area): price grows in proportion to
    area, so on the log scale the relationship becomes a straight line.
    Tree models can learn curved relationships themselves, so they use the
    plain values.
    """
    linear_prep = ColumnTransformer([
        ("categories", _one_hot(), CATEGORICAL),
        ("log_area", FunctionTransformer(np.log, feature_names_out="one-to-one"), ["total_sqft"]),
    ], remainder="passthrough")
    tree_prep = ColumnTransformer([("categories", _one_hot(), CATEGORICAL)], remainder="passthrough")

    return {
        "Linear Regression": Pipeline([("prep", linear_prep), ("model", _log_target(LinearRegression()))]),
        "Ridge Regression": Pipeline([("prep", linear_prep), ("model", _log_target(Ridge(alpha=1.0)))]),
        "Random Forest": Pipeline([("prep", tree_prep), ("model", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", tree_prep), ("model", GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42))]),
    }


def compare_models(X, y, folds=5):
    """5-fold cross-validation for every model. Returns one row per model."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv,
                                scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])
        rows.append({
            "Model": name,
            "R2": scores["test_r2"].mean(),
            "R2 std": scores["test_r2"].std(),
            "MAE (lakhs)": -scores["test_neg_mean_absolute_error"].mean(),
            "RMSE (lakhs)": -scores["test_neg_root_mean_squared_error"].mean(),
        })
    return pd.DataFrame(rows).sort_values("MAE (lakhs)").reset_index(drop=True)


def linear_effects(model):
    """
    For the linear model: how much each feature changes the price, in %.
    Because the model works on log(price), a coefficient b means
    roughly (e^b - 1) x 100 % change in price.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["model"].regressor_.coef_
    effects = pd.DataFrame({"feature": names, "coef": coefs})
    effects["feature"] = (effects["feature"]
                          .str.replace("categories__location_", "location: ", regex=False)
                          .str.replace("categories__area_type_", "area type: ", regex=False)
                          .str.replace("log_area__total_sqft", "log(total_sqft)", regex=False)
                          .str.replace("remainder__", "", regex=False))
    effects["price_change_%"] = (np.exp(effects["coef"]) - 1) * 100
    return effects


def predict_price(model, location, total_sqft, bhk, bath, balcony=1,
                  area_type="Super built-up Area", ready_to_move=1):
    """Predicted price in lakhs for one house."""
    house = pd.DataFrame([{
        "location": location, "area_type": area_type, "total_sqft": total_sqft,
        "bhk": bhk, "bath": bath, "balcony": balcony, "ready_to_move": ready_to_move,
    }])[FEATURES]
    return float(model.predict(house)[0])
