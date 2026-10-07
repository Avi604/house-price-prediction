"""
Trains and compares the models, then saves the best one for the Streamlit app.

Run from the project folder:
    python train.py

Creates:
    models/house_price_model.joblib   the trained best model
    models/metrics.json               comparison results, test scores, price range
    models/test_predictions.csv       actual vs predicted prices on the test set
"""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_prep import clean_data
from src.model import FEATURES, TARGET, build_models, compare_models

os.makedirs("models", exist_ok=True)

# 1. Clean data
df = clean_data()
X, y = df[FEATURES], df[TARGET]
print(f"Clean data: {len(df)} houses, {df['location'].nunique()} locations")

# 2. Compare all models with 5-fold cross-validation
results = compare_models(X, y)
print("\nModel comparison (5-fold cross-validation):")
print(results.round(3).to_string(index=False))

# 3. The best model is the one with the lowest average error (MAE)
best_name = results.loc[0, "Model"]
print(f"\nBest model: {best_name}")

# 4. Check it on a separate test set (20% of houses it never saw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_models()[best_name].fit(X_train, y_train)
pred = model.predict(X_test)
test = {
    "R2": r2_score(y_test, pred),
    "MAE (lakhs)": mean_absolute_error(y_test, pred),
    "RMSE (lakhs)": float(np.sqrt(mean_squared_error(y_test, pred))),
    "Median % error": float(np.median(np.abs(y_test - pred) / y_test) * 100),
}
print("Test set:", {k: round(v, 3) for k, v in test.items()})

# 5. Likely price range: for 80% of test houses, actual / predicted
#    was between these two values
ratio = y_test / pred
low, high = np.percentile(ratio, [10, 90])
print(f"80% of real prices were between {low:.2f}x and {high:.2f}x of the prediction")

# 6. Train the final model on ALL houses and save everything
final_model = build_models()[best_name].fit(X, y)
joblib.dump(final_model, "models/house_price_model.joblib")

X_test.assign(actual=y_test.values, predicted=pred).to_csv("models/test_predictions.csv", index=False)

with open("models/metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "houses": len(df),
        "locations": int(df["location"].nunique()),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "comparison": results.round(4).to_dict(orient="records"),
        "test": {k: round(v, 4) for k, v in test.items()},
        "range_factors": [round(float(low), 3), round(float(high), 3)],
    }, f, indent=2)

print("\nSaved models/house_price_model.joblib, models/metrics.json and models/test_predictions.csv")
