"""
Data cleaning for the Bengaluru House Price dataset.

Each step is a small function so it can be shown one by one in the notebook
and reused by train.py and the Streamlit app.
"""
from pathlib import Path

import pandas as pd

# full path, so the code works no matter which folder it is started from
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "Bengaluru_House_Data.csv"

# How many square feet one unit is worth, for areas written in other units
UNIT_TO_SQFT = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Acres": 43560.0,
    "Cents": 435.6,
    "Guntha": 1089.0,
    "Perch": 272.25,
    "Grounds": 2400.0,
}


def load_raw(path=RAW_PATH):
    """Read the original CSV file."""
    return pd.read_csv(path)


def convert_sqft(value):
    """
    Turn total_sqft text into a number of square feet.
      "1056"          -> 1056.0
      "1133 - 1384"   -> 1258.5   (average of the range)
      "34.46Sq. Meter"-> 370.9    (converted to sq ft)
    Returns None if it cannot be understood.
    """
    text = str(value).strip()
    if " - " in text:
        low, high = text.split(" - ")
        try:
            return (float(low) + float(high)) / 2
        except ValueError:
            return None
    for unit, factor in UNIT_TO_SQFT.items():
        if text.endswith(unit):
            try:
                return float(text.replace(unit, "")) * factor
            except ValueError:
                return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bhk(size):
    """'2 BHK' -> 2, '4 Bedroom' -> 4, '1 RK' -> 1."""
    try:
        return int(str(size).split()[0])
    except ValueError:
        return None


def basic_clean(df):
    """Remove duplicates, fix data types and handle missing values."""
    df = df.drop_duplicates().copy()

    # society is missing for about 40% of houses, so it is dropped
    df = df.drop(columns=["society"])

    df["location"] = df["location"].str.strip()
    df["area_type"] = df["area_type"].str.replace("  ", " ").str.strip()
    df["bhk"] = df["size"].apply(parse_bhk)
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df["ready_to_move"] = (df["availability"] == "Ready To Move").astype(int)

    # rows without location, size, area or bathrooms cannot be used
    df = df.dropna(subset=["location", "bhk", "total_sqft", "bath"])

    # a missing balcony count is filled with the most common value
    df["balcony"] = df["balcony"].fillna(df["balcony"].mode()[0])

    df["bhk"] = df["bhk"].astype(int)
    df["bath"] = df["bath"].astype(int)
    df["balcony"] = df["balcony"].astype(int)
    return df.drop(columns=["size", "availability"])


def add_price_per_sqft(df):
    """price is in lakhs, so 1 lakh = 1,00,000 rupees."""
    df = df.copy()
    df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]
    return df


def group_rare_locations(df, min_houses=10):
    """Locations with 10 or fewer houses are grouped as 'Other'."""
    df = df.copy()
    counts = df["location"].value_counts()
    rare = counts[counts <= min_houses].index
    df.loc[df["location"].isin(rare), "location"] = "Other"
    return df


def remove_outliers(df):
    """
    Remove houses that are not realistic:
      1. more than 10,000 sq ft (farm land or typing errors,
         e.g. one '3 BHK' listed as 6,53,400 sq ft)
      2. less than 300 sq ft per bedroom
      3. more bathrooms than bedrooms + 2
      4. price per sq ft far from the normal range of its location
         (outside mean +/- 1 standard deviation)
    """
    df = df[df["total_sqft"] <= 10000]
    df = df[df["total_sqft"] / df["bhk"] >= 300]
    df = df[df["bath"] <= df["bhk"] + 2]

    kept = []
    for _, group in df.groupby("location"):
        mean = group["price_per_sqft"].mean()
        std = group["price_per_sqft"].std()
        if pd.isna(std):
            kept.append(group)
            continue
        kept.append(group[(group["price_per_sqft"] >= mean - std) &
                          (group["price_per_sqft"] <= mean + std)])
    return pd.concat(kept).reset_index(drop=True)


def clean_data(path=RAW_PATH):
    """Run all cleaning steps in order and return the final table."""
    df = load_raw(path)
    df = basic_clean(df)
    df = add_price_per_sqft(df)
    df = group_rare_locations(df)
    df = remove_outliers(df)
    return df
