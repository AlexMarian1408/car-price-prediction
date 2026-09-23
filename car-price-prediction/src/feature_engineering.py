"""Feature engineering for used-car price prediction."""
from __future__ import annotations

from datetime import datetime
import pandas as pd


def add_features(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Create meaningful derived features without using the target."""
    current_year = current_year or datetime.now().year
    data = df.copy()

    data["car_age"] = (current_year - data["year"]).clip(lower=0)
    age_for_division = data["car_age"].replace(0, 1)
    data["mileage_per_year"] = data["mileage_km"] / age_for_division
    data["engine_volume_liters"] = data["engine_volume_cm3"] / 1000
    data["is_newer_car"] = (data["year"] >= 2015).astype(int)
    data["is_high_mileage"] = (data["mileage_km"] >= 300_000).astype(int)
    data["make_model"] = data["make"].astype(str) + "_" + data["model"].astype(str)

    return data


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data

    df = add_features(clean_data(load_data("data/cars.csv")))
    print(df[["year", "car_age", "mileage_km", "mileage_per_year",
              "engine_volume_cm3", "engine_volume_liters",
              "is_newer_car", "is_high_mileage", "make_model"]].head())
