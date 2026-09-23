"""Data cleaning utilities for the used-car price regression project."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COLUMN_RENAME = {
    "priceUSD": "price_usd",
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "engine_volume_cm3",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw cars CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Clean column names, duplicates, invalid values and basic formatting.

    Missing predictor values are intentionally kept for the sklearn imputers.
    """
    current_year = current_year or datetime.now().year
    data = df.copy().rename(columns=COLUMN_RENAME)

    # Standardize text values.
    for column in data.select_dtypes(include="object").columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
        )

    # Ensure numerical columns are numerical.
    for column in ["price_usd", "year", "mileage_km", "engine_volume_cm3"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Remove exact duplicate advertisements.
    data = data.drop_duplicates()

    # Target and year must be valid.
    data = data[data["price_usd"].notna() & (data["price_usd"] > 0)]
    data = data[data["year"].between(1900, current_year)]

    # Implausible mileage values are treated as missing, then imputed later.
    data.loc[data["mileage_km"] < 0, "mileage_km"] = np.nan
    data.loc[data["mileage_km"] > 1_000_000, "mileage_km"] = np.nan

    # Values such as 16000/20000 cm3 are consistent with a x10 data-entry error.
    engine_typo = data["engine_volume_cm3"] > 8000
    data.loc[engine_typo, "engine_volume_cm3"] = (
        data.loc[engine_typo, "engine_volume_cm3"] / 10
    )
    data.loc[
        (data["engine_volume_cm3"] < 500) | (data["engine_volume_cm3"] > 8000),
        "engine_volume_cm3",
    ] = np.nan

    return data.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean cars.csv")
    parser.add_argument("--input", default="data/cars.csv")
    parser.add_argument("--output", default=None, help="Optional cleaned CSV path")
    args = parser.parse_args()

    raw = load_data(args.input)
    cleaned = clean_data(raw)
    print(f"Rows before cleaning: {len(raw):,}")
    print(f"Rows after cleaning:  {len(cleaned):,}")
    print("Missing values after cleaning:")
    print(cleaned.isna().sum().sort_values(ascending=False).head(10))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
