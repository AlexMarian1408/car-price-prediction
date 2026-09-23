"""Evaluate the saved final model on the reproducible hold-out set."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_cleaning import load_data, clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved car-price model")
    parser.add_argument("--data", default="data/cars.csv")
    parser.add_argument("--model", default="models/car_price_model.joblib")
    parser.add_argument("--metrics", default="models/evaluation_metrics.json")
    args = parser.parse_args()

    df = add_features(clean_data(load_data(args.data)))
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = joblib.load(args.model)
    predictions = model.predict(X_test)

    metrics = {
        "MAE_USD": float(mean_absolute_error(y_test, predictions)),
        "RMSE_USD": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    comparison = pd.DataFrame({"actual": y_test.iloc[:10].values, "predicted": predictions[:10]})
    print("\nSample predictions:")
    print(comparison.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
