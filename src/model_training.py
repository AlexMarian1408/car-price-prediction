"""Train and save a regression model for used-car prices."""
from __future__ import annotations

from pathlib import Path
import argparse
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_cleaning import load_data, clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target, build_preprocessor


def build_model(X):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, encoding="ordinal")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.08,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final car-price model")
    parser.add_argument("--data", default="data/cars.csv")
    parser.add_argument("--output", default="models/car_price_model.joblib")
    args = parser.parse_args()

    df = add_features(clean_data(load_data(args.data)))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = build_model(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)

    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2:    {r2:.4f}")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
