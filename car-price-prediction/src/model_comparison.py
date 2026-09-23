"""Compare at least three regression algorithms and save the best model."""
from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_cleaning import load_data, clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target, build_preprocessor


def candidate_models(X_train):
    return {
        "Ridge": Pipeline([
            ("preprocessor", build_preprocessor(X_train, encoding="onehot")),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", build_preprocessor(X_train, encoding="ordinal")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", build_preprocessor(X_train, encoding="ordinal")),
            ("model", HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                l2_regularization=1.0,
                random_state=42,
            )),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regression models")
    parser.add_argument("--data", default="data/cars.csv")
    parser.add_argument("--model-output", default="models/car_price_model.joblib")
    parser.add_argument("--results-output", default="models/model_comparison.csv")
    args = parser.parse_args()

    df = add_features(clean_data(load_data(args.data)))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    results = []
    fitted = {}
    for name, pipeline in candidate_models(X_train).items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results.append({
            "model": name,
            "MAE_USD": mean_absolute_error(y_test, pred),
            "RMSE_USD": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })
        fitted[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        ["RMSE_USD", "MAE_USD"], ascending=True
    ).reset_index(drop=True)
    best_name = results_df.loc[0, "model"]

    Path(args.results_output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_output, index=False)
    joblib.dump(fitted[best_name], args.model_output)

    print(results_df.round({"MAE_USD": 2, "RMSE_USD": 2, "R2": 4}).to_string(index=False))
    print(f"\nFinal model selected by lowest RMSE: {best_name}")
    print(f"Saved final model: {args.model_output}")


if __name__ == "__main__":
    main()
