"""Preprocessing pipelines for regression models."""
from __future__ import annotations

from typing import Literal
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET_COLUMN = "price_usd"


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def get_column_groups(X: pd.DataFrame):
    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numerical]
    return numerical, categorical


def build_preprocessor(
    X: pd.DataFrame,
    encoding: Literal["onehot", "ordinal"] = "onehot",
) -> ColumnTransformer:
    """Create leakage-safe preprocessing fitted only on training data."""
    numerical, categorical = get_column_groups(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if encoding == "onehot":
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
        )
    else:
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical),
            ("cat", categorical_pipeline, categorical),
        ],
        sparse_threshold=0 if encoding == "ordinal" else 0.3,
    )
