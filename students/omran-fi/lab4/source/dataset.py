from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.datasets import load_diabetes, fetch_openml


DatasetName = Literal["diabetes", "openml:house_prices"]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def load_dataset(name: DatasetName = "diabetes", seed: int = 42) -> Dataset:
    if name == "diabetes":
        data = load_diabetes(as_frame=True)

        df = data.frame
        target_col = "target"  # ✅ الصحيح في diabetes

        X = df.drop(columns=[target_col]).to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)

        return Dataset(
            name="diabetes",
            X=X,
            y=y,
            feature_names=list(df.drop(columns=[target_col]).columns),
        )

    if name == "openml:house_prices":
        bunch = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        df = bunch.frame.copy()

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        target_col = "SalePrice"

        if target_col not in numeric_cols:
            raise RuntimeError("SalePrice not found in numeric columns")

        numeric_cols.remove(target_col)

        df = df[numeric_cols + [target_col]].dropna()
        df = df.sample(n=min(len(df), 3000), random_state=seed)

        X = df[numeric_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)

        return Dataset(
            name="openml_house_prices",
            X=X,
            y=y,
            feature_names=numeric_cols,
        )

    raise ValueError(f"Unknown dataset name: {name}")
