# retain_dss/models/evaluator.py
from typing import Dict
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from retain_dss.models.trainer import get_feature_cols, _ROUTE_CONFIG


def evaluate_route(
    df: pd.DataFrame,
    models: Dict[str, XGBRegressor],
    route_name: str,
) -> pd.DataFrame:
    feature_cols = get_feature_cols(route_name)
    X = df[feature_cols].values
    rows = []
    for target, model in models.items():
        y = df[target].values
        y_pred = model.predict(X)
        rows.append({
            "target": target,
            "r2":   r2_score(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        })
    return pd.DataFrame(rows)


def feature_importance(
    models: Dict[str, XGBRegressor],
    route_name: str,
) -> pd.DataFrame:
    feature_cols = get_feature_cols(route_name)
    rows = []
    for target, model in models.items():
        for col, val in zip(feature_cols, model.feature_importances_):
            rows.append({"target": target, "feature": col, "importance": float(val)})
    return pd.DataFrame(rows)
