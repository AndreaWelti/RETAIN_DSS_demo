# retain_dss/models/predictor.py
from typing import Dict
import numpy as np
from xgboost import XGBRegressor
from retain_dss.models.trainer import get_feature_cols


def predict(
    material_inputs: Dict[str, float],
    process_params: Dict[str, float],
    models: Dict[str, XGBRegressor],
    route_name: str,
) -> Dict[str, float]:
    feature_cols = get_feature_cols(route_name)
    row = {**material_inputs, **process_params}
    X = np.array([[row[col] for col in feature_cols]])
    return {target: float(model.predict(X)[0]) for target, model in models.items()}
