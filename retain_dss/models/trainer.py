# retain_dss/models/trainer.py
from pathlib import Path
from typing import Dict, List
import joblib
import pandas as pd
from xgboost import XGBRegressor
from retain_dss.data.schema import (
    MATERIAL_INPUTS, PROCESS_PARAMS_R1, PROCESS_PARAMS_R2, PROCESS_PARAMS_R3,
    OUTPUTS_R1, OUTPUTS_R2, OUTPUTS_R3,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"

_ROUTE_CONFIG = {
    "route1": {"params": PROCESS_PARAMS_R1, "outputs": OUTPUTS_R1},
    "route2": {"params": PROCESS_PARAMS_R2, "outputs": OUTPUTS_R2},
    "route3": {"params": PROCESS_PARAMS_R3, "outputs": OUTPUTS_R3},
}


def get_feature_cols(route_name: str) -> List[str]:
    return list(MATERIAL_INPUTS.keys()) + list(_ROUTE_CONFIG[route_name]["params"].keys())


def train_route(df: pd.DataFrame, route_name: str) -> Dict[str, XGBRegressor]:
    feature_cols = get_feature_cols(route_name)
    models = {}
    for target in _ROUTE_CONFIG[route_name]["outputs"]:
        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(df[feature_cols].values, df[target].values)
        models[target] = model
    return models


def save_models(
    models: Dict[str, XGBRegressor],
    route_name: str,
    models_dir: Path = MODELS_DIR,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for target, model in models.items():
        joblib.dump(model, models_dir / f"{route_name}_{target}.pkl")


def load_models(
    route_name: str,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, XGBRegressor]:
    return {
        target: joblib.load(models_dir / f"{route_name}_{target}.pkl")
        for target in _ROUTE_CONFIG[route_name]["outputs"]
    }
