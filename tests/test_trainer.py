# tests/test_trainer.py
from pathlib import Path
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route, save_models, load_models
from retain_dss.data.schema import OUTPUTS_R1

def test_train_route1_returns_all_models(tmp_path):
    df = generate_route1(n=200, seed=0)
    models = train_route(df, "route1")
    assert set(models.keys()) == set(OUTPUTS_R1)

def test_save_and_load_models(tmp_path):
    df = generate_route1(n=100, seed=0)
    models = train_route(df, "route1")
    save_models(models, "route1", models_dir=tmp_path)
    loaded = load_models("route1", models_dir=tmp_path)
    assert set(loaded.keys()) == set(OUTPUTS_R1)

def test_model_predict_shape():
    df = generate_route1(n=100, seed=0)
    models = train_route(df, "route1")
    feature_cols = list(df.columns[:14])
    X = df[feature_cols].values[:5]
    for target, model in models.items():
        pred = model.predict(X)
        assert pred.shape == (5,), f"{target}: wrong shape"
