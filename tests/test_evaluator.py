# tests/test_evaluator.py
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route
from retain_dss.models.evaluator import evaluate_route, feature_importance

def test_evaluate_r2_above_threshold():
    df = generate_route1(500, 42)
    models = train_route(df, "route1")
    metrics = evaluate_route(df, models, "route1")
    assert (metrics["r2"] > 0.80).all(), metrics[metrics["r2"] <= 0.80]

def test_feature_importance_returns_dataframe():
    df = generate_route1(200, 0)
    models = train_route(df, "route1")
    imp = feature_importance(models, "route1")
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) > 0
