# tests/test_predictor.py
from retain_dss.data.schema import MATERIAL_INPUTS, PROCESS_PARAMS_R1
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route
from retain_dss.models.predictor import predict

def test_predict_route1_keys():
    df = generate_route1(100, 0)
    models = train_route(df, "route1")
    mat = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in MATERIAL_INPUTS.items()}
    proc = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in PROCESS_PARAMS_R1.items()}
    result = predict(mat, proc, models, "route1")
    assert "pvc_purity" in result
    assert "mass_yield" in result

def test_predict_route1_values_in_range():
    df = generate_route1(200, 0)
    models = train_route(df, "route1")
    mat = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in MATERIAL_INPUTS.items()}
    proc = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in PROCESS_PARAMS_R1.items()}
    result = predict(mat, proc, models, "route1")
    assert 50 <= result["pvc_purity"] <= 100
    assert 0  <= result["mass_yield"]  <= 100
