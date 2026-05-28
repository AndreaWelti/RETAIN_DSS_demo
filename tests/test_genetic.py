# tests/test_genetic.py
import numpy as np
from retain_dss.data.schema import EconomicParams, MATERIAL_INPUTS
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route
from retain_dss.optimizer.genetic import run_nsga2

PRICES = EconomicParams()
MAT_INPUTS = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in MATERIAL_INPUTS.items()}

def test_run_nsga2_route1_returns_result():
    df = generate_route1(300, 42)
    models = train_route(df, "route1")
    res = run_nsga2(MAT_INPUTS, models, "route1", PRICES, pop_size=20, n_gen=10, seed=0)
    assert res.X is not None
    assert res.F is not None

def test_pareto_front_shape():
    df = generate_route1(300, 42)
    models = train_route(df, "route1")
    res = run_nsga2(MAT_INPUTS, models, "route1", PRICES, pop_size=20, n_gen=10, seed=0)
    assert res.X.shape[1] == 6   # 6 process parameters for route1
    assert res.F.shape[1] == 5   # 5 objectives

def test_pareto_front_non_dominated():
    df = generate_route1(300, 42)
    models = train_route(df, "route1")
    res = run_nsga2(MAT_INPUTS, models, "route1", PRICES, pop_size=20, n_gen=5, seed=0)
    F = res.F
    for i in range(len(F)):
        for j in range(len(F)):
            if i == j:
                continue
            dominated = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
            assert not dominated, f"Solution {i} dominated by {j}"
