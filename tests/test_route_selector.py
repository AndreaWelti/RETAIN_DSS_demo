# tests/test_route_selector.py
from retain_dss.data.schema import EconomicParams, MATERIAL_INPUTS
from retain_dss.data.generator import generate_route1, generate_route2, generate_route3
from retain_dss.models.trainer import train_route
from retain_dss.optimizer.genetic import run_nsga2
from retain_dss.optimizer.route_selector import select_best_route

PRICES = EconomicParams()
MAT = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in MATERIAL_INPUTS.items()}

def _get_all_results():
    results = {}
    for route, gen, seed in [("route1", generate_route1, 42),
                              ("route2", generate_route2, 43),
                              ("route3", generate_route3, 44)]:
        df = gen(300, seed)
        models = train_route(df, route)
        results[route] = run_nsga2(MAT, models, route, PRICES, pop_size=20, n_gen=10, seed=0)
    return results

def test_select_returns_valid_route():
    results = _get_all_results()
    out = select_best_route(results)
    assert out["recommended_route"] in {"route1", "route2", "route3"}

def test_select_returns_hypervolumes_for_all_routes():
    results = _get_all_results()
    out = select_best_route(results)
    assert set(out["hypervolumes"].keys()) == {"route1", "route2", "route3"}

def test_select_best_params_shape():
    results = _get_all_results()
    out = select_best_route(results)
    assert len(out["best_params"]) == 6
