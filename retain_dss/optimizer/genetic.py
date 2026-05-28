# retain_dss/optimizer/genetic.py
from typing import Dict
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from retain_dss.data.schema import (
    EconomicParams, PROCESS_PARAMS_R1, PROCESS_PARAMS_R2, PROCESS_PARAMS_R3,
)
from retain_dss.models.predictor import predict
from retain_dss.optimizer.objectives import build_objectives

_ROUTE_PARAMS = {
    "route1": PROCESS_PARAMS_R1,
    "route2": PROCESS_PARAMS_R2,
    "route3": PROCESS_PARAMS_R3,
}


class RecyclingProblem(Problem):
    def __init__(
        self,
        material_inputs: Dict[str, float],
        models: dict,
        route_name: str,
        prices: EconomicParams,
    ):
        self.material_inputs = material_inputs
        self.models = models
        self.route_name = route_name
        self.prices = prices
        self._param_keys = list(_ROUTE_PARAMS[route_name].keys())
        xl = np.array([_ROUTE_PARAMS[route_name][k]["range"][0] for k in self._param_keys])
        xu = np.array([_ROUTE_PARAMS[route_name][k]["range"][1] for k in self._param_keys])
        super().__init__(n_var=len(self._param_keys), n_obj=5, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            process_params = dict(zip(self._param_keys, x))
            preds = predict(self.material_inputs, process_params, self.models, self.route_name)
            obj = build_objectives(preds, self.route_name, self.prices)
            F.append([
                -obj.pvc_purity,
                -obj.mass_yield,
                -obj.material_quality,
                obj.total_energy,
                -obj.net_margin,
            ])
        out["F"] = np.array(F, dtype=float)


def run_nsga2(
    material_inputs: Dict[str, float],
    models: dict,
    route_name: str,
    prices: EconomicParams,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
):
    problem = RecyclingProblem(material_inputs, models, route_name, prices)
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
    )
    return minimize(
        problem, algorithm,
        get_termination("n_gen", n_gen),
        seed=seed, verbose=False,
    )
