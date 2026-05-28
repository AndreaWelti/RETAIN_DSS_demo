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
    EconomicParams, MATERIAL_INPUTS,
    PROCESS_PARAMS_R1, PROCESS_PARAMS_R2, PROCESS_PARAMS_R3,
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
        n = len(X)
        # Build full feature matrix [n, n_mat_inputs + n_process_params] in one shot
        mat_vals = np.tile(
            [self.material_inputs[k] for k in MATERIAL_INPUTS],
            (n, 1),
        )
        X_full = np.hstack([mat_vals, X])  # same column order as training

        # Batch-predict all targets at once (XGBoost is vectorised internally)
        preds = {target: model.predict(X_full) for target, model in self.models.items()}

        zeros = np.zeros(n)
        elec       = preds.get("elec_consumption",     zeros)
        thermal    = preds.get("thermal_consumption",  zeros)
        solvent    = preds.get("solvent_consumed",      zeros)
        extractant = preds.get("extractant_consumed",  zeros)
        pvc_purity = preds.get("pvc_purity",           zeros)
        mass_yield = preds.get("mass_yield",           zeros)
        pet_key    = "pet_recovery" if "pet_recovery" in preds else "pet_purity"
        pet_mass   = preds.get(pet_key,                zeros) / 100 * mass_yield / 100
        plast_mass = preds.get("plasticizer_recovery", zeros) / 100 * mass_yield / 100
        stab_mass  = preds.get("stabilizer_recovery",  zeros) / 100 * mass_yield / 100

        opex    = (elec * self.prices.electricity_price
                   + thermal    * self.prices.thermal_energy_price
                   + solvent    * self.prices.solvent_price
                   + extractant * self.prices.extractant_price)
        revenue = (pvc_purity / 100 * mass_yield / 100 * self.prices.price_pvc_recycled
                   + pet_mass   * self.prices.price_pet_recycled
                   + plast_mass * self.prices.price_plasticizer_recovered
                   + stab_mass  * self.prices.price_stabilizer_recovered)

        if self.route_name == "route1":
            quality = 1.0 / (preds.get("particle_size_out_d50", np.ones(n)) + 0.01)
        else:
            quality = preds.get("tensile_strength_output", zeros)

        out["F"] = np.column_stack([
            -pvc_purity,
            -mass_yield,
            -quality,
            elec + thermal,
            -(revenue - opex),
        ])


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
