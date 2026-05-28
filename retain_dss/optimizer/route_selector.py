# retain_dss/optimizer/route_selector.py
from typing import Dict
import numpy as np
from pymoo.indicators.hv import HV


def select_best_route(results: Dict[str, object]) -> Dict:
    """Compare 3 Pareto fronts using hypervolume indicator, return best route."""
    all_F = np.vstack([r.F for r in results.values()])
    f_min = all_F.min(axis=0)
    f_max = all_F.max(axis=0)
    scale = f_max - f_min + 1e-9

    ref_point = np.ones(5) * 1.1
    hv_calc = HV(ref_point=ref_point)

    hvs = {}
    for name, res in results.items():
        F_norm = (res.F - f_min) / scale
        hvs[name] = float(hv_calc(F_norm))

    best_route = max(hvs, key=hvs.get)
    best_res = results[best_route]
    best_idx = int(np.argmin(best_res.F[:, 0]))  # best pvc_purity (negated, so argmin)

    return {
        "recommended_route": best_route,
        "hypervolumes": hvs,
        "best_params": best_res.X[best_idx],
        "best_objectives_raw": best_res.F[best_idx],
        "pareto_X": best_res.X,
        "pareto_F": best_res.F,
        "all_results": results,
    }
