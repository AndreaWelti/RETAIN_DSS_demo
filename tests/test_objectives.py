# tests/test_objectives.py
import pytest
from retain_dss.data.schema import EconomicParams
from retain_dss.optimizer.objectives import compute_economic_kpis, build_objectives

DEFAULT_PRICES = EconomicParams()

def test_net_margin_positive_for_clean_process():
    preds = {
        "pvc_purity": 95.0, "mass_yield": 90.0,
        "elec_consumption": 100.0, "thermal_consumption": 0.0,
        "pet_purity": 80.0,
    }
    eco = compute_economic_kpis(preds, "route1", DEFAULT_PRICES)
    assert eco["net_margin"] > 0

def test_net_margin_lower_for_high_energy():
    base = {"pvc_purity": 90.0, "mass_yield": 85.0,
            "elec_consumption": 100.0, "thermal_consumption": 0.0, "pet_purity": 75.0}
    high = {**base, "elec_consumption": 800.0}
    eco_base = compute_economic_kpis(base, "route1", DEFAULT_PRICES)
    eco_high = compute_economic_kpis(high, "route1", DEFAULT_PRICES)
    assert eco_base["net_margin"] > eco_high["net_margin"]

def test_build_objectives_route1_quality_is_inverse_particle_size():
    preds = {
        "pvc_purity": 80.0, "mass_yield": 75.0,
        "elec_consumption": 200.0, "thermal_consumption": 0.0,
        "pet_purity": 70.0, "particle_size_out_d50": 0.5,
    }
    obj = build_objectives(preds, "route1", DEFAULT_PRICES)
    assert obj.material_quality == pytest.approx(1.0 / (0.5 + 0.01), rel=1e-3)

def test_build_objectives_route2_quality_is_tensile():
    preds = {
        "pvc_purity": 94.0, "mass_yield": 87.0,
        "elec_consumption": 80.0, "thermal_consumption": 600.0,
        "tensile_strength_output": 185.0, "pet_recovery": 88.0, "solvent_consumed": 10.0,
    }
    obj = build_objectives(preds, "route2", DEFAULT_PRICES)
    assert obj.material_quality == pytest.approx(185.0)
