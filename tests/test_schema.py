# tests/test_schema.py
from retain_dss.data.schema import (
    MATERIAL_INPUTS, PROCESS_PARAMS_R1, PROCESS_PARAMS_R2, PROCESS_PARAMS_R3,
    OUTPUTS_R1, OUTPUTS_R2, OUTPUTS_R3, EconomicParams
)

def test_material_inputs_keys():
    expected = {"pvc_content", "pet_content", "additive_content", "contamination_level",
                "particle_size_d50", "moisture_content", "material_age", "tensile_strength_input"}
    assert set(MATERIAL_INPUTS.keys()) == expected

def test_all_ranges_valid():
    for name, spec in {**MATERIAL_INPUTS, **PROCESS_PARAMS_R1,
                       **PROCESS_PARAMS_R2, **PROCESS_PARAMS_R3}.items():
        lo, hi = spec["range"]
        assert lo < hi, f"{name}: lo={lo} >= hi={hi}"

def test_outputs_r1_count():
    assert len(OUTPUTS_R1) == 5

def test_outputs_r2_count():
    assert len(OUTPUTS_R2) == 7

def test_outputs_r3_count():
    assert len(OUTPUTS_R3) == 9

def test_economic_params_defaults():
    p = EconomicParams()
    assert p.electricity_price == 0.12
    assert p.price_pvc_recycled == 400.0
