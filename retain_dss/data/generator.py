# retain_dss/data/generator.py
import numpy as np
import pandas as pd
from retain_dss.data.schema import MATERIAL_INPUTS


def _sample_inputs(n: int, rng: np.random.Generator) -> pd.DataFrame:
    data = {}
    for name, spec in MATERIAL_INPUTS.items():
        lo, hi = spec["range"]
        data[name] = rng.uniform(lo, hi, n)
    return pd.DataFrame(data)


def generate_route1(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic dataset for Route 1 — Mechanical recycling (T3.1)."""
    rng = np.random.default_rng(seed)
    inp = _sample_inputs(n, rng)

    shredder_speed    = rng.uniform(200,  1200, n)
    mill_gap          = rng.uniform(0.1,     5, n)
    sieve_mesh_size   = rng.uniform(0.5,    10, n)
    separation_cycles = rng.uniform(1,       5, n)
    process_temp      = rng.uniform(20,     80, n)
    throughput_rate   = rng.uniform(10,    500, n)

    pvc_purity = np.clip(
        55 + 25 * (separation_cycles / 5) + 10 * (sieve_mesh_size / 10) ** 0.5
        - 20 * (inp["contamination_level"] / 15)
        - 8  * (inp["moisture_content"] / 8)
        + rng.normal(0, 2, n),
        60, 95,
    )
    pet_purity = np.clip(
        50 + 20 * (separation_cycles / 5) + 15 * (1 - mill_gap / 5)
        - 15 * (inp["contamination_level"] / 15)
        + rng.normal(0, 2, n),
        55, 92,
    )
    mass_yield = np.clip(
        80 - 10 * (separation_cycles / 5) - 5 * (inp["moisture_content"] / 8)
        + rng.normal(0, 3, n),
        40, 90,
    )
    elec_consumption = np.clip(
        100 + 300 * (shredder_speed / 1200) ** 1.5 + 80 * separation_cycles
        - 40 * (throughput_rate / 500)
        + rng.normal(0, 20, n),
        50, 800,
    )
    particle_size_out_d50 = np.clip(
        mill_gap * 0.55 + sieve_mesh_size * 0.25 + rng.normal(0, 0.1, n),
        0.1, 5,
    )

    df = inp.copy()
    df["shredder_speed"]        = shredder_speed
    df["mill_gap"]              = mill_gap
    df["sieve_mesh_size"]       = sieve_mesh_size
    df["separation_cycles"]     = separation_cycles
    df["process_temp"]          = process_temp
    df["throughput_rate"]       = throughput_rate
    df["pvc_purity"]            = pvc_purity
    df["pet_purity"]            = pet_purity
    df["mass_yield"]            = mass_yield
    df["elec_consumption"]      = elec_consumption
    df["thermal_consumption"]   = np.zeros(n)
    df["particle_size_out_d50"] = particle_size_out_d50
    return df


def generate_route2(n: int = 500, seed: int = 43) -> pd.DataFrame:
    """Generate synthetic dataset for Route 2 — Solvent/Texyloop recycling (T3.2)."""
    rng = np.random.default_rng(seed)
    inp = _sample_inputs(n, rng)

    solvent_concentration = rng.uniform(10,  80, n)
    dissolution_temp      = rng.uniform(60, 160, n)
    dissolution_time      = rng.uniform(15, 240, n)
    solid_liquid_ratio    = rng.uniform(20, 200, n)
    precipitation_temp    = rng.uniform(0,   40, n)
    washing_cycles        = rng.uniform(1,    5, n)

    pvc_purity = np.clip(
        55 + 35 * ((dissolution_temp - 60) / 100)
        + 12 * (washing_cycles / 5)
        - 22 * (inp["contamination_level"] / 15)
        - 6  * (inp["moisture_content"] / 8)
        + 8  * (solvent_concentration / 80) ** 0.7
        + rng.normal(0, 1.5, n),
        85, 99.5,
    )
    pet_recovery = np.clip(
        60 + 30 * ((dissolution_temp - 60) / 100)
        + 15 * (dissolution_time / 240) ** 0.5
        - 10 * (solid_liquid_ratio / 200)
        + rng.normal(0, 3, n),
        70, 98,
    )
    mass_yield = np.clip(
        65 + 20 * (solvent_concentration / 80) + 10 * (washing_cycles / 5)
        - 8 * (inp["contamination_level"] / 15)
        + rng.normal(0, 3, n),
        55, 95,
    )
    elec_consumption = np.clip(
        50 + 30 * (washing_cycles / 5) + rng.normal(0, 5, n),
        30, 150,
    )
    thermal_consumption = np.clip(
        200 + 600 * ((dissolution_temp - 60) / 100)
        + 2 * dissolution_time
        - 150 * (precipitation_temp / 40)
        + rng.normal(0, 30, n),
        200, 1500,
    )
    tensile_strength_output = np.clip(
        inp["tensile_strength_input"] * (pvc_purity / 100) ** 1.2
        * (0.8 + 0.2 * (washing_cycles / 5))
        - 2 * inp["material_age"]
        + rng.normal(0, 10, n),
        30, 250,
    )
    solvent_consumed = np.clip(
        solvent_concentration * (1 - 0.7 * (washing_cycles / 5)) * 0.3
        + rng.normal(0, 2, n),
        2, 40,
    )

    df = inp.copy()
    df["solvent_concentration"]   = solvent_concentration
    df["dissolution_temp"]        = dissolution_temp
    df["dissolution_time"]        = dissolution_time
    df["solid_liquid_ratio"]      = solid_liquid_ratio
    df["precipitation_temp"]      = precipitation_temp
    df["washing_cycles"]          = washing_cycles
    df["pvc_purity"]              = pvc_purity
    df["pet_recovery"]            = pet_recovery
    df["mass_yield"]              = mass_yield
    df["elec_consumption"]        = elec_consumption
    df["thermal_consumption"]     = thermal_consumption
    df["tensile_strength_output"] = tensile_strength_output
    df["solvent_consumed"]        = solvent_consumed
    return df


def generate_route3(n: int = 500, seed: int = 44) -> pd.DataFrame:
    """Generate synthetic dataset for Route 3 — Selective additive extraction (T3.3)."""
    rng = np.random.default_rng(seed)
    inp = _sample_inputs(n, rng)

    extractant_type  = rng.integers(0, 3, n).astype(float)
    extractant_conc  = rng.uniform(5,   50, n)
    extraction_temp  = rng.uniform(40, 120, n)
    extraction_time  = rng.uniform(10, 180, n)
    ph_level         = rng.uniform(2,   12, n)
    agitation_speed  = rng.uniform(100, 800, n)
    additive_wash    = rng.uniform(1,    3, n)

    type_factor = np.where(extractant_type == 0, 1.0,
                  np.where(extractant_type == 1, 1.15, 0.85))

    ph_eff_plasticizer = np.exp(-0.1  * (ph_level - 7) ** 2)
    ph_eff_stabilizer  = np.exp(-0.08 * (ph_level - 9) ** 2)

    plasticizer_recovery = np.clip(
        30 + 55 * (extractant_conc / 50) ** 0.7 * type_factor * ph_eff_plasticizer
        + 10 * (extraction_temp - 40) / 80
        + 8  * (extraction_time / 180) ** 0.5
        + rng.normal(0, 3, n),
        30, 95,
    )
    stabilizer_recovery = np.clip(
        20 + 50 * (extractant_conc / 50) ** 0.6 * type_factor * ph_eff_stabilizer
        + 15 * (extraction_temp - 40) / 80
        + rng.normal(0, 3, n),
        20, 90,
    )
    pvc_purity = np.clip(
        72 + 18 * (plasticizer_recovery / 95) + 8 * (stabilizer_recovery / 90)
        - 15 * (inp["contamination_level"] / 15)
        + rng.normal(0, 2, n),
        70, 98,
    )
    mass_yield = np.clip(
        78 - 12 * (extractant_conc / 50) - 5 * (extraction_time / 180)
        + rng.normal(0, 3, n),
        50, 92,
    )
    elec_consumption = np.clip(
        50 + 150 * (agitation_speed / 800) ** 1.3 + 0.5 * extraction_time
        + rng.normal(0, 8, n),
        50, 400,
    )
    thermal_consumption = np.clip(
        80 + 400 * ((extraction_temp - 40) / 80) + 2.5 * extraction_time
        + rng.normal(0, 20, n),
        80, 900,
    )
    tensile_strength_output = np.clip(
        inp["tensile_strength_input"] * (pvc_purity / 100) ** 1.1
        - 1.5 * inp["material_age"]
        + rng.normal(0, 8, n),
        40, 220,
    )
    additive_purity = np.clip(
        50 + 35 * ph_eff_plasticizer * type_factor
        + 10 * (additive_wash / 3)
        + rng.normal(0, 4, n),
        50, 99,
    )
    extractant_consumed = np.clip(
        extractant_conc * 0.1 * (1 - 0.5 * (agitation_speed / 800))
        + rng.normal(0, 1, n),
        0.5, 15,
    )

    df = inp.copy()
    df["extractant_type"]         = extractant_type
    df["extractant_conc"]         = extractant_conc
    df["extraction_temp"]         = extraction_temp
    df["extraction_time"]         = extraction_time
    df["ph_level"]                = ph_level
    df["agitation_speed"]         = agitation_speed
    df["pvc_purity"]              = pvc_purity
    df["mass_yield"]              = mass_yield
    df["elec_consumption"]        = elec_consumption
    df["thermal_consumption"]     = thermal_consumption
    df["tensile_strength_output"] = tensile_strength_output
    df["plasticizer_recovery"]    = plasticizer_recovery
    df["stabilizer_recovery"]     = stabilizer_recovery
    df["additive_purity"]         = additive_purity
    df["extractant_consumed"]     = extractant_consumed
    return df
