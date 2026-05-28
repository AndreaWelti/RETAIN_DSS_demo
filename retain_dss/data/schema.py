# retain_dss/data/schema.py
from dataclasses import dataclass
from typing import Dict, List

MATERIAL_INPUTS: Dict[str, dict] = {
    "pvc_content":            {"unit": "%wt",   "range": (40.0, 70.0)},
    "pet_content":            {"unit": "%wt",   "range": (20.0, 45.0)},
    "additive_content":       {"unit": "%wt",   "range": (5.0,  20.0)},
    "contamination_level":    {"unit": "%wt",   "range": (0.0,  15.0)},
    "particle_size_d50":      {"unit": "mm",    "range": (0.5,  50.0)},
    "moisture_content":       {"unit": "%wt",   "range": (0.0,   8.0)},
    "material_age":           {"unit": "years", "range": (1.0,  20.0)},
    "tensile_strength_input": {"unit": "MPa",   "range": (50.0, 300.0)},
}

PROCESS_PARAMS_R1: Dict[str, dict] = {
    "shredder_speed":    {"unit": "rpm",  "range": (200.0, 1200.0)},
    "mill_gap":          {"unit": "mm",   "range": (0.1,      5.0)},
    "sieve_mesh_size":   {"unit": "mm",   "range": (0.5,     10.0)},
    "separation_cycles": {"unit": "n",    "range": (1.0,      5.0)},
    "process_temp":      {"unit": "°C",   "range": (20.0,    80.0)},
    "throughput_rate":   {"unit": "kg/h", "range": (10.0,   500.0)},
}

PROCESS_PARAMS_R2: Dict[str, dict] = {
    "solvent_concentration": {"unit": "%vol", "range": (10.0,  80.0)},
    "dissolution_temp":      {"unit": "°C",   "range": (60.0, 160.0)},
    "dissolution_time":      {"unit": "min",  "range": (15.0, 240.0)},
    "solid_liquid_ratio":    {"unit": "g/L",  "range": (20.0, 200.0)},
    "precipitation_temp":    {"unit": "°C",   "range": (0.0,   40.0)},
    "washing_cycles":        {"unit": "n",    "range": (1.0,    5.0)},
}

PROCESS_PARAMS_R3: Dict[str, dict] = {
    "extractant_type":  {"unit": "cat",  "range": (0.0,   2.0)},
    "extractant_conc":  {"unit": "%vol", "range": (5.0,  50.0)},
    "extraction_temp":  {"unit": "°C",   "range": (40.0, 120.0)},
    "extraction_time":  {"unit": "min",  "range": (10.0, 180.0)},
    "ph_level":         {"unit": "—",    "range": (2.0,   12.0)},
    "agitation_speed":  {"unit": "rpm",  "range": (100.0, 800.0)},
}

OUTPUTS_R1: List[str] = [
    "pvc_purity", "pet_purity", "mass_yield",
    "elec_consumption", "particle_size_out_d50",
]
OUTPUTS_R2: List[str] = [
    "pvc_purity", "pet_recovery", "mass_yield",
    "elec_consumption", "thermal_consumption",
    "tensile_strength_output", "solvent_consumed",
]
OUTPUTS_R3: List[str] = [
    "pvc_purity", "mass_yield", "elec_consumption", "thermal_consumption",
    "tensile_strength_output", "plasticizer_recovery", "stabilizer_recovery",
    "additive_purity", "extractant_consumed",
]

@dataclass
class EconomicParams:
    electricity_price: float = 0.12
    thermal_energy_price: float = 0.06
    solvent_price: float = 1.50
    extractant_price: float = 2.00
    price_pvc_recycled: float = 400.0
    price_pet_recycled: float = 350.0
    price_plasticizer_recovered: float = 800.0
    price_stabilizer_recovered: float = 1200.0
