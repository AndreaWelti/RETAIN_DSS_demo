# RETAIN DSS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python decision support tool for PVC/PET tarpaulin recycling: synthetic database, ML surrogate models, NSGA-II multi-objective optimizer, Streamlit UI with Ashby charts, and 4 Jupyter notebooks.

**Architecture:** `retain_dss` Python package with 4 layers (data → models → optimizer → ui). Streamlit app and Jupyter notebooks both import the same package. Synthetic data generation encodes physically plausible correlations; real data can replace CSV files without changing any model or optimizer code.

**Tech Stack:** Python 3.10+, XGBoost, pymoo (NSGA-II), Streamlit, Plotly, pandas, numpy, scikit-learn, joblib, pytest

---

## File Map

```
retain_dss/
  __init__.py
  data/
    __init__.py
    schema.py            ← variable defs, ranges, EconomicParams dataclass
    generator.py         ← synthetic CSV generation for 3 routes
    loader.py            ← CSV read/write with fixed paths
  models/
    __init__.py
    trainer.py           ← XGBoost training, save/load .pkl
    predictor.py         ← single-sample inference wrapper
    evaluator.py         ← R², RMSE, feature importance
  optimizer/
    __init__.py
    objectives.py        ← RouteObjectives dataclass + economic KPI formulas
    genetic.py           ← pymoo NSGA-II Problem subclass + run_nsga2()
    route_selector.py    ← hypervolume comparison of 3 Pareto fronts
  ui/
    __init__.py
    app.py               ← Streamlit 4-page application
data/synthetic/
  route1_mechanical.csv
  route2_solvent.csv
  route3_extraction.csv
models/saved/
  *.pkl
notebooks/
  01_data_generation.ipynb
  02_model_training.ipynb
  03_optimization.ipynb
  04_sensitivity_analysis.ipynb
tests/
  test_schema.py
  test_generator.py
  test_loader.py
  test_trainer.py
  test_predictor.py
  test_evaluator.py
  test_objectives.py
  test_genetic.py
  test_route_selector.py
requirements.txt
pyproject.toml
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `retain_dss/__init__.py`
- Create: `retain_dss/data/__init__.py`
- Create: `retain_dss/models/__init__.py`
- Create: `retain_dss/optimizer/__init__.py`
- Create: `retain_dss/ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/synthetic/.gitkeep`
- Create: `models/saved/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
xgboost>=2.0
pymoo>=0.6
streamlit>=1.32
plotly>=5.20
joblib>=1.3
pytest>=8.0
nbformat>=5.9
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "retain_dss"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["retain_dss*"]
```

- [ ] **Step 3: Create all __init__.py files and directory placeholders**

```bash
# Run from project root
mkdir -p retain_dss/data retain_dss/models retain_dss/optimizer retain_dss/ui
mkdir -p tests data/synthetic models/saved notebooks
touch retain_dss/__init__.py retain_dss/data/__init__.py
touch retain_dss/models/__init__.py retain_dss/optimizer/__init__.py
touch retain_dss/ui/__init__.py tests/__init__.py
touch data/synthetic/.gitkeep models/saved/.gitkeep
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e . -r requirements.txt
```

Expected: no errors, `import retain_dss` succeeds in Python.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt pyproject.toml retain_dss/ tests/ data/ models/ notebooks/
git commit -m "feat: project scaffolding for retain_dss"
```

---

## Task 2: Schema Module

**Files:**
- Create: `retain_dss/data/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: retain_dss.data.schema`

- [ ] **Step 3: Write schema.py**

```python
# retain_dss/data/schema.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    "extractant_type":  {"unit": "cat",  "range": (0.0,   2.0)},  # integer 0/1/2
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schema.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add retain_dss/data/schema.py tests/test_schema.py
git commit -m "feat: schema module with variable definitions and EconomicParams"
```

---

## Task 3: Synthetic Data Generator

**Files:**
- Create: `retain_dss/data/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generator.py
import numpy as np
import pandas as pd
from retain_dss.data.generator import generate_route1, generate_route2, generate_route3
from retain_dss.data.schema import MATERIAL_INPUTS, OUTPUTS_R1, OUTPUTS_R2, OUTPUTS_R3

def test_route1_shape():
    df = generate_route1(n=100, seed=0)
    assert len(df) == 100

def test_route1_output_columns():
    df = generate_route1(n=50, seed=0)
    for col in OUTPUTS_R1:
        assert col in df.columns, f"missing {col}"

def test_route1_no_nan():
    df = generate_route1(n=200, seed=1)
    assert not df.isnull().any().any()

def test_route1_pvc_purity_range():
    df = generate_route1(n=500, seed=2)
    assert df["pvc_purity"].between(60, 95).all()

def test_route2_pvc_purity_range():
    df = generate_route2(n=500, seed=3)
    assert df["pvc_purity"].between(85, 99.5).all()

def test_route3_ph_bell_curve():
    # plasticizer_recovery should peak near pH 7
    import pandas as pd
    df_low_ph  = generate_route3(n=200, seed=10)
    df_low_ph  = df_low_ph[df_low_ph["ph_level"].between(6, 8)]
    df_high_ph = generate_route3(n=200, seed=10)
    df_high_ph = df_high_ph[df_high_ph["ph_level"].between(10, 12)]
    assert df_low_ph["plasticizer_recovery"].mean() > df_high_ph["plasticizer_recovery"].mean()

def test_route3_output_columns():
    df = generate_route3(n=50, seed=0)
    for col in OUTPUTS_R3:
        assert col in df.columns, f"missing {col}"

def test_all_routes_reproducible():
    assert generate_route1(n=10, seed=42).equals(generate_route1(n=10, seed=42))
    assert generate_route2(n=10, seed=42).equals(generate_route2(n=10, seed=42))
    assert generate_route3(n=10, seed=42).equals(generate_route3(n=10, seed=42))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generator.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write generator.py**

```python
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
    df["shredder_speed"]       = shredder_speed
    df["mill_gap"]             = mill_gap
    df["sieve_mesh_size"]      = sieve_mesh_size
    df["separation_cycles"]    = separation_cycles
    df["process_temp"]         = process_temp
    df["throughput_rate"]      = throughput_rate
    df["pvc_purity"]           = pvc_purity
    df["pet_purity"]           = pet_purity
    df["mass_yield"]           = mass_yield
    df["elec_consumption"]     = elec_consumption
    df["thermal_consumption"]  = np.zeros(n)
    df["particle_size_out_d50"] = particle_size_out_d50
    return df


def generate_route2(n: int = 500, seed: int = 43) -> pd.DataFrame:
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
    df["solvent_concentration"]  = solvent_concentration
    df["dissolution_temp"]       = dissolution_temp
    df["dissolution_time"]       = dissolution_time
    df["solid_liquid_ratio"]     = solid_liquid_ratio
    df["precipitation_temp"]     = precipitation_temp
    df["washing_cycles"]         = washing_cycles
    df["pvc_purity"]             = pvc_purity
    df["pet_recovery"]           = pet_recovery
    df["mass_yield"]             = mass_yield
    df["elec_consumption"]       = elec_consumption
    df["thermal_consumption"]    = thermal_consumption
    df["tensile_strength_output"] = tensile_strength_output
    df["solvent_consumed"]       = solvent_consumed
    return df


def generate_route3(n: int = 500, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inp = _sample_inputs(n, rng)

    extractant_type  = rng.integers(0, 3, n).astype(float)
    extractant_conc  = rng.uniform(5,   50, n)
    extraction_temp  = rng.uniform(40, 120, n)
    extraction_time  = rng.uniform(10, 180, n)
    ph_level         = rng.uniform(2,   12, n)
    agitation_speed  = rng.uniform(100, 800, n)
    additive_wash    = rng.uniform(1,    3, n)  # internal washing cycles for additive purity

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generator.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add retain_dss/data/generator.py tests/test_generator.py
git commit -m "feat: synthetic data generators for 3 recycling routes"
```

---

## Task 4: Data Loader

**Files:**
- Create: `retain_dss/data/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
import pandas as pd
import pytest
from retain_dss.data.loader import save_route, load_route
from retain_dss.data.generator import generate_route1

def test_round_trip(tmp_path, monkeypatch):
    import retain_dss.data.loader as loader_mod
    monkeypatch.setattr(loader_mod, "DATA_DIR", tmp_path)
    df = generate_route1(n=50, seed=0)
    save_route(df, "route1_mechanical", data_dir=tmp_path)
    loaded = load_route("route1_mechanical", data_dir=tmp_path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True))

def test_save_creates_file(tmp_path):
    df = generate_route1(n=10, seed=0)
    path = save_route(df, "route1_mechanical", data_dir=tmp_path)
    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write loader.py**

```python
# retain_dss/data/loader.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic"


def save_route(df: pd.DataFrame, route_name: str, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{route_name}.csv"
    df.to_csv(path, index=False)
    return path


def load_route(route_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{route_name}.csv"
    return pd.read_csv(path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Generate and save the synthetic datasets**

```python
# run this once from project root
from retain_dss.data.generator import generate_route1, generate_route2, generate_route3
from retain_dss.data.loader import save_route

save_route(generate_route1(500, 42),  "route1_mechanical")
save_route(generate_route2(500, 43),  "route2_solvent")
save_route(generate_route3(500, 44),  "route3_extraction")
print("Datasets saved.")
```

Run: `python -c "exec(open('retain_dss/data/loader.py').read())"` — or paste snippet in a Python REPL.

```bash
python - <<'EOF'
from retain_dss.data.generator import generate_route1, generate_route2, generate_route3
from retain_dss.data.loader import save_route
save_route(generate_route1(500, 42), "route1_mechanical")
save_route(generate_route2(500, 43), "route2_solvent")
save_route(generate_route3(500, 44), "route3_extraction")
print("Done — 3 CSVs in data/synthetic/")
EOF
```

Expected: `Done — 3 CSVs in data/synthetic/`

- [ ] **Step 6: Commit**

```bash
git add retain_dss/data/loader.py tests/test_loader.py data/synthetic/
git commit -m "feat: data loader + generate synthetic CSVs"
```

---

## Task 5: Surrogate Model Trainer

**Files:**
- Create: `retain_dss/models/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trainer.py
from pathlib import Path
import pytest
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route, save_models, load_models
from retain_dss.data.schema import OUTPUTS_R1

def test_train_route1_returns_all_models(tmp_path):
    df = generate_route1(n=200, seed=0)
    models = train_route(df, "route1")
    assert set(models.keys()) == set(OUTPUTS_R1)

def test_save_and_load_models(tmp_path):
    df = generate_route1(n=100, seed=0)
    models = train_route(df, "route1")
    save_models(models, "route1", models_dir=tmp_path)
    loaded = load_models("route1", models_dir=tmp_path)
    assert set(loaded.keys()) == set(OUTPUTS_R1)

def test_model_predict_shape(tmp_path):
    import numpy as np
    df = generate_route1(n=100, seed=0)
    models = train_route(df, "route1")
    X = df[list(df.columns[:14])].values[:5]
    for target, model in models.items():
        pred = model.predict(X)
        assert pred.shape == (5,), f"{target}: wrong shape"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write trainer.py**

```python
# retain_dss/models/trainer.py
from pathlib import Path
from typing import Dict, List
import joblib
import pandas as pd
from xgboost import XGBRegressor
from retain_dss.data.schema import (
    MATERIAL_INPUTS, PROCESS_PARAMS_R1, PROCESS_PARAMS_R2, PROCESS_PARAMS_R3,
    OUTPUTS_R1, OUTPUTS_R2, OUTPUTS_R3,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"

_ROUTE_CONFIG = {
    "route1": {"params": PROCESS_PARAMS_R1, "outputs": OUTPUTS_R1},
    "route2": {"params": PROCESS_PARAMS_R2, "outputs": OUTPUTS_R2},
    "route3": {"params": PROCESS_PARAMS_R3, "outputs": OUTPUTS_R3},
}


def get_feature_cols(route_name: str) -> List[str]:
    return list(MATERIAL_INPUTS.keys()) + list(_ROUTE_CONFIG[route_name]["params"].keys())


def train_route(df: pd.DataFrame, route_name: str) -> Dict[str, XGBRegressor]:
    feature_cols = get_feature_cols(route_name)
    models = {}
    for target in _ROUTE_CONFIG[route_name]["outputs"]:
        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(df[feature_cols].values, df[target].values)
        models[target] = model
    return models


def save_models(
    models: Dict[str, XGBRegressor],
    route_name: str,
    models_dir: Path = MODELS_DIR,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for target, model in models.items():
        joblib.dump(model, models_dir / f"{route_name}_{target}.pkl")


def load_models(
    route_name: str,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, XGBRegressor]:
    return {
        target: joblib.load(models_dir / f"{route_name}_{target}.pkl")
        for target in _ROUTE_CONFIG[route_name]["outputs"]
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trainer.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Train and save all 20 models**

```bash
python - <<'EOF'
from retain_dss.data.loader import load_route
from retain_dss.models.trainer import train_route, save_models
for route in ["route1_mechanical", "route2_solvent", "route3_extraction"]:
    key = route.split("_")[0]  # route1/route2/route3
    df = load_route(route)
    models = train_route(df, key)
    save_models(models, key)
    print(f"{key}: {len(models)} models saved")
EOF
```

Expected:
```
route1: 5 models saved
route2: 7 models saved
route3: 8 models saved  (thermal_consumption=0 for r1 handled via ≈0 output)
```

- [ ] **Step 6: Commit**

```bash
git add retain_dss/models/trainer.py tests/test_trainer.py models/saved/
git commit -m "feat: XGBoost surrogate model trainer + 20 serialised models"
```

---

## Task 6: Predictor and Evaluator

**Files:**
- Create: `retain_dss/models/predictor.py`
- Create: `retain_dss/models/evaluator.py`
- Create: `tests/test_predictor.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predictor.py
from retain_dss.data.schema import MATERIAL_INPUTS, PROCESS_PARAMS_R1
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route
from retain_dss.models.predictor import predict

def test_predict_route1_keys():
    df = generate_route1(100, 0)
    models = train_route(df, "route1")
    mat = {k: 55.0 for k in MATERIAL_INPUTS}
    mat.update({"pvc_content": 55, "pet_content": 30, "additive_content": 10,
                "contamination_level": 3, "particle_size_d50": 10,
                "moisture_content": 2, "material_age": 5, "tensile_strength_input": 180})
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
```

```python
# tests/test_evaluator.py
from retain_dss.data.generator import generate_route1
from retain_dss.models.trainer import train_route
from retain_dss.models.evaluator import evaluate_route, feature_importance

def test_evaluate_r2_above_threshold():
    df = generate_route1(500, 42)
    models = train_route(df, "route1")
    metrics = evaluate_route(df, models, "route1")
    assert (metrics["r2"] > 0.80).all(), metrics[metrics["r2"] <= 0.80]

def test_feature_importance_returns_dataframe():
    df = generate_route1(200, 0)
    models = train_route(df, "route1")
    imp = feature_importance(models, "route1")
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_predictor.py tests/test_evaluator.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write predictor.py**

```python
# retain_dss/models/predictor.py
from typing import Dict
import numpy as np
from xgboost import XGBRegressor
from retain_dss.models.trainer import get_feature_cols


def predict(
    material_inputs: Dict[str, float],
    process_params: Dict[str, float],
    models: Dict[str, XGBRegressor],
    route_name: str,
) -> Dict[str, float]:
    feature_cols = get_feature_cols(route_name)
    row = {**material_inputs, **process_params}
    X = np.array([[row[col] for col in feature_cols]])
    return {target: float(model.predict(X)[0]) for target, model in models.items()}
```

- [ ] **Step 4: Write evaluator.py**

```python
# retain_dss/models/evaluator.py
from typing import Dict
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from retain_dss.models.trainer import get_feature_cols, _ROUTE_CONFIG


def evaluate_route(
    df: pd.DataFrame,
    models: Dict[str, XGBRegressor],
    route_name: str,
) -> pd.DataFrame:
    feature_cols = get_feature_cols(route_name)
    X = df[feature_cols].values
    rows = []
    for target, model in models.items():
        y = df[target].values
        y_pred = model.predict(X)
        rows.append({
            "target": target,
            "r2":   r2_score(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        })
    return pd.DataFrame(rows)


def feature_importance(
    models: Dict[str, XGBRegressor],
    route_name: str,
) -> pd.DataFrame:
    feature_cols = get_feature_cols(route_name)
    rows = []
    for target, model in models.items():
        for col, val in zip(feature_cols, model.feature_importances_):
            rows.append({"target": target, "feature": col, "importance": float(val)})
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_predictor.py tests/test_evaluator.py -v
```

Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add retain_dss/models/predictor.py retain_dss/models/evaluator.py \
        tests/test_predictor.py tests/test_evaluator.py
git commit -m "feat: surrogate model predictor and evaluator"
```

---

## Task 7: Optimizer — Objectives and Economic KPIs

**Files:**
- Create: `retain_dss/optimizer/objectives.py`
- Create: `tests/test_objectives.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_objectives.py
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
    import pytest
    preds = {
        "pvc_purity": 94.0, "mass_yield": 87.0,
        "elec_consumption": 80.0, "thermal_consumption": 600.0,
        "tensile_strength_output": 185.0, "pet_recovery": 88.0, "solvent_consumed": 10.0,
    }
    obj = build_objectives(preds, "route2", DEFAULT_PRICES)
    assert obj.material_quality == pytest.approx(185.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_objectives.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write objectives.py**

```python
# retain_dss/optimizer/objectives.py
from dataclasses import dataclass
from typing import Dict
from retain_dss.data.schema import EconomicParams


@dataclass
class RouteObjectives:
    pvc_purity: float        # %, maximise
    mass_yield: float        # %, maximise
    material_quality: float  # MPa or 1/(mm+0.01), maximise
    total_energy: float      # kWh/t, minimise
    net_margin: float        # €/t, maximise


def compute_economic_kpis(
    predictions: Dict[str, float],
    route_name: str,
    prices: EconomicParams,
) -> Dict[str, float]:
    elec       = predictions.get("elec_consumption", 0.0)
    thermal    = predictions.get("thermal_consumption", 0.0)
    solvent    = predictions.get("solvent_consumed", 0.0)
    extractant = predictions.get("extractant_consumed", 0.0)

    opex = (
        elec       * prices.electricity_price
        + thermal  * prices.thermal_energy_price
        + solvent  * prices.solvent_price
        + extractant * prices.extractant_price
    )

    pvc_mass = predictions.get("pvc_purity", 0.0) / 100 * predictions.get("mass_yield", 0.0) / 100
    pet_mass = predictions.get(
        "pet_recovery", predictions.get("pet_purity", 0.0)
    ) / 100 * predictions.get("mass_yield", 0.0) / 100
    plast_mass = predictions.get("plasticizer_recovery", 0.0) / 100 * predictions.get("mass_yield", 0.0) / 100
    stab_mass  = predictions.get("stabilizer_recovery",  0.0) / 100 * predictions.get("mass_yield", 0.0) / 100

    revenue = (
        pvc_mass   * prices.price_pvc_recycled
        + pet_mass * prices.price_pet_recycled
        + plast_mass * prices.price_plasticizer_recovered
        + stab_mass  * prices.price_stabilizer_recovered
    )
    return {"opex": opex, "revenue": revenue, "net_margin": revenue - opex}


def build_objectives(
    predictions: Dict[str, float],
    route_name: str,
    prices: EconomicParams,
) -> RouteObjectives:
    eco = compute_economic_kpis(predictions, route_name, prices)
    if route_name == "route1":
        ps = predictions.get("particle_size_out_d50", 1.0)
        quality = 1.0 / (ps + 0.01)
    else:
        quality = predictions.get("tensile_strength_output", 0.0)
    return RouteObjectives(
        pvc_purity=predictions.get("pvc_purity", 0.0),
        mass_yield=predictions.get("mass_yield", 0.0),
        material_quality=quality,
        total_energy=elec + thermal if (
            elec := predictions.get("elec_consumption", 0.0),
            thermal := predictions.get("thermal_consumption", 0.0)
        ) else 0.0,
        net_margin=eco["net_margin"],
    )
```

> **Note:** The walrus-operator trick above for `total_energy` is awkward. Use the cleaner version below:

```python
def build_objectives(
    predictions: Dict[str, float],
    route_name: str,
    prices: EconomicParams,
) -> RouteObjectives:
    eco = compute_economic_kpis(predictions, route_name, prices)
    if route_name == "route1":
        quality = 1.0 / (predictions.get("particle_size_out_d50", 1.0) + 0.01)
    else:
        quality = predictions.get("tensile_strength_output", 0.0)
    total_energy = (
        predictions.get("elec_consumption", 0.0)
        + predictions.get("thermal_consumption", 0.0)
    )
    return RouteObjectives(
        pvc_purity=predictions.get("pvc_purity", 0.0),
        mass_yield=predictions.get("mass_yield", 0.0),
        material_quality=quality,
        total_energy=total_energy,
        net_margin=eco["net_margin"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_objectives.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add retain_dss/optimizer/objectives.py tests/test_objectives.py
git commit -m "feat: optimizer objectives and economic KPI formulas"
```

---

## Task 8: NSGA-II Genetic Algorithm

**Files:**
- Create: `retain_dss/optimizer/genetic.py`
- Create: `tests/test_genetic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genetic.py
import numpy as np
from retain_dss.data.schema import EconomicParams, MATERIAL_INPUTS, PROCESS_PARAMS_R1
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
    # No solution should be dominated by another
    for i in range(len(F)):
        for j in range(len(F)):
            if i == j:
                continue
            dominated = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
            assert not dominated, f"Solution {i} is dominated by {j}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_genetic.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write genetic.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_genetic.py -v
```

Expected: 3 PASSED (may take ~30 s due to optimization runs)

- [ ] **Step 5: Commit**

```bash
git add retain_dss/optimizer/genetic.py tests/test_genetic.py
git commit -m "feat: NSGA-II optimizer for recycling process parameters"
```

---

## Task 9: Route Selector

**Files:**
- Create: `retain_dss/optimizer/route_selector.py`
- Create: `tests/test_route_selector.py`

- [ ] **Step 1: Write the failing test**

```python
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
    for route, gen in [("route1", generate_route1), ("route2", generate_route2),
                       ("route3", generate_route3)]:
        seed = {"route1": 42, "route2": 43, "route3": 44}[route]
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_route_selector.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write route_selector.py**

```python
# retain_dss/optimizer/route_selector.py
from typing import Dict
import numpy as np
from pymoo.indicators.hv import HV


def select_best_route(results: Dict[str, object]) -> Dict:
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
    best_idx = int(np.argmin(best_res.F[:, 0]))  # best pvc_purity (obj 0 negated)

    return {
        "recommended_route": best_route,
        "hypervolumes": hvs,
        "best_params": best_res.X[best_idx],
        "best_objectives_raw": best_res.F[best_idx],
        "pareto_X": best_res.X,
        "pareto_F": best_res.F,
        "all_results": results,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_route_selector.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASSED

- [ ] **Step 6: Commit**

```bash
git add retain_dss/optimizer/route_selector.py tests/test_route_selector.py
git commit -m "feat: route selector using hypervolume indicator on Pareto fronts"
```

---

## Task 10: Streamlit Application

**Files:**
- Create: `retain_dss/ui/app.py`

- [ ] **Step 1: Write app.py**

```python
# retain_dss/ui/app.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from retain_dss.data.schema import (
    MATERIAL_INPUTS, PROCESS_PARAMS_R1, PROCESS_PARAMS_R2,
    PROCESS_PARAMS_R3, EconomicParams,
)
from retain_dss.data.loader import load_route
from retain_dss.models.trainer import train_route, load_models, MODELS_DIR
from retain_dss.models.predictor import predict
from retain_dss.optimizer.genetic import run_nsga2
from retain_dss.optimizer.route_selector import select_best_route
from retain_dss.optimizer.objectives import build_objectives, compute_economic_kpis

st.set_page_config(page_title="RETAIN DSS", layout="wide")

ROUTE_LABELS = {
    "route1": "T3.1 — Meccanica",
    "route2": "T3.2 — Solvente (Texyloop)",
    "route3": "T3.3 — Estrazione additivi",
}
ROUTE_COLORS = {"route1": "#3b82f6", "route2": "#f59e0b", "route3": "#4ade80"}
ROUTE_PARAMS = {"route1": PROCESS_PARAMS_R1, "route2": PROCESS_PARAMS_R2, "route3": PROCESS_PARAMS_R3}

PAGES = ["📥 Input materiale", "💶 Prezzi & scenario", "⚙️ Ottimizzazione", "📊 Risultati"]
page = st.sidebar.radio("Navigazione", PAGES)


# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    models = {}
    for route in ["route1", "route2", "route3"]:
        try:
            models[route] = load_models(route)
        except FileNotFoundError:
            df = load_route(f"route{route[-1]}_{'mechanical' if route=='route1' else 'solvent' if route=='route2' else 'extraction'}")
            m = train_route(df, route)
            from retain_dss.models.trainer import save_models
            save_models(m, route)
            models[route] = m
    return models


def _ashby_chart(all_results, selection, x_key, y_key, size_key, log_scale):
    fig = go.Figure()
    for route_name, res in all_results.items():
        F = res.F  # negated objectives: col0=-pvc_purity, 1=-mass_yield, 2=-quality, 3=energy, 4=-margin
        obj_map = {
            "pvc_purity": -F[:, 0],
            "mass_yield": -F[:, 1],
            "material_quality": -F[:, 2],
            "total_energy": F[:, 3],
            "net_margin": -F[:, 4],
        }
        x = obj_map.get(x_key, -F[:, 0])
        y = obj_map.get(y_key, -F[:, 2])
        sz = obj_map.get(size_key, -F[:, 1])
        sz_norm = 8 + 20 * (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
        color = ROUTE_COLORS[route_name]
        label = ROUTE_LABELS[route_name]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=sz_norm, color=color, opacity=0.6, line=dict(width=0.5, color="white")),
            name=label,
        ))
        # Ellipse (convex hull approximation via parametric)
        cx, cy = x.mean(), y.mean()
        rx, ry = max(x.std() * 1.5, (x.max()-x.min())/2 + 0.5), max(y.std() * 1.5, (y.max()-y.min())/2 + 0.5)
        t = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scatter(
            x=cx + rx * np.cos(t), y=cy + ry * np.sin(t), mode="lines",
            line=dict(color=color, width=1.5, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # Star on recommended solution
    if selection:
        best_route = selection["recommended_route"]
        bp = selection["best_params"]
        best_F = selection["best_objectives_raw"]
        obj_map_best = {
            "pvc_purity": -best_F[0], "mass_yield": -best_F[1],
            "material_quality": -best_F[2], "total_energy": best_F[3], "net_margin": -best_F[4],
        }
        fig.add_trace(go.Scatter(
            x=[obj_map_best.get(x_key, -best_F[0])],
            y=[obj_map_best.get(y_key, -best_F[2])],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="#fcd34d", line=dict(width=1, color="white")),
            text=["★ " + ROUTE_LABELS[best_route].split("—")[0].strip()],
            textposition="top center",
            name="Soluzione ottimale",
        ))

    axis_kw = dict(type="log") if log_scale else {}
    fig.update_layout(
        xaxis=dict(title=x_key.replace("_", " "), **axis_kw),
        yaxis=dict(title=y_key.replace("_", " "), **axis_kw),
        template="plotly_dark", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Page 1: Input materiale ─────────────────────────────────────────────────
if page == PAGES[0]:
    st.title("📥 Proprietà materiale in ingresso")
    st.markdown("Inserisci le proprietà del telo PVC/PET da riciclare.")
    mat_inputs = {}
    cols = st.columns(2)
    for i, (name, spec) in enumerate(MATERIAL_INPUTS.items()):
        lo, hi = spec["range"]
        mid = (lo + hi) / 2
        mat_inputs[name] = cols[i % 2].slider(
            f"{name} [{spec['unit']}]", min_value=float(lo), max_value=float(hi),
            value=float(mid), step=float((hi - lo) / 100),
        )
    st.session_state["mat_inputs"] = mat_inputs
    st.success("Valori salvati. Prosegui con ➡️ Prezzi & scenario")


# ── Page 2: Prezzi & scenario ────────────────────────────────────────────────
elif page == PAGES[1]:
    st.title("💶 Prezzi & scenario economico")
    st.markdown("Modifica i prezzi di riferimento per l'analisi economica.")
    p = EconomicParams()
    c1, c2 = st.columns(2)
    prices = EconomicParams(
        electricity_price      = c1.number_input("Energia elettrica (€/kWh_el)", value=p.electricity_price,  step=0.01),
        thermal_energy_price   = c1.number_input("Energia termica (€/kWh_th)",   value=p.thermal_energy_price, step=0.01),
        solvent_price          = c1.number_input("Solvente (€/L)",               value=p.solvent_price,       step=0.1),
        extractant_price       = c1.number_input("Estrattore (€/L)",             value=p.extractant_price,    step=0.1),
        price_pvc_recycled     = c2.number_input("PVC riciclato (€/t)",          value=p.price_pvc_recycled,  step=10.0),
        price_pet_recycled     = c2.number_input("PET riciclato (€/t)",          value=p.price_pet_recycled,  step=10.0),
        price_plasticizer_recovered = c2.number_input("Plastificante recuperato (€/t)", value=p.price_plasticizer_recovered, step=50.0),
        price_stabilizer_recovered  = c2.number_input("Stabilizzante recuperato (€/t)", value=p.price_stabilizer_recovered,  step=50.0),
    )
    st.session_state["prices"] = prices
    st.success("Prezzi salvati. Prosegui con ➡️ Ottimizzazione")


# ── Page 3: Ottimizzazione ───────────────────────────────────────────────────
elif page == PAGES[2]:
    st.title("⚙️ Ottimizzazione multi-obiettivo")
    mat_inputs = st.session_state.get("mat_inputs")
    prices     = st.session_state.get("prices", EconomicParams())

    if mat_inputs is None:
        st.warning("Inserisci prima le proprietà del materiale (pagina 1).")
    else:
        pop_size = st.slider("Dimensione popolazione GA", 20, 200, 100, 10)
        n_gen    = st.slider("Numero generazioni GA",    20, 500, 200, 10)

        if st.button("🚀 Ottimizza le 3 rotte", type="primary"):
            models   = get_models()
            results  = {}
            progress = st.progress(0, text="Avvio ottimizzazione…")
            for i, route in enumerate(["route1", "route2", "route3"]):
                progress.progress((i * 33), text=f"Ottimizzando {ROUTE_LABELS[route]}…")
                results[route] = run_nsga2(mat_inputs, models[route], route, prices, pop_size, n_gen)
            progress.progress(100, text="Completato!")
            selection = select_best_route(results)
            st.session_state["results"]   = results
            st.session_state["selection"] = selection
            st.success(f"✓ Rotta raccomandata: **{ROUTE_LABELS[selection['recommended_route']]}**")


# ── Page 4: Risultati ────────────────────────────────────────────────────────
elif page == PAGES[3]:
    st.title("📊 Risultati — Ashby Chart & Pareto front")
    selection = st.session_state.get("selection")
    results   = st.session_state.get("results")

    if selection is None:
        st.warning("Esegui prima l'ottimizzazione (pagina 3).")
    else:
        best_route = selection["recommended_route"]
        st.markdown(f"### ★ Rotta raccomandata: **{ROUTE_LABELS[best_route]}**")

        # KPI cards
        cols = st.columns(3)
        for i, (route, label) in enumerate(ROUTE_LABELS.items()):
            res = results[route]
            best_idx = int(np.argmin(res.F[:, 0]))
            F = res.F[best_idx]
            border = "border: 2px solid " + ROUTE_COLORS[route]
            is_best = "⭐ " if route == best_route else ""
            cols[i].markdown(
                f"<div style='{border};padding:12px;border-radius:8px'>"
                f"<b>{is_best}{label}</b><br>"
                f"PVC purezza: <b>{-F[0]:.1f}%</b><br>"
                f"Resa: <b>{-F[1]:.1f}%</b><br>"
                f"Energia: <b>{F[3]:.0f} kWh/t</b><br>"
                f"Margine: <b>€{-F[4]:.0f}/t</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        # Ashby chart controls
        KPI_OPTS = ["pvc_purity", "mass_yield", "material_quality", "total_energy", "net_margin"]
        cc1, cc2, cc3, cc4 = st.columns(4)
        x_key    = cc1.selectbox("Asse X",   KPI_OPTS, index=0)
        y_key    = cc2.selectbox("Asse Y",   KPI_OPTS, index=2)
        size_key = cc3.selectbox("Dim. bolla", KPI_OPTS, index=1)
        log_sc   = cc4.checkbox("Scala log")

        fig = _ashby_chart(results, selection, x_key, y_key, size_key, log_sc)
        st.plotly_chart(fig, use_container_width=True)

        # Best parameters table
        st.subheader("Parametri ottimali — rotta raccomandata")
        route_params = ROUTE_PARAMS[best_route]
        param_keys   = list(route_params.keys())
        best_params  = selection["best_params"]
        param_df = pd.DataFrame({
            "Parametro": param_keys,
            "Valore ottimale": [f"{v:.2f}" for v in best_params],
            "Unità": [route_params[k]["unit"] for k in param_keys],
            "Range": [str(route_params[k]["range"]) for k in param_keys],
        })
        st.dataframe(param_df, use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Test the Streamlit app manually**

```bash
streamlit run retain_dss/ui/app.py
```

Open http://localhost:8501. Walk through all 4 pages:
1. Set material inputs → confirm values saved
2. Adjust energy prices → confirm values saved
3. Click "Ottimizza" (pop=20, gen=10 for speed) → confirm progress bar completes
4. Check Results page: KPI cards visible, Ashby chart renders, parameters table shows

Expected: no Python errors in terminal, all 4 pages render correctly

- [ ] **Step 3: Commit**

```bash
git add retain_dss/ui/app.py
git commit -m "feat: Streamlit 4-page DSS app with Ashby chart and KPI cards"
```

---

## Task 11: Jupyter Notebooks

**Files:**
- Create: `notebooks/01_data_generation.ipynb`
- Create: `notebooks/02_model_training.ipynb`
- Create: `notebooks/03_optimization.ipynb`
- Create: `notebooks/04_sensitivity_analysis.ipynb`

- [ ] **Step 1: Create notebook 01_data_generation.ipynb**

Use `nbformat` to create (or create manually in Jupyter). Content per cell:

**Cell 1 — Setup:**
```python
from retain_dss.data.generator import generate_route1, generate_route2, generate_route3
from retain_dss.data.loader import save_route
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
```

**Cell 2 — Generate datasets:**
```python
df1 = generate_route1(500, 42)
df2 = generate_route2(500, 43)
df3 = generate_route3(500, 44)
save_route(df1, "route1_mechanical")
save_route(df2, "route2_solvent")
save_route(df3, "route3_extraction")
print(f"Route 1: {df1.shape}, Route 2: {df2.shape}, Route 3: {df3.shape}")
```

**Cell 3 — Distribution of key outputs:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (df, label) in zip(axes, [(df1,"Meccanica"),(df2,"Solvente"),(df3,"Estrazione")]):
    ax.hist(df["pvc_purity"], bins=30, color="#6366f1", edgecolor="white", alpha=0.8)
    ax.set_title(f"PVC Purity — {label}")
    ax.set_xlabel("pvc_purity (%)")
plt.tight_layout()
plt.show()
```

**Cell 4 — Correlation heatmap for Route 2:**
```python
cols_of_interest = ["solvent_concentration","dissolution_temp","dissolution_time",
                    "washing_cycles","pvc_purity","mass_yield","thermal_consumption"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df2[cols_of_interest].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix — Route 2 (Solvent)")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Create notebook 02_model_training.ipynb**

**Cell 1 — Setup:**
```python
from retain_dss.data.loader import load_route
from retain_dss.models.trainer import train_route, save_models
from retain_dss.models.evaluator import evaluate_route, feature_importance
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
```

**Cell 2 — Train all routes:**
```python
route_map = {
    "route1": "route1_mechanical",
    "route2": "route2_solvent",
    "route3": "route3_extraction",
}
all_models = {}
for key, fname in route_map.items():
    df = load_route(fname)
    models = train_route(df, key)
    save_models(models, key)
    all_models[key] = models
    print(f"{key}: {len(models)} models trained")
```

**Cell 3 — Metrics table:**
```python
import pandas as pd
rows = []
for key, fname in route_map.items():
    df = load_route(fname)
    metrics = evaluate_route(df, all_models[key], key)
    metrics["route"] = key
    rows.append(metrics)
metrics_df = pd.concat(rows, ignore_index=True)
print(metrics_df.to_string(index=False))
```

**Cell 4 — Feature importance for Route 2:**
```python
imp = feature_importance(all_models["route2"], "route2")
top = imp[imp["target"] == "pvc_purity"].sort_values("importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top["feature"], top["importance"], color="#f59e0b")
ax.set_title("Feature importance — Route 2, pvc_purity")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Create notebook 03_optimization.ipynb**

**Cell 1 — Setup:**
```python
from retain_dss.data.schema import MATERIAL_INPUTS, EconomicParams
from retain_dss.models.trainer import load_models
from retain_dss.optimizer.genetic import run_nsga2
from retain_dss.optimizer.route_selector import select_best_route
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
```

**Cell 2 — Define material inputs and prices:**
```python
mat_inputs = {k: (v["range"][0] + v["range"][1]) / 2 for k, v in MATERIAL_INPUTS.items()}
mat_inputs.update({"pvc_content": 58, "contamination_level": 4, "moisture_content": 2,
                   "material_age": 7, "tensile_strength_input": 180})
prices = EconomicParams()
print("Material inputs:", mat_inputs)
```

**Cell 3 — Run NSGA-II for all routes:**
```python
models = {r: load_models(r) for r in ["route1", "route2", "route3"]}
results = {}
for route in ["route1", "route2", "route3"]:
    print(f"Optimising {route}...")
    results[route] = run_nsga2(mat_inputs, models[route], route, prices,
                               pop_size=100, n_gen=200, seed=42)
    print(f"  → {len(results[route].X)} Pareto solutions")
selection = select_best_route(results)
print(f"\nRecommended route: {selection['recommended_route']}")
print(f"Hypervolumes: {selection['hypervolumes']}")
```

**Cell 4 — Ashby chart (Plotly):**
```python
COLORS = {"route1": "#3b82f6", "route2": "#f59e0b", "route3": "#4ade80"}
fig = go.Figure()
for route_name, res in results.items():
    F = res.F
    x = -F[:, 0]   # pvc_purity
    y = -F[:, 2]   # material_quality
    sz = 8 + 20 * (-F[:, 1] - (-F[:, 1]).min()) / ((-F[:, 1]).max() - (-F[:, 1]).min() + 1e-9)
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
        marker=dict(size=sz, color=COLORS[route_name], opacity=0.7),
        name=route_name))
    cx, cy = x.mean(), y.mean()
    rx, ry = x.std()*1.5 + 0.5, y.std()*1.5 + 0.5
    t = np.linspace(0, 2*np.pi, 60)
    fig.add_trace(go.Scatter(x=cx+rx*np.cos(t), y=cy+ry*np.sin(t), mode="lines",
        line=dict(color=COLORS[route_name], dash="dash"), showlegend=False))
best_F = selection["best_objectives_raw"]
fig.add_trace(go.Scatter(x=[-best_F[0]], y=[-best_F[2]], mode="markers+text",
    marker=dict(symbol="star", size=20, color="#fcd34d"),
    text=["★ Best"], textposition="top center", name="Optimal"))
fig.update_layout(xaxis_title="PVC Purity (%)", yaxis_title="Material Quality",
                  template="plotly_dark", height=500)
fig.show()
```

- [ ] **Step 4: Create notebook 04_sensitivity_analysis.ipynb**

**Cell 1 — Setup:**
```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
from retain_dss.data.schema import MATERIAL_INPUTS, EconomicParams
from retain_dss.models.trainer import load_models
from retain_dss.optimizer.genetic import run_nsga2
from retain_dss.optimizer.route_selector import select_best_route
```

**Cell 2 — Sweep electricity price:**
```python
mat_inputs = {k: (v["range"][0]+v["range"][1])/2 for k, v in MATERIAL_INPUTS.items()}
models = {r: load_models(r) for r in ["route1","route2","route3"]}
elec_prices = np.linspace(0.05, 0.40, 8)
margins = []
recommended = []
for ep in elec_prices:
    prices = EconomicParams(electricity_price=ep)
    results = {r: run_nsga2(mat_inputs, models[r], r, prices, 50, 50, 42) for r in ["route1","route2","route3"]}
    sel = select_best_route(results)
    margins.append(-sel["best_objectives_raw"][4])
    recommended.append(sel["recommended_route"])
    print(f"  elec={ep:.2f}: best margin = {margins[-1]:.1f} €/t, route = {recommended[-1]}")
```

**Cell 3 — Plot sensitivity:**
```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=elec_prices, y=margins, mode="lines+markers",
    marker=dict(color="#6366f1", size=8), line=dict(width=2)))
fig.update_layout(xaxis_title="Electricity price (€/kWh_el)",
                  yaxis_title="Best net margin (€/t)",
                  template="plotly_dark", title="Sensitivity to electricity price")
fig.show()
```

**Cell 4 — Sweep PVC market price:**
```python
pvc_prices = np.linspace(200, 700, 8)
margins_pvc = []
for pp in pvc_prices:
    prices = EconomicParams(price_pvc_recycled=pp)
    results = {r: run_nsga2(mat_inputs, models[r], r, prices, 50, 50, 42) for r in ["route1","route2","route3"]}
    sel = select_best_route(results)
    margins_pvc.append(-sel["best_objectives_raw"][4])
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=pvc_prices, y=margins_pvc, mode="lines+markers",
    marker=dict(color="#f59e0b", size=8), line=dict(width=2)))
fig2.update_layout(xaxis_title="PVC recycled price (€/t)", yaxis_title="Best net margin (€/t)",
                   template="plotly_dark", title="Sensitivity to PVC market price")
fig2.show()
```

- [ ] **Step 5: Verify notebooks run end-to-end**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_generation.ipynb --output notebooks/01_data_generation_executed.ipynb
```

Expected: no errors, output notebook created with all cells executed.

- [ ] **Step 6: Commit**

```bash
git add notebooks/
git commit -m "feat: 4 Jupyter notebooks — data, models, optimization, sensitivity"
```

---

## Task 12: Final Integration Test

- [ ] **Step 1: Run complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASSED (16+ tests across 9 test files)

- [ ] **Step 2: End-to-end smoke test**

```bash
python - <<'EOF'
from retain_dss.data.schema import MATERIAL_INPUTS, EconomicParams
from retain_dss.models.trainer import load_models
from retain_dss.optimizer.genetic import run_nsga2
from retain_dss.optimizer.route_selector import select_best_route

mat = {k: (v["range"][0]+v["range"][1])/2 for k,v in MATERIAL_INPUTS.items()}
prices = EconomicParams()
models = {r: load_models(r) for r in ["route1","route2","route3"]}
results = {r: run_nsga2(mat, models[r], r, prices, 20, 10, 42) for r in ["route1","route2","route3"]}
sel = select_best_route(results)
print("Recommended route:", sel["recommended_route"])
print("HV scores:", {k: round(v,4) for k,v in sel["hypervolumes"].items()})
print("Best pvc_purity:", round(-sel["best_objectives_raw"][0], 1), "%")
print("Best net_margin: €", round(-sel["best_objectives_raw"][4], 1), "/t")
EOF
```

Expected output (values will vary):
```
Recommended route: route2
HV scores: {'route1': 0.31xx, 'route2': 0.41xx, 'route3': 0.38xx}
Best pvc_purity: 93.x %
Best net_margin: € 1xx.x /t
```

- [ ] **Step 3: Launch Streamlit app for final visual check**

```bash
streamlit run retain_dss/ui/app.py
```

Go through all 4 pages with non-default inputs. Confirm:
- Ashby chart renders with 3 ellipses and a star
- KPI cards show all 3 routes
- Parameters table displays 6 rows for the recommended route

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete RETAIN DSS dummy simulation — all layers integrated"
```

---

## Self-Review Checklist

| Spec requirement | Task |
|---|---|
| Synthetic database — 3 routes, ~500 records, 8 input + 6 process variables | Task 3 |
| EconomicParams configurable, electricity + thermal energy separate | Tasks 2, 7 |
| ~20 XGBoost surrogate models, one per KPI per route | Tasks 5, 6 |
| NSGA-II multi-objective, 5 objectives, pymoo | Tasks 7, 8 |
| Route selector via hypervolume | Task 9 |
| Streamlit 4-page app with sliders, prices, progress, Ashby chart | Task 10 |
| Ashby chart: ellipses, bubbles, star, configurable axes, log scale | Task 10 |
| 4 Jupyter notebooks covering all layers | Task 11 |
| Sensitivity analysis on energy/market prices | Task 11, Cell 4 |
| Real data can replace synthetic CSVs without code change | loader.py API |
