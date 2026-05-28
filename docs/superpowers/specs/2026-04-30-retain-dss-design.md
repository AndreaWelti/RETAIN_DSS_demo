# RETAIN T3.4 — Decision Support Tool: Design Specification

**Date:** 2026-04-30
**Project:** RETAIN (EU Grant 101181056) — Task T3.4
**Scope:** Dummy simulation of the Python-based recycling process optimization tool

---

## 1. Context

Task T3.4 develops a Python-based decision support tool (DSS) for recycling PVC/PET tarpaulins within WP3. Three recycling routes are in scope:

- **T3.1 — Mechanical recycling**: shredding, milling, shaving, granulometric separation
- **T3.2 — Solvent-based recycling**: Texyloop/soak-and-separate for PVC/PET separation
- **T3.3 — Selective additive extraction**: recovery of plasticizers, stabilizers, pigments

This simulation is fully synthetic (no real process data yet) but structured so that real data from T3.1–T3.3 experiments can replace the synthetic dataset without changing any downstream code.

---

## 2. Architecture

Package name: `retain_dss`
Approach: layered Python package + Streamlit UI + Jupyter notebooks

```
retain_dss/
  data/
    generator.py     # synthetic database generation
    schema.py        # variable definitions and ranges
    loader.py        # CSV read/write utilities
  models/
    trainer.py       # surrogate model training (XGBoost/LightGBM)
    predictor.py     # fast inference
    evaluator.py     # R², RMSE, feature importance
  optimizer/
    genetic.py       # NSGA-II via pymoo
    objectives.py    # 5 objective functions
    route_selector.py # compares 3 Pareto fronts, returns best route
  ui/
    app.py           # Streamlit operator-facing application
notebooks/
  01_data_generation.ipynb
  02_model_training.ipynb
  03_optimization.ipynb
  04_sensitivity_analysis.ipynb
data/synthetic/
  route1_mechanical.csv
  route2_solvent.csv
  route3_extraction.csv
models/saved/
  *.pkl              # serialised surrogate models
requirements.txt
```

**Users:**
- Researchers/data scientists: Jupyter notebooks importing `retain_dss`
- Process engineers/operators: Streamlit web app (no code required)

---

## 3. Synthetic Database Schema

~500 records per route. Each record represents one process experiment.

### 3.1 Input material properties (common to all 3 routes)

| Variable | Unit | Range |
|---|---|---|
| pvc_content | %wt | 40–70 |
| pet_content | %wt | 20–45 |
| additive_content | %wt | 5–20 |
| contamination_level | %wt | 0–15 |
| particle_size_d50 | mm | 0.5–50 |
| moisture_content | %wt | 0–8 |
| material_age | years | 1–20 |
| tensile_strength_input | MPa | 50–300 |

### 3.2 Process parameters per route

**Route 1 — Mechanical (T3.1)**

| Variable | Unit | Range |
|---|---|---|
| shredder_speed | rpm | 200–1200 |
| mill_gap | mm | 0.1–5 |
| sieve_mesh_size | mm | 0.5–10 |
| separation_cycles | n | 1–5 |
| process_temp | °C | 20–80 |
| throughput_rate | kg/h | 10–500 |

**Route 2 — Solvent/Texyloop (T3.2)**

| Variable | Unit | Range |
|---|---|---|
| solvent_concentration | %vol | 10–80 |
| dissolution_temp | °C | 60–160 |
| dissolution_time | min | 15–240 |
| solid_liquid_ratio | g/L | 20–200 |
| precipitation_temp | °C | 0–40 |
| washing_cycles | n | 1–5 |

**Route 3 — Selective extraction (T3.3)**

| Variable | Unit | Range |
|---|---|---|
| extractant_type | categorical | 3 types |
| extractant_conc | %vol | 5–50 |
| extraction_temp | °C | 40–120 |
| extraction_time | min | 10–180 |
| ph_level | — | 2–12 |
| agitation_speed | rpm | 100–800 |

### 3.3 Output properties per route

Physical outputs (predicted by surrogate models):

| Variable | Route 1 | Route 2 | Route 3 |
|---|---|---|---|
| pvc_purity (%) | ✓ | ✓ | ✓ |
| pet_purity / pet_recovery (%) | ✓ | ✓ | — |
| mass_yield (%) | ✓ | ✓ | ✓ |
| elec_consumption (kWh_el/t) | ✓ | ✓ | ✓ |
| thermal_consumption (kWh_th/t) | ≈0 | ✓ | ✓ |
| particle_size_out_d50 (mm) | ✓ | — | — |
| tensile_strength_output (MPa) | — | ✓ | ✓ |
| plasticizer_recovery (%) | — | — | ✓ |
| stabilizer_recovery (%) | — | — | ✓ |
| additive_purity (%) | — | — | ✓ |
| solvent_consumed (L/t, losses) | — | ✓ | — |
| extractant_consumed (L/t, losses) | — | — | ✓ |

> Note: Route 1 lacks tensile strength as output — `particle_size_out_d50` serves as the material quality proxy for that route. `pvc_yield` used in revenue formula = `pvc_purity × mass_yield / 100`.

Economic outputs (derived analytically — no ML model):

| Variable | Formula |
|---|---|
| opex_per_ton (€/t) | elec × p_elec + thermal × p_thermal + solvent_consumed × p_solvent |
| revenue_per_ton (€/t) | pvc_yield × p_pvc + pet_yield × p_pet + [additive_yield × p_additive] |
| net_margin_per_ton (€/t) | revenue − opex |

### 3.4 Configurable economic parameters

| Parameter | Default | Unit |
|---|---|---|
| electricity_price | 0.12 | €/kWh_el |
| thermal_energy_price | 0.06 | €/kWh_th |
| solvent_price | 1.50 | €/L |
| extractant_price | 2.00 | €/L |
| price_pvc_recycled | 400 | €/t |
| price_pet_recycled | 350 | €/t |
| price_plasticizer_recovered | 800 | €/t |
| price_stabilizer_recovered | 1200 | €/t |

All parameters are editable in the Streamlit UI and passed to the optimizer at runtime — no model retraining required.

---

## 4. ML Surrogate Models

**Algorithm:** Gradient Boosting (XGBoost default, LightGBM as alternative)
**Total models:** ~20 (one per physical output KPI × route — Route 1: 5, Route 2: 7, Route 3: 8)
**Input features:** 8 material properties + 6 process parameters = 14 features
**Training split:** 80/20, 5-fold cross-validation
**Metrics reported:** R², RMSE, feature importance plot
**Serialisation:** joblib `.pkl` files in `models/saved/`

Synthetic data generation encodes physically plausible non-linear correlations with Gaussian noise. Examples:
- PVC purity increases with dissolution temperature and washing cycles but decreases with high contamination and moisture
- Mechanical route energy consumption scales with shredder speed and number of separation cycles
- Additive recovery peaks at a specific pH window and drops outside it

---

## 5. Genetic Algorithm Optimizer (NSGA-II)

**Library:** `pymoo`

**Configuration:**

| Parameter | Value |
|---|---|
| Algorithm | NSGA-II |
| Population size | 100 |
| Generations | 200 |
| Crossover | SBX (η=15) |
| Mutation | Polynomial (η=20) |
| Decision variables | 6 process parameters per route (bounded) |
| Constraints | Physical parameter ranges |

**5 objectives (all normalised to [0,1]):**

| # | Objective | Direction |
|---|---|---|
| 1 | pvc_purity | maximise |
| 2 | mass_yield | maximise |
| 3 | material_quality (tensile_strength_output for R2/R3; particle_size_out_d50 inverted for R1) | maximise |
| 4 | total_energy (elec + thermal) | minimise |
| 5 | net_margin_per_ton | maximise |

**Execution flow:**
1. User provides 8 input material properties
2. NSGA-II runs independently on all 3 routes (parallelisable)
3. Each run produces a Pareto front over the 5 objectives
4. `route_selector` compares the 3 Pareto fronts using hypervolume indicator
5. Returns: recommended route + optimal process parameters + all 3 Pareto fronts for inspection

---

## 6. User Interfaces

### 6.1 Streamlit Application (`ui/app.py`)

Four pages (sidebar navigation):

1. **Input materiale** — sliders for 8 input material properties
2. **Prezzi & scenario** — editable fields for all 8 economic parameters
3. **Ottimizzazione** — "Ottimizza" button, progress bar, 3 parallel route runs
4. **Risultati** — recommended route card, optimal parameters table, Ashby chart, Pareto front

### 6.2 Ashby Chart (Results page)

Inspired by Granta EduPack materials selection charts:

- **X/Y axes:** user-selectable from output KPIs (purity, yield, tensile strength, energy, margin)
- **Bubble size:** proportional to a third KPI (default: mass yield)
- **Ellipses:** one per route, enclosing the feasible output space of the Pareto front solutions
- **Scatter points:** individual Pareto solutions, coloured by route
- **★ marker:** recommended optimal solution
- **Pareto front line:** dashed curve connecting non-dominated solutions
- **Scale:** linear or log–log selectable
- **Multiple chart tabs:** user can open side-by-side views of different property combinations
- **Bottom strip:** KPI comparison cards for all 3 routes

### 6.3 Jupyter Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_generation` | Generate and save synthetic CSV datasets; visualise distributions and correlations |
| `02_model_training` | Train 15 surrogate models; show R², RMSE, feature importance per route |
| `03_optimization` | Run NSGA-II on all 3 routes; visualise Pareto fronts in 2D/3D; Ashby chart |
| `04_sensitivity_analysis` | How results change when electricity/market prices vary (what-if scenarios) |

---

## 7. Dependencies

```
python >= 3.10
numpy
pandas
scikit-learn
xgboost
lightgbm
pymoo
streamlit
plotly
joblib
```

---

## 8. Data Flow Summary

```
[Input: material properties + economic prices]
        ↓
[retain_dss.data] → synthetic CSV (or real data swap-in)
        ↓
[retain_dss.models] → 15 surrogate models trained
        ↓
[retain_dss.optimizer] → NSGA-II × 3 routes (parallel)
        ↓                       ↓
  Pareto fronts ×3       Economic KPIs (analytical)
        ↓
[route_selector] → recommended route + optimal parameters
        ↓
[UI: Streamlit app / Jupyter notebooks]
→ Ashby chart, Pareto front, KPI comparison strip
```
