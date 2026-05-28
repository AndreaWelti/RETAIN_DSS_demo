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

# Hide Streamlit's default running-man animation; replace with hourglass via st.spinner
st.markdown(
    """<style>[data-testid="stStatusWidget"] {visibility: hidden;}</style>""",
    unsafe_allow_html=True,
)


def _fmt(name: str) -> str:
    """Convert snake_case key to Title Case label without underscores."""
    return name.replace("_", " ").title()


# ── Descriptions shown in help tooltips ("?" buttons) ─────────────────────────

MATERIAL_DESCRIPTIONS = {
    "pvc_content": (
        "Mass fraction of poly(vinyl chloride) polymer in the composite tarpaulin (%wt of total dry weight). "
        "Typical industrial tarpaulins contain 40–70 %wt PVC. "
        "Higher values favour Routes 1 and 2, which target direct PVC recovery."
    ),
    "pet_content": (
        "Mass fraction of poly(ethylene terephthalate) fibres forming the woven or knitted scrim embedded "
        "within the PVC coating. PET provides tensile and tear resistance. "
        "High-performance tarpaulins (truck covers, stadium membranes) tend toward higher PET fractions."
    ),
    "additive_content": (
        "Mass fraction of functional additives compounded into the PVC matrix: "
        "plasticisers (DINP, citrate esters — 10–50 phr), heat stabilisers (Ca/Zn, organotin), "
        "UV absorbers, flame retardants, and pigments. "
        "This fraction is the primary economic target of Route 3 (selective extraction), "
        "since recovered plasticisers and stabilisers command much higher market prices than bulk polymer."
    ),
    "contamination_level": (
        "Mass fraction of extraneous materials not part of the original tarpaulin, "
        "accumulated during service life and handling. Contaminants include: "
        "soil, clay, sand (ground contact); oils and greases (machinery contact); "
        "adhesive and paint residues; biological matter (mould, organic debris); "
        "metallic fragments (grommets, staples, wire eyelets). "
        "Contamination is the single strongest negative predictor of output purity across all three routes."
    ),
    "particle_size_d50": (
        "Median particle diameter of the pre-shredded input material (d₅₀ of mass-based particle size distribution). "
        "Large values (>20 mm) indicate coarse primary shredding only; "
        "small values (<5 mm) indicate fine pre-grinding. "
        "For Route 1 this sets the starting point for milling. "
        "For Routes 2 and 3 it controls dissolution/extraction kinetics via specific surface area."
    ),
    "moisture_content": (
        "Water absorbed primarily in the PET fibre network and at the PVC–PET interface, "
        "from outdoor storage, rainfall, or washing. "
        "Negatively impacts purity (dilution, incomplete dissolution) in Routes 2 and 3, "
        "and causes steam expansion and sticking during milling in Route 1. "
        "Values above 5 %wt indicate that a pre-drying step would be beneficial."
    ),
    "material_age": (
        "Estimated service lifetime of the tarpaulin before collection, in years. "
        "Used as a proxy for cumulative thermal and UV degradation: chain scission reduces molecular weight, "
        "plasticiser migrates out of the PVC matrix, and surface chalking occurs. "
        "Older materials yield lower tensile strength in the recovered polymer (Routes 2 and 3)."
    ),
    "tensile_strength_input": (
        "Ultimate tensile strength of the composite tarpaulin as received, before any shredding. "
        "Encodes the combined effect of PET fibre integrity, PVC degradation state, and residual plasticiser content. "
        "Primary predictor of recovered polymer quality: output tensile strength scales "
        "proportionally with input tensile strength, modulated by output purity."
    ),
}

ECONOMIC_DESCRIPTIONS = {
    "electricity_price": (
        "Unit cost of electrical energy purchased from the grid (€/kWh_el). "
        "Applies to all three routes (shredder, agitator, pumps). "
        "Dominant operating cost for Route 1 (high shredder/mill consumption) "
        "and Route 3 (agitation scales as speed^1.3)."
    ),
    "thermal_energy_price": (
        "Unit cost of thermal energy delivered to the process — steam or hot-oil circuit (€/kWh_th). "
        "Applies to Routes 2 and 3 only (heating of dissolution/extraction vessels). "
        "Not applicable to Route 1, which requires no process heating."
    ),
    "solvent_price": (
        "Net unit cost of make-up solvent consumed per tonne of input (€/L), "
        "after in-process recovery and recycling. "
        "Applies to Route 2 (Texyloop) only. "
        "Strong influence on net margin at high solvent concentration or low washing cycles."
    ),
    "extractant_price": (
        "Unit cost of extractant chemical consumed per tonne (€/L). "
        "Applies to Route 3 (selective additive extraction) only. "
        "Higher per-litre cost than solvent, but consumed in smaller volumes "
        "due to the targeted extraction mechanism."
    ),
    "price_pvc_recycled": (
        "Market price of recycled PVC polymer meeting product quality specifications (€/t). "
        "Primary revenue driver for Routes 1 and 2. "
        "Sensitive to virgin PVC price and sustainability premiums in current markets."
    ),
    "price_pet_recycled": (
        "Market price of recycled PET fibre or pellet recovered as a co-product (€/t). "
        "Relevant for Routes 1 and 2, where PET purity or PET recovery is predicted."
    ),
    "price_plasticizer_recovered": (
        "Market value of recovered plasticiser (e.g., DINP, citrate esters) (€/t). "
        "Significantly higher than bulk polymer value due to speciality chemical positioning. "
        "Exclusively relevant for Route 3 — contributes zero revenue in Routes 1 and 2."
    ),
    "price_stabilizer_recovered": (
        "Market value of recovered heat stabiliser (Ca/Zn complexes, organotin compounds) (€/t). "
        "Highest unit value among all co-products. "
        "Exclusively relevant for Route 3 — contributes zero revenue in Routes 1 and 2."
    ),
}

ROUTE_DESCRIPTIONS = {
    "route1": (
        "**T3.1 — Mechanical Recycling**\n\n"
        "Separates PVC from PET through purely physical size reduction (shredding, milling) "
        "and classification (electrostatic or density-based separation). "
        "No solvents or process heating required — lowest energy option. "
        "Output: PVC-rich granulate and PET-rich fraction. "
        "PVC purity range: 60–95%."
    ),
    "route2": (
        "**T3.2 — Solvent / Texyloop Recycling**\n\n"
        "Selectively dissolves PVC in a hot organic solvent, leaving PET undissolved. "
        "PVC is re-precipitated by cooling, washed, dried, and pelletised. "
        "Solvent is recovered in a closed loop. Delivers the highest PVC purity (85–99.5%) "
        "at the cost of higher energy and solvent consumption."
    ),
    "route3": (
        "**T3.3 — Selective Additive Extraction**\n\n"
        "Uses a liquid extractant to selectively remove and recover plasticisers and stabilisers "
        "from the PVC matrix. PVC and PET remain solid throughout — they are not dissolved. "
        "Produces three co-product streams: cleaned PVC/PET composite, recovered plasticiser, "
        "and recovered stabiliser. Highest economic value when additive content is high."
    ),
}

PROCESS_PARAM_DESCRIPTIONS = {
    "route1": {
        "shredder_speed":    "Rotational speed of the primary shredder rotor. Higher speeds deliver finer fragments but increase electrical energy consumption (∝ speed^1.5).",
        "mill_gap":          "Clearance between milling surfaces. Primary determinant of output particle size: d50_out ≈ 0.55 × mill_gap + 0.25 × sieve_mesh_size.",
        "sieve_mesh_size":   "Aperture of the classification screen. Sets the upper bound on product particle size; larger apertures increase yield but reduce purity.",
        "separation_cycles": "Number of passes through the electrostatic or density separator. Each cycle improves purity at the cost of mass yield losses.",
        "process_temp":      "Ambient temperature of the processing environment. Moderate warming (40–60 °C) slightly improves PVC/PET liberation.",
        "throughput_rate":   "Mass flow rate fed to the process line. Higher throughput reduces unit energy consumption but shortens separation residence time.",
    },
    "route2": {
        "solvent_concentration": "Volume fraction of the selective PVC solvent. Higher concentration accelerates dissolution but increases solvent and energy costs.",
        "dissolution_temp":      "Temperature of the dissolution vessel. Main driver of PVC purity — above ~100 °C, PVC dissolves rapidly while PET remains solid.",
        "dissolution_time":      "Residence time in the dissolution vessel. Longer times ensure complete PVC dissolution; diminishing returns above ~120 min.",
        "solid_liquid_ratio":    "Mass of feedstock per litre of solvent. Higher ratios reduce cost but increase viscosity and reduce purity.",
        "precipitation_temp":    "Temperature to which the PVC-laden solution is cooled to re-precipitate PVC. Lower temperature → finer, purer precipitate.",
        "washing_cycles":        "Number of solvent-wash and water-rinse cycles applied to the precipitated PVC cake. Each cycle improves purity by removing residual solvent and impurities.",
    },
    "route3": {
        "extractant_type":   "Chemical family of the extractant: 0 = polar organic (good for plasticisers); 1 = enhanced polar (+15% efficiency); 2 = aqueous alkaline (lower efficiency, lower cost).",
        "extractant_conc":   "Concentration of extractant agent. Non-linear effect on recovery (power-law 0.7) due to equilibrium partitioning limits.",
        "extraction_temp":   "Operating temperature of the extraction vessel. Higher temperature accelerates diffusion kinetics and improves plasticiser/stabiliser recovery.",
        "extraction_time":   "Contact time with the extractant. Approaches equilibrium above ~120 min for typical particle sizes.",
        "ph_level":          "pH of the extraction bath. Plasticiser recovery peaks near pH 7 (Gaussian factor); stabiliser recovery peaks near pH 9. Intermediate pH trades off both.",
        "agitation_speed":   "Agitator speed. Thins the diffusion boundary layer at particle surfaces, accelerating extraction. Electrical energy ∝ speed^1.3.",
    },
}

KPI_DESCRIPTIONS = {
    "pvc_purity":       "Purity of the recovered PVC fraction (%). Higher is better. Primary product quality specification.",
    "mass_yield":       "Fraction of input mass recovered as usable output (%). Higher is better. Affects all revenue streams.",
    "material_quality": "Structural quality of recovered polymer: tensile strength [MPa] for Routes 2 and 3; inverse of output particle size [mm⁻¹] for Route 1.",
    "total_energy":     "Total energy consumption — electrical + thermal (kWh/t). Lower is better. Sum of all energy inputs per tonne of feedstock.",
    "net_margin":       "Net economic margin (€/t = Revenue − OPEX). Higher is better. Revenue from recovered materials minus energy and consumable costs.",
}

ROUTE_LABELS = {
    "route1": "T3.1 — Mechanical",
    "route2": "T3.2 — Solvent (Texyloop)",
    "route3": "T3.3 — Additive Extraction",
}
ROUTE_COLORS = {"route1": "#3b82f6", "route2": "#f59e0b", "route3": "#4ade80"}
ROUTE_PARAMS = {"route1": PROCESS_PARAMS_R1, "route2": PROCESS_PARAMS_R2, "route3": PROCESS_PARAMS_R3}

PAGES = ["📥 Material Input", "💶 Prices & Scenario", "⚙️ Optimisation", "📊 Results"]
page = st.sidebar.radio("Navigation", PAGES)


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
            name="Optimal solution",
        ))

    axis_kw = dict(type="log") if log_scale else {}
    fig.update_layout(
        xaxis=dict(title=x_key.replace("_", " "), **axis_kw),
        yaxis=dict(title=y_key.replace("_", " "), **axis_kw),
        template="plotly_dark", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Page 1: Material Input ──────────────────────────────────────────────────
if page == PAGES[0]:
    st.title("📥 Material Input Properties")
    st.markdown("Enter the properties of the PVC/PET tarpaulin to be recycled.")
    mat_inputs = {}
    cols = st.columns(2)
    for i, (name, spec) in enumerate(MATERIAL_INPUTS.items()):
        lo, hi = spec["range"]
        mid = (lo + hi) / 2
        mat_inputs[name] = cols[i % 2].slider(
            f"{_fmt(name)} [{spec['unit']}]", min_value=float(lo), max_value=float(hi),
            value=float(mid), step=float((hi - lo) / 100),
            help=MATERIAL_DESCRIPTIONS.get(name, ""),
        )
    st.session_state["mat_inputs"] = mat_inputs
    st.success("Values saved. Continue with ➡️ Prices & Scenario")


# ── Page 2: Prices & Scenario ────────────────────────────────────────────────
elif page == PAGES[1]:
    st.title("💶 Prices & Economic Scenario")
    st.markdown("Edit the reference prices for the economic analysis.")
    p = EconomicParams()
    c1, c2 = st.columns(2)
    prices = EconomicParams(
        electricity_price      = c1.number_input("Electricity (€/kWh_el)",            value=p.electricity_price,           step=0.01,  help=ECONOMIC_DESCRIPTIONS["electricity_price"]),
        thermal_energy_price   = c1.number_input("Thermal energy (€/kWh_th)",          value=p.thermal_energy_price,        step=0.01,  help=ECONOMIC_DESCRIPTIONS["thermal_energy_price"]),
        solvent_price          = c1.number_input("Solvent (€/L)",                       value=p.solvent_price,               step=0.1,   help=ECONOMIC_DESCRIPTIONS["solvent_price"]),
        extractant_price       = c1.number_input("Extractant (€/L)",                    value=p.extractant_price,            step=0.1,   help=ECONOMIC_DESCRIPTIONS["extractant_price"]),
        price_pvc_recycled     = c2.number_input("Recycled PVC (€/t)",                  value=p.price_pvc_recycled,          step=10.0,  help=ECONOMIC_DESCRIPTIONS["price_pvc_recycled"]),
        price_pet_recycled     = c2.number_input("Recycled PET (€/t)",                  value=p.price_pet_recycled,          step=10.0,  help=ECONOMIC_DESCRIPTIONS["price_pet_recycled"]),
        price_plasticizer_recovered = c2.number_input("Recovered plasticiser (€/t)",    value=p.price_plasticizer_recovered, step=50.0,  help=ECONOMIC_DESCRIPTIONS["price_plasticizer_recovered"]),
        price_stabilizer_recovered  = c2.number_input("Recovered stabiliser (€/t)",     value=p.price_stabilizer_recovered,  step=50.0,  help=ECONOMIC_DESCRIPTIONS["price_stabilizer_recovered"]),
    )
    st.session_state["prices"] = prices
    st.success("Prices saved. Continue with ➡️ Optimisation")


# ── Page 3: Optimisation ─────────────────────────────────────────────────────
elif page == PAGES[2]:
    st.title("⚙️ Multi-Objective Optimisation")
    mat_inputs = st.session_state.get("mat_inputs")
    prices     = st.session_state.get("prices", EconomicParams())

    if mat_inputs is None:
        st.warning("Please enter material properties first (page 1).")
    else:
        # Route description cards
        st.markdown("#### Recycling routes included in the optimisation")
        rc1, rc2, rc3 = st.columns(3)
        for col, (rkey, rlabel) in zip([rc1, rc2, rc3], ROUTE_LABELS.items()):
            with col:
                with st.expander(rlabel, expanded=False):
                    st.markdown(ROUTE_DESCRIPTIONS[rkey])
        st.divider()

        pop_size = st.slider("GA population size", 20, 200, 100, 10)
        n_gen    = st.slider("Number of GA generations", 20, 500, 200, 10)

        if st.button("🚀 Optimise all 3 routes", type="primary"):
            models   = get_models()
            results  = {}
            progress = st.progress(0, text="⏳ Starting optimisation…")
            with st.spinner("⏳ Running NSGA-II — this may take a minute…"):
                for i, route in enumerate(["route1", "route2", "route3"]):
                    progress.progress((i + 1) * 33, text=f"⏳ Optimising {ROUTE_LABELS[route]}…")
                    results[route] = run_nsga2(mat_inputs, models[route], route, prices, pop_size, n_gen)
            progress.progress(100, text="✅ Done!")
            selection = select_best_route(results)
            st.session_state["results"]   = results
            st.session_state["selection"] = selection
            st.success(f"✓ Recommended route: **{ROUTE_LABELS[selection['recommended_route']]}**")


# ── Page 4: Results ──────────────────────────────────────────────────────────
elif page == PAGES[3]:
    st.title("📊 Results — Ashby Chart & Pareto Front")
    selection = st.session_state.get("selection")
    results   = st.session_state.get("results")

    if selection is None:
        st.warning("Please run the optimisation first (page 3).")
    else:
        best_route = selection["recommended_route"]
        st.markdown(f"### ★ Recommended route: **{ROUTE_LABELS[best_route]}**")

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
                f"PVC purity: <b>{-F[0]:.1f}%</b><br>"
                f"Mass yield: <b>{-F[1]:.1f}%</b><br>"
                f"Energy: <b>{F[3]:.0f} kWh/t</b><br>"
                f"Net margin: <b>€{-F[4]:.0f}/t</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        # Ashby chart controls
        KPI_OPTS = ["pvc_purity", "mass_yield", "material_quality", "total_energy", "net_margin"]
        cc1, cc2, cc3, cc4 = st.columns(4)
        x_key    = cc1.selectbox("X axis",      KPI_OPTS, index=0)
        cc1.caption(KPI_DESCRIPTIONS.get(x_key, ""))
        y_key    = cc2.selectbox("Y axis",      KPI_OPTS, index=2)
        cc2.caption(KPI_DESCRIPTIONS.get(y_key, ""))
        size_key = cc3.selectbox("Bubble size", KPI_OPTS, index=1)
        cc3.caption(KPI_DESCRIPTIONS.get(size_key, ""))
        log_sc   = cc4.checkbox("Log scale")

        fig = _ashby_chart(results, selection, x_key, y_key, size_key, log_sc)
        st.plotly_chart(fig, use_container_width=True)

        # Best parameters table
        st.subheader("Optimal parameters — recommended route")
        route_params  = ROUTE_PARAMS[best_route]
        param_keys    = list(route_params.keys())
        best_params   = selection["best_params"]
        route_descs   = PROCESS_PARAM_DESCRIPTIONS.get(best_route, {})
        param_df = pd.DataFrame({
            "Parameter":     [_fmt(k) for k in param_keys],
            "Optimal value": [f"{v:.2f}" for v in best_params],
            "Unit":          [route_params[k]["unit"] for k in param_keys],
            "Range":         [f"{route_params[k]['range'][0]} – {route_params[k]['range'][1]}" for k in param_keys],
            "Description":   [route_descs.get(k, "") for k in param_keys],
        })
        st.dataframe(param_df, use_container_width=True, hide_index=True)
