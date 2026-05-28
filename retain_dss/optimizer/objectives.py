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

    pvc_mass = (
        predictions.get("pvc_purity", 0.0) / 100
        * predictions.get("mass_yield", 0.0) / 100
    )
    pet_key = "pet_recovery" if "pet_recovery" in predictions else "pet_purity"
    pet_mass = (
        predictions.get(pet_key, 0.0) / 100
        * predictions.get("mass_yield", 0.0) / 100
    )
    plast_mass = (
        predictions.get("plasticizer_recovery", 0.0) / 100
        * predictions.get("mass_yield", 0.0) / 100
    )
    stab_mass = (
        predictions.get("stabilizer_recovery", 0.0) / 100
        * predictions.get("mass_yield", 0.0) / 100
    )

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
