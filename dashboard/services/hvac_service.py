"""
Service layer - wraps hvac library and handles unit conversions.

Converts Pint Quantity objects to JSON-serializable format for the API.

@author: Bola
"""
from datetime import date, time
import sys
from pathlib import Path

# this is a bit hacky but we need the hvac package importable
# even when running from a different directory
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from hvac import Quantity
from hvac.energy_estimation.load import HeatingLoad
from hvac.radiant_emitter.panel_radiator import PanelRadiator
from hvac.sun.surface import Location, Surface

Q_ = Quantity  # shorthand, pint convention


def quantity_to_json(q: Quantity) -> dict:
    """Convert a Pint Quantity to JSON-friendly dict."""
    if q is None:
        return None
    try:
        import math
        val = float(q.m)
        # NaN breaks JSON serialization, learned that the hard way
        if math.isnan(val):
            val = None
        return {"value": val, "unit": str(q.u)}
    except Exception:
        # shouldn't happen but just in case
        return {"value": None, "unit": ""}


def heating_load_calculate(
    T_int_degC: float,
    T_ext_min_degC: float,
    H_trm_W_per_K: float,
    V_dot_ven_m3_hr: float,
    Q_dot_ihg_W: float,
    eta_sys_pct: float,
    T_ext_current_degC: float | None = None,
    num_hours: float | None = None,
) -> dict:
    """
    Compute heating load using hvac.energy_estimation.load.HeatingLoad.
    All numeric inputs are in SI-friendly units; we convert to Quantity inside.
    """
    load = HeatingLoad(
        T_int=Q_(T_int_degC, "degC"),
        T_ext_min=Q_(T_ext_min_degC, "degC"),
        H_trm=Q_(H_trm_W_per_K, "W/K"),
        V_dot_ven=Q_(V_dot_ven_m3_hr, "m**3/hr"),
        Q_dot_ihg=Q_(Q_dot_ihg_W, "W"),
        eta_sys=Q_(eta_sys_pct, "pct"),
    )
    if T_ext_current_degC is not None:
        load.T_ext = Q_(T_ext_current_degC, "degC")
    if num_hours is not None:
        load.num_hours = Q_(num_hours, "hr")

    return {
        "H_ven": quantity_to_json(load.H_ven.to("W/K")),
        "H_tot": quantity_to_json(load.H_tot.to("W/K")),
        "T_bal": quantity_to_json(load.T_bal.to("degC")),
        "Q_dot_out": quantity_to_json(load.Q_dot_out.to("kW")),
        "Q_dot_trm": quantity_to_json(load.Q_dot_trm.to("kW")),
        "Q_dot_ven": quantity_to_json(load.Q_dot_ven.to("kW")),
        "Q_dot_in": quantity_to_json(load.Q_dot_in.to("kW")),
        "Q_in": quantity_to_json(load.Q_in.to("kWh")),
    }


def heating_load_characteristic(
    T_int_degC: float,
    T_ext_min_degC: float,
    H_trm_W_per_K: float,
    V_dot_ven_m3_hr: float,
    Q_dot_ihg_W: float,
    eta_sys_pct: float,
) -> dict:
    """Get load curve (T_ext vs Q_dot_in) for charting."""
    load = HeatingLoad(
        T_int=Q_(T_int_degC, "degC"),
        T_ext_min=Q_(T_ext_min_degC, "degC"),
        H_trm=Q_(H_trm_W_per_K, "W/K"),
        V_dot_ven=Q_(V_dot_ven_m3_hr, "m**3/hr"),
        Q_dot_ihg=Q_(Q_dot_ihg_W, "W"),
        eta_sys=Q_(eta_sys_pct, "pct"),
    )
    T_ext_rng, Q_dot_in_rng = load.get_characteristic()
    T_ext_list = T_ext_rng.to("degC").m
    Q_list = Q_dot_in_rng.to("kW").m
    if hasattr(T_ext_list, "__len__"):
        T_ext_list = [float(x) for x in T_ext_list]
    else:
        T_ext_list = [float(T_ext_list)]
    if hasattr(Q_list, "__len__"):
        Q_list = [float(x) for x in Q_list]
    else:
        Q_list = [float(Q_list)]
    return {"T_ext_degC": T_ext_list, "Q_dot_in_kW": Q_list}


def solar_position(
    latitude_deg: float,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    surface_azimuth_deg: float = 0.0,
    surface_slope_deg: float = 0.0,
) -> dict:
    """
    Get solar position and basic angles for a location and optional surface.
    Surface: azimuth (S=0°, E=-90°, W=+90°), slope from horizontal.
    """
    loc = Location(fi=Q_(latitude_deg, "deg"))
    loc.date = date(year, month, day)
    loc.solar_time = time(hour, minute, 0)
    surface = Surface(
        location=loc,
        gamma=Q_(surface_azimuth_deg, "deg"),
        beta=Q_(surface_slope_deg, "deg"),
    )
    return {
        "sunrise": str(loc.sunrise) if loc.sunrise else None,
        "sunset": str(loc.sunset) if loc.sunset else None,
        "solar_altitude_deg": quantity_to_json(loc.sun.alpha.to("deg")),
        "zenith_deg": quantity_to_json(loc.sun.theta.to("deg")),
        "solar_azimuth_deg": quantity_to_json(loc.sun.gamma.to("deg")),
        "surface_sunrise": str(surface.sunrise) if surface.sunrise else None,
        "surface_sunset": str(surface.sunset) if surface.sunset else None,
        "profile_angle_deg": quantity_to_json(surface.alpha_p.to("deg")),
    }


def panel_radiator_design(
    Qe_dot_nom_W: float,
    Tw_sup_nom_degC: float,
    Tw_ret_nom_degC: float,
    Ti_nom_degC: float,
    n_exp: float,
) -> dict:
    """
    Create a PanelRadiator from nominal specs and return key design values.
    """
    rad = PanelRadiator(
        Qe_dot_nom=Q_(Qe_dot_nom_W, "W"),
        Tw_sup_nom=Q_(Tw_sup_nom_degC, "degC"),
        Tw_ret_nom=Q_(Tw_ret_nom_degC, "degC"),
        Ti_nom=Q_(Ti_nom_degC, "degC"),
        n_exp=n_exp,
    )
    return {
        "Vw_dot_nom": quantity_to_json(rad.Vw_dot_nom.to("L/s")),
        "Qe_dot_nom": quantity_to_json(rad.Qe_dot_nom.to("W")),
        "Tw_sup_nom": quantity_to_json(rad.Tw_sup_nom.to("degC")),
        "Tw_ret_nom": quantity_to_json(rad.Tw_ret_nom.to("degC")),
        "Ti_nom": quantity_to_json(rad.Ti_nom.to("degC")),
    }


def panel_radiator_operating(
    Qe_dot_nom_W: float,
    Tw_sup_nom_degC: float,
    Tw_ret_nom_degC: float,
    Ti_nom_degC: float,
    n_exp: float,
    Ti_degC: float,
    Qe_dot_W: float | None = None,
    Tw_sup_degC: float | None = None,
    Vw_dot_L_s: float | None = None,
) -> dict:
    """
    Get operating point: set Ti and exactly one of Qe_dot, Tw_sup, Vw_dot;
    returns the single solved quantity.
    """
    rad = PanelRadiator(
        Qe_dot_nom=Q_(Qe_dot_nom_W, "W"),
        Tw_sup_nom=Q_(Tw_sup_nom_degC, "degC"),
        Tw_ret_nom=Q_(Tw_ret_nom_degC, "degC"),
        Ti_nom=Q_(Ti_nom_degC, "degC"),
        n_exp=n_exp,
    )
    Ti = Q_(Ti_degC, "degC")
    Qe = Q_(Qe_dot_W, "W") if Qe_dot_W is not None else None
    Tw_sup = Q_(Tw_sup_degC, "degC") if Tw_sup_degC is not None else None
    Vw = (Q_(Vw_dot_L_s / 1000.0, "m**3/s") if Vw_dot_L_s is not None else None)
    out = rad(Ti=Ti, Qe_dot=Qe, Tw_sup=Tw_sup, Vw_dot=Vw)
    if out is None:
        return {}
    # Return in user-friendly units
    if str(out.u) == "watt":
        return {"solved": quantity_to_json(out.to("W")), "quantity": "Qe_dot"}
    if "kelvin" in str(out.u) or "degC" in str(out.u):
        return {"solved": quantity_to_json(out.to("degC")), "quantity": "Tw_sup"}
    return {"solved": quantity_to_json(out.to("L/s")), "quantity": "Vw_dot"}
