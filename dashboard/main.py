from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from dashboard.services.hvac_service import (
    heating_load_calculate,
    heating_load_characteristic,
    solar_position,
    panel_radiator_design,
    panel_radiator_operating,
)
from dashboard.services.ml_service import get_ml_service
from dashboard.simulation.simulator import get_simulator

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_simulator()
    except Exception as e:
        pass
    yield


app = FastAPI(title="HVAC System Project", description="HVAC simulation and control dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class HeatingLoadRequest(BaseModel):
    T_int_degC: float = 20.0
    T_ext_min_degC: float = -8.0
    H_trm_W_per_K: float = 0.143
    V_dot_ven_m3_hr: float = 100.0
    Q_dot_ihg_W: float = 500.0
    eta_sys_pct: float = 80.0
    T_ext_current_degC: float | None = 5.0
    num_hours: float | None = 10.0


class SolarRequest(BaseModel):
    latitude_deg: float = 51.0
    year: int = 2024
    month: int = 6
    day: int = 21
    hour: int = 12
    minute: int = 0
    surface_azimuth_deg: float = 0.0
    surface_slope_deg: float = 30.0


class RadiatorDesignRequest(BaseModel):
    Qe_dot_nom_W: float = 1000.0
    Tw_sup_nom_degC: float = 70.0
    Tw_ret_nom_degC: float = 55.0
    Ti_nom_degC: float = 20.0
    n_exp: float = 1.3


class RadiatorOperatingRequest(BaseModel):
    Qe_dot_nom_W: float = 1000.0
    Tw_sup_nom_degC: float = 70.0
    Tw_ret_nom_degC: float = 55.0
    Ti_nom_degC: float = 20.0
    n_exp: float = 1.3
    Ti_degC: float = 20.0
    Qe_dot_W: float | None = None
    Tw_sup_degC: float | None = None
    Vw_dot_L_s: float | None = None


class SimConfigUpdate(BaseModel):
    R: float | None = None
    C: float | None = None
    dt_sec: int | None = None
    Q_internal_occupied: float | None = None
    Q_internal_unoccupied: float | None = None
    max_cooling: float | None = None
    max_heating: float | None = None
    heating_type: str | None = None
    electricity_band: str | None = None
    setpoint_occupied: float | None = None
    setpoint_unoccupied: float | None = None
    strategy: str | None = None
    deadband: float | None = None
    Kp: float | None = None
    Ki: float | None = None
    Kd: float | None = None
    mpc_horizon_steps: int | None = None
    mpc_weight_comfort: float | None = None
    mpc_weight_energy: float | None = None
    occupancy_override: bool | None = None
    setpoint_override: float | None = None
    weather_source: str | None = None
    weather_lat: float | None = None
    weather_lon: float | None = None


class MLPredictionRequest(BaseModel):
    floor_area: float = 1000.0
    height: float = 3.0
    window_ratio: float = 0.3
    insulation_r: float = 3.5
    occupancy: float = 20.0
    outdoor_temp: float = 15.0
    solar_irradiance: float = 500.0
    wind_speed: float = 5.0


class MLTrainingRequest(BaseModel):
    n_samples: int = 1000


class SetpointOptimizationRequest(BaseModel):
    current_conditions: dict
    target_energy: float


class SimControlRequest(BaseModel):
    pause: bool | None = None
    resume: bool | None = None
    occupancy_override: bool | None = None
    setpoint_override: float | None = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("overview.html", {"request": request})


@app.get("/heating-load", response_class=HTMLResponse)
async def heating_load_page(request: Request):
    return templates.TemplateResponse("heating_load.html", {"request": request})


@app.get("/solar", response_class=HTMLResponse)
async def solar_page(request: Request):
    return templates.TemplateResponse("solar.html", {"request": request})


@app.get("/radiator", response_class=HTMLResponse)
async def radiator_page(request: Request):
    return templates.TemplateResponse("radiator.html", {"request": request})


@app.get("/live", response_class=HTMLResponse)
async def live_control_page(request: Request):
    return templates.TemplateResponse("live_control.html", {"request": request})


def _get_sim():
    try:
        return get_simulator()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Simulation not available: {e!s}") from e


@app.get("/api/sim/status")
async def api_sim_status():
    return _get_sim().get_status()


@app.get("/api/sim/config")
async def api_sim_config():
    return _get_sim().get_config()


@app.post("/api/sim/config")
async def api_sim_config_update(body: SimConfigUpdate):
    sim = _get_sim()
    d = body.model_dump(exclude_unset=True)
    if d:
        sim.update_config(**d)
    return sim.get_config()


@app.post("/api/sim/control")
async def api_sim_control(body: SimControlRequest):
    sim = _get_sim()
    if body.pause is True:
        sim.pause()
    if body.resume is True:
        sim.resume()
    if body.occupancy_override is not None:
        sim.update_config(occupancy_override=body.occupancy_override)
    if body.setpoint_override is not None:
        sim.update_config(setpoint_override=body.setpoint_override)
    return sim.get_status()


@app.get("/api/sim/history")
async def api_sim_history(limit: int = 5000):
    limit = max(1, min(5000, limit))
    return _get_sim().get_history(limit=limit)


@app.post("/api/heating-load")
async def api_heating_load(body: HeatingLoadRequest):
    try:
        return heating_load_calculate(
            T_int_degC=body.T_int_degC,
            T_ext_min_degC=body.T_ext_min_degC,
            H_trm_W_per_K=body.H_trm_W_per_K,
            V_dot_ven_m3_hr=body.V_dot_ven_m3_hr,
            Q_dot_ihg_W=body.Q_dot_ihg_W,
            eta_sys_pct=body.eta_sys_pct,
            T_ext_current_degC=body.T_ext_current_degC,
            num_hours=body.num_hours,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/heating-load/characteristic")
async def api_heating_load_characteristic(body: HeatingLoadRequest):
    try:
        return heating_load_characteristic(
            T_int_degC=body.T_int_degC,
            T_ext_min_degC=body.T_ext_min_degC,
            H_trm_W_per_K=body.H_trm_W_per_K,
            V_dot_ven_m3_hr=body.V_dot_ven_m3_hr,
            Q_dot_ihg_W=body.Q_dot_ihg_W,
            eta_sys_pct=body.eta_sys_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/solar")
async def api_solar(body: SolarRequest):
    try:
        return solar_position(
            latitude_deg=body.latitude_deg,
            year=body.year,
            month=body.month,
            day=body.day,
            hour=body.hour,
            minute=body.minute,
            surface_azimuth_deg=body.surface_azimuth_deg,
            surface_slope_deg=body.surface_slope_deg,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/radiator/design")
async def api_radiator_design(body: RadiatorDesignRequest):
    try:
        return panel_radiator_design(
            Qe_dot_nom_W=body.Qe_dot_nom_W,
            Tw_sup_nom_degC=body.Tw_sup_nom_degC,
            Tw_ret_nom_degC=body.Tw_ret_nom_degC,
            Ti_nom_degC=body.Ti_nom_degC,
            n_exp=body.n_exp,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/radiator/operating")
async def api_radiator_operating(body: RadiatorOperatingRequest):
    given = sum(1 for x in [body.Qe_dot_W, body.Tw_sup_degC, body.Vw_dot_L_s] if x is not None)
    if given != 1:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of: Qe_dot_W, Tw_sup_degC, Vw_dot_L_s",
        )
    try:
        return panel_radiator_operating(
            Qe_dot_nom_W=body.Qe_dot_nom_W,
            Tw_sup_nom_degC=body.Tw_sup_nom_degC,
            Tw_ret_nom_degC=body.Tw_ret_nom_degC,
            Ti_nom_degC=body.Ti_nom_degC,
            n_exp=body.n_exp,
            Ti_degC=body.Ti_degC,
            Qe_dot_W=body.Qe_dot_W,
            Tw_sup_degC=body.Tw_sup_degC,
            Vw_dot_L_s=body.Vw_dot_L_s,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/overview")
async def api_overview():
    return {
        "modules": [
            {"name": "Live Control", "path": "/live", "description": "Real-time RC simulation, rule-based / PID / MPC control"},
            {"name": "Heating Load", "path": "/heating-load", "description": "Building heating load and balance temperature"},
            {"name": "Solar", "path": "/solar", "description": "Solar position and surface angles"},
            {"name": "Radiator", "path": "/radiator", "description": "Panel radiator design and operating point"},
        ],
        "version": "0.1.3",
    }


@app.get("/api/ml/status")
async def api_ml_status():
    ml_service = get_ml_service()
    return ml_service.get_model_status()


@app.post("/api/ml/train")
async def api_ml_train(body: MLTrainingRequest):
    ml_service = get_ml_service()
    return ml_service.train_models(n_samples=body.n_samples)


@app.post("/api/ml/predict/energy")
async def api_ml_predict_energy(body: MLPredictionRequest):
    ml_service = get_ml_service()
    params = body.model_dump()
    return ml_service.predict_energy(params)


@app.post("/api/ml/predict/loads")
async def api_ml_predict_loads(body: MLPredictionRequest):
    ml_service = get_ml_service()
    params = body.model_dump()
    return ml_service.predict_loads(params)


@app.post("/api/ml/optimize/setpoint")
async def api_ml_optimize_setpoint(body: SetpointOptimizationRequest):
    ml_service = get_ml_service()
    return ml_service.optimize_setpoint(
        body.current_conditions, 
        body.target_energy
    )


@app.get("/api/ml/forecast")
async def api_ml_forecast(hours_ahead: int = 24):
    try:
        sim = get_simulator()
        history = sim.get_history(limit=100)
        
        ml_service = get_ml_service()
        return ml_service.generate_forecast(history, hours_ahead)
    except Exception as e:
        return {"error": str(e)}
