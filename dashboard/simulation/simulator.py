"""
--- HVAC RC Thermal Model ---
Resistor-Capacitor (RC) model for building thermal dynamics.

"""
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .rc_model import step_rc
from .controllers import (
    rule_based,
    rule_based_hvac,
    PIDController,
    mpc_simple,
    mpc_simple_hvac,
    RuleBasedConfig,
    PIDConfig,
    MPCConfig,
    HeatingType,
    HEATING_EFFICIENCY,
    FUEL_COSTS,
    ElectricityBand,
    ELECTRICITY_RATES_NGN,
)
from .mpc_advanced import AdvancedMPC, MPCAdvancedConfig, create_advanced_mpc
from .weather import get_outdoor_T
from .occupancy import is_business_hours, get_setpoint


@dataclass
class SimConfig:
    """Sim parameters. Defaults are roughly based on our office building."""
    # --- RC thermal model params ---
    R: float = 0.01  # K/W  (lower = better insulated)
    C: float = 1e4   # J/K  (thermal mass - was 1e6 but that was way too slow)
    dt_sec: float = 1.0  # sim timestep. smaller = smoother but more CPU
    Q_internal_occupied: float = 500.0   # internal gains when ppl around (W)
    Q_internal_unoccupied: float = 100.0 # just equipment (W)
    
    # --- Cooling System ---
    max_cooling: float = 10000.0  # chiller capacity (W)
    cooling_cop: float = 3.0      # coefficient of performance (electricity -> cooling)
    
    # --- Heating System ---
    max_heating: float = 8000.0   # heating capacity (W)
    heating_type: str = "gas_furnace"  # gas_furnace, electric, heat_pump_air, heat_pump_ground, boiler_gas, boiler_oil
    heating_efficiency: float = 0.92   # AFUE for furnace/boiler, COP for heat pump
    
    # --- Nigerian Electricity Pricing (NERC Tariff) ---
    # Currency: Nigerian Naira (₦)
    # Default: Band A (20+ hours supply) = ₦225/kWh
    # 1 kWh at Band A costs ₦225
    # ₦1000 = 4.44 kWh at Band A
    electricity_band: str = "band_a"  # band_a, band_b, band_c, band_d, band_e
    electricity_cost_per_kwh: float = 225.0  # ₦/kWh (Band A default)
    fuel_cost_per_kwh: float = 80.0    # ₦/kWh equivalent for gas heating
    currency: str = "NGN"  # Nigerian Naira
    
    # --- Setpoints ---
    setpoint_occupied: float = 22.0  # deg C
    setpoint_unoccupied: float = 18.0
    # Strategy: rule, pid, mpc (simple), mpc_advanced (production)
    strategy: Literal["rule", "pid", "mpc", "mpc_advanced"] = "rule"
    # Rule-based
    deadband: float = 1.0
    # PID - tuned for 10kW system with C=1e4 J/K
    Kp: float = 2000.0    # W per °C error
    Ki: float = 50.0      # W per °C·s accumulated
    Kd: float = 100.0     # W per °C/s rate
    # MPC (simple)
    mpc_horizon_steps: int = 6
    mpc_weight_comfort: float = 1.0
    mpc_weight_energy: float = 0.1
    # MPC Advanced (production-grade)
    mpc_adv_prediction_horizon: int = 60    # 60 min lookahead
    mpc_adv_control_horizon: int = 15       # 15 min control window
    mpc_adv_weight_comfort: float = 100.0
    mpc_adv_weight_energy: float = 1.0
    mpc_adv_weight_demand: float = 50.0
    mpc_adv_precool_hours: float = 1.5
    mpc_adv_optimizer: str = "scipy"        # "grid" or "scipy"
    # Overrides
    occupancy_override: bool | None = None  # None = auto (business hours)
    setpoint_override: float | None = None  # None = use occupied/unoccupied
    weather_source: Literal["synthetic", "api"] = "synthetic"
    weather_lat: float = 6.5244
    weather_lon: float = 3.3792


@dataclass
class SimState:
    """Current simulation state. Gets updated every tick."""
    T_in: float = 25.0       # indoor temp - start a bit warm so you can see cooling kick in
    T_out: float = 25.0      # outdoor (from weather module)
    setpoint: float = 22.0
    cooling: float = 0.0     # current cooling output in W
    heating: float = 0.0     # current heating output in W
    hvac_mode: str = "off"   # "heating", "cooling", or "off"
    occupied: bool = True
    strategy: str = "rule"   # active control strategy
    sim_time_sec: float = 0.0
    paused: bool = False     # pause button in UI
    error: float = 0.0       # for PID - difference from setpoint
    # Energy tracking
    total_heating_kwh: float = 0.0
    total_cooling_kwh: float = 0.0
    total_cost: float = 0.0
    # MPC Advanced diagnostics
    mpc_mode: str = ""       # pre-cooling, occupied, setback
    mpc_cost_breakdown: dict = field(default_factory=dict)
    electricity_rate: float = 0.0
    predicted_trajectory: list = field(default_factory=list)


class Simulator:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "hvac_log.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self.config = SimConfig()
        self.state = SimState()
        self._pid = PIDController(self.config.Kp, self.config.Ki, self.config.Kd)
        self._mpc_advanced = self._create_mpc_advanced()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_time_sec REAL,
                    T_in REAL, T_out REAL, setpoint REAL,
                    cooling REAL, heating REAL, hvac_mode TEXT,
                    occupied INTEGER, strategy TEXT,
                    created_at REAL
                )
            """)
            conn.commit()
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = conn

    def _log_row(self, sim_time: float, T_in: float, T_out: float, setpoint: float,
                 cooling: float, heating: float, hvac_mode: str,
                 occupied: bool, strategy: str) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                "INSERT INTO log (sim_time_sec, T_in, T_out, setpoint, cooling, heating, hvac_mode, occupied, strategy, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sim_time, T_in, T_out, setpoint, cooling, heating, hvac_mode, 1 if occupied else 0, strategy, time.time()),
            )
            self._conn.commit()

    def get_history(self, limit: int = 5000) -> list[dict]:
        with self._lock:
            if self._conn is None:
                return []
            try:
                cur = self._conn.execute(
                    "SELECT sim_time_sec, T_in, T_out, setpoint, cooling, heating, hvac_mode, occupied, strategy, created_at FROM log ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cur.fetchall()
            except Exception:
                return []
        return [
            {
                "sim_time_sec": r[0],
                "T_in": r[1],
                "T_out": r[2],
                "setpoint": r[3],
                "cooling": r[4],
                "heating": r[5],
                "hvac_mode": r[6],
                "occupied": bool(r[7]),
                "strategy": r[8],
                "created_at": r[9],
            }
            for r in reversed(rows)
        ]

    def update_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            # If heating_type changed, update efficiency and fuel cost
            if "heating_type" in kwargs:
                ht = kwargs["heating_type"]
                if ht in HEATING_EFFICIENCY:
                    self.config.heating_efficiency = HEATING_EFFICIENCY[ht]
                if ht in FUEL_COSTS:
                    self.config.fuel_cost_per_kwh = FUEL_COSTS[ht]
            # If electricity_band changed, update electricity rate
            if "electricity_band" in kwargs:
                band = kwargs["electricity_band"]
                if band in ELECTRICITY_RATES_NGN:
                    self.config.electricity_cost_per_kwh = ELECTRICITY_RATES_NGN[band]
            if "Kp" in kwargs or "Ki" in kwargs or "Kd" in kwargs:
                self._pid = PIDController(
                    self.config.Kp,
                    self.config.Ki,
                    self.config.Kd,
                )
            if "strategy" in kwargs and kwargs["strategy"] == "pid":
                self._pid.reset()
            # Recreate MPC advanced if relevant params changed
            mpc_adv_keys = {"mpc_adv_prediction_horizon", "mpc_adv_control_horizon",
                           "mpc_adv_weight_comfort", "mpc_adv_weight_energy",
                           "mpc_adv_weight_demand", "mpc_adv_precool_hours",
                           "mpc_adv_optimizer", "R", "C", "max_cooling", "max_heating"}
            if mpc_adv_keys & set(kwargs.keys()):
                self._mpc_advanced = self._create_mpc_advanced()

    def _create_mpc_advanced(self) -> AdvancedMPC:
        """Create/recreate the advanced MPC controller with current config."""
        cfg = self.config
        return create_advanced_mpc(
            max_cooling=cfg.max_cooling,
            R=cfg.R,
            C=cfg.C,
            prediction_horizon=cfg.mpc_adv_prediction_horizon,
            control_horizon=cfg.mpc_adv_control_horizon,
            weight_comfort=cfg.mpc_adv_weight_comfort,
            weight_energy=cfg.mpc_adv_weight_energy,
            weight_demand=cfg.mpc_adv_weight_demand,
            precool_hours=cfg.mpc_adv_precool_hours,
            optimizer=cfg.mpc_adv_optimizer,
            setpoint_occupied=cfg.setpoint_occupied,
            setpoint_unoccupied=cfg.setpoint_unoccupied,
            dt_sec=cfg.dt_sec,
        )

    def _step(self) -> None:
        try:
            cfg = self.config
            st = self.state
            if st.paused:
                return
            # Occupancy
            if cfg.occupancy_override is not None:
                occupied = cfg.occupancy_override
            else:
                occupied = is_business_hours()
            st.occupied = occupied
            # Setpoint
            if cfg.setpoint_override is not None:
                setpoint = cfg.setpoint_override
            else:
                setpoint = get_setpoint(occupied, cfg.setpoint_occupied, cfg.setpoint_unoccupied)
            st.setpoint = setpoint
            # Outdoor T
            T_out = get_outdoor_T(cfg.weather_source, cfg.weather_lat, cfg.weather_lon)
            st.T_out = T_out
            Q_internal = cfg.Q_internal_occupied if occupied else cfg.Q_internal_unoccupied
            
            # Control - now returns both heating and cooling
            heating = 0.0
            cooling = 0.0
            
            if cfg.strategy == "rule":
                heating, cooling = rule_based_hvac(
                    st.T_in, setpoint, cfg.deadband,
                    st.heating, st.cooling,
                    cfg.max_heating, cfg.max_cooling,
                )
                st.mpc_mode = ""
                st.mpc_cost_breakdown = {}
            elif cfg.strategy == "pid":
                heating, cooling = self._pid.compute_hvac(
                    st.T_in, setpoint, st.sim_time_sec,
                    cfg.max_heating, cfg.max_cooling
                )
                st.mpc_mode = ""
                st.mpc_cost_breakdown = {}
            elif cfg.strategy == "mpc_advanced":
                # MPC advanced currently only handles cooling
                # TODO: extend mpc_advanced for heating
                cooling, diagnostics = self._mpc_advanced.compute(
                    st.T_in, T_out, Q_internal, occupied, st.sim_time_sec
                )
                st.mpc_mode = diagnostics.get("mode", "")
                st.mpc_cost_breakdown = {
                    "comfort": round(diagnostics.get("comfort", 0), 2),
                    "energy": round(diagnostics.get("energy", 0), 4),
                    "demand": round(diagnostics.get("demand", 0), 2),
                    "total": round(diagnostics.get("total", 0), 2),
                }
                st.electricity_rate = diagnostics.get("electricity_rate", 0)
                st.predicted_trajectory = diagnostics.get("control_trajectory", [])
                setpoint = diagnostics.get("setpoint", setpoint)
                st.setpoint = setpoint
                # If it's cold and no cooling needed, try heating
                if cooling == 0 and st.T_in < setpoint - cfg.deadband:
                    heating = cfg.max_heating
            else:  # mpc (simple)
                heating, cooling = mpc_simple_hvac(
                    st.T_in, T_out, cfg.R, cfg.C, Q_internal, setpoint,
                    cfg.mpc_horizon_steps, cfg.dt_sec,
                    cfg.mpc_weight_comfort, cfg.mpc_weight_energy,
                    cfg.max_heating, cfg.max_cooling,
                    cfg.heating_efficiency, cfg.cooling_cop,
                )
                st.mpc_mode = ""
                st.mpc_cost_breakdown = {}
            
            # Update state
            st.heating = heating
            st.cooling = cooling
            st.strategy = cfg.strategy
            st.error = st.T_in - setpoint
            
            # Determine HVAC mode
            if heating > 0:
                st.hvac_mode = "heating"
            elif cooling > 0:
                st.hvac_mode = "cooling"
            else:
                st.hvac_mode = "off"
            
            # Track energy consumption and cost
            dt_hours = cfg.dt_sec / 3600.0
            if heating > 0:
                heating_kwh = (heating / 1000.0) * dt_hours
                st.total_heating_kwh += heating_kwh
                # Cost depends on heating type efficiency
                fuel_kwh = heating_kwh / cfg.heating_efficiency if cfg.heating_efficiency > 0 else heating_kwh
                st.total_cost += fuel_kwh * cfg.fuel_cost_per_kwh
            if cooling > 0:
                cooling_kwh = (cooling / 1000.0) * dt_hours
                st.total_cooling_kwh += cooling_kwh
                # Electricity cost for cooling (input power = cooling / COP)
                electricity_kwh = cooling_kwh / cfg.cooling_cop if cfg.cooling_cop > 0 else cooling_kwh
                st.total_cost += electricity_kwh * cfg.electricity_cost_per_kwh
            
            # RC step with both heating and cooling
            T_in_new, _ = step_rc(
                st.T_in, T_out, cfg.R, cfg.C, Q_internal, cooling, cfg.dt_sec, heating
            )
            st.T_in = T_in_new
            st.sim_time_sec += cfg.dt_sec
            self._log_row(
                st.sim_time_sec, st.T_in, st.T_out, setpoint,
                cooling, heating, st.hvac_mode, occupied, cfg.strategy,
            )
        except Exception:
            # Don't kill the simulator thread; next step will retry
            pass

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._step()
            elapsed = time.monotonic() - t0
            sleep = max(0, self.config.dt_sec - elapsed)
            if sleep > 0:
                self._stop.wait(timeout=sleep)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._started = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def pause(self) -> None:
        self.state.paused = True

    def resume(self) -> None:
        self.state.paused = False

    def get_status(self) -> dict:
        with self._lock:
            st = self.state
            cfg = self.config
        status = {
            "T_in": round(st.T_in, 2),
            "T_out": round(st.T_out, 2),
            "setpoint": round(st.setpoint, 2),
            "error": round(st.error, 2),
            # Heating
            "heating": round(st.heating, 1),
            "heating_kw": round(st.heating / 1000, 2),
            # Cooling
            "cooling": round(st.cooling, 1),
            "cooling_kw": round(st.cooling / 1000, 2),
            # HVAC mode
            "hvac_mode": st.hvac_mode,
            # Energy totals
            "total_heating_kwh": round(st.total_heating_kwh, 2),
            "total_cooling_kwh": round(st.total_cooling_kwh, 2),
            "total_cost": round(st.total_cost, 2),
            # Other
            "occupied": st.occupied,
            "strategy": st.strategy,
            "sim_time_sec": round(st.sim_time_sec, 1),
            "paused": st.paused,
            "current_time": time.time(),
        }
        # Add MPC advanced diagnostics if active
        if st.strategy == "mpc_advanced":
            status.update({
                "mpc_mode": st.mpc_mode,
                "mpc_cost_breakdown": st.mpc_cost_breakdown,
                "electricity_rate": round(st.electricity_rate, 3),
                "predicted_trajectory": [round(x, 1) for x in st.predicted_trajectory[:5]],
            })
        return status

    def get_config(self) -> dict:
        with self._lock:
            c = self.config
        return {
            "R": c.R,
            "C": c.C,
            "dt_sec": c.dt_sec,
            "Q_internal_occupied": c.Q_internal_occupied,
            "Q_internal_unoccupied": c.Q_internal_unoccupied,
            # Cooling
            "max_cooling": c.max_cooling,
            "cooling_cop": c.cooling_cop,
            # Heating
            "max_heating": c.max_heating,
            "heating_type": c.heating_type,
            "heating_efficiency": c.heating_efficiency,
            # Nigerian Electricity Pricing (NERC)
            "electricity_band": c.electricity_band,
            "electricity_cost_per_kwh": c.electricity_cost_per_kwh,
            "fuel_cost_per_kwh": c.fuel_cost_per_kwh,
            "currency": c.currency,
            # Setpoints
            "setpoint_occupied": c.setpoint_occupied,
            "setpoint_unoccupied": c.setpoint_unoccupied,
            "strategy": c.strategy,
            "deadband": c.deadband,
            "Kp": c.Kp, "Ki": c.Ki, "Kd": c.Kd,
            # Simple MPC
            "mpc_horizon_steps": c.mpc_horizon_steps,
            "mpc_weight_comfort": c.mpc_weight_comfort,
            "mpc_weight_energy": c.mpc_weight_energy,
            # Advanced MPC
            "mpc_adv_prediction_horizon": c.mpc_adv_prediction_horizon,
            "mpc_adv_control_horizon": c.mpc_adv_control_horizon,
            "mpc_adv_weight_comfort": c.mpc_adv_weight_comfort,
            "mpc_adv_weight_energy": c.mpc_adv_weight_energy,
            "mpc_adv_weight_demand": c.mpc_adv_weight_demand,
            "mpc_adv_precool_hours": c.mpc_adv_precool_hours,
            "mpc_adv_optimizer": c.mpc_adv_optimizer,
            # Overrides
            "occupancy_override": c.occupancy_override,
            "setpoint_override": c.setpoint_override,
            "weather_source": c.weather_source,
            "weather_lat": c.weather_lat,
            "weather_lon": c.weather_lon,
        }


# Singleton used by FastAPI
_simulator: Simulator | None = None
_sim_lock = threading.Lock()


def get_simulator() -> Simulator:
    global _simulator
    with _sim_lock:
        if _simulator is None:
            _simulator = Simulator()
            _simulator.start()
        return _simulator
