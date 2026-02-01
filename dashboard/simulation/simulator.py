"""
Real-time HVAC simulation using a simple RC thermal model.

Runs in a background thread, logs to SQLite. The RC model is a simplified
approach but works well enough for demonstrating the control strategies.

@author: Bola
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
    PIDController,
    mpc_simple,
    RuleBasedConfig,
    PIDConfig,
    MPCConfig,
)
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
    max_cooling: float = 10000.0  # chiller capacity, 10kW is probably overkill
    # --- Setpoints ---
    setpoint_occupied: float = 22.0  # deg C
    setpoint_unoccupied: float = 18.0
    # Strategy
    strategy: Literal["rule", "pid", "mpc"] = "rule"
    # Rule-based
    deadband: float = 1.0
    # PID
    Kp: float = 0.5
    Ki: float = 0.05
    Kd: float = 0.01
    # MPC
    mpc_horizon_steps: int = 6
    mpc_weight_comfort: float = 1.0
    mpc_weight_energy: float = 0.1
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
    cooling: float = 0.0     # current cooling output in W (negative = heating... wait no thats wrong TODO check this)
    occupied: bool = True
    strategy: str = "rule"   # active control strategy
    sim_time_sec: float = 0.0
    paused: bool = False     # pause button in UI
    error: float = 0.0       # for PID - difference from setpoint


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
                    cooling REAL, occupied INTEGER, strategy TEXT,
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
                 cooling: float, occupied: bool, strategy: str) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                "INSERT INTO log (sim_time_sec, T_in, T_out, setpoint, cooling, occupied, strategy, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (sim_time, T_in, T_out, setpoint, cooling, 1 if occupied else 0, strategy, time.time()),
            )
            self._conn.commit()

    def get_history(self, limit: int = 5000) -> list[dict]:
        with self._lock:
            if self._conn is None:
                return []
            try:
                cur = self._conn.execute(
                    "SELECT sim_time_sec, T_in, T_out, setpoint, cooling, occupied, strategy, created_at FROM log ORDER BY id DESC LIMIT ?",
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
                "occupied": bool(r[5]),
                "strategy": r[6],
                "created_at": r[7],
            }
            for r in reversed(rows)
        ]

    def update_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            if "Kp" in kwargs or "Ki" in kwargs or "Kd" in kwargs:
                self._pid = PIDController(
                    self.config.Kp,
                    self.config.Ki,
                    self.config.Kd,
                )
            if "strategy" in kwargs and kwargs["strategy"] == "pid":
                self._pid.reset()

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
            # Control
            if cfg.strategy == "rule":
                cooling = rule_based(
                    st.T_in, setpoint, cfg.deadband,
                    st.cooling, cfg.max_cooling,
                )
            elif cfg.strategy == "pid":
                cooling = self._pid.compute(st.T_in, setpoint, st.sim_time_sec, cfg.max_cooling)
            else:  # mpc
                cooling = mpc_simple(
                    st.T_in, T_out, cfg.R, cfg.C, Q_internal, setpoint,
                    cfg.mpc_horizon_steps, cfg.dt_sec,
                    cfg.mpc_weight_comfort, cfg.mpc_weight_energy, cfg.max_cooling,
                )
            st.cooling = cooling
            st.strategy = cfg.strategy
            st.error = st.T_in - setpoint
            # RC step
            T_in_new, _ = step_rc(
                st.T_in, T_out, cfg.R, cfg.C, Q_internal, cooling, cfg.dt_sec,
            )
            st.T_in = T_in_new
            st.sim_time_sec += cfg.dt_sec
            self._log_row(
                st.sim_time_sec, st.T_in, st.T_out, setpoint,
                cooling, occupied, cfg.strategy,
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
        return {
            "T_in": round(st.T_in, 2),
            "T_out": round(st.T_out, 2),
            "setpoint": round(st.setpoint, 2),
            "error": round(st.error, 2),
            "cooling": round(st.cooling, 1),
            "occupied": st.occupied,
            "strategy": st.strategy,
            "sim_time_sec": round(st.sim_time_sec, 1),
            "paused": st.paused,
            "current_time": time.time(),
        }

    def get_config(self) -> dict:
        with self._lock:
            c = self.config
        return {
            "R": c.R,
            "C": c.C,
            "dt_sec": c.dt_sec,
            "Q_internal_occupied": c.Q_internal_occupied,
            "Q_internal_unoccupied": c.Q_internal_unoccupied,
            "max_cooling": c.max_cooling,
            "setpoint_occupied": c.setpoint_occupied,
            "setpoint_unoccupied": c.setpoint_unoccupied,
            "strategy": c.strategy,
            "deadband": c.deadband,
            "Kp": c.Kp, "Ki": c.Ki, "Kd": c.Kd,
            "mpc_horizon_steps": c.mpc_horizon_steps,
            "mpc_weight_comfort": c.mpc_weight_comfort,
            "mpc_weight_energy": c.mpc_weight_energy,
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
