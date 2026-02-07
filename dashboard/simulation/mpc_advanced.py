"""
Advanced Model Predictive Control (MPC) for HVAC Systems.

Production-grade implementation with:
- Weather forecast integration (24-hour lookahead)
- Occupancy schedule prediction
- Time-of-use electricity pricing
- Pre-cooling/pre-heating optimization
- Solar gain estimation
- Variable control trajectory (not constant over horizon)
- Soft constraints with slack variables
- Receding horizon with warm-start

"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable
import numpy as np

from .rc_model import step_rc


@dataclass
class MPCAdvancedConfig:
    """Configuration for advanced MPC controller."""
    # Horizon settings
    prediction_horizon: int = 60          # steps to look ahead
    control_horizon: int = 15             # steps where control can change
    dt_sec: float = 60.0                  # timestep (1 min default)
    
    # Comfort settings
    setpoint_occupied: float = 22.0       # °C
    setpoint_unoccupied: float = 26.0     # °C (allow drift when empty)
    setpoint_night: float = 28.0          # °C (even more relaxed at night)
    comfort_band: float = 1.0             # ±1°C acceptable deviation
    
    # Cost weights (these define the optimization objective)
    weight_comfort: float = 100.0         # penalty per °C² deviation
    weight_energy: float = 1.0            # penalty per kWh
    weight_demand: float = 50.0           # penalty for peak demand (kW²)
    weight_rate_change: float = 10.0      # penalty for rapid control changes
    
    # Equipment constraints
    max_cooling: float = 10000.0          # W - chiller capacity
    min_cooling: float = 0.0              # W - can't heat in cooling mode
    max_rate_change: float = 2000.0       # W/min - equipment ramp limit
    
    # Building model (RC parameters)
    R: float = 0.01                       # K/W thermal resistance
    C: float = 1e4                        # J/K thermal capacitance
    
    # Electricity pricing ($/kWh by hour, 24 values)
    # Default: higher during peak (12-20), lower at night
    electricity_rates: list[float] = field(default_factory=lambda: [
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,  # 00-06: off-peak
        0.12, 0.15, 0.18, 0.20, 0.22, 0.25,  # 06-12: rising
        0.28, 0.30, 0.30, 0.28, 0.25, 0.22,  # 12-18: peak
        0.20, 0.18, 0.15, 0.12, 0.10, 0.08,  # 18-24: falling
    ])
    
    # Occupancy schedule (probability by hour)
    # Default: office building schedule
    occupancy_schedule: list[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # 00-06: empty
        0.1, 0.5, 0.9, 1.0, 1.0, 1.0,        # 06-12: arriving
        0.9, 1.0, 1.0, 1.0, 0.9, 0.7,        # 12-18: full/leaving
        0.3, 0.1, 0.05, 0.0, 0.0, 0.0,       # 18-24: mostly gone
    ])
    
    # Pre-cooling settings
    precool_hours: float = 1.5            # start cooling this many hours before occupancy
    precool_target_offset: float = -1.0   # cool to setpoint - offset
    
    # Solar gain estimation
    solar_gain_peak: float = 500.0        # W peak solar gain through windows
    solar_peak_hour: float = 14.0         # hour of peak solar (2pm)
    
    # Optimization settings
    optimizer: str = "scipy"              # "grid" or "scipy"
    grid_resolution: int = 21             # for grid search
    max_iterations: int = 100             # for scipy optimizer


class WeatherForecast:
    """
    Weather forecast provider.
    Uses simple model if no API available, can be extended for real forecast APIs.
    """
    def __init__(
        self,
        base_temp: float = 25.0,
        daily_amplitude: float = 5.0,
        trend: float = 0.0,  # °C/hour warming/cooling trend
    ):
        self.base_temp = base_temp
        self.daily_amplitude = daily_amplitude
        self.trend = trend
        self._forecast_cache: dict[int, float] = {}
    
    def update_current(self, T_current: float, hour: float):
        """Update forecast model with current observation."""
        # Simple exponential smoothing
        expected = self._predict_hour(hour)
        self.base_temp += 0.1 * (T_current - expected)
    
    def _predict_hour(self, hour: float) -> float:
        """Predict temperature for a given hour of day."""
        # Sinusoidal model: peak at 3pm (15:00), min at 6am
        daily_variation = math.sin(2 * math.pi * (hour - 6) / 24)
        return self.base_temp + self.daily_amplitude * daily_variation
    
    def get_forecast(self, hours_ahead: int = 24, dt_hours: float = 1/60) -> list[float]:
        """
        Get temperature forecast for next N hours.
        Returns list of temperatures at dt_hours intervals.
        """
        now = datetime.now(timezone.utc)
        current_hour = now.hour + now.minute / 60
        
        n_steps = int(hours_ahead / dt_hours)
        forecast = []
        
        for i in range(n_steps):
            future_hour = (current_hour + i * dt_hours) % 24
            hours_from_now = i * dt_hours
            T = self._predict_hour(future_hour) + self.trend * hours_from_now
            forecast.append(T)
        
        return forecast


class OccupancyPredictor:
    """
    Occupancy prediction based on schedule and calendar.
    Can be extended with ML models, calendar integration, etc.
    """
    def __init__(self, schedule: list[float], utc_offset: float = 1.0):
        """
        schedule: 24-element list of occupancy probability by hour
        utc_offset: local timezone offset from UTC
        """
        self.schedule = schedule
        self.utc_offset = utc_offset
        self._overrides: dict[str, bool] = {}  # date -> forced occupancy
    
    def add_override(self, date_str: str, occupied: bool):
        """Override occupancy for a specific date (e.g., holidays)."""
        self._overrides[date_str] = occupied
    
    def predict(self, hours_ahead: int = 24, dt_hours: float = 1/60) -> list[float]:
        """
        Predict occupancy probability for next N hours.
        Returns list of probabilities [0, 1] at dt_hours intervals.
        """
        now = datetime.now(timezone.utc) + timedelta(hours=self.utc_offset)
        n_steps = int(hours_ahead / dt_hours)
        predictions = []
        
        for i in range(n_steps):
            future_time = now + timedelta(hours=i * dt_hours)
            date_str = future_time.strftime("%Y-%m-%d")
            
            # Check for override (holiday, etc.)
            if date_str in self._overrides:
                predictions.append(1.0 if self._overrides[date_str] else 0.0)
                continue
            
            # Weekend check
            if future_time.weekday() >= 5:  # Saturday=5, Sunday=6
                predictions.append(0.05)  # minimal occupancy
                continue
            
            # Use hourly schedule with interpolation
            hour = future_time.hour + future_time.minute / 60
            hour_floor = int(hour) % 24
            hour_ceil = (hour_floor + 1) % 24
            frac = hour - hour_floor
            
            prob = (1 - frac) * self.schedule[hour_floor] + frac * self.schedule[hour_ceil]
            predictions.append(prob)
        
        return predictions


def estimate_solar_gain(
    hour: float,
    peak_gain: float = 500.0,
    peak_hour: float = 14.0,
) -> float:
    """
    Estimate solar heat gain through windows.
    Simple Gaussian model centered on peak_hour.
    """
    if hour < 6 or hour > 20:  # nighttime
        return 0.0
    
    # Gaussian centered at peak_hour, ~4 hour standard deviation
    gain = peak_gain * math.exp(-0.5 * ((hour - peak_hour) / 4) ** 2)
    return max(0.0, gain)


class AdvancedMPC:
    """
    Production-grade Model Predictive Controller.
    
    Features:
    - Multi-step variable control trajectory
    - Weather and occupancy forecasting
    - Time-of-use electricity pricing
    - Pre-cooling optimization
    - Constraint handling with soft penalties
    - Warm-starting from previous solution
    """
    
    def __init__(self, config: MPCAdvancedConfig | None = None):
        self.config = config or MPCAdvancedConfig()
        self.weather = WeatherForecast()
        self.occupancy = OccupancyPredictor(
            self.config.occupancy_schedule,
            utc_offset=1.0
        )
        
        # Warm-start: previous solution
        self._last_trajectory: np.ndarray | None = None
        self._last_cost: float = float('inf')
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "total_energy_kwh": 0.0,
            "total_cost": 0.0,
            "avg_comfort_error": 0.0,
            "precool_events": 0,
        }
    
    def _get_setpoint_trajectory(
        self,
        occupancy_forecast: list[float],
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate setpoint trajectory based on occupancy forecast.
        Includes pre-cooling logic.
        """
        cfg = self.config
        setpoints = np.zeros(n_steps)
        dt_hours = cfg.dt_sec / 3600
        
        for i in range(n_steps):
            occ = occupancy_forecast[i] if i < len(occupancy_forecast) else 0.0
            
            # Look ahead for pre-cooling
            precool_steps = int(cfg.precool_hours / dt_hours)
            future_occ = 0.0
            if i + precool_steps < len(occupancy_forecast):
                future_occ = max(occupancy_forecast[i:i + precool_steps])
            
            # Determine setpoint
            if occ > 0.5:
                # Occupied: tight control
                setpoints[i] = cfg.setpoint_occupied
            elif future_occ > 0.5:
                # Pre-cooling: prepare for occupancy
                setpoints[i] = cfg.setpoint_occupied + cfg.precool_target_offset
            elif occ > 0.1:
                # Partially occupied
                setpoints[i] = cfg.setpoint_occupied + 1.0
            else:
                # Unoccupied
                now = datetime.now(timezone.utc)
                hour = (now.hour + i * dt_hours) % 24
                if 6 <= hour <= 22:
                    setpoints[i] = cfg.setpoint_unoccupied
                else:
                    setpoints[i] = cfg.setpoint_night
        
        return setpoints
    
    def _get_electricity_rates(self, n_steps: int) -> np.ndarray:
        """Get electricity rates for each step in horizon."""
        cfg = self.config
        rates = np.zeros(n_steps)
        dt_hours = cfg.dt_sec / 3600
        now = datetime.now(timezone.utc)
        
        for i in range(n_steps):
            hour = int((now.hour + i * dt_hours) % 24)
            rates[i] = cfg.electricity_rates[hour]
        
        return rates
    
    def _simulate_trajectory(
        self,
        control_trajectory: np.ndarray,
        T_in_start: float,
        T_out_forecast: list[float],
        Q_internal_forecast: list[float],
        solar_forecast: list[float],
    ) -> np.ndarray:
        """
        Simulate building response to a control trajectory.
        Returns predicted indoor temperature trajectory.
        """
        cfg = self.config
        n_steps = len(control_trajectory)
        T_trajectory = np.zeros(n_steps + 1)
        T_trajectory[0] = T_in_start
        
        for i in range(n_steps):
            T_out = T_out_forecast[i] if i < len(T_out_forecast) else T_out_forecast[-1]
            Q_int = Q_internal_forecast[i] if i < len(Q_internal_forecast) else 500.0
            Q_solar = solar_forecast[i] if i < len(solar_forecast) else 0.0
            
            # Total internal gains
            Q_total_internal = Q_int + Q_solar
            
            T_new, _ = step_rc(
                T_trajectory[i],
                T_out,
                cfg.R,
                cfg.C,
                Q_total_internal,
                control_trajectory[i],
                cfg.dt_sec,
            )
            T_trajectory[i + 1] = T_new
        
        return T_trajectory[1:]  # exclude initial state
    
    def _compute_cost(
        self,
        control_trajectory: np.ndarray,
        T_trajectory: np.ndarray,
        setpoint_trajectory: np.ndarray,
        electricity_rates: np.ndarray,
        occupancy_forecast: list[float],
    ) -> tuple[float, dict]:
        """
        Compute total cost for a control trajectory.
        Returns (total_cost, cost_breakdown).
        """
        cfg = self.config
        n_steps = len(control_trajectory)
        
        # Comfort cost (weighted by occupancy)
        comfort_errors = T_trajectory - setpoint_trajectory
        occupancy_weights = np.array([
            occupancy_forecast[i] if i < len(occupancy_forecast) else 0.0
            for i in range(n_steps)
        ])
        # Higher penalty when occupied
        comfort_weights = 0.1 + 0.9 * occupancy_weights
        comfort_cost = cfg.weight_comfort * np.sum(
            comfort_weights * comfort_errors ** 2
        )
        
        # Energy cost (time-of-use pricing)
        energy_kwh = control_trajectory * cfg.dt_sec / 3600 / 1000  # W*s -> kWh
        energy_cost = cfg.weight_energy * np.sum(energy_kwh * electricity_rates)
        
        # Peak demand cost (penalize high instantaneous power)
        peak_demand_kw = np.max(control_trajectory) / 1000
        demand_cost = cfg.weight_demand * peak_demand_kw ** 2
        
        # Rate of change cost (smooth operation)
        rate_changes = np.diff(control_trajectory)
        rate_cost = cfg.weight_rate_change * np.sum(rate_changes ** 2) / 1e6
        
        # Constraint violation penalties (soft constraints)
        constraint_cost = 0.0
        # Over capacity
        over_cap = np.maximum(0, control_trajectory - cfg.max_cooling)
        constraint_cost += 1000 * np.sum(over_cap ** 2)
        # Under minimum
        under_min = np.maximum(0, cfg.min_cooling - control_trajectory)
        constraint_cost += 1000 * np.sum(under_min ** 2)
        # Rate limit violations
        rate_violations = np.maximum(0, np.abs(rate_changes) - cfg.max_rate_change)
        constraint_cost += 100 * np.sum(rate_violations ** 2)
        
        total_cost = comfort_cost + energy_cost + demand_cost + rate_cost + constraint_cost
        
        breakdown = {
            "comfort": comfort_cost,
            "energy": energy_cost,
            "demand": demand_cost,
            "rate_change": rate_cost,
            "constraints": constraint_cost,
            "total": total_cost,
            "avg_error": np.mean(np.abs(comfort_errors)),
            "max_error": np.max(np.abs(comfort_errors)),
            "energy_kwh": np.sum(energy_kwh),
            "peak_kw": peak_demand_kw,
        }
        
        return total_cost, breakdown
    
    def _optimize_grid(
        self,
        T_in: float,
        T_out_forecast: list[float],
        Q_internal_forecast: list[float],
        solar_forecast: list[float],
        setpoint_trajectory: np.ndarray,
        electricity_rates: np.ndarray,
        occupancy_forecast: list[float],
    ) -> tuple[np.ndarray, dict]:
        """
        Grid search optimization (simpler but less efficient).
        Good for debugging and small problems.
        """
        cfg = self.config
        n_control = cfg.control_horizon
        n_pred = cfg.prediction_horizon
        
        # Extend control to prediction horizon (hold last value)
        def extend_control(ctrl):
            extended = np.zeros(n_pred)
            extended[:len(ctrl)] = ctrl
            extended[len(ctrl):] = ctrl[-1]
            return extended
        
        best_cost = float('inf')
        best_control = np.zeros(n_control)
        best_breakdown = {}
        
        # Simplified: optimize average level + trend
        # (Full grid over n_control dimensions would be intractable)
        for level in np.linspace(0, cfg.max_cooling, cfg.grid_resolution):
            for trend in np.linspace(-0.3, 0.3, 7):  # -30% to +30% change over horizon
                control = np.array([
                    np.clip(
                        level * (1 + trend * i / n_control),
                        cfg.min_cooling,
                        cfg.max_cooling
                    )
                    for i in range(n_control)
                ])
                
                extended = extend_control(control)
                T_traj = self._simulate_trajectory(
                    extended, T_in, T_out_forecast, Q_internal_forecast, solar_forecast
                )
                
                cost, breakdown = self._compute_cost(
                    extended, T_traj, setpoint_trajectory,
                    electricity_rates, occupancy_forecast
                )
                
                if cost < best_cost:
                    best_cost = cost
                    best_control = control.copy()
                    best_breakdown = breakdown
        
        return best_control, best_breakdown
    
    def _optimize_scipy(
        self,
        T_in: float,
        T_out_forecast: list[float],
        Q_internal_forecast: list[float],
        solar_forecast: list[float],
        setpoint_trajectory: np.ndarray,
        electricity_rates: np.ndarray,
        occupancy_forecast: list[float],
    ) -> tuple[np.ndarray, dict]:
        """
        Scipy-based optimization (SLSQP or L-BFGS-B).
        More efficient for larger problems.
        """
        from scipy.optimize import minimize
        
        cfg = self.config
        n_control = cfg.control_horizon
        n_pred = cfg.prediction_horizon
        
        def extend_control(ctrl):
            extended = np.zeros(n_pred)
            extended[:len(ctrl)] = ctrl
            extended[len(ctrl):] = ctrl[-1]
            return extended
        
        def objective(control_flat):
            control = np.clip(control_flat, cfg.min_cooling, cfg.max_cooling)
            extended = extend_control(control)
            T_traj = self._simulate_trajectory(
                extended, T_in, T_out_forecast, Q_internal_forecast, solar_forecast
            )
            cost, _ = self._compute_cost(
                extended, T_traj, setpoint_trajectory,
                electricity_rates, occupancy_forecast
            )
            return cost
        
        # Initial guess: warm start from previous solution or heuristic
        if self._last_trajectory is not None:
            # Shift previous solution forward
            x0 = np.zeros(n_control)
            x0[:-1] = self._last_trajectory[1:]
            x0[-1] = self._last_trajectory[-1]
        else:
            # Heuristic: proportional to error
            error = T_in - setpoint_trajectory[0]
            x0 = np.full(n_control, np.clip(error * 2000, 0, cfg.max_cooling))
        
        bounds = [(cfg.min_cooling, cfg.max_cooling)] * n_control
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': cfg.max_iterations, 'ftol': 1e-6}
        )
        
        best_control = np.clip(result.x, cfg.min_cooling, cfg.max_cooling)
        extended = extend_control(best_control)
        T_traj = self._simulate_trajectory(
            extended, T_in, T_out_forecast, Q_internal_forecast, solar_forecast
        )
        _, breakdown = self._compute_cost(
            extended, T_traj, setpoint_trajectory,
            electricity_rates, occupancy_forecast
        )
        
        return best_control, breakdown
    
    def compute(
        self,
        T_in: float,
        T_out: float,
        Q_internal: float,
        occupied: bool,
        sim_time_sec: float = 0.0,
    ) -> tuple[float, dict]:
        """
        Main MPC computation. Returns (cooling_output_watts, diagnostics).
        
        This is called every control timestep. It:
        1. Updates forecasts with current observations
        2. Generates setpoint trajectory
        3. Optimizes control trajectory
        4. Returns first control action (receding horizon)
        """
        cfg = self.config
        self.stats["total_calls"] += 1
        
        # Update weather model with observation
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60
        self.weather.update_current(T_out, hour)
        
        # Get forecasts
        hours_ahead = cfg.prediction_horizon * cfg.dt_sec / 3600
        dt_hours = cfg.dt_sec / 3600
        
        T_out_forecast = self.weather.get_forecast(hours_ahead, dt_hours)
        occupancy_forecast = self.occupancy.predict(hours_ahead, dt_hours)
        
        # Solar gain forecast
        solar_forecast = []
        for i in range(cfg.prediction_horizon):
            h = (hour + i * dt_hours) % 24
            solar_forecast.append(estimate_solar_gain(h, cfg.solar_gain_peak, cfg.solar_peak_hour))
        
        # Internal gains based on occupancy
        Q_internal_forecast = [
            cfg.C * 0.00005 if occ < 0.1 else Q_internal  # minimal when empty
            for occ in occupancy_forecast[:cfg.prediction_horizon]
        ]
        
        # Generate setpoint trajectory
        setpoint_trajectory = self._get_setpoint_trajectory(
            occupancy_forecast, cfg.prediction_horizon
        )
        
        # Electricity rates
        electricity_rates = self._get_electricity_rates(cfg.prediction_horizon)
        
        # Optimize
        if cfg.optimizer == "scipy":
            try:
                control_trajectory, breakdown = self._optimize_scipy(
                    T_in, T_out_forecast, Q_internal_forecast, solar_forecast,
                    setpoint_trajectory, electricity_rates, occupancy_forecast
                )
            except ImportError:
                # Fallback if scipy not available
                control_trajectory, breakdown = self._optimize_grid(
                    T_in, T_out_forecast, Q_internal_forecast, solar_forecast,
                    setpoint_trajectory, electricity_rates, occupancy_forecast
                )
        else:
            control_trajectory, breakdown = self._optimize_grid(
                T_in, T_out_forecast, Q_internal_forecast, solar_forecast,
                setpoint_trajectory, electricity_rates, occupancy_forecast
            )
        
        # Save for warm-start
        self._last_trajectory = control_trajectory
        self._last_cost = breakdown["total"]
        
        # Update statistics
        cooling_kw = control_trajectory[0] / 1000
        self.stats["total_energy_kwh"] += cooling_kw * cfg.dt_sec / 3600
        self.stats["total_cost"] += breakdown["energy"] / cfg.weight_energy
        
        # Detect pre-cooling
        current_setpoint = setpoint_trajectory[0]
        if current_setpoint < cfg.setpoint_occupied and not occupied:
            self.stats["precool_events"] += 1
            breakdown["mode"] = "pre-cooling"
        elif occupied:
            breakdown["mode"] = "occupied"
        else:
            breakdown["mode"] = "setback"
        
        # Diagnostics
        diagnostics = {
            **breakdown,
            "setpoint": current_setpoint,
            "T_out_forecast": T_out_forecast[:5],  # next 5 steps
            "occupancy_forecast": occupancy_forecast[:5],
            "electricity_rate": electricity_rates[0],
            "control_trajectory": control_trajectory[:5].tolist(),
            "optimizer": cfg.optimizer,
        }
        
        # Return first action (receding horizon principle)
        return float(control_trajectory[0]), diagnostics


# Factory function for easy integration
def create_advanced_mpc(
    max_cooling: float = 10000.0,
    R: float = 0.01,
    C: float = 1e4,
    **kwargs
) -> AdvancedMPC:
    """Create an AdvancedMPC with custom configuration."""
    config = MPCAdvancedConfig(
        max_cooling=max_cooling,
        R=R,
        C=C,
        **kwargs
    )
    return AdvancedMPC(config)
