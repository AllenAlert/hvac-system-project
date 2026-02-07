
import math
from dataclasses import dataclass, field
from typing import Literal


# ============ Heating System Types ============

class HeatingType:
    """Heating system types with their typical efficiencies."""
    GAS_FURNACE = "gas_furnace"           # 80-98% AFUE
    ELECTRIC_RESISTANCE = "electric"       # 100% efficient (but expensive)
    HEAT_PUMP_AIR = "heat_pump_air"        # COP 2.5-4.0
    HEAT_PUMP_GROUND = "heat_pump_ground"  # COP 3.5-5.0
    BOILER_GAS = "boiler_gas"             # 80-95% AFUE
    BOILER_OIL = "boiler_oil"             # 80-90% AFUE

HEATING_EFFICIENCY = {
    HeatingType.GAS_FURNACE: 0.92,
    HeatingType.ELECTRIC_RESISTANCE: 1.0,
    HeatingType.HEAT_PUMP_AIR: 3.0,       # COP (varies with outdoor temp)
    HeatingType.HEAT_PUMP_GROUND: 4.0,    # COP
    HeatingType.BOILER_GAS: 0.90,
    HeatingType.BOILER_OIL: 0.85,
}

# Fuel costs ($/kWh equivalent)
FUEL_COSTS = {
    HeatingType.GAS_FURNACE: 0.04,
    HeatingType.ELECTRIC_RESISTANCE: 0.12,
    HeatingType.HEAT_PUMP_AIR: 0.12,
    HeatingType.HEAT_PUMP_GROUND: 0.12,
    HeatingType.BOILER_GAS: 0.04,
    HeatingType.BOILER_OIL: 0.06,
}


@dataclass
class RuleBasedConfig:
    setpoint_occupied: float = 22.0  # deg C
    setpoint_unoccupied: float = 18.0
    deadband: float = 1.0  # prevents short-cycling (on-off-on-off...)


@dataclass
class PIDConfig:
    Kp: float = 2000.0    # W per °C error - aggressive proportional
    Ki: float = 50.0      # W per °C·s - integral builds up over ~minutes
    Kd: float = 100.0     # W per °C/s - dampens oscillation
    setpoint_occupied: float = 22.0
    setpoint_unoccupied: float = 18.0


@dataclass
class MPCConfig:
    # model predictive control - looks ahead and optimizes
    # honestly this is pretty simplified compared to real MPC
    horizon_steps: int = 6  # 6 * 60s = 6 min lookahead
    weight_comfort: float = 1.0  # how much we care about temp error
    weight_energy: float = 0.1   # how much we care about energy use
    setpoint_occupied: float = 22.0
    setpoint_unoccupied: float = 18.0


def rule_based_hvac(
    T_in: float,
    setpoint: float,
    deadband: float,
    current_heating: float,
    current_cooling: float,
    max_heating: float,
    max_cooling: float,
) -> tuple[float, float]:
    """
    Classic bang-bang thermostat with hysteresis for both heating and cooling.
    Returns (heating_output, cooling_output) in Watts.
    
    The deadband prevents rapid cycling which would damage equipment.
    There's also a neutral zone between heating and cooling.
    """
    # Define temperature bands
    cool_on = setpoint + deadband      # start cooling above this
    cool_off = setpoint + deadband / 2  # stop cooling below this
    heat_on = setpoint - deadband       # start heating below this
    heat_off = setpoint - deadband / 2  # stop heating above this
    
    heating = 0.0
    cooling = 0.0
    
    # Cooling logic
    if T_in >= cool_on:
        cooling = max_cooling
    elif T_in <= cool_off:
        cooling = 0.0
    else:
        # In deadband - maintain current state
        cooling = current_cooling
    
    # Heating logic
    if T_in <= heat_on:
        heating = max_heating
    elif T_in >= heat_off:
        heating = 0.0
    else:
        # In deadband - maintain current state
        heating = current_heating
    
    # Safety: never heat and cool at the same time
    if heating > 0 and cooling > 0:
        # Temperature is somehow in both bands (shouldn't happen with proper deadband)
        # Prefer the one that moves us toward setpoint
        if T_in > setpoint:
            heating = 0.0
        else:
            cooling = 0.0
    
    return heating, cooling


def rule_based(
    T_in: float,
    setpoint: float,
    deadband: float,
    cooling_current: float,
    max_cooling: float,
) -> float:
    """
    Legacy function for backward compatibility - cooling only.
    """
    _, cooling = rule_based_hvac(
        T_in, setpoint, deadband,
        0.0, cooling_current,
        0.0, max_cooling
    )
    return cooling


class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._integral = 0.0
        self._last_error: float | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None
        self._last_t = None

    def compute(
        self,
        T_in: float,
        setpoint: float,
        t: float,
        max_cooling: float,
    ) -> float:
        """Legacy cooling-only compute for backward compatibility."""
        _, cooling = self.compute_hvac(T_in, setpoint, t, 0.0, max_cooling)
        return cooling
    
    def compute_hvac(
        self,
        T_in: float,
        setpoint: float,
        t: float,
        max_heating: float,
        max_cooling: float,
    ) -> tuple[float, float]:
        """
        Compute both heating and cooling outputs.
        Returns (heating_output, cooling_output) in Watts.
        
        Positive error (T_in > setpoint) = too hot = need cooling
        Negative error (T_in < setpoint) = too cold = need heating
        """
        error = T_in - setpoint
        dt = 0.0
        if self._last_t is not None:
            dt = t - self._last_t
        if dt <= 0:
            dt = 60.0  # assume 60 s if first call

        self._integral += error * dt
        # Anti-windup: limit integral to prevent overshoot
        # At Ki=50, max integral contribution = 50 * 200 = 10000W (full capacity)
        self._integral = max(-200, min(200, self._integral))

        derivative = 0.0
        if self._last_error is not None and dt > 0:
            derivative = (error - self._last_error) / dt

        self._last_error = error
        self._last_t = t

        # PID output
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        
        # Split output into heating and cooling
        # u > 0: need cooling (temperature too high)
        # u < 0: need heating (temperature too low)
        heating = 0.0
        cooling = 0.0
        
        if u > 0:
            cooling = min(u, max_cooling)
        elif u < 0:
            heating = min(-u, max_heating)
        
        return heating, cooling


def mpc_simple(
    T_in: float,
    T_out: float,
    R: float,
    C: float,
    Q_internal: float,
    setpoint: float,
    horizon_steps: int,
    dt_sec: float,
    weight_comfort: float,
    weight_energy: float,
    max_cooling: float,
) -> float:
    """
    Legacy cooling-only MPC for backward compatibility.
    """
    _, cooling = mpc_simple_hvac(
        T_in, T_out, R, C, Q_internal, setpoint,
        horizon_steps, dt_sec, weight_comfort, weight_energy,
        0.0, max_cooling
    )
    return cooling


def mpc_simple_hvac(
    T_in: float,
    T_out: float,
    R: float,
    C: float,
    Q_internal: float,
    setpoint: float,
    horizon_steps: int,
    dt_sec: float,
    weight_comfort: float,
    weight_energy: float,
    max_heating: float,
    max_cooling: float,
    heating_efficiency: float = 0.92,
    cooling_cop: float = 3.0,
) -> tuple[float, float]:
    """
    Simplified MPC for both heating and cooling.
    Grid search over heating/cooling combinations.
    Returns (heating_output, cooling_output) in Watts.
    
    Parameters:
    - heating_efficiency: Furnace/boiler efficiency (0.8-1.0)
    - cooling_cop: Cooling coefficient of performance (2.5-5.0)
    """
    from .rc_model import step_rc

    best_heating = 0.0
    best_cooling = 0.0
    best_cost = 1e30
    
    # Discrete options for heating and cooling
    n_options = 11
    
    for i in range(n_options):
        for j in range(n_options):
            # Don't heat and cool simultaneously
            if i > 0 and j > 0:
                continue
                
            Q_heat = (i / (n_options - 1)) * max_heating if max_heating > 0 else 0.0
            Q_cool = (j / (n_options - 1)) * max_cooling if max_cooling > 0 else 0.0
            
            T = T_in
            cost = 0.0
            
            for step in range(horizon_steps):
                T, _ = step_rc(T, T_out, R, C, Q_internal, Q_cool, dt_sec, Q_heat)
                error = T - setpoint
                
                # Comfort cost
                cost += weight_comfort * (error ** 2)
                
                # Energy cost (account for efficiency)
                # Heating: actual fuel use = output / efficiency
                # Cooling: electricity use = output / COP
                heating_energy = Q_heat / heating_efficiency if heating_efficiency > 0 else Q_heat
                cooling_energy = Q_cool / cooling_cop if cooling_cop > 0 else Q_cool
                cost += weight_energy * (heating_energy + cooling_energy)
            
            if cost < best_cost:
                best_cost = cost
                best_heating = Q_heat
                best_cooling = Q_cool
    
    return best_heating, best_cooling
