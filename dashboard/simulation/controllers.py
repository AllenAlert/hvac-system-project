
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

# ============ Nigerian Electricity Tariff Bands (NERC 2024) ============
# Rates in Naira per kWh
class ElectricityBand:
    """Nigerian NERC electricity tariff bands based on supply hours."""
    BAND_A = "band_a"  # 20+ hours supply - ₦225/kWh
    BAND_B = "band_b"  # 16-20 hours - ₦63/kWh
    BAND_C = "band_c"  # 12-16 hours - ₦50/kWh
    BAND_D = "band_d"  # 8-12 hours - ₦43/kWh
    BAND_E = "band_e"  # 4-8 hours - ₦40/kWh

ELECTRICITY_RATES_NGN = {
    ElectricityBand.BAND_A: 225.0,   # ₦225/kWh (premium - 20+ hours)
    ElectricityBand.BAND_B: 63.0,    # ₦63/kWh (16-20 hours)
    ElectricityBand.BAND_C: 50.0,    # ₦50/kWh (12-16 hours)
    ElectricityBand.BAND_D: 43.0,    # ₦43/kWh (8-12 hours)
    ElectricityBand.BAND_E: 40.0,    # ₦40/kWh (4-8 hours)
}

# Fuel costs in Naira per kWh equivalent
# Gas: ~₦80/kWh, Diesel/Oil: ~₦150/kWh (generator fuel equivalent)
FUEL_COSTS = {
    HeatingType.GAS_FURNACE: 80.0,          # ₦80/kWh (natural gas)
    HeatingType.ELECTRIC_RESISTANCE: 225.0,  # ₦225/kWh (Band A electricity)
    HeatingType.HEAT_PUMP_AIR: 225.0,        # ₦225/kWh (Band A electricity)
    HeatingType.HEAT_PUMP_GROUND: 225.0,     # ₦225/kWh (Band A electricity)
    HeatingType.BOILER_GAS: 80.0,            # ₦80/kWh (natural gas)
    HeatingType.BOILER_OIL: 150.0,           # ₦150/kWh (diesel equivalent)
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
    
    Uses proper hysteresis to prevent rapid on/off cycling:
    - Cooling: turns ON above (setpoint + deadband), stays ON until below (setpoint - deadband/2)
    - Heating: turns ON below (setpoint - deadband), stays ON until above (setpoint + deadband/2)
    """
    # Define temperature bands with proper hysteresis
    # Key insight: once ON, system should run until PAST the setpoint to avoid cycling
    cool_on = setpoint + deadband           # start cooling above this (e.g., 23°C)
    cool_off = setpoint - deadband / 2      # stop cooling below this (e.g., 21.5°C) - overshoot allowed
    heat_on = setpoint - deadband           # start heating below this (e.g., 21°C)
    heat_off = setpoint + deadband / 2      # stop heating above this (e.g., 22.5°C) - overshoot allowed
    
    heating = 0.0
    cooling = 0.0
    
    # Cooling logic - with proper hysteresis
    if current_cooling > 0:
        # Already cooling - keep cooling until we're well below setpoint
        if T_in <= cool_off:
            cooling = 0.0
        else:
            cooling = max_cooling
    else:
        # Not cooling - only start if above cool_on threshold
        if T_in >= cool_on:
            cooling = max_cooling
        else:
            cooling = 0.0
    
    # Heating logic - with proper hysteresis
    if current_heating > 0:
        # Already heating - keep heating until we're well above setpoint
        if T_in >= heat_off:
            heating = 0.0
        else:
            heating = max_heating
    else:
        # Not heating - only start if below heat_on threshold
        if T_in <= heat_on:
            heating = max_heating
        else:
            heating = 0.0
    
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
        error = T_in - setpoint
        dt = 0.0
        if self._last_t is not None:
            dt = t - self._last_t
        if dt <= 0:
            dt = 60.0  # assume 60 s if first call

        self._integral += error * dt
  
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
