
import math
from dataclasses import dataclass, field
from typing import Literal


class HeatingType:
    GAS_FURNACE = "gas_furnace"
    ELECTRIC_RESISTANCE = "electric"
    HEAT_PUMP_AIR = "heat_pump_air"
    HEAT_PUMP_GROUND = "heat_pump_ground"
    BOILER_GAS = "boiler_gas"
    BOILER_OIL = "boiler_oil"

HEATING_EFFICIENCY = {
    HeatingType.GAS_FURNACE: 0.92,
    HeatingType.ELECTRIC_RESISTANCE: 1.0,
    HeatingType.HEAT_PUMP_AIR: 3.0,
    HeatingType.HEAT_PUMP_GROUND: 4.0,
    HeatingType.BOILER_GAS: 0.90,
    HeatingType.BOILER_OIL: 0.85,
}

class ElectricityBand:
    BAND_A = "band_a"
    BAND_B = "band_b"
    BAND_C = "band_c"
    BAND_D = "band_d"
    BAND_E = "band_e"

ELECTRICITY_RATES_NGN = {
    ElectricityBand.BAND_A: 225.0,
    ElectricityBand.BAND_B: 63.0,
    ElectricityBand.BAND_C: 50.0,
    ElectricityBand.BAND_D: 43.0,
    ElectricityBand.BAND_E: 40.0,
}

FUEL_COSTS = {
    HeatingType.GAS_FURNACE: 80.0,
    HeatingType.ELECTRIC_RESISTANCE: 225.0,
    HeatingType.HEAT_PUMP_AIR: 225.0,
    HeatingType.HEAT_PUMP_GROUND: 225.0,
    HeatingType.BOILER_GAS: 80.0,
    HeatingType.BOILER_OIL: 150.0,
}


@dataclass
class RuleBasedConfig:
    setpoint_occupied: float = 22.0
    setpoint_unoccupied: float = 18.0
    deadband: float = 1.0


@dataclass
class PIDConfig:
    Kp: float = 2000.0
    Ki: float = 50.0
    Kd: float = 100.0
    setpoint_occupied: float = 22.0
    setpoint_unoccupied: float = 18.0


@dataclass
class MPCConfig:
    horizon_steps: int = 6
    weight_comfort: float = 1.0
    weight_energy: float = 0.1
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
    cool_on = setpoint + deadband
    cool_off = setpoint - deadband / 2
    heat_on = setpoint - deadband
    heat_off = setpoint + deadband / 2
    
    heating = 0.0
    cooling = 0.0
    
    if current_cooling > 0:
        if T_in <= cool_off:
            cooling = 0.0
        else:
            cooling = max_cooling
    else:
        if T_in >= cool_on:
            cooling = max_cooling
        else:
            cooling = 0.0
    
    if current_heating > 0:
        if T_in >= heat_off:
            heating = 0.0
        else:
            heating = max_heating
    else:
        if T_in <= heat_on:
            heating = max_heating
        else:
            heating = 0.0
    
    if heating > 0 and cooling > 0:
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
            dt = 60.0

        self._integral += error * dt
        self._integral = max(-200, min(200, self._integral))

        derivative = 0.0
        if self._last_error is not None and dt > 0:
            derivative = (error - self._last_error) / dt

        self._last_error = error
        self._last_t = t

        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        
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
    
    n_options = 11
    
    for i in range(n_options):
        for j in range(n_options):
            if i > 0 and j > 0:
                continue
                
            Q_heat = (i / (n_options - 1)) * max_heating if max_heating > 0 else 0.0
            Q_cool = (j / (n_options - 1)) * max_cooling if max_cooling > 0 else 0.0
            
            T = T_in
            cost = 0.0
            
            for step in range(horizon_steps):
                T, _ = step_rc(T, T_out, R, C, Q_internal, Q_cool, dt_sec, Q_heat)
                error = T - setpoint
                
                cost += weight_comfort * (error ** 2)
                
                heating_energy = Q_heat / heating_efficiency if heating_efficiency > 0 else Q_heat
                cooling_energy = Q_cool / cooling_cop if cooling_cop > 0 else Q_cool
                cost += weight_energy * (heating_energy + cooling_energy)
            
            if cost < best_cost:
                best_cost = cost
                best_heating = Q_heat
                best_cooling = Q_cool
    
    return best_heating, best_cooling
