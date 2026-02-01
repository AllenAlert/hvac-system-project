"""
Control strategies for the HVAC simulation.

Implements:
 - rule_based: simple thermostat with deadband (what most real buildings use)
 - PID: classic control theory approach
 - MPC: model predictive control (looks ahead and optimizes)

@author: Bola
"""
import math
from dataclasses import dataclass, field


@dataclass
class RuleBasedConfig:
    setpoint_occupied: float = 22.0  # deg C
    setpoint_unoccupied: float = 18.0
    deadband: float = 1.0  # prevents short-cycling (on-off-on-off...)


@dataclass
class PIDConfig:
    Kp: float = 0.5
    Ki: float = 0.05
    Kd: float = 0.01
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


def rule_based(
    T_in: float,
    setpoint: float,
    deadband: float,
    cooling_current: float,
    max_cooling: float,
) -> float:
    """
    Classic bang-bang thermostat with hysteresis.
    The deadband prevents rapid cycling which would kill the compressor.
    """
    high = setpoint + deadband / 2.0
    low = setpoint - deadband / 2.0
    
    # above threshold -> full cooling
    if T_in >= high:
        return max_cooling
    # below threshold -> off
    if T_in <= low:
        return 0.0
    # in the deadband - keep doing whatever we were doing
    # (this is the hysteresis part)
    return cooling_current if cooling_current > 0 else 0.0


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
        error = T_in - setpoint  # positive = too hot, need more cooling
        dt = 0.0
        if self._last_t is not None:
            dt = t - self._last_t
        if dt <= 0:
            dt = 60.0  # assume 60 s if first call

        self._integral += error * dt
        self._integral = max(-10000, min(10000, self._integral))  # anti-windup

        derivative = 0.0
        if self._last_error is not None and dt > 0:
            derivative = (error - self._last_error) / dt

        self._last_error = error
        self._last_t = t

        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        # u positive = need cooling; clamp to [0, max_cooling]
        cooling = max(0.0, min(max_cooling, u))
        return cooling


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
    Simplified MPC: grid search over constant cooling level for horizon.
    Minimizes sum of (comfort error)^2 * weight_comfort + cooling * weight_energy.
    Returns first-step cooling (W).
    """
    from .rc_model import step_rc

    best_cooling = 0.0
    best_cost = 1e30
    # Discrete options for cooling
    n_options = 11
    for i in range(n_options):
        Q_cool = (i / (n_options - 1)) * max_cooling
        T = T_in
        cost = 0.0
        for step in range(horizon_steps):
            T, _ = step_rc(T, T_out, R, C, Q_internal, Q_cool, dt_sec)
            error = T - setpoint
            cost += weight_comfort * (error ** 2) + weight_energy * Q_cool
        if cost < best_cost:
            best_cost = cost
            best_cooling = Q_cool
    return best_cooling
