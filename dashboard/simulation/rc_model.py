"""
RC (resistor-capacitor) thermal model for building simulation.

Simplified single-zone model:
- R = thermal resistance (how well insulated)
- C = thermal capacitance (thermal mass)

Good for demonstrating control strategies and quick simulations.

@author: Bola
"""


def step_rc(
    T_in: float,
    T_out: float,
    R: float,
    C: float,
    Q_internal: float,
    Q_cooling: float,
    dt_sec: float,
) -> tuple[float, float]:
    """
    Advance the model by one timestep.
    Returns (new indoor temp, rate of change).
    """
    # basic heat balance:
    # C * dT/dt = (T_out - T_in)/R + Q_internal - Q_cooling
    # 
    # positive Q_cooling = removing heat = cooling
    # negative would be heating (but we just set Q_cooling=0 and let it drift)
    
    dT_dt = (1.0 / C) * ((T_out - T_in) / R + Q_internal - Q_cooling)
    T_in_new = T_in + dT_dt * dt_sec
    
    # DEBUG:
    # if abs(dT_dt) > 0.1:
    #     print(f"big dT: {dT_dt:.3f} K/s, T_in={T_in:.1f}, T_out={T_out:.1f}")
    
    return T_in_new, dT_dt
