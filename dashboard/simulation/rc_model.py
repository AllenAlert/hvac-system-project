"""
RC (resistor-capacitor) thermal model for building simulation.

Simplified single-zone model:
- R = thermal resistance (how well insulated)
- C = thermal capacitance (thermal mass)

"""


def step_rc(
    T_in: float,
    T_out: float,
    R: float,
    C: float,
    Q_internal: float,
    Q_cooling: float,
    dt_sec: float,
    Q_heating: float = 0.0,
) -> tuple[float, float]:
    """
    Advance the model by one timestep.
    Returns (new indoor temp, rate of change).
    
    Parameters:
    - T_in: Current indoor temperature (°C)
    - T_out: Outdoor temperature (°C)
    - R: Thermal resistance (K/W)
    - C: Thermal capacitance (J/K)
    - Q_internal: Internal heat gains (W) - people, equipment, lights
    - Q_cooling: Cooling power (W) - removes heat
    - dt_sec: Timestep (seconds)
    - Q_heating: Heating power (W) - adds heat
    """
    # Heat balance equation:
    # C * dT/dt = (T_out - T_in)/R + Q_internal + Q_heating - Q_cooling
    # 
    # Positive terms add heat (raise temperature):
    #   - Heat flow from outside when T_out > T_in
    #   - Internal gains (people, equipment)
    #   - Heating system
    # Negative terms remove heat (lower temperature):
    #   - Heat flow to outside when T_out < T_in  
    #   - Cooling system
    
    dT_dt = (1.0 / C) * ((T_out - T_in) / R + Q_internal + Q_heating - Q_cooling)
    T_in_new = T_in + dT_dt * dt_sec
    
    return T_in_new, dT_dt
