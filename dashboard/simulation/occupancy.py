from datetime import datetime, timezone
from typing import Literal


def is_business_hours(utc_offset_hours: float = 1.0) -> bool:
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    local = now + timedelta(hours=utc_offset_hours)
    h = local.hour + local.minute / 60.0
    return 8.0 <= h < 18.0


def get_setpoint(occupied: bool, setpoint_occupied: float, setpoint_unoccupied: float) -> float:
    return setpoint_occupied if occupied else setpoint_unoccupied
