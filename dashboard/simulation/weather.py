import math
import time
from datetime import datetime, timezone
from typing import Literal


def synthetic_outdoor_T_degC(
    hour_of_day: float,
    day_of_year: int = 180,
    T_avg_degC: float = 25.0,
    amplitude_degC: float = 5.0,
    latitude_approx: float = 45.0,
) -> float:
    daily = math.sin(2 * math.pi * (hour_of_day - 6) / 24)
    seasonal = math.sin(2 * math.pi * (day_of_year - 80) / 365) * 10
    T = T_avg_degC + amplitude_degC * daily + 0.3 * seasonal
    return round(T, 2)


def get_synthetic_T_now() -> float:
    now = datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0 + now.second / 3600.0
    doy = now.timetuple().tm_yday
    return synthetic_outdoor_T_degC(hour, doy, T_avg_degC=24.0, amplitude_degC=4.0)


_WEATHER_CACHE: dict[str, tuple[float, float]] = {}
_CACHE_SEC = 600


def fetch_weather_api(lat: float = 6.5244, lon: float = 3.3792) -> float | None:
    import urllib.request
    import json
    
    key = f"{lat},{lon}"
    now = time.time()
    
    if key in _WEATHER_CACHE:
        t, ts = _WEATHER_CACHE[key]
        if now - ts < _CACHE_SEC:
            return t
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        T = float(data["current"]["temperature_2m"])
        _WEATHER_CACHE[key] = (T, now)
        return T
    except Exception:
        return None


def get_outdoor_T(source: Literal["synthetic", "api"], lat: float = 6.5244, lon: float = 3.3792) -> float:
    if source == "api":
        T = fetch_weather_api(lat, lon)
        if T is not None:
            return T
    return get_synthetic_T_now()
