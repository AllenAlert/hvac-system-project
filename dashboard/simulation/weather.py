"""
Outdoor temperature module.

Provides synthetic weather (sine wave) or real weather from Open-Meteo API.

"""
import math
import time
from datetime import datetime, timezone
from typing import Literal


# ============ synthetic weather ============
# just a sine wave, peaks around 3pm, coldest around 6am

def synthetic_outdoor_T_degC(
    hour_of_day: float,
    day_of_year: int = 180,
    T_avg_degC: float = 25.0,
    amplitude_degC: float = 5.0,
    latitude_approx: float = 45.0,  # not actually used rn, TODO
) -> float:
    """
    Fake outdoor temp based on time of day and rough season.
    """
    # peak at ~3pm (hour 15), min at ~6am (hour 6)
    daily = math.sin(2 * math.pi * (hour_of_day - 6) / 24)
    
    # seasonal adjustment - warmer in summer (day 172 ish)
    seasonal = math.sin(2 * math.pi * (day_of_year - 80) / 365) * 10
    
    T = T_avg_degC + amplitude_degC * daily + 0.3 * seasonal
    return round(T, 2)


def get_synthetic_T_now() -> float:
    now = datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0 + now.second / 3600.0
    doy = now.timetuple().tm_yday
    return synthetic_outdoor_T_degC(hour, doy, T_avg_degC=24.0, amplitude_degC=4.0)


# ============ real weather API ============
# using open-meteo because it's free and doesn't need an API key
# (weatherapi and openweather both need registration)

_WEATHER_CACHE: dict[str, tuple[float, float]] = {}  # key -> (temp, fetch_time)
_CACHE_SEC = 600  # cache for 10 min, don't want to hammer the API


def fetch_weather_api(lat: float = 6.5244, lon: float = 3.3792) -> float | None:
    """Fetch current temp from Open-Meteo. Returns degC or None if it fails."""
    import urllib.request
    import json
    
    key = f"{lat},{lon}"
    now = time.time()
    
    # check cache first
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
        # network error, timeout, whatever - just return None
        # caller should fall back to synthetic
        return None


def get_outdoor_T(source: Literal["synthetic", "api"], lat: float = 6.5244, lon: float = 3.3792) -> float:
    """Get current outdoor temperature. Fallback to synthetic if API fails."""
    if source == "api":
        T = fetch_weather_api(lat, lon)
        if T is not None:
            return T
    return get_synthetic_T_now()
