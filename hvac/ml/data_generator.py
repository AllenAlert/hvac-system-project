import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from hvac import Quantity
# Simplified data generator - does not require full building modules
import random

class HVACDataGenerator:
    """Generate training data using HVAC physics calculations."""
    
    def __init__(self, location = None):
        self.location = location  # Optional location for future use
        
    def generate_building_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate dataset with building parameters and energy consumption."""
        data = []
        
        for _ in range(n_samples):
            # Random building parameters
            floor_area = random.uniform(100, 5000)  # m²
            height = random.uniform(2.5, 4.0)  # m
            window_ratio = random.uniform(0.1, 0.6)
            insulation_r = random.uniform(1.0, 6.0)  # R-value
            occupancy = random.uniform(5, 50)  # people/1000m²
            
            # Weather parameters
            outdoor_temp = random.uniform(-20, 40)  # °C
            solar_irradiance = random.uniform(0, 1000)  # W/m²
            wind_speed = random.uniform(0, 15)  # m/s
            
            # Calculate energy consumption (simplified)
            heating_load = self._calculate_heating_load(
                floor_area, height, window_ratio, insulation_r, outdoor_temp
            )
            cooling_load = self._calculate_cooling_load(
                floor_area, occupancy, solar_irradiance, outdoor_temp
            )
            
            data.append({
                'floor_area': floor_area,
                'height': height,
                'window_ratio': window_ratio,
                'insulation_r': insulation_r,
                'occupancy': occupancy,
                'outdoor_temp': outdoor_temp,
                'solar_irradiance': solar_irradiance,
                'wind_speed': wind_speed,
                'heating_load': heating_load,
                'cooling_load': cooling_load,
                'total_energy': heating_load + cooling_load
            })
            
        return pd.DataFrame(data)
    
    def _calculate_heating_load(self, area: float, height: float, 
                              window_ratio: float, r_value: float, 
                              outdoor_temp: float) -> float:
        """Simplified heating load calculation."""
        volume = area * height
        u_value = 1.0 / r_value
        temp_diff = max(0, 20 - outdoor_temp)  # Indoor target 20°C
        
        # Transmission losses
        wall_area = area * 0.4 * (1 - window_ratio)  # Simplified
        window_area = area * 0.4 * window_ratio
        
        transmission_loss = (wall_area * u_value + window_area * 3.0) * temp_diff
        
        # Ventilation losses
        ventilation_loss = volume * 0.34 * 0.5 * temp_diff  # 0.5 ACH
        
        return max(0, transmission_loss + ventilation_loss)
    
    def _calculate_cooling_load(self, area: float, occupancy: float,
                              solar_irradiance: float, outdoor_temp: float) -> float:
        """Simplified cooling load calculation."""
        if outdoor_temp < 24:  # No cooling needed
            return 0
            
        # Internal gains
        people_gain = (occupancy / 1000) * area * 100  # W
        solar_gain = area * 0.3 * solar_irradiance * 0.001  # kW
        
        # Temperature difference load
        temp_load = area * 0.05 * max(0, outdoor_temp - 24)
        
        return people_gain * 0.001 + solar_gain + temp_load  # kW
    
    def generate_time_series(self, building_params: Dict, 
                           hours: int = 8760) -> pd.DataFrame:
        """Generate hourly time series data for a specific building."""
        data = []
        
        for hour in range(hours):
            # Simulate weather patterns
            outdoor_temp = 15 + 10 * np.sin(2 * np.pi * hour / 8760) + random.gauss(0, 3)
            solar = max(0, 800 * np.sin(np.pi * (hour % 24) / 12) + random.gauss(0, 100))
            
            # Calculate loads
            heating = self._calculate_heating_load(
                building_params['floor_area'],
                building_params['height'],
                building_params['window_ratio'],
                building_params['insulation_r'],
                outdoor_temp
            )
            
            cooling = self._calculate_cooling_load(
                building_params['floor_area'],
                building_params['occupancy'],
                solar,
                outdoor_temp
            )
            
            data.append({
                'hour': hour,
                'outdoor_temp': outdoor_temp,
                'solar_irradiance': solar,
                'heating_load': heating,
                'cooling_load': cooling,
                'total_energy': heating + cooling
            })
            
        return pd.DataFrame(data)