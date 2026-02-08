"""
HVAC System Robustness Testing Suite

Tests system performance under:
1. Extreme weather conditions
2. Unexpected occupancy changes
3. Prediction errors (forecast mismatch)
4. System parameter changes (model mismatch)

Compares Rule-based, PID, and MPC controllers.
"""

import sys
import time
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.simulation.rc_model import step_rc
from dashboard.simulation.controllers import (
    PIDController, rule_based_hvac, mpc_simple_hvac,
    HeatingType, HEATING_EFFICIENCY
)


# ============ Test Configuration ============

@dataclass
class TestConfig:
    """Configuration for a single robustness test."""
    name: str
    description: str
    duration_hours: float = 24.0
    dt_sec: float = 60.0
    
    # Building parameters (nominal)
    R: float = 0.01          # K/W
    C: float = 1e4           # J/K
    
    # HVAC capacity
    max_heating: float = 8000.0
    max_cooling: float = 10000.0
    
    # Comfort
    setpoint: float = 22.0
    comfort_band: float = 1.0  # acceptable deviation
    
    # Initial conditions
    T_in_initial: float = 22.0
    T_out_initial: float = 25.0


@dataclass
class TestResults:
    """Results from a robustness test."""
    test_name: str
    controller: str
    
    # Comfort metrics
    comfort_violations_pct: float = 0.0  # % time outside comfort band
    max_deviation: float = 0.0           # max |T_in - setpoint|
    avg_deviation: float = 0.0           # mean |T_in - setpoint|
    rms_error: float = 0.0               # root mean square error
    
    # Energy metrics
    total_heating_kwh: float = 0.0
    total_cooling_kwh: float = 0.0
    peak_power_kw: float = 0.0
    
    # Stability metrics
    oscillation_count: int = 0           # number of heating/cooling switches
    settling_time_min: float = 0.0       # time to reach steady state
    
    # Recovery metrics
    recovery_time_min: float = 0.0       # time to recover from disturbance
    overshoot_deg: float = 0.0           # max overshoot during recovery
    
    # Raw data for plotting
    time_series: list = field(default_factory=list)
    T_in_series: list = field(default_factory=list)
    T_out_series: list = field(default_factory=list)
    heating_series: list = field(default_factory=list)
    cooling_series: list = field(default_factory=list)


# ============ Disturbance Generators ============

class WeatherDisturbance:
    """Generate extreme weather scenarios."""
    
    @staticmethod
    def heat_wave(t_hours: float, base_temp: float = 25.0) -> float:
        """Extreme heat wave: 35-42°C during day."""
        hour = t_hours % 24
        daily_var = 5 * math.sin(2 * math.pi * (hour - 6) / 24)
        return 38.0 + daily_var + random.gauss(0, 0.5)
    
    @staticmethod
    def cold_snap(t_hours: float, base_temp: float = 25.0) -> float:
        """Extreme cold: -10 to -5°C."""
        hour = t_hours % 24
        daily_var = 3 * math.sin(2 * math.pi * (hour - 6) / 24)
        return -8.0 + daily_var + random.gauss(0, 0.5)
    
    @staticmethod
    def rapid_front(t_hours: float, base_temp: float = 25.0) -> float:
        """Cold front passing: 30°C drops to 10°C in 2 hours."""
        if t_hours < 8:
            return 30.0
        elif t_hours < 10:
            # Rapid drop
            progress = (t_hours - 8) / 2
            return 30.0 - 20.0 * progress
        else:
            return 10.0 + 2 * math.sin(2 * math.pi * (t_hours - 10) / 24)
    
    @staticmethod
    def high_variability(t_hours: float, base_temp: float = 25.0) -> float:
        """High frequency temperature swings (±5°C every hour)."""
        base = 25.0 + 5 * math.sin(2 * math.pi * t_hours / 24)
        rapid_swing = 5 * math.sin(2 * math.pi * t_hours)  # hourly
        noise = random.gauss(0, 1)
        return base + rapid_swing + noise
    
    @staticmethod
    def normal(t_hours: float, base_temp: float = 25.0) -> float:
        """Normal weather: 20-30°C daily cycle."""
        hour = t_hours % 24
        return base_temp + 5 * math.sin(2 * math.pi * (hour - 6) / 24)


class OccupancyDisturbance:
    """Generate occupancy disturbance scenarios."""
    
    @staticmethod
    def sudden_crowd(t_hours: float, base_occupancy: bool = False) -> tuple[bool, float]:
        """Sudden influx of people (conference, event)."""
        # Normal: 8-18 occupied
        # Disturbance: 200 people arrive at hour 10, stay until hour 14
        hour = t_hours % 24
        if 10 <= hour < 14:
            return True, 5000.0  # 5kW internal gains (crowd)
        elif 8 <= hour < 18:
            return True, 500.0   # normal occupancy
        return False, 100.0
    
    @staticmethod
    def unexpected_vacancy(t_hours: float, base_occupancy: bool = False) -> tuple[bool, float]:
        """Building unexpectedly empty (holiday, emergency)."""
        return False, 50.0  # minimal gains
    
    @staticmethod
    def irregular_schedule(t_hours: float, base_occupancy: bool = False) -> tuple[bool, float]:
        """Irregular occupancy pattern (night shift, 24/7 ops)."""
        hour = t_hours % 24
        # Two shifts: 6-14, 22-6
        if 6 <= hour < 14 or 22 <= hour or hour < 6:
            return True, 500.0
        return False, 100.0
    
    @staticmethod
    def random_bursts(t_hours: float, base_occupancy: bool = False) -> tuple[bool, float]:
        """Random occupancy bursts."""
        # Use deterministic "random" based on time
        seed = int(t_hours * 100) % 1000
        random.seed(seed)
        if random.random() < 0.3:  # 30% chance of burst
            return True, random.uniform(500, 3000)
        random.seed()  # reset
        return False, 100.0
    
    @staticmethod
    def normal(t_hours: float, base_occupancy: bool = False) -> tuple[bool, float]:
        """Normal office schedule."""
        hour = t_hours % 24
        if 8 <= hour < 18:
            return True, 500.0
        return False, 100.0


class PredictionError:
    """Simulate forecast/prediction errors."""
    
    @staticmethod
    def weather_bias(actual_temp: float, bias: float = 5.0) -> float:
        """Systematic weather forecast bias."""
        return actual_temp + bias  # forecast always too high/low
    
    @staticmethod
    def weather_noise(actual_temp: float, noise_std: float = 3.0) -> float:
        """Random weather forecast error."""
        return actual_temp + random.gauss(0, noise_std)
    
    @staticmethod
    def occupancy_wrong(actual_occ: bool) -> bool:
        """Occupancy prediction is wrong 50% of the time."""
        if random.random() < 0.5:
            return not actual_occ
        return actual_occ


class ParameterDrift:
    """Simulate system parameter changes (model mismatch)."""
    
    @staticmethod
    def degraded_insulation(nominal_R: float, t_hours: float) -> float:
        """Insulation degrades over time (R decreases)."""
        degradation = 0.3 * min(1.0, t_hours / 12)  # 30% degradation over 12 hours
        return nominal_R * (1 - degradation)
    
    @staticmethod
    def hvac_capacity_loss(nominal_capacity: float, t_hours: float) -> float:
        """HVAC capacity reduces (dirty filters, wear)."""
        loss = 0.25 * min(1.0, t_hours / 8)  # 25% loss over 8 hours
        return nominal_capacity * (1 - loss)
    
    @staticmethod
    def thermal_mass_change(nominal_C: float, t_hours: float) -> float:
        """Thermal mass changes (furniture added, water in pipes)."""
        # Increases by 50% over 6 hours
        increase = 0.5 * min(1.0, t_hours / 6)
        return nominal_C * (1 + increase)


# ============ Test Runner ============

class RobustnessTestRunner:
    """Runs robustness tests on different controllers."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: list[TestResults] = []
    
    def run_test(
        self,
        controller_name: str,
        weather_fn: Callable[[float, float], float],
        occupancy_fn: Callable[[float, bool], tuple[bool, float]],
        R_fn: Callable[[float, float], float] | None = None,
        C_fn: Callable[[float, float], float] | None = None,
        max_cooling_fn: Callable[[float, float], float] | None = None,
        max_heating_fn: Callable[[float, float], float] | None = None,
    ) -> TestResults:
        """
        Run a single test with specified disturbances.
        
        Args:
            controller_name: "rule", "pid", "mpc"
            weather_fn: (t_hours, base_temp) -> T_out
            occupancy_fn: (t_hours, base_occ) -> (occupied, Q_internal)
            R_fn: (nominal_R, t_hours) -> actual_R (optional)
            C_fn: (nominal_C, t_hours) -> actual_C (optional)
            max_cooling_fn: capacity degradation function (optional)
            max_heating_fn: capacity degradation function (optional)
        """
        cfg = self.config
        n_steps = int(cfg.duration_hours * 3600 / cfg.dt_sec)
        
        # Initialize state
        T_in = cfg.T_in_initial
        heating = 0.0
        cooling = 0.0
        
        # Controller setup
        if controller_name == "pid":
            pid = PIDController(Kp=2000.0, Ki=50.0, Kd=100.0)
        
        # Result tracking
        result = TestResults(
            test_name=cfg.name,
            controller=controller_name,
        )
        
        comfort_violations = 0
        deviations = []
        mode_switches = 0
        last_mode = "off"
        peak_power = 0.0
        
        # Disturbance detection for recovery time
        disturbance_start = None
        recovered = True
        
        for step in range(n_steps):
            t_sec = step * cfg.dt_sec
            t_hours = t_sec / 3600
            
            # Get actual conditions (possibly perturbed)
            T_out = weather_fn(t_hours, cfg.T_out_initial)
            occupied, Q_internal = occupancy_fn(t_hours, False)
            
            # Get actual system parameters (possibly drifting)
            actual_R = R_fn(cfg.R, t_hours) if R_fn else cfg.R
            actual_C = C_fn(cfg.C, t_hours) if C_fn else cfg.C
            actual_max_cooling = max_cooling_fn(cfg.max_cooling, t_hours) if max_cooling_fn else cfg.max_cooling
            actual_max_heating = max_heating_fn(cfg.max_heating, t_hours) if max_heating_fn else cfg.max_heating
            
            # Controller uses NOMINAL parameters (doesn't know about drift)
            setpoint = cfg.setpoint
            
            # Control action
            if controller_name == "rule":
                heating, cooling = rule_based_hvac(
                    T_in, setpoint, 1.0,
                    heating, cooling,
                    cfg.max_heating, cfg.max_cooling  # nominal
                )
            elif controller_name == "pid":
                heating, cooling = pid.compute_hvac(
                    T_in, setpoint, t_sec,
                    cfg.max_heating, cfg.max_cooling  # nominal
                )
            elif controller_name == "mpc":
                heating, cooling = mpc_simple_hvac(
                    T_in, T_out, cfg.R, cfg.C, Q_internal, setpoint,
                    horizon_steps=6, dt_sec=cfg.dt_sec,
                    weight_comfort=1.0, weight_energy=0.1,
                    max_heating=cfg.max_heating, max_cooling=cfg.max_cooling,
                    heating_efficiency=0.92, cooling_cop=3.0
                )
            
            # Apply actual capacity limits (reality may differ from controller's belief)
            cooling = min(cooling, actual_max_cooling)
            heating = min(heating, actual_max_heating)
            
            # Simulate physics with ACTUAL parameters
            T_in_new, _ = step_rc(
                T_in, T_out, actual_R, actual_C,
                Q_internal, cooling, cfg.dt_sec, heating
            )
            
            # Track metrics
            deviation = abs(T_in - setpoint)
            deviations.append(deviation)
            
            if deviation > cfg.comfort_band:
                comfort_violations += 1
                if recovered:
                    disturbance_start = t_hours
                    recovered = False
            else:
                if not recovered:
                    result.recovery_time_min = (t_hours - disturbance_start) * 60
                    recovered = True
            
            # Mode switching
            if heating > 0:
                current_mode = "heating"
            elif cooling > 0:
                current_mode = "cooling"
            else:
                current_mode = "off"
            
            if current_mode != last_mode and current_mode != "off" and last_mode != "off":
                mode_switches += 1
            last_mode = current_mode
            
            # Peak power
            peak_power = max(peak_power, heating + cooling)
            
            # Energy
            result.total_heating_kwh += heating * cfg.dt_sec / 3600 / 1000
            result.total_cooling_kwh += cooling * cfg.dt_sec / 3600 / 1000
            
            # Store time series
            result.time_series.append(t_hours)
            result.T_in_series.append(T_in)
            result.T_out_series.append(T_out)
            result.heating_series.append(heating)
            result.cooling_series.append(cooling)
            
            # Update state
            T_in = T_in_new
        
        # Compute final metrics
        result.comfort_violations_pct = 100 * comfort_violations / n_steps
        result.max_deviation = max(deviations)
        result.avg_deviation = sum(deviations) / len(deviations)
        result.rms_error = math.sqrt(sum(d**2 for d in deviations) / len(deviations))
        result.oscillation_count = mode_switches
        result.peak_power_kw = peak_power / 1000
        
        # Overshoot: max deviation after disturbance
        if deviations:
            result.overshoot_deg = max(deviations)
        
        self.results.append(result)
        return result
    
    def run_all_controllers(
        self,
        weather_fn: Callable,
        occupancy_fn: Callable,
        **kwargs
    ) -> dict[str, TestResults]:
        """Run test for all controller types."""
        results = {}
        for controller in ["rule", "pid", "mpc"]:
            print(f"  Testing {controller}...", end=" ", flush=True)
            result = self.run_test(controller, weather_fn, occupancy_fn, **kwargs)
            results[controller] = result
            print(f"done (violations: {result.comfort_violations_pct:.1f}%)")
        return results


# ============ Test Scenarios ============

def run_extreme_weather_tests() -> dict:
    """Test suite for extreme weather conditions."""
    print("\n" + "="*60)
    print("EXTREME WEATHER TESTS")
    print("="*60)
    
    scenarios = {
        "heat_wave": ("Heat Wave (35-42°C)", WeatherDisturbance.heat_wave),
        "cold_snap": ("Cold Snap (-10°C)", WeatherDisturbance.cold_snap),
        "rapid_front": ("Rapid Cold Front", WeatherDisturbance.rapid_front),
        "high_variability": ("High Variability", WeatherDisturbance.high_variability),
    }
    
    all_results = {}
    
    for key, (name, weather_fn) in scenarios.items():
        print(f"\n--- {name} ---")
        config = TestConfig(
            name=name,
            description=f"Testing under {name.lower()} conditions",
            duration_hours=24.0
        )
        runner = RobustnessTestRunner(config)
        results = runner.run_all_controllers(
            weather_fn=weather_fn,
            occupancy_fn=OccupancyDisturbance.normal,
        )
        all_results[key] = results
    
    return all_results


def run_occupancy_disturbance_tests() -> dict:
    """Test suite for occupancy disturbances."""
    print("\n" + "="*60)
    print("OCCUPANCY DISTURBANCE TESTS")
    print("="*60)
    
    scenarios = {
        "sudden_crowd": ("Sudden Crowd (Conference)", OccupancyDisturbance.sudden_crowd),
        "unexpected_vacancy": ("Unexpected Vacancy", OccupancyDisturbance.unexpected_vacancy),
        "irregular_schedule": ("Irregular Schedule (Shifts)", OccupancyDisturbance.irregular_schedule),
        "random_bursts": ("Random Bursts", OccupancyDisturbance.random_bursts),
    }
    
    all_results = {}
    
    for key, (name, occ_fn) in scenarios.items():
        print(f"\n--- {name} ---")
        config = TestConfig(
            name=name,
            description=f"Testing under {name.lower()} conditions",
            duration_hours=24.0
        )
        runner = RobustnessTestRunner(config)
        results = runner.run_all_controllers(
            weather_fn=WeatherDisturbance.normal,
            occupancy_fn=occ_fn,
        )
        all_results[key] = results
    
    return all_results


def run_parameter_drift_tests() -> dict:
    """Test suite for system parameter changes (model mismatch)."""
    print("\n" + "="*60)
    print("PARAMETER DRIFT / MODEL MISMATCH TESTS")
    print("="*60)
    
    all_results = {}
    
    # Test 1: Insulation degradation
    print("\n--- Insulation Degradation (R drops 30%) ---")
    config = TestConfig(
        name="Insulation Degradation",
        description="R parameter drops by 30% over 12 hours",
        duration_hours=24.0
    )
    runner = RobustnessTestRunner(config)
    results = runner.run_all_controllers(
        weather_fn=WeatherDisturbance.normal,
        occupancy_fn=OccupancyDisturbance.normal,
        R_fn=ParameterDrift.degraded_insulation,
    )
    all_results["insulation_degradation"] = results
    
    # Test 2: HVAC capacity loss
    print("\n--- HVAC Capacity Loss (25% reduction) ---")
    config = TestConfig(
        name="HVAC Capacity Loss",
        description="Cooling capacity drops 25% over 8 hours",
        duration_hours=24.0
    )
    runner = RobustnessTestRunner(config)
    results = runner.run_all_controllers(
        weather_fn=WeatherDisturbance.heat_wave,  # stress test with heat wave
        occupancy_fn=OccupancyDisturbance.normal,
        max_cooling_fn=ParameterDrift.hvac_capacity_loss,
    )
    all_results["capacity_loss"] = results
    
    # Test 3: Thermal mass change
    print("\n--- Thermal Mass Change (C increases 50%) ---")
    config = TestConfig(
        name="Thermal Mass Change",
        description="Building thermal mass increases 50%",
        duration_hours=24.0
    )
    runner = RobustnessTestRunner(config)
    results = runner.run_all_controllers(
        weather_fn=WeatherDisturbance.normal,
        occupancy_fn=OccupancyDisturbance.normal,
        C_fn=ParameterDrift.thermal_mass_change,
    )
    all_results["thermal_mass_change"] = results
    
    # Test 4: Combined drift
    print("\n--- Combined Parameter Drift ---")
    config = TestConfig(
        name="Combined Drift",
        description="Multiple parameters drifting simultaneously",
        duration_hours=24.0
    )
    runner = RobustnessTestRunner(config)
    results = runner.run_all_controllers(
        weather_fn=WeatherDisturbance.normal,
        occupancy_fn=OccupancyDisturbance.normal,
        R_fn=ParameterDrift.degraded_insulation,
        C_fn=ParameterDrift.thermal_mass_change,
        max_cooling_fn=ParameterDrift.hvac_capacity_loss,
    )
    all_results["combined_drift"] = results
    
    return all_results


def run_combined_stress_test() -> dict:
    """Ultimate stress test: multiple disturbances simultaneously."""
    print("\n" + "="*60)
    print("COMBINED STRESS TEST (WORST CASE)")
    print("="*60)
    
    print("\n--- Heat Wave + Sudden Crowd + Capacity Loss ---")
    config = TestConfig(
        name="Ultimate Stress Test",
        description="Heat wave + sudden crowd + degraded HVAC",
        duration_hours=24.0
    )
    runner = RobustnessTestRunner(config)
    results = runner.run_all_controllers(
        weather_fn=WeatherDisturbance.heat_wave,
        occupancy_fn=OccupancyDisturbance.sudden_crowd,
        max_cooling_fn=ParameterDrift.hvac_capacity_loss,
        R_fn=ParameterDrift.degraded_insulation,
    )
    
    return {"ultimate_stress": results}


# ============ Results Summary ============

def print_results_table(all_results: dict):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Scenario':<25} {'Controller':<8} {'Violations%':<12} {'MaxDev°C':<10} {'RMS°C':<8} {'Energy kWh':<12} {'Switches':<8}")
    print("-"*80)
    
    for scenario_name, controllers in all_results.items():
        for controller_name, result in controllers.items():
            total_energy = result.total_heating_kwh + result.total_cooling_kwh
            print(f"{scenario_name:<25} {controller_name:<8} {result.comfort_violations_pct:<12.1f} {result.max_deviation:<10.2f} {result.rms_error:<8.2f} {total_energy:<12.2f} {result.oscillation_count:<8}")


def print_controller_comparison(all_results: dict):
    """Compare controllers across all tests."""
    print("\n" + "="*60)
    print("CONTROLLER COMPARISON (AVERAGED ACROSS ALL TESTS)")
    print("="*60)
    
    # Aggregate by controller
    controller_stats = {"rule": [], "pid": [], "mpc": []}
    
    for scenario_results in all_results.values():
        for controller, result in scenario_results.items():
            controller_stats[controller].append({
                "violations": result.comfort_violations_pct,
                "rms": result.rms_error,
                "energy": result.total_heating_kwh + result.total_cooling_kwh,
                "switches": result.oscillation_count,
            })
    
    print(f"\n{'Controller':<12} {'Avg Violations%':<16} {'Avg RMS°C':<12} {'Avg Energy kWh':<16} {'Avg Switches':<12}")
    print("-"*70)
    
    for controller, stats in controller_stats.items():
        if stats:
            avg_viol = sum(s["violations"] for s in stats) / len(stats)
            avg_rms = sum(s["rms"] for s in stats) / len(stats)
            avg_energy = sum(s["energy"] for s in stats) / len(stats)
            avg_switches = sum(s["switches"] for s in stats) / len(stats)
            print(f"{controller:<12} {avg_viol:<16.1f} {avg_rms:<12.2f} {avg_energy:<16.2f} {avg_switches:<12.1f}")


def save_results_json(all_results: dict, filename: str = "robustness_results.json"):
    """Save results to JSON file."""
    output_path = Path(__file__).parent / filename
    
    # Convert to serializable format
    serializable = {}
    for scenario, controllers in all_results.items():
        serializable[scenario] = {}
        for controller, result in controllers.items():
            serializable[scenario][controller] = {
                "comfort_violations_pct": result.comfort_violations_pct,
                "max_deviation": result.max_deviation,
                "avg_deviation": result.avg_deviation,
                "rms_error": result.rms_error,
                "total_heating_kwh": result.total_heating_kwh,
                "total_cooling_kwh": result.total_cooling_kwh,
                "peak_power_kw": result.peak_power_kw,
                "oscillation_count": result.oscillation_count,
                "recovery_time_min": result.recovery_time_min,
            }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# ============ Main ============

def main():
    """Run complete robustness test suite."""
    print("="*60)
    print("HVAC SYSTEM ROBUSTNESS TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Run test suites
    all_results.update(run_extreme_weather_tests())
    all_results.update(run_occupancy_disturbance_tests())
    all_results.update(run_parameter_drift_tests())
    all_results.update(run_combined_stress_test())
    
    # Print summary
    print_results_table(all_results)
    print_controller_comparison(all_results)
    
    # Save results
    save_results_json(all_results)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    results = main()
