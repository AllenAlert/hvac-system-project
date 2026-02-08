"""
Export robustness test results to Excel with charts.
Compares Rule-based, PID, Simple MPC, and Advanced MPC controllers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import random
from datetime import datetime
from dataclasses import dataclass, field

import xlsxwriter

from dashboard.simulation.rc_model import step_rc
from dashboard.simulation.controllers import (
    PIDController, rule_based_hvac, mpc_simple_hvac,
)
from dashboard.simulation.mpc_advanced import AdvancedMPC, MPCAdvancedConfig


# ============ Test Configuration ============

@dataclass
class TestConfig:
    name: str
    duration_hours: float = 24.0
    dt_sec: float = 60.0
    R: float = 0.01
    C: float = 1e4
    max_heating: float = 8000.0
    max_cooling: float = 10000.0
    setpoint: float = 22.0
    comfort_band: float = 1.0
    T_in_initial: float = 22.0
    T_out_initial: float = 25.0


@dataclass 
class TestResults:
    test_name: str
    controller: str
    comfort_violations_pct: float = 0.0
    max_deviation: float = 0.0
    avg_deviation: float = 0.0
    rms_error: float = 0.0
    total_heating_kwh: float = 0.0
    total_cooling_kwh: float = 0.0
    peak_power_kw: float = 0.0
    oscillation_count: int = 0
    
    # Time series for plotting
    time_hours: list = field(default_factory=list)
    T_in: list = field(default_factory=list)
    T_out: list = field(default_factory=list)
    setpoint: list = field(default_factory=list)
    heating: list = field(default_factory=list)
    cooling: list = field(default_factory=list)


# ============ Weather Scenarios ============

def weather_normal(t_hours: float) -> float:
    hour = t_hours % 24
    return 25.0 + 5 * math.sin(2 * math.pi * (hour - 6) / 24)

def weather_heat_wave(t_hours: float) -> float:
    hour = t_hours % 24
    return 38.0 + 5 * math.sin(2 * math.pi * (hour - 6) / 24) + random.gauss(0, 0.3)

def weather_cold_snap(t_hours: float) -> float:
    hour = t_hours % 24
    return -8.0 + 3 * math.sin(2 * math.pi * (hour - 6) / 24) + random.gauss(0, 0.3)

def weather_rapid_change(t_hours: float) -> float:
    """Temperature drops from 30°C to 10°C between hour 8-10."""
    if t_hours < 8:
        return 30.0
    elif t_hours < 10:
        return 30.0 - 10 * (t_hours - 8)
    else:
        return 10.0 + 2 * math.sin(2 * math.pi * (t_hours - 10) / 24)


# ============ Occupancy Scenarios ============

def occupancy_normal(t_hours: float) -> tuple[bool, float]:
    hour = t_hours % 24
    if 8 <= hour < 18:
        return True, 500.0
    return False, 100.0

def occupancy_sudden_crowd(t_hours: float) -> tuple[bool, float]:
    hour = t_hours % 24
    if 10 <= hour < 14:
        return True, 4000.0  # Big event
    elif 8 <= hour < 18:
        return True, 500.0
    return False, 100.0


# ============ Test Runner ============

def run_simulation(
    config: TestConfig,
    controller_type: str,
    weather_fn,
    occupancy_fn,
) -> TestResults:
    """Run simulation with specified controller and conditions."""
    
    n_steps = int(config.duration_hours * 3600 / config.dt_sec)
    result = TestResults(test_name=config.name, controller=controller_type)
    
    # Initialize
    T_in = config.T_in_initial
    heating = 0.0
    cooling = 0.0
    
    # Controllers
    pid = PIDController(Kp=2000.0, Ki=50.0, Kd=100.0) if controller_type == "pid" else None
    
    if controller_type == "mpc_advanced":
        mpc_adv = AdvancedMPC(MPCAdvancedConfig(
            max_cooling=config.max_cooling,
            R=config.R,
            C=config.C,
            prediction_horizon=30,
            control_horizon=10,
        ))
    
    deviations = []
    mode_switches = 0
    last_mode = "off"
    peak_power = 0.0
    
    for step in range(n_steps):
        t_sec = step * config.dt_sec
        t_hours = t_sec / 3600
        
        T_out = weather_fn(t_hours)
        occupied, Q_internal = occupancy_fn(t_hours)
        setpoint = config.setpoint
        
        # Control action based on controller type
        if controller_type == "rule":
            heating, cooling = rule_based_hvac(
                T_in, setpoint, 1.0,
                heating, cooling,
                config.max_heating, config.max_cooling
            )
        elif controller_type == "pid":
            heating, cooling = pid.compute_hvac(
                T_in, setpoint, t_sec,
                config.max_heating, config.max_cooling
            )
        elif controller_type == "mpc_simple":
            heating, cooling = mpc_simple_hvac(
                T_in, T_out, config.R, config.C, Q_internal, setpoint,
                horizon_steps=6, dt_sec=config.dt_sec,
                weight_comfort=1.0, weight_energy=0.1,
                max_heating=config.max_heating, max_cooling=config.max_cooling,
                heating_efficiency=0.92, cooling_cop=3.0
            )
        elif controller_type == "mpc_advanced":
            cooling, _ = mpc_adv.compute(T_in, T_out, Q_internal, occupied, t_sec)
            # Advanced MPC currently only does cooling, add heating if cold
            if cooling == 0 and T_in < setpoint - 1.0:
                heating = config.max_heating
            else:
                heating = 0.0
        
        # Simulate physics
        T_in_new, _ = step_rc(
            T_in, T_out, config.R, config.C,
            Q_internal, cooling, config.dt_sec, heating
        )
        
        # Track metrics
        deviation = abs(T_in - setpoint)
        deviations.append(deviation)
        
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
        
        peak_power = max(peak_power, heating + cooling)
        result.total_heating_kwh += heating * config.dt_sec / 3600 / 1000
        result.total_cooling_kwh += cooling * config.dt_sec / 3600 / 1000
        
        # Store time series (every 5 minutes for plotting)
        if step % 5 == 0:
            result.time_hours.append(t_hours)
            result.T_in.append(T_in)
            result.T_out.append(T_out)
            result.setpoint.append(setpoint)
            result.heating.append(heating / 1000)  # kW
            result.cooling.append(cooling / 1000)  # kW
        
        T_in = T_in_new
    
    # Compute final metrics
    n_violations = sum(1 for d in deviations if d > config.comfort_band)
    result.comfort_violations_pct = 100 * n_violations / len(deviations)
    result.max_deviation = max(deviations)
    result.avg_deviation = sum(deviations) / len(deviations)
    result.rms_error = math.sqrt(sum(d**2 for d in deviations) / len(deviations))
    result.oscillation_count = mode_switches
    result.peak_power_kw = peak_power / 1000
    
    return result


def create_excel_report(output_path: str):
    """Create Excel workbook with all test results and charts."""
    
    workbook = xlsxwriter.Workbook(output_path)
    
    # Formats
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
    number_fmt = workbook.add_format({'num_format': '0.00', 'border': 1})
    pct_fmt = workbook.add_format({'num_format': '0.0%', 'border': 1})
    title_fmt = workbook.add_format({'bold': True, 'font_size': 14})
    
    controllers = ["rule", "pid", "mpc_simple", "mpc_advanced"]
    controller_names = {
        "rule": "Rule-Based",
        "pid": "PID", 
        "mpc_simple": "MPC Simple",
        "mpc_advanced": "MPC Advanced"
    }
    
    scenarios = [
        ("Normal Weather", weather_normal, occupancy_normal),
        ("Heat Wave", weather_heat_wave, occupancy_normal),
        ("Cold Snap", weather_cold_snap, occupancy_normal),
        ("Rapid Change", weather_rapid_change, occupancy_normal),
        ("Sudden Crowd", weather_normal, occupancy_sudden_crowd),
    ]
    
    # ============ Summary Sheet ============
    summary = workbook.add_worksheet("Summary")
    summary.set_column('A:A', 20)
    summary.set_column('B:E', 15)
    
    summary.write(0, 0, "HVAC Robustness Test Results", title_fmt)
    summary.write(1, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Summary table headers
    row = 4
    summary.write(row, 0, "Scenario", header_fmt)
    summary.write(row, 1, "Controller", header_fmt)
    summary.write(row, 2, "Violations %", header_fmt)
    summary.write(row, 3, "RMS Error °C", header_fmt)
    summary.write(row, 4, "Energy kWh", header_fmt)
    summary.write(row, 5, "Peak kW", header_fmt)
    summary.write(row, 6, "Switches", header_fmt)
    
    all_results = {}
    summary_row = 5
    
    for scenario_name, weather_fn, occupancy_fn in scenarios:
        print(f"\nRunning scenario: {scenario_name}")
        all_results[scenario_name] = {}
        
        config = TestConfig(
            name=scenario_name,
            duration_hours=24.0,
            dt_sec=60.0,
        )
        
        for ctrl in controllers:
            print(f"  Controller: {ctrl}...", end=" ", flush=True)
            result = run_simulation(config, ctrl, weather_fn, occupancy_fn)
            all_results[scenario_name][ctrl] = result
            print(f"done ({result.comfort_violations_pct:.1f}% violations)")
            
            # Write to summary
            total_energy = result.total_heating_kwh + result.total_cooling_kwh
            summary.write(summary_row, 0, scenario_name)
            summary.write(summary_row, 1, controller_names[ctrl])
            summary.write(summary_row, 2, result.comfort_violations_pct / 100, pct_fmt)
            summary.write(summary_row, 3, result.rms_error, number_fmt)
            summary.write(summary_row, 4, total_energy, number_fmt)
            summary.write(summary_row, 5, result.peak_power_kw, number_fmt)
            summary.write(summary_row, 6, result.oscillation_count)
            summary_row += 1
    
    # ============ Individual Scenario Sheets with Charts ============
    
    for scenario_name, weather_fn, occupancy_fn in scenarios:
        sheet_name = scenario_name.replace(" ", "_")[:31]
        ws = workbook.add_worksheet(sheet_name)
        
        # Write data headers
        headers = ["Time (h)", "T_out °C"]
        for ctrl in controllers:
            headers.extend([f"{controller_names[ctrl]} T_in", f"{controller_names[ctrl]} Heat kW", f"{controller_names[ctrl]} Cool kW"])
        headers.append("Setpoint °C")
        
        for col, h in enumerate(headers):
            ws.write(0, col, h, header_fmt)
        
        # Get reference result for time/T_out/setpoint
        ref_result = all_results[scenario_name]["rule"]
        n_rows = len(ref_result.time_hours)
        
        # Write time series data
        for row in range(n_rows):
            ws.write(row + 1, 0, ref_result.time_hours[row])
            ws.write(row + 1, 1, ref_result.T_out[row])
            
            col = 2
            for ctrl in controllers:
                r = all_results[scenario_name][ctrl]
                ws.write(row + 1, col, r.T_in[row] if row < len(r.T_in) else "")
                ws.write(row + 1, col + 1, r.heating[row] if row < len(r.heating) else "")
                ws.write(row + 1, col + 2, r.cooling[row] if row < len(r.cooling) else "")
                col += 3
            
            ws.write(row + 1, col, ref_result.setpoint[row])
        
        # ============ Temperature Chart ============
        temp_chart = workbook.add_chart({'type': 'line'})
        temp_chart.set_title({'name': f'{scenario_name} - Temperature'})
        temp_chart.set_x_axis({'name': 'Time (hours)'})
        temp_chart.set_y_axis({'name': 'Temperature (°C)'})
        temp_chart.set_size({'width': 720, 'height': 400})
        
        # Outdoor temp
        temp_chart.add_series({
            'name': 'T_outdoor',
            'categories': [sheet_name, 1, 0, n_rows, 0],
            'values': [sheet_name, 1, 1, n_rows, 1],
            'line': {'color': 'gray', 'dash_type': 'dash'},
        })
        
        # Setpoint
        setpoint_col = 2 + len(controllers) * 3
        temp_chart.add_series({
            'name': 'Setpoint',
            'categories': [sheet_name, 1, 0, n_rows, 0],
            'values': [sheet_name, 1, setpoint_col, n_rows, setpoint_col],
            'line': {'color': 'green', 'dash_type': 'dash', 'width': 2},
        })
        
        # Controller temperatures
        colors = ['#C00000', '#0070C0', '#00B050', '#7030A0']  # Red, Blue, Green, Purple
        for i, ctrl in enumerate(controllers):
            col = 2 + i * 3  # T_in column for this controller
            temp_chart.add_series({
                'name': controller_names[ctrl],
                'categories': [sheet_name, 1, 0, n_rows, 0],
                'values': [sheet_name, 1, col, n_rows, col],
                'line': {'color': colors[i], 'width': 1.5},
            })
        
        ws.insert_chart('R2', temp_chart)
        
        # ============ Power Chart ============
        power_chart = workbook.add_chart({'type': 'line'})
        power_chart.set_title({'name': f'{scenario_name} - HVAC Power'})
        power_chart.set_x_axis({'name': 'Time (hours)'})
        power_chart.set_y_axis({'name': 'Power (kW)', 'min': 0})
        power_chart.set_size({'width': 720, 'height': 400})
        
        for i, ctrl in enumerate(controllers):
            heat_col = 2 + i * 3 + 1
            cool_col = 2 + i * 3 + 2
            
            # Heating (positive, warm colors)
            power_chart.add_series({
                'name': f'{controller_names[ctrl]} Heat',
                'categories': [sheet_name, 1, 0, n_rows, 0],
                'values': [sheet_name, 1, heat_col, n_rows, heat_col],
                'line': {'color': colors[i], 'width': 1.5},
            })
            
            # Cooling (shown as negative for clarity? or separate?)
            power_chart.add_series({
                'name': f'{controller_names[ctrl]} Cool',
                'categories': [sheet_name, 1, 0, n_rows, 0],
                'values': [sheet_name, 1, cool_col, n_rows, cool_col],
                'line': {'color': colors[i], 'dash_type': 'dash', 'width': 1.5},
            })
        
        ws.insert_chart('R24', power_chart)
    
    # ============ Controller Comparison Chart ============
    comp_ws = workbook.add_worksheet("Comparison")
    
    # Write comparison data
    comp_ws.write(0, 0, "Controller Comparison Summary", title_fmt)
    
    comp_ws.write(2, 0, "Controller", header_fmt)
    comp_ws.write(2, 1, "Avg Violations %", header_fmt)
    comp_ws.write(2, 2, "Avg RMS °C", header_fmt)
    comp_ws.write(2, 3, "Avg Energy kWh", header_fmt)
    
    for i, ctrl in enumerate(controllers):
        violations = []
        rms_errors = []
        energies = []
        
        for scenario_name in all_results:
            r = all_results[scenario_name][ctrl]
            violations.append(r.comfort_violations_pct)
            rms_errors.append(r.rms_error)
            energies.append(r.total_heating_kwh + r.total_cooling_kwh)
        
        row = 3 + i
        comp_ws.write(row, 0, controller_names[ctrl])
        comp_ws.write(row, 1, sum(violations) / len(violations), number_fmt)
        comp_ws.write(row, 2, sum(rms_errors) / len(rms_errors), number_fmt)
        comp_ws.write(row, 3, sum(energies) / len(energies), number_fmt)
    
    # Bar chart for comparison
    bar_chart = workbook.add_chart({'type': 'column'})
    bar_chart.set_title({'name': 'Controller Performance Comparison'})
    bar_chart.set_size({'width': 600, 'height': 350})
    
    bar_chart.add_series({
        'name': 'Avg RMS Error (°C)',
        'categories': ['Comparison', 3, 0, 6, 0],
        'values': ['Comparison', 3, 2, 6, 2],
        'fill': {'color': '#4472C4'},
    })
    
    comp_ws.insert_chart('F2', bar_chart)
    
    # Energy comparison
    energy_chart = workbook.add_chart({'type': 'column'})
    energy_chart.set_title({'name': 'Energy Consumption Comparison'})
    energy_chart.set_size({'width': 600, 'height': 350})
    
    energy_chart.add_series({
        'name': 'Avg Energy (kWh)',
        'categories': ['Comparison', 3, 0, 6, 0],
        'values': ['Comparison', 3, 3, 6, 3],
        'fill': {'color': '#70AD47'},
    })
    
    comp_ws.insert_chart('F20', energy_chart)
    
    workbook.close()
    print(f"\n✓ Excel report saved to: {output_path}")


if __name__ == "__main__":
    output_file = Path(__file__).parent / "robustness_results.xlsx"
    create_excel_report(str(output_file))
