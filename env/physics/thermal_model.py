"""
env/physics/thermal_model.py
============================
Mô hình nhiệt động học RC (Resistor-Capacitor).
Tích hợp Variable COP dựa trên chênh lệch nhiệt độ.
"""

import numpy as np

def update_thermal_physics(
    current_temp: float,
    outdoor_temp: float,
    hvac_input_kw: float,
    ghi_w_m2: float,
    base_load_kw: float,
    wind_speed: float,
    config: dict,
) -> tuple[float, dict]:
    """
    Cập nhật nhiệt độ phòng dựa trên cân bằng nhiệt mở rộng (RC 1R1C + Gains).
    Version 5.0: Enhanced Physics-Aware Damping & Clamping.
    """
    # 1. Physical Parameters
    R_envelope = config.get("physics", {}).get("thermal", {}).get("thermal_resistance", 2.0)
    C = config.get("physics", {}).get("thermal", {}).get("thermal_capacitance", 40.0) # More inertia
    dt_hours = config.get("environment", {}).get("dt_hours", 0.25)
    
    # 2. Heat Gains (kW)
    q_envelope = (outdoor_temp - current_temp) / R_envelope
    
    solar_gain_coeff = config.get("physics", {}).get("thermal", {}).get("solar_gain_coeff", 0.5)
    q_solar = solar_gain_coeff * ghi_w_m2 / 1000.0
    
    base_int_kw = config.get("physics", {}).get("thermal", {}).get("internal_gain_base_kw", 0.1)
    frac_int = config.get("physics", {}).get("thermal", {}).get("internal_gain_fraction", 0.2)
    q_internal = base_int_kw + frac_int * base_load_kw
    
    infiltration_base = config.get("physics", {}).get("thermal", {}).get("infiltration_coeff", 0.05)
    q_infiltration = infiltration_base * (1.0 + 0.1 * wind_speed) * (outdoor_temp - current_temp)
    
    # 3. HVAC with Physics-Aware Damping
    # HVAC becomes less effective as indoor temp drops (approach thermodynamic floor)
    floor_temp = 16.0
    efficiency_factor = np.clip((current_temp - floor_temp) / (22.0 - floor_temp), 0.0, 1.0)
    
    cop_nom = config.get("physics", {}).get("thermal", {}).get("cop_nominal", 3.0)
    cop_min = config.get("physics", {}).get("thermal", {}).get("cop_min", 1.0)
    k_deg = config.get("physics", {}).get("thermal", {}).get("cop_degradation_k", 0.02)
    
    delta_t = max(0.0, outdoor_temp - current_temp)
    cop = max(cop_min, cop_nom * (1.0 - k_deg * delta_t))
    q_hvac = hvac_input_kw * cop * efficiency_factor # Diminishing returns near 16°C
    
    # 4. Energy Balance & Integration
    net_heat_kw = q_envelope + q_solar + q_internal + q_infiltration - q_hvac
    d_temp = net_heat_kw * dt_hours / C
    
    # Physical Clamping [16°C - 35°C]
    next_indoor_temp = float(np.clip(current_temp + d_temp, 16.0, 35.0))
    
    # 5. Symmetric Comfort Metrics
    comfort_low = 22.0
    comfort_high = 26.0
    
    comfort_violation = max(0.0, comfort_low - next_indoor_temp, next_indoor_temp - comfort_high)
    severe_overheat = max(0.0, next_indoor_temp - 30.0)
    severe_overcool = max(0.0, 20.0 - next_indoor_temp)
    
    thermal_info = {
        "indoor_temp_prev": current_temp,
        "indoor_temp_next": next_indoor_temp,
        "outdoor_temp": outdoor_temp,
        "q_envelope_kw": q_envelope,
        "q_solar_kw": q_solar,
        "q_internal_kw": q_internal,
        "q_infiltration_kw": q_infiltration,
        "q_hvac_kw": q_hvac,
        "cop": cop,
        "comfort_violation": comfort_violation,
        "severe_overheat": severe_overheat,
        "severe_overcool": severe_overcool,
        "d_temp": d_temp
    }
    
    return next_indoor_temp, thermal_info
