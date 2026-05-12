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
    config: dict,
) -> tuple[float, dict]:
    """
    Cập nhật nhiệt độ phòng dựa trên cân bằng nhiệt mở rộng (RC 1R1C + Gains).
    
    Công thức: C * dT/dt = Q_envelope + Q_solar + Q_internal + Q_infiltration - Q_hvac
    - Q_envelope: Truyền nhiệt qua vỏ nhà (kW)
    - Q_solar: Nhiệt mặt trời qua kính (kW)
    - Q_internal: Nhiệt từ người & thiết bị (kW)
    - Q_infiltration: Nhiệt do rò rỉ khí (kW)
    - Q_hvac: Năng lượng làm mát (kW thermal)
    """
    R_envelope = config.get("physics", {}).get("thermal", {}).get("thermal_resistance", 2.5)
    C = config.get("physics", {}).get("thermal", {}).get("thermal_capacitance", 12.0)
    dt_hours = config.get("environment", {}).get("dt_hours", 0.25)
    
    # 1. Truyền nhiệt qua vỏ nhà
    q_envelope = (outdoor_temp - current_temp) / R_envelope
    
    # 2. Nhiệt mặt trời (kW)
    solar_gain_coeff = config.get("physics", {}).get("thermal", {}).get("solar_gain_coeff", 0.5)
    q_solar = solar_gain_coeff * ghi_w_m2 / 1000.0
    
    # 3. Nhiệt nội bộ
    base_int_kw = config.get("physics", {}).get("thermal", {}).get("internal_gain_base_kw", 0.1)
    frac_int = config.get("physics", {}).get("thermal", {}).get("internal_gain_fraction", 0.2)
    q_internal = base_int_kw + frac_int * base_load_kw
    
    # 4. Thông gió/xâm nhập khí
    infiltration_coeff = config.get("physics", {}).get("thermal", {}).get("infiltration_coeff", 0.05)
    q_infiltration = infiltration_coeff * (outdoor_temp - current_temp)
    
    # 5. Variable COP và HVAC cooling
    cop_nom = config.get("physics", {}).get("thermal", {}).get("cop_nominal", 2.5)
    cop_min = config.get("physics", {}).get("thermal", {}).get("cop_min", 1.0)
    k_deg = config.get("physics", {}).get("thermal", {}).get("cop_degradation_k", 0.02)
    
    delta_t = max(0.0, outdoor_temp - current_temp)
    cop = max(cop_min, cop_nom * (1.0 - k_deg * delta_t))
    q_hvac = hvac_input_kw * cop
    
    # 6. Cân bằng năng lượng và cập nhật nhiệt độ
    net_heat_kw = q_envelope + q_solar + q_internal + q_infiltration - q_hvac
    d_temp = net_heat_kw * dt_hours / C
    next_indoor_temp = float(np.clip(current_temp + d_temp, 10.0, 45.0))
    
    # 7. Tính toán Comfort Violations
    comfort_low = config.get("physics", {}).get("thermal", {}).get("comfort_low", 22.0)
    comfort_high = config.get("physics", {}).get("thermal", {}).get("comfort_high", 26.0)
    severe_overheat_threshold = config.get("physics", {}).get("thermal", {}).get("severe_overheat_threshold", 27.0)
    
    comfort_violation = max(0.0, comfort_low - next_indoor_temp, next_indoor_temp - comfort_high)
    severe_overheat = max(0.0, next_indoor_temp - severe_overheat_threshold)
    
    thermal_info = {
        "indoor_temp_prev": current_temp,
        "indoor_temp_next": next_indoor_temp,
        "outdoor_temp": outdoor_temp,
        "ghi_w_m2": ghi_w_m2,
        "cop": cop,
        "q_envelope_kw": q_envelope,
        "q_solar_kw": q_solar,
        "q_internal_kw": q_internal,
        "q_infiltration_kw": q_infiltration,
        "q_hvac_kw": q_hvac,
        "net_heat_kw": net_heat_kw,
        "d_temp": d_temp,
        "comfort_violation": comfort_violation,
        "severe_overheat": severe_overheat
    }
    
    return next_indoor_temp, thermal_info
