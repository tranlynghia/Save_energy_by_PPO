"""
env/physics/battery_model.py
============================
Mô hình hóa Pin lưu trữ (ESS) chuẩn vật lý.
Quy ước dấu:
- batt_grid_kw > 0: Sạc (Lấy điện từ lưới, tăng Net Load)
- batt_grid_kw < 0: Xả (Cấp điện cho nhà, giảm Net Load)
"""

import numpy as np

def update_battery_physics(
    current_soc: float,
    requested_batt_kw: float,
    outdoor_temp: float,
    config: dict,
) -> tuple[float, float, float, dict]:
    """
    Cập nhật trạng thái Pin dựa trên yêu cầu từ Agent và các ràng buộc vật lý.
    """
    capacity_kwh = config.get("physics", {}).get("battery", {}).get("capacity_kwh", 10.0)
    max_charge_kw = config.get("physics", {}).get("battery", {}).get("max_charge_kw", 5.0)
    max_discharge_kw = config.get("physics", {}).get("battery", {}).get("max_discharge_kw", 5.0)
    min_soc = config.get("physics", {}).get("battery", {}).get("min_soc", 0.05)
    max_soc = config.get("physics", {}).get("battery", {}).get("max_soc", 0.95)
    eta_charge = config.get("physics", {}).get("battery", {}).get("charge_efficiency", 0.95)
    eta_discharge = config.get("physics", {}).get("battery", {}).get("discharge_efficiency", 0.95)
    temp_min_c = config.get("physics", {}).get("battery", {}).get("temp_min_c", 0.0)
    temp_max_c = config.get("physics", {}).get("battery", {}).get("temp_max_c", 45.0)
    dt_hours = config.get("environment", {}).get("dt_hours", 0.25)

    # 1. Ràng buộc TCVN (Nhiệt độ - Derating)
    temp_derating_factor = 1.0
    status = "ok"
    if outdoor_temp < temp_min_c or outdoor_temp > temp_max_c:
        temp_derating_factor = 0.0
        status = "temp_lock"
    elif outdoor_temp > 40.0:
        temp_derating_factor = 0.5
        status = "temp_derating"

    # 2. Tính toán SOC limit
    energy_headroom_kwh = (max_soc - current_soc) * capacity_kwh
    soc_headroom_limit_kw = (energy_headroom_kwh / dt_hours) / eta_charge

    energy_available_kwh = (current_soc - min_soc) * capacity_kwh
    soc_available_limit_kw = (energy_available_kwh / dt_hours) * eta_discharge

    # 3. Tính toán công suất thực tế
    actual_batt_kw = requested_batt_kw * temp_derating_factor

    if actual_batt_kw > 0:
        # Sạc
        batt_grid_kw = min(actual_batt_kw, max_charge_kw, soc_headroom_limit_kw)
        batt_cell_kw = batt_grid_kw * eta_charge
    elif actual_batt_kw < 0:
        # Xả
        batt_grid_kw = -min(abs(actual_batt_kw), max_discharge_kw, soc_available_limit_kw)
        batt_cell_kw = batt_grid_kw / eta_discharge
    else:
        batt_grid_kw = 0.0
        batt_cell_kw = 0.0

    delta_soc = (batt_cell_kw * dt_hours) / capacity_kwh
    next_soc = float(np.clip(current_soc + delta_soc, min_soc, max_soc))

    # 4. Tính toán battery health / telemetry
    healthy_soc_low = config.get("physics", {}).get("battery", {}).get("healthy_soc_low", 0.3)
    healthy_soc_high = config.get("physics", {}).get("battery", {}).get("healthy_soc_high", 0.8)

    if next_soc < healthy_soc_low:
        soc_health_violation = healthy_soc_low - next_soc
    elif next_soc > healthy_soc_high:
        soc_health_violation = next_soc - healthy_soc_high
    else:
        soc_health_violation = 0.0

    battery_throughput_kwh = abs(batt_cell_kw) * dt_hours
    hit_min_soc = bool(next_soc <= min_soc + 1e-4)
    hit_max_soc = bool(next_soc >= max_soc - 1e-4)

    battery_info = {
        "requested_batt_kw": requested_batt_kw,
        "batt_grid_kw": batt_grid_kw,
        "batt_cell_kw": batt_cell_kw,
        "actual_batt_kw": actual_batt_kw,
        "current_soc": current_soc,
        "next_soc": next_soc,
        "delta_soc": delta_soc,
        "capacity_kwh": capacity_kwh,
        "eta_charge": eta_charge,
        "eta_discharge": eta_discharge,
        "battery_throughput_kwh": battery_throughput_kwh,
        "soc_health_violation": soc_health_violation,
        "hit_min_soc": hit_min_soc,
        "hit_max_soc": hit_max_soc,
        "temp_derating_factor": temp_derating_factor,
        "status": status
    }

    return next_soc, batt_grid_kw, batt_cell_kw, battery_info
