"""
utils/reward.py
===============
Trung tâm tính toán Reward đa mục tiêu (Research-Grade).
Tập trung vào: Chống lạm dụng pin, duy trì tiện nghi, và ổn định điều khiển.
"""

import numpy as np

# Ranh giới clipping để ổn định PPO (nhưng vẫn giữ raw để log)
REWARD_CLIP_MIN = -20.0
REWARD_CLIP_MAX =  5.0

def calculate_hems_reward(telemetry: dict, config: dict) -> tuple[float, dict]:
    """
    Tính toán phần thưởng toàn diện cho HEMS dựa trên telemetry vật lý.
    Bao gồm cả Bonus và Penalty để định hướng học tập.
    """
    cost_norm = config.get("reward", {}).get("cost_normalization", 5000.0)
    
    # 1. Chi phí điện (r_eco) và Saving (r_saving)
    electricity_cost = telemetry.get("electricity_cost", 0.0)
    baseline_cost = telemetry.get("baseline_cost", electricity_cost)
    
    r_eco = - (electricity_cost / cost_norm)
    
    w_saving = config.get("reward", {}).get("weights", {}).get("saving", 1.0)
    saving_vnd = baseline_cost - electricity_cost
    r_saving = w_saving * (saving_vnd / cost_norm)

    # 2. Comfort (r_comfort) và Comfort Bonus
    w_comfort = config.get("reward", {}).get("weights", {}).get("comfort", 2.0)
    comfort_violation = telemetry.get("comfort_violation", 0.0)
    r_comfort = - w_comfort * (comfort_violation ** 2)
    
    comfort_in_zone_bonus = config.get("reward", {}).get("bonuses", {}).get("comfort_in_zone", 0.2)
    r_comfort_bonus = comfort_in_zone_bonus if comfort_violation == 0 else 0.0

    # 3. Severe overheating (r_severe)
    w_severe = config.get("reward", {}).get("weights", {}).get("severe", 4.0)
    severe_overheat = telemetry.get("severe_overheat", 0.0)
    r_severe = - w_severe * (severe_overheat ** 2)

    # 4. Peak (r_peak)
    w_peak = config.get("reward", {}).get("weights", {}).get("peak", 0.5)
    peak_limit_kw = config.get("reward", {}).get("peak_limit_kw", 4.0)
    grid_import_kw = telemetry.get("grid_import_kw", 0.0)
    r_peak = - w_peak * max(0.0, grid_import_kw - peak_limit_kw) ** 2

    # 5. Battery degradation proxy (r_deg)
    w_deg = config.get("reward", {}).get("weights", {}).get("degradation", 0.1)
    battery_throughput_kwh = telemetry.get("battery_throughput_kwh", 0.0)
    r_deg = - w_deg * battery_throughput_kwh

    # 6. SoC health (r_soc) và SoC Bonus
    w_soc = config.get("reward", {}).get("weights", {}).get("soc_health", 1.0)
    soc_health_violation = telemetry.get("soc_health_violation", 0.0)
    r_soc = - w_soc * (soc_health_violation ** 2)
    
    soc = telemetry.get("soc", 0.5)
    healthy_soc_low = config.get("reward", {}).get("battery", {}).get("healthy_soc_low", 0.3)
    healthy_soc_high = config.get("reward", {}).get("battery", {}).get("healthy_soc_high", 0.8)
    soc_healthy_bonus = config.get("reward", {}).get("bonuses", {}).get("soc_healthy", 0.05)
    
    r_soc_bonus = soc_healthy_bonus if healthy_soc_low <= soc <= healthy_soc_high else 0.0

    # 7. Smooth action (r_smooth)
    w_smooth = config.get("reward", {}).get("weights", {}).get("smooth", 0.05)
    prev_action_norm = telemetry.get("prev_action_norm", np.zeros(2))
    current_action_norm = telemetry.get("current_action_norm", np.zeros(2))
    r_smooth = - w_smooth * np.sum((current_action_norm - prev_action_norm) ** 2)
    
    # 8. Battery switch penalty (r_switch)
    w_switch = config.get("reward", {}).get("weights", {}).get("switch", 0.05)
    prev_batt_grid_kw = telemetry.get("prev_batt_grid_kw", 0.0)
    batt_grid_kw = telemetry.get("batt_grid_kw", 0.0)
    eps = 0.1
    
    battery_switch = (
        abs(prev_batt_grid_kw) > eps
        and abs(batt_grid_kw) > eps
        and np.sign(prev_batt_grid_kw) != np.sign(batt_grid_kw)
    )
    r_switch = -w_switch if battery_switch else 0.0

    # 9. Export revenue (r_export) - optional, disabled by default
    export_revenue = telemetry.get("grid_export_kw", 0.0) * config.get("environment", {}).get("dt_hours", 0.25) * 0.0
    r_export = export_revenue / cost_norm

    # Tổng hợp phần thưởng
    raw_reward = (
        r_comfort_bonus + r_saving + r_soc_bonus +
        r_eco + r_comfort + r_severe + r_peak +
        r_deg + r_soc + r_smooth + r_switch + r_export
    )
    
    reward_min = config.get("reward", {}).get("reward_min", -20.0)
    reward_max = config.get("reward", {}).get("reward_max", 5.0)
    clipped_reward = float(np.clip(raw_reward, reward_min, reward_max))
    is_clipped = bool(raw_reward < reward_min or raw_reward > reward_max)

    reward_breakdown = {
        "r_eco": r_eco,
        "r_saving": r_saving,
        "r_comfort_bonus": r_comfort_bonus,
        "r_comfort": r_comfort,
        "r_severe": r_severe,
        "r_peak": r_peak,
        "r_deg": r_deg,
        "r_soc": r_soc,
        "r_soc_bonus": r_soc_bonus,
        "r_smooth": r_smooth,
        "r_switch": r_switch,
        
        "raw_reward": raw_reward,
        "clipped_reward": clipped_reward,
        "is_reward_clipped": is_clipped,
        
        "electricity_cost": electricity_cost,
        "baseline_cost": baseline_cost,
        "saving_vnd": saving_vnd,
        "comfort_violation": comfort_violation,
        "severe_overheat": severe_overheat,
        "battery_throughput_kwh": battery_throughput_kwh,
        "soc_health_violation": soc_health_violation,
        "grid_import_kw": grid_import_kw,
        "soc": soc,
        "batt_grid_kw": batt_grid_kw,
        "battery_switch": int(battery_switch)
    }

    return clipped_reward, reward_breakdown
