"""
utils/reward.py
===============
HEMS Optimization Reward Strategy (v9.0 - Sequential Flow & Arbitrage).
Mục tiêu: PV -> Load -> Battery -> Grid routing and aggressive economic incentives.
"""

import numpy as np

def calculate_hems_reward(telemetry: dict, config: dict) -> tuple[float, dict]:
    """
    HEMS Optimization Reward Strategy (v9.0).
    Scaled and Coupled rewards based on sequential energy flow decomposition.
    """
    # 1. Inputs
    grid_import = telemetry.get("grid_import_kw", 0.0)
    pv_export = telemetry.get("grid_export_kw", 0.0) # pv_export_kw passed as grid_export_kw
    
    pv_to_load = telemetry.get("pv_to_load", 0.0)
    pv_to_battery = telemetry.get("pv_to_battery", 0.0)
    battery_to_load = telemetry.get("battery_to_load", 0.0)
    
    pv_power = telemetry.get("pv_kw", 0.0)
    soc = telemetry.get("soc", 0.0)
    hour = telemetry.get("hour", 0.0)
    indoor_temp = telemetry.get("indoor_temp", 24.0)
    price = telemetry.get("price_vnd_kwh", 2000.0)
    high_price_threshold = config.get("pricing", {}).get("tou_peak", 3000.0)

    # --- 2. Scaled Economic Rewards ---
    # Rebalanced for less hacking: battery_use is rewarded more, grid is penalized.
    r_pv_to_load = 1.0 * pv_to_load
    r_pv_to_battery = 0.5 * pv_to_battery # Reduce incentive to just charge and hold
    r_battery_use = 1.5 * battery_to_load # Increase incentive to discharge
    
    r_grid = -1.0 * grid_import
    r_pv_waste = -0.5 * pv_export

    # --- 3. Independence & Self-Consumption ---
    # Continuous self-consumption objective without huge spikes
    pv_self_cons_ratio = (pv_to_load + pv_to_battery) / max(pv_power, 1e-6)
    r_self_consumption = 1.5 * pv_self_cons_ratio if pv_power > 0.1 else 0.0
    r_grid_avoid = 0.0 # Remove sparse bonus

    # --- 4. Thermal Comfort ---
    dist = max(0.0, 22.0 - indoor_temp, indoor_temp - 26.0)
    # Quadratic comfort penalty, increases fast if out of bounds. Scaled down.
    r_comfort = -0.8 * (dist ** 2)
    r_comfort_bonus = 0.0 # Remove sparse bonus

    # --- 5. Battery Health & TOU (Peak Shaving) ---
    # Continuous penalty for low SoC to avoid sparse spikes.
    # Replace sparse empty battery with continuous penalty if below 20%
    r_empty_battery = -2.0 * max(0.0, 0.20 - soc)
    
    # TOU-Aware Battery (Peak Shift)
    r_peak_shift = 0.0
    if price >= high_price_threshold:
        r_peak_shift = 2.0 * battery_to_load # Aggressive discharge reward in peak
        r_grid -= 1.5 * grid_import # Extra penalty for grid in peak

    # --- 6. Total Calculation ---
    raw_reward = (
        r_pv_to_load + r_pv_to_battery + r_battery_use + 
        r_grid + r_pv_waste + 
        r_self_consumption +
        r_comfort + 
        r_empty_battery + r_peak_shift
    )
    
    # Normalize and smooth reward to roughly [-5, +5] range
    clipped_reward = float(np.clip(raw_reward, -5.0, 5.0))

    reward_breakdown = {
        "r_grid": r_grid,
        "r_pv_to_load": r_pv_to_load,
        "r_pv_to_battery": r_pv_to_battery,
        "r_battery_use": r_battery_use,
        "r_pv_waste": r_pv_waste,
        "r_grid_avoid": r_grid_avoid,
        "r_self_consumption": r_self_consumption,
        "r_comfort": r_comfort,
        "r_comfort_bonus": r_comfort_bonus,
        "r_empty_battery": r_empty_battery,
        "r_peak_shift": r_peak_shift,
        "raw_reward": raw_reward,
        "clipped_reward": clipped_reward,
        # Behavioral telemetry for logger/plotter
        "pv_to_load": pv_to_load,
        "pv_to_battery": pv_to_battery,
        "battery_to_load": battery_to_load,
        "pv_waste_ratio": pv_export / (pv_power + 1e-6) if pv_power > 0.1 else 0.0
    }

    return clipped_reward, reward_breakdown
