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

def calculate_hems_reward(
    env_info, 
    action_norm, 
    prev_action_norm, 
    is_terminal,
    config
):
    """
    Tính toán phần thưởng toàn diện cho HEMS.
    
    :param env_info:        Dict chứa các thông số vật lý (grid_import, temp, soc, v.v.)
    :param action_norm:     Hành động hiện tại (normalized [-1, 1])
    :param prev_action_norm: Hành động trước đó (normalized [-1, 1])
    :param is_terminal:     Đánh dấu kết thúc episode
    :param config:          Cấu hình trọng số
    
    :return: (clipped_reward, reward_breakdown)
    """
    
    # --- 1. Economic Reward (r_eco) ---
    # Chỉ tính dựa trên chi phí tiền điện thực tế
    cost_norm = config.get("reward.cost_normalization", 5000.0)
    electricity_cost = env_info["electricity_cost"]
    r_eco = -(electricity_cost / cost_norm)

    # --- 2. Comfort Reward (r_comfort) ---
    # Vùng an toàn 22-26 C. Ngoài vùng này phạt Quadratic.
    indoor_temp = env_info["indoor_temp"]
    w_comfort = config.get("reward.weights.comfort", 1.0)
    
    comfort_vio = max(0.0, 22.0 - indoor_temp, indoor_temp - 26.0)
    r_comfort = -w_comfort * (comfort_vio ** 2)
    
    # Hard constraint violation
    if indoor_temp > 27.0 or indoor_temp < 21.0:
        r_comfort -= 10.0

    # --- 3. Battery Degradation & Health (r_deg + r_soc) ---
    batt_grid_kw = env_info["batt_grid_kw"]
    soc = env_info["soc"]
    w_deg = config.get("reward.weights.degradation", 0.1)
    w_soc = config.get("reward.weights.soc_health", 2.0)
    
    # Throughput penalty
    r_deg = -w_deg * abs(batt_grid_kw)
    
    # SoC Healthy Zone (30% - 80%)
    r_soc = 0.0
    if soc < 0.3:
        r_soc = -w_soc * (0.3 - soc)**2
    elif soc > 0.8:
        r_soc = -w_soc * (soc - 0.8)**2

    # --- 4. Anti-Chattering (r_switch) ---
    # Phạt nếu đảo chiều sạc/xả liên tục
    r_switch = 0.0
    w_switch = config.get("reward.weights.switch", 0.5)
    prev_batt_p = env_info.get("prev_batt_grid_kw", 0.0)
    
    if np.sign(prev_batt_p) != np.sign(batt_grid_kw) and abs(prev_batt_p) > 0.1 and abs(batt_grid_kw) > 0.1:
        r_switch = -w_switch

    # --- 5. Action Smoothness (r_smooth) ---
    # Tính trên không gian chuẩn hóa [-1, 1]
    w_smooth = config.get("reward.weights.smooth", 0.5)
    r_smooth = -w_smooth * np.sum((action_norm - prev_action_norm)**2)

    # --- 6. Peak Shaving (r_peak) ---
    peak_limit = config.get("reward.peak_limit", 4.0)
    grid_import = env_info["grid_import_kw"]
    r_peak = -1.0 * max(0.0, grid_import - peak_limit)**2

    # --- 7. Terminal SoC (r_terminal) ---
    r_terminal = 0.0
    if is_terminal:
        r_terminal = -2.0 * abs(soc - 0.5)

    # --- Tổng hợp ---
    raw_reward = r_eco + r_comfort + r_deg + r_soc + r_switch + r_smooth + r_peak + r_terminal
    
    clipped_reward = float(np.clip(raw_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))
    is_clipped = bool(raw_reward < REWARD_CLIP_MIN or raw_reward > REWARD_CLIP_MAX)

    reward_breakdown = {
        "r_eco": r_eco,
        "r_comfort": r_comfort,
        "r_deg": r_deg,
        "r_soc": r_soc,
        "r_switch": r_switch,
        "r_smooth": r_smooth,
        "r_peak": r_peak,
        "r_terminal": r_terminal,
        "raw_reward": raw_reward,
        "clipped_reward": clipped_reward,
        "is_reward_clipped": is_clipped,
        "comfort_violation": comfort_vio
    }

    return clipped_reward, reward_breakdown
