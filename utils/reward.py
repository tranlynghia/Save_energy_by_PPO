"""
utils/reward.py
===============
Hàm phần thưởng đa mục tiêu với các cải tiến baseline:
  - NaN guard: kiểm tra đầu vào trước khi tính
  - Reward clipping: giới hạn tổng reward trong khoảng hợp lý
  - Reward decomposition: trả về từng thành phần riêng biệt để log
  - Stable scaling: chia nhỏ để tránh gradient explosion
"""

import numpy as np


# ---------------------------------------------------------------------------
# Giới hạn clip reward (sau khi đã scale) để tránh outlier phá mạng
# ---------------------------------------------------------------------------
REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX =  2.0


def _check_nan(value: float, name: str) -> float:
    """Trả về 0.0 và cảnh báo nếu giá trị là NaN/Inf."""
    if not np.isfinite(value):
        print(f"[RewardWARN] {name} = {value} → replaced with 0.0")
        return 0.0
    return float(value)


def calculate_multi_objective_reward(state, action, power_consumed: float, env_params: dict):
    """
    Tính toán Hàm phần thưởng đa mục tiêu theo mô hình toán học (Mục 7.4).

    Các cải tiến so với phiên bản demo:
    1. NaN guard: kiểm tra mọi đầu vào
    2. Reward clipping: [-10, 2] sau khi scale
    3. Decomposed logging: trả về từng thành phần riêng

    :param state:           Mảng numpy trạng thái hiện tại.
    :param action:          Từ điển hành động.
    :param power_consumed:  Tổng công suất điện sử dụng (kW).
    :param env_params:      Từ điển chứa các thông số thực tế của môi trường.
    :return: (tổng_phần_thưởng_clipped, từ_điển_chi_tiết)
    """

    # ------------------------------------------------------------------
    # Guard: kiểm tra đầu vào hợp lệ
    # ------------------------------------------------------------------
    power_consumed = _check_nan(power_consumed, "power_consumed")
    power_consumed = float(np.clip(power_consumed, 0.0, 1e6))  # không âm, không quá lớn

    # ------------------------------------------------------------------
    # 1. Đọc trọng số từ env_params
    # ------------------------------------------------------------------
    w_eco     = float(env_params.get("w_eco",     1.0))
    w_peak    = float(env_params.get("w_peak",    2.0))
    w_comfort = float(env_params.get("w_comfort", 10.0))
    w_task    = float(env_params.get("w_task",    10.0))

    # ==================================================================
    # R_eco: Chi phí điện (Cost Reward - Quy mô Tòa nhà)
    #   Chia cho 80000.0 (2000 * 40) để đưa về dải [-5, 0]
    # ==================================================================
    current_price = _check_nan(float(env_params.get("current_price", 2.0)), "current_price")
    r_eco = -(power_consumed * current_price) / 80000.0

    # ==================================================================
    # R_peak: Phạt Công Suất Đỉnh (Peak Penalty - Quy mô Tòa nhà)
    # ==================================================================
    peak_limit   = float(env_params.get("power_limit", 160.0))
    excess_power = max(0.0, power_consumed - peak_limit)
    r_peak = -(excess_power * 1.0)  # Giảm hệ số từ 20 -> 1 do excess_power đã lớn gấp 40 lần

    # ==================================================================
    # R_comfort: Phạt Tiện nghi (Comfort Reward)
    #   Dùng hàm bậc 2 nhẹ nhàng, tránh nổ số để PPO tập trung vào Cost
    # ==================================================================
    current_temp = _check_nan(float(env_params.get("current_indoor_temp", 25.0)), "current_indoor_temp")
    COMFORT_LOW  = 23.0
    COMFORT_HIGH = 27.0

    if current_temp < COMFORT_LOW:
        diff = COMFORT_LOW - current_temp
        temp_penalty = -(diff ** 1.5) # Giảm độ dốc từ bậc 2 xuống 1.5
    elif current_temp > COMFORT_HIGH:
        diff = current_temp - COMFORT_HIGH
        temp_penalty = -(diff ** 1.5)
    else:
        temp_penalty = 0.0

    fridge_temp    = _check_nan(float(env_params.get("current_fridge_temp", 4.0)), "current_fridge_temp")
    fridge_penalty = -2.0 if (fridge_temp > 8.0 or fridge_temp < 2.0) else 0.0 # Giảm từ -10 -> -2

    r_comfort = temp_penalty + fridge_penalty

    # ==================================================================
    # R_task: Hình phạt quá hạn (Deadline Penalty)
    # ==================================================================
    washer_deadline  = env_params.get("washer_deadline", 10)
    washer_completed = env_params.get("washer_completed", False)
    task_penalty = 0.0
    if not washer_completed:
        if washer_deadline <= 0:
            task_penalty = -5.0 # Giảm mạnh từ -50 -> -5
        elif washer_deadline < 12: 
            task_penalty = -1.0

    r_task = task_penalty

    # ==================================================================
    # R_deg: Battery Degradation Penalty (Pin hao mòn)
    # ==================================================================
    from configs.config_loader import ConfigLoader
    alpha = float(ConfigLoader.get("stochastic.degradation_weights.alpha", 2.0))
    beta  = float(ConfigLoader.get("stochastic.degradation_weights.beta",  1.0))

    battery_power = _check_nan(float(env_params.get("battery_power", 0.0)), "battery_power")
    current_soc   = _check_nan(float(env_params.get("current_soc",   0.5)), "current_soc")

    r_deg = -(alpha * abs(battery_power) + beta * (current_soc - 0.5) ** 2)

    # ==================================================================
    # TỔNG PHẦN THƯỞNG (trước khi scale)
    # ==================================================================
    weighted_eco     = w_eco     * r_eco
    weighted_peak    = w_peak    * r_peak
    weighted_comfort = w_comfort * r_comfort
    weighted_task    = w_task    * r_task

    total_reward = weighted_eco + weighted_peak + weighted_comfort + weighted_task + r_deg

    # ------------------------------------------------------------------
    # Stable Scaling: Giảm scaling factor xuống vì r_eco và penalties đã nhỏ
    # ------------------------------------------------------------------
    reward_scale = 10.0  # Giảm từ 100 -> 10 để signal rõ hơn cho PPO
    total_reward_scaled = total_reward / reward_scale

    # ------------------------------------------------------------------
    # Reward Clipping: giới hạn trong khoảng [CLIP_MIN, CLIP_MAX]
    # ------------------------------------------------------------------
    total_reward_clipped = float(np.clip(total_reward_scaled, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

    # ------------------------------------------------------------------
    # NaN guard cuối cùng
    # ------------------------------------------------------------------
    if not np.isfinite(total_reward_clipped):
        print(f"[RewardERROR] total_reward_clipped = {total_reward_clipped} → clamped to 0.0")
        total_reward_clipped = 0.0

    # ------------------------------------------------------------------
    # Reward Decomposition Dict (dùng cho TensorBoard + CSV logger)
    # ------------------------------------------------------------------
    reward_info = {
        # Thành phần chưa scale (để debug)
        "r_eco":           weighted_eco,
        "r_peak":          weighted_peak,
        "r_comfort":       weighted_comfort,
        "r_task":          weighted_task,
        "r_deg":           r_deg,
        # Tổng
        "total":           total_reward,
        "scaled_total":    total_reward_scaled,
        "clipped_total":   total_reward_clipped,
        # Thành phần gốc (chưa nhân trọng số) — dùng cho phân tích
        "cost_reward":     r_eco,
        "comfort_reward":  r_comfort,
        "battery_penalty": r_deg,
        "peak_penalty":    r_peak,
    }

    return total_reward_clipped, reward_info
