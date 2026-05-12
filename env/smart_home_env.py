"""
env/smart_home_env.py — Refactored Research-Grade HEMS Environment
==================================================================
Version: 4.0 (Senior Engineer Refactor)
- Chính xác hóa Power Balance.
- Hợp nhất Reward logic về utils/reward.py.
- Tách bạch Grid-side và Cell-side battery power.
"""

import yaml
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

from env.data.data_manager import DataManager
from env.physics.battery_model import update_battery_physics
from env.physics.thermal_model import update_thermal_physics
from utils.reward import calculate_hems_reward

# Load config from config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

class SmartHomeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_manager=None, max_episode_steps=96, random_start=True, seed=None):
        super().__init__()
        self.data_manager = data_manager or DataManager()
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start

        # Action: [batt_kw, hvac_kw] - Nhận giá trị vật lý từ Wrapper
        self.action_space = spaces.Box(
            low=np.array([-5.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 4.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-1.2, high=1.2, shape=(20,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._rng = np.random.default_rng(seed)
        
        self._data_step = int(self._rng.integers(0, self.data_manager.max_steps - self.max_episode_steps - 5)) if self.random_start else 0
        self._episode_step = 0
        
        # Trạng thái vật lý
        self._indoor_temp = 24.0 + self._rng.uniform(-1.0, 1.0)
        self._soc = float(self._rng.uniform(0.3, 0.7))
        self._prev_batt_grid_kw = 0.0
        self._prev_action_norm = np.zeros(2, dtype=np.float32)
        
        return self._compute_observation(), {}

    def step(self, action):
        # 1. Tiền xử lý Action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        action_norm = np.zeros(2, dtype=np.float32)
        action_norm[0] = action[0] / 5.0 # batt
        action_norm[1] = action[1] / 4.0 # hvac
        
        # 2. Lấy data:
        data = self.data_manager.get_step_data(self._data_step)
        outdoor_temp = data["outdoor_temp"]
        ghi_w_m2 = data.get("ghi_w_m2", 0.0)
        base_load_kw = data["base_load_kw"]
        pv_kw = data["pv_kw"]
        price = data["price_vnd_kwh"]

        # 3. Update battery:
        next_soc, batt_grid_kw, batt_cell_kw, battery_info = update_battery_physics(
            self._soc, action[0], outdoor_temp, CONFIG
        )

        # 4. Update thermal:
        next_indoor_temp, thermal_info = update_thermal_physics(
            self._indoor_temp, outdoor_temp, action[1], ghi_w_m2, base_load_kw, CONFIG
        )

        # 5. Power balance tại AC bus:
        net_load_kw = base_load_kw + action[1] + batt_grid_kw - pv_kw
        grid_import_kw = max(0.0, net_load_kw)
        grid_export_kw = max(0.0, -net_load_kw)

        dt_hours = CONFIG.get("environment", {}).get("dt_hours", 0.25)
        electricity_cost = grid_import_kw * dt_hours * price

        # Baseline demo tính cost để phục vụ reward saving
        rule_hvac_kw = 2.5 if self._indoor_temp > 26.0 else 0.0
        baseline_net_load_kw = base_load_kw + rule_hvac_kw - pv_kw
        baseline_grid_import_kw = max(0.0, baseline_net_load_kw)
        baseline_cost = baseline_grid_import_kw * dt_hours * price

        # 6. Gọi reward
        telemetry = {
            "electricity_cost": electricity_cost,
            "baseline_cost": baseline_cost,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "indoor_temp": next_indoor_temp,
            "outdoor_temp": outdoor_temp,
            "comfort_violation": thermal_info["comfort_violation"],
            "severe_overheat": thermal_info["severe_overheat"],
            "soc": next_soc,
            "batt_grid_kw": batt_grid_kw,
            "prev_batt_grid_kw": self._prev_batt_grid_kw,
            "battery_throughput_kwh": battery_info["battery_throughput_kwh"],
            "soc_health_violation": battery_info["soc_health_violation"],
            "hvac_input_kw": action[1],
            "prev_action_norm": self._prev_action_norm,
            "current_action_norm": action_norm
        }

        reward, reward_breakdown = calculate_hems_reward(telemetry, CONFIG)

        # 7. Update state
        self._soc = next_soc
        self._indoor_temp = next_indoor_temp
        self._prev_batt_grid_kw = batt_grid_kw
        self._prev_action_norm = action_norm.copy()
        
        self._data_step += 1
        self._episode_step += 1
        is_terminal = (self._episode_step >= self.max_episode_steps - 1)

        # 8. Info
        info = {
            "outdoor_temp": outdoor_temp,
            "ghi_w_m2": ghi_w_m2,
            "pv_kw": pv_kw,
            "base_load_kw": base_load_kw,
            "price_vnd_kwh": price,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "net_load_kw": net_load_kw,
            "indoor_temp": next_indoor_temp,
            "soc": next_soc,
            "batt_grid_kw": batt_grid_kw,
            "batt_cell_kw": batt_cell_kw,
            "hvac_input_kw": action[1],
            "cop": thermal_info["cop"],
            "q_envelope_kw": thermal_info["q_envelope_kw"],
            "q_solar_kw": thermal_info["q_solar_kw"],
            "q_internal_kw": thermal_info["q_internal_kw"],
            "q_infiltration_kw": thermal_info["q_infiltration_kw"],
            "q_hvac_kw": thermal_info["q_hvac_kw"],
            "d_temp": thermal_info["d_temp"],
            "comfort_violation": thermal_info["comfort_violation"],
            "severe_overheat": thermal_info["severe_overheat"],
            "weather_source": data.get("weather_source", "unknown"),
            "reward_breakdown": reward_breakdown
        }
        info.update(reward_breakdown)
        
        # backward compatibility
        info["batt_power_kw"] = batt_grid_kw
        info["electricity_cost"] = electricity_cost
        
        return self._compute_observation(), reward, False, is_terminal, info

    def _compute_observation(self):
        # Giữ nguyên cấu trúc 20 chiều như yêu cầu
        data = self.data_manager.get_step_data(self._data_step)
        obs = np.zeros(20, dtype=np.float32)
        
        def _norm(v, lo, hi): return np.clip(2.0*(v-lo)/(hi-lo) - 1.0, -1.2, 1.2)
        
        obs[0] = _norm(self._indoor_temp, 15, 40)
        obs[1] = _norm(data["outdoor_temp"], 15, 40)
        obs[2] = self._soc * 2.0 - 1.0
        obs[3] = _norm(data["base_load_kw"], 0, 5)
        obs[4] = _norm(data["pv_kw"], 0, 5)
        obs[5] = _norm(data["price_vnd_kwh"], 1000, 3500)
        obs[6] = _norm(data["base_load_kw"] - data["pv_kw"], -5, 5)
        obs[7] = self._prev_action_norm[1] # hvac state
        # ... các forecast khác giữ nguyên logic index
        obs[8:11] = [_norm(data[f"pv_fc_{h}h"], 0, 5) for h in [6, 12, 24]]
        obs[11:14] = [_norm(data[f"load_fc_{h}h"], 0, 5) for h in [6, 12, 24]]
        obs[14] = data["hour_sin"]
        obs[15] = data["hour_cos"]
        obs[16] = _norm(self._indoor_temp - 24.0, -4, 4)
        obs[17:20] = [_norm(data[f"price_fc_{h}h"], 1000, 3500) for h in [6, 12, 24]]
        
        return np.nan_to_num(obs)
