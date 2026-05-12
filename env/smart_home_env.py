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

        # Observation Space: [outdoor_temp, ghi, soc, base_load, indoor_temp, hour_sin, hour_cos, price, next_pv, next_temp]
        obs_dim = 10
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * obs_dim, dtype=np.float32),
            high=np.array([1.0] * obs_dim, dtype=np.float32),
            dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._rng = np.random.default_rng(seed)
        
        # Support random start day/hour
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
        ghi_w_m2 = data["ghi_w_m2"]
        base_load_kw = data["base_load_kw"]
        pv_kw = data["pv_kw"]
        price = data["price_vnd_kwh"]

        # 3. Sequential Energy Flow (v9.0 Strategy)
        total_load = base_load_kw + action[1]
        remaining_load = total_load
        
        # 3.1 PV serves load first
        pv_to_load = min(pv_kw, remaining_load)
        remaining_load -= pv_to_load
        pv_surplus = max(0.0, pv_kw - pv_to_load)
        
        # 3.2 PV surplus charges battery
        # Determine battery charge request: If SoC is low, force charge from PV
        if self._soc <= 0.12: # min_soc + 2%
            batt_charge_request = 5.0 # Max request
        else:
            batt_charge_request = max(0.0, action[0])
            
        # Use update_battery_physics to handle constraints and SoC update
        # We pass the batt_charge_request but cap it by pv_surplus for grid_batt calculation later
        next_soc, actual_batt_grid_kw, _, _ = update_battery_physics(
            self._soc, batt_charge_request, outdoor_temp, CONFIG
        )
        
        # Routing: In this flow, we only charge from PV
        pv_to_battery = min(max(0.0, actual_batt_grid_kw), pv_surplus)
        
        # Re-run battery physics with actual pv_to_battery to get correct SoC if clipped
        # Actually, we can just update SoC manually if we want to be simple, but let's be consistent
        next_soc, actual_batt_grid_kw, _, _ = update_battery_physics(
            self._soc, pv_to_battery, outdoor_temp, CONFIG
        )
        self._soc = next_soc
        
        # 3.3 Battery serves remaining load
        batt_discharge_request = min(0.0, action[0]) # discharge is negative
        # Discharge limit is capped by remaining_load
        # We want to discharge min(remaining_load, abs(request))
        discharge_target = -min(remaining_load, abs(batt_discharge_request))
        
        next_soc, actual_batt_grid_kw, _, _ = update_battery_physics(
            self._soc, discharge_target, outdoor_temp, CONFIG
        )
        battery_to_load = max(0.0, -actual_batt_grid_kw)
        remaining_load -= battery_to_load
        self._soc = next_soc
        
        # 4. Final Balance
        grid_import_kw = max(0.0, remaining_load)
        pv_export_kw = max(0.0, pv_surplus - pv_to_battery)
        
        # Total battery power at grid level for logging (net of charge/discharge)
        # Note: In this sequential flow, they happen in sequence, so it's a bit different
        # but for power balance:
        batt_grid_kw = pv_to_battery - battery_to_load

        # 5. Update thermal:
        next_indoor_temp, thermal_info = update_thermal_physics(
            self._indoor_temp, outdoor_temp, action[1], ghi_w_m2, base_load_kw, data.get("wind_speed", 2.0), CONFIG
        )

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
            "grid_import_kw":   grid_import_kw,
            "grid_export_kw":   pv_export_kw, # renamed to match balance
            "pv_kw":            pv_kw,
            "pv_to_load":       pv_to_load,
            "pv_to_battery":    pv_to_battery,
            "battery_to_load":  battery_to_load,
            "hvac_input_kw":    action[1],
            "base_load_kw":     base_load_kw,
            "batt_grid_kw":     batt_grid_kw,
            "indoor_temp":      next_indoor_temp,
            "price_vnd_kwh":    price,
            "hour":             data["hour"],
            "soc":              self._soc
        }

        reward, reward_breakdown = calculate_hems_reward(telemetry, CONFIG)

        # 7. Update state
        self._indoor_temp = next_indoor_temp
        self._prev_batt_grid_kw = batt_grid_kw
        self._prev_action_norm = action_norm.copy()
        # SoC already updated during sequential flow
        
        self._data_step += 1
        self._episode_step += 1
        is_terminal = (self._episode_step >= self.max_episode_steps - 1)

        # 8. Info & Telemetry for Logging
        info = {
            "outdoor_temp": outdoor_temp,
            "ghi_w_m2": ghi_w_m2,
            "pv_kw": pv_kw,
            "pv_to_load": reward_breakdown.get("pv_to_load", 0.0),
            "pv_to_battery": reward_breakdown.get("pv_to_battery", 0.0),
            "battery_to_load": reward_breakdown.get("battery_to_load", 0.0),
            "pv_waste_ratio": reward_breakdown.get("pv_waste_ratio", 0.0),
            "self_consumption_ratio": reward_breakdown.get("self_consumption", 0.0),
            "grid_dependency": grid_import_kw / (base_load_kw + action[1] + max(0, batt_grid_kw) + 1e-6),
            "base_load_kw": base_load_kw,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": pv_export_kw,
            "indoor_temp": next_indoor_temp,
            "soc": self._soc,
            "batt_grid_kw": batt_grid_kw,
            "hvac_input_kw": action[1],
            "cop": thermal_info["cop"],
            "q_envelope_kw": thermal_info.get("q_envelope_kw", 0.0),
            "q_solar_kw": thermal_info["q_solar_kw"],
            "q_hvac_kw": thermal_info["q_hvac_kw"],
            "comfort_violation": thermal_info["comfort_violation"],
            "severe_overheat": thermal_info["severe_overheat"],
            "severe_overcool": thermal_info.get("severe_overcool", 0.0),
            "reward_breakdown": reward_breakdown
        }
        info.update(reward_breakdown)
        
        # backward compatibility
        info["batt_power_kw"] = batt_grid_kw
        info["electricity_cost"] = electricity_cost
        
        return self._compute_observation(), reward, False, is_terminal, info

    def _compute_observation(self):
        data = self.data_manager.get_step_data(self._data_step)
        next_data = self.data_manager.get_step_data(self._data_step + 4) # +1h (4 steps)
        
        obs = np.zeros(10, dtype=np.float32)
        
        def _norm(v, lo, hi): return np.clip(2.0*(v-lo)/(hi-lo) - 1.0, -1.0, 1.0)
        
        # 1-5: Core state
        obs[0] = _norm(self._indoor_temp, 16, 35)
        obs[1] = _norm(data["outdoor_temp"], 10, 45)
        obs[2] = self._soc * 2.0 - 1.0
        obs[3] = _norm(data["base_load_kw"], 0, 5)
        obs[4] = _norm(data["pv_kw"], 0, 5)
        
        # 6-10: Temporal & Forecasts
        obs[5] = data["hour_sin"]
        obs[6] = data["hour_cos"]
        obs[7] = _norm(data["price_vnd_kwh"], 1000, 4000)
        obs[8] = _norm(next_data["pv_kw"], 0, 5)
        obs[9] = _norm(next_data["outdoor_temp"], 10, 45)
        
        return np.nan_to_num(obs).astype(np.float32)
