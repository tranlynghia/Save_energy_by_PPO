"""
utils/logger.py
===============
Nâng cấp RewardDecompositionLogger v4.0.
Ghi log chi tiết: Raw vs Clipped reward, Comfort violations, Battery health.
"""

import os
import csv
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardDecompositionLogger(BaseCallback):
    """
    Callback ghi lại chi tiết các thành phần reward và telemetry vật lý.
    """

    def __init__(self, log_dir: str, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = int(time.time())
        self.csv_path = os.path.join(log_dir, f"reward_decomposition_{timestamp}.csv")
        self._episode_rewards_per_env: list[list[dict]] = []
        
        # Header CSV (chuẩn v5.0)
        self.headers = [
            "timestep", "episode", "env_idx",
            "r_eco", "r_saving", "r_comfort_bonus", "r_comfort", "r_severe", 
            "r_deg", "r_soc", "r_soc_bonus", "r_switch", "r_smooth", "r_peak", "r_export",
            "raw_reward", "clipped_reward", "is_reward_clipped", "clip_fraction", 
            "comfort_violation", "severe_overheat", "grid_import_kw", "batt_grid_kw", 
            "soc", "indoor_temp", "electricity_cost", "baseline_cost", "saving_vnd", 
            "battery_switch", "ep_len"
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        self._episode_count = 0

    def _on_training_start(self) -> None:
        self._episode_rewards_per_env = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if info is None: continue
            rb = info.get("reward_breakdown", None)
            if rb is None: continue
            
            is_clipped = 1.0 if rb.get("is_reward_clipped", False) else 0.0
            batt_grid_kw = info.get("batt_power_kw", 0.0)

            data = {
                "r_eco":            rb.get("r_eco", 0.0),
                "r_saving":         rb.get("r_saving", 0.0),
                "r_comfort_bonus":  rb.get("r_comfort_bonus", 0.0),
                "r_comfort":        rb.get("r_comfort", 0.0),
                "r_severe":         rb.get("r_severe", 0.0),
                "r_deg":            rb.get("r_deg", 0.0),
                "r_soc":            rb.get("r_soc", 0.0),
                "r_soc_bonus":      rb.get("r_soc_bonus", 0.0),
                "r_switch":         rb.get("r_switch", 0.0),
                "r_smooth":         rb.get("r_smooth", 0.0),
                "r_peak":           rb.get("r_peak", 0.0),
                "r_export":         rb.get("r_export", 0.0),
                
                "raw_reward":       rb.get("raw_reward", 0.0),
                "clipped_reward":   rb.get("clipped_reward", 0.0),
                "is_reward_clipped":is_clipped,
                "clip_fraction":    is_clipped,
                
                "comfort_violation":rb.get("comfort_violation", 0.0),
                "severe_overheat":  rb.get("severe_overheat", 0.0),
                "grid_import_kw":   info.get("grid_import_kw", 0.0),
                "batt_grid_kw":     batt_grid_kw,
                "soc":              info.get("soc", 0.0),
                "indoor_temp":      info.get("indoor_temp", 0.0),
                "electricity_cost": rb.get("electricity_cost", 0.0),
                "baseline_cost":    rb.get("baseline_cost", 0.0),
                "saving_vnd":       rb.get("saving_vnd", 0.0),
                "battery_switch":   rb.get("battery_switch", 0.0)
            }
            self._episode_rewards_per_env[i].append(data)

            # TensorBoard step-level
            if self.logger is not None:
                for k, v in data.items():
                    self.logger.record(f"step_reward/{k}", v)

            if dones[i]:
                self._log_episode(i)

        return True

    def _log_episode(self, env_idx: int):
        rewards_list = self._episode_rewards_per_env[env_idx]
        n = len(rewards_list)
        if n == 0: return

        self._episode_count += 1
        
        # Tính trung bình các cột cho episode
        means = {}
        for k in self.headers[3:-1]: 
            means[k] = float(np.mean([s[k] for s in rewards_list]))

        # Các chỉ số health
        comfort_violation_rate = float(np.mean([1.0 if s["comfort_violation"] > 0 else 0.0 for s in rewards_list]))
        severe_overheat_hours = float(np.sum([0.25 for s in rewards_list if s["severe_overheat"] > 0])) # 0.25h per step
        
        avg_soc = means["soc"]
        # Use throughput directly from telemetry or sum
        battery_throughput = float(np.sum([abs(s["batt_grid_kw"]) * 0.25 for s in rewards_list]))
        
        battery_switch_count = int(np.sum([s["battery_switch"] for s in rewards_list]))
        reward_clip_fraction = means["is_reward_clipped"]
        
        total_saving_vnd = float(np.sum([s["saving_vnd"] for s in rewards_list]))
        avg_saving_vnd = means["saving_vnd"]
        episode_return = float(np.sum([s["clipped_reward"] for s in rewards_list]))
        avg_step_reward = means["clipped_reward"]

        # TensorBoard episode-level
        if self.logger is not None:
            for k, v in means.items():
                self.logger.record(f"episode_reward/{k}", v)
            
            # Ghi health metrics
            self.logger.record("health/comfort_violation_rate", comfort_violation_rate)
            self.logger.record("health/severe_overheat_hours", severe_overheat_hours)
            self.logger.record("health/avg_soc", avg_soc)
            self.logger.record("health/battery_throughput", battery_throughput)
            self.logger.record("health/battery_switch_count", battery_switch_count)
            self.logger.record("health/reward_clip_fraction", reward_clip_fraction)
            self.logger.record("health/total_saving_vnd", total_saving_vnd)
            self.logger.record("health/avg_saving_vnd", avg_saving_vnd)
            self.logger.record("health/episode_return", episode_return)
            self.logger.record("health/avg_step_reward", avg_step_reward)

            self.logger.dump(self.num_timesteps)

        # Ghi CSV
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    self._episode_count,
                    env_idx,
                    *[means[k] for k in self.headers[3:-1]],
                    n
                ])
        except Exception as e:
            print(f"[LoggerError] {e}")

        self._episode_rewards_per_env[env_idx] = []
