"""
utils/logger.py
===============
Nâng cấp RewardDecompositionLogger v8.0.
Ghi log chi tiết các chỉ số Temporal Intelligence: PV waste, early dump, energy reserve.
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
        
        # Header CSV (Temporal Focused v8.0)
        self.headers = [
            "timestep", "episode", "env_idx",
            "r_grid", "r_pv_to_battery", "r_battery_use", "r_peak_shaving",
            "r_pv_waste", "r_early_dump", "r_energy_reserve",
            "r_comfort", "r_comfort_bonus", "r_severe_hot", "r_severe_cold",
            "raw_reward", "clipped_reward", "is_reward_clipped",
            "grid_import_kw", "pv_kw", "soc", "indoor_temp", "ep_len"
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
            
            is_clipped = 1.0 if rb.get("clipped_reward", 0.0) != rb.get("raw_reward", 0.0) else 0.0

            data = {
                "r_grid":           rb.get("r_grid", 0.0),
                "r_pv_to_battery":  rb.get("r_pv_to_battery", 0.0),
                "r_battery_use":    rb.get("r_battery_use", 0.0),
                "r_peak_shaving":   rb.get("r_peak_shaving", 0.0),
                "r_pv_waste":       rb.get("r_pv_waste", 0.0),
                "r_early_dump":     rb.get("r_early_dump", 0.0),
                "r_energy_reserve": rb.get("r_energy_reserve", 0.0),
                "r_comfort":        rb.get("r_comfort", 0.0),
                "r_comfort_bonus":  rb.get("r_comfort_bonus", 0.0),
                "r_severe_hot":     rb.get("r_severe_hot", 0.0),
                "r_severe_cold":    rb.get("r_severe_cold", 0.0),
                
                "raw_reward":       rb.get("raw_reward", 0.0),
                "clipped_reward":   rb.get("clipped_reward", 0.0),
                "is_reward_clipped":is_clipped,
                
                "grid_import_kw":   info.get("grid_import_kw", 0.0),
                "pv_kw":            info.get("pv_kw", 0.0),
                "soc":              info.get("soc", 0.0),
                "indoor_temp":      info.get("indoor_temp", 0.0)
            }
            self._episode_rewards_per_env[i].append(data)

            if self.logger is not None:
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        self.logger.record(f"step_reward/{k}", v)

            if dones[i]:
                self._log_episode(i)

        return True

    def _log_episode(self, env_idx: int):
        rewards_list = self._episode_rewards_per_env[env_idx]
        n = len(rewards_list)
        if n == 0: return

        self._episode_count += 1
        means = {}
        target_keys = self.headers[3:-1]
        for k in target_keys: 
            if k in rewards_list[0]:
                means[k] = float(np.mean([s[k] for s in rewards_list]))
            else:
                means[k] = 0.0

        episode_return = float(np.sum([s["clipped_reward"] for s in rewards_list]))

        if self.logger is not None:
            for k, v in means.items():
                self.logger.record(f"episode_reward/{k}", v)
            self.logger.record("health/episode_return", episode_return)
            self.logger.dump(self.num_timesteps)

        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, self._episode_count, env_idx,
                    *[means[k] for k in target_keys], n
                ])
        except Exception as e:
            print(f"[LoggerError] {e}")

        self._episode_rewards_per_env[env_idx] = []
