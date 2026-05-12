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
        
        # Header CSV (chuẩn v4.0)
        self.headers = [
            "timestep", "episode", "env_idx",
            "r_eco", "r_comfort", "r_deg", "r_soc", "r_switch", "r_smooth", "r_peak", "r_terminal",
            "raw_reward", "clipped_total", "is_clipped", "comfort_vio", "ep_len"
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
            
            data = {
                "r_eco":            rb.get("r_eco", 0.0),
                "r_comfort":        rb.get("r_comfort", 0.0),
                "r_deg":            rb.get("r_deg", 0.0),
                "r_soc":            rb.get("r_soc", 0.0),
                "r_switch":         rb.get("r_switch", 0.0),
                "r_smooth":         rb.get("r_smooth", 0.0),
                "r_peak":           rb.get("r_peak", 0.0),
                "r_terminal":       rb.get("r_terminal", 0.0),
                "raw_reward":       rb.get("raw_reward", 0.0),
                "clipped_total":    rb.get("clipped_reward", 0.0),
                "is_clipped":       1.0 if rb.get("is_reward_clipped", False) else 0.0,
                "comfort_vio":      rb.get("comfort_violation", 0.0),
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
        for k in self.headers[3:-1]: # từ r_eco đến comfort_vio
            means[k] = float(np.mean([s[k] for s in rewards_list]))

        # TensorBoard episode-level
        if self.logger is not None:
            for k, v in means.items():
                self.logger.record(f"episode_reward/{k}", v)
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
