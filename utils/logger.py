"""
utils/logger.py
===============
Custom SB3 callback để ghi log phân rã Reward (reward decomposition)
vào TensorBoard VÀ CSV. Chạy song song với CheckpointCallback.
"""

import os
import csv
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardDecompositionLogger(BaseCallback):
    """
    Callback ghi lại các thành phần reward riêng biệt (eco, comfort, peak, task, deg)
    vào TensorBoard (mỗi step) VÀ CSV (khi hết episode hoặc định kỳ).
    """

    def __init__(self, log_dir: str, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "reward_decomposition.csv")
        self._episode_rewards_per_env: list[list[dict]] = []
        self._start_time = time.time()

        # Tạo header CSV
        # Handle PermissionError if file is open in another program (e.g., Excel)
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                self._write_header(f)
        except PermissionError:
            # Nếu file bị khóa, tạo file mới với timestamp
            timestamp = int(time.time())
            self.csv_path = os.path.join(log_dir, f"reward_decomposition_{timestamp}.csv")
            print(f"[Warning] File gốc bị khóa. Ghi log vào file mới: {self.csv_path}")
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                self._write_header(f)
        
        self._episode_count = 0

    def _write_header(self, f):
        writer = csv.writer(f)
        writer.writerow([
            "timestep", "episode", "env_idx",
            "r_eco", "r_comfort", "r_peak", "r_task", "r_deg",
            "cost_reward", "comfort_reward", "battery_penalty", "peak_penalty",
            "total_scaled", "clipped_total", "ep_len",
        ])

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        self._episode_rewards_per_env = [[] for _ in range(num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if info is None: continue
            rb = info.get("reward_breakdown", None)
            if rb is None: continue
            
            data = {
                "r_eco":            rb.get("r_eco",            0.0),
                "r_comfort":        rb.get("r_comfort",        0.0),
                "r_peak":           rb.get("r_peak",           0.0),
                "r_task":           rb.get("r_task",           0.0),
                "r_deg":            rb.get("r_deg",            0.0),
                "cost_reward":      rb.get("cost_reward",      0.0),
                "comfort_reward":   rb.get("comfort_reward",   0.0),
                "battery_penalty":  rb.get("battery_penalty",  0.0),
                "peak_penalty":     rb.get("peak_penalty",     0.0),
                "total_scaled":     rb.get("scaled_total",     0.0),
                "clipped_total":    rb.get("clipped_total",    0.0),
            }
            self._episode_rewards_per_env[i].append(data)

            # Ghi TensorBoard TỨC THÌ (mỗi step) để thấy đường biểu diễn chạy liên tục
            if self.logger is not None:
                # Ghi giá trị trung bình của các env tại step này
                for k, v in data.items():
                    self.logger.record(f"step_reward/{k}", v)

            # Ghi log khi episode kết thúc
            if dones[i]:
                self._log_episode(i, reason="episode_end")

        # Ghi log định kỳ mỗi log_freq steps (tổng của tất cả env) để tránh file trống quá lâu
        if self.num_timesteps % self.log_freq == 0:
            for i in range(len(self._episode_rewards_per_env)):
                if len(self._episode_rewards_per_env[i]) > 0:
                    self._log_episode(i, reason="periodic")

        return True

    def _on_training_end(self) -> None:
        """Khi kết thúc training, flush toàn bộ dữ liệu còn lại vào CSV."""
        for i in range(len(self._episode_rewards_per_env)):
            if self._episode_rewards_per_env[i]:
                self._log_episode(i, reason="training_end")

    def _log_episode(self, env_idx: int, reason: str = "episode_end"):
        rewards_list = self._episode_rewards_per_env[env_idx]
        n = len(rewards_list)
        if n == 0: return

        self._episode_count += 1
        keys = [
            "r_eco", "r_comfort", "r_peak", "r_task", "r_deg",
            "cost_reward", "comfort_reward", "battery_penalty", "peak_penalty",
            "total_scaled", "clipped_total",
        ]
        means = {k: float(np.mean([s[k] for s in rewards_list])) for k in keys}

        # Ghi TensorBoard (Episode-level summary)
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
                    means["r_eco"], means["r_comfort"], means["r_peak"], means["r_task"], means["r_deg"],
                    means["cost_reward"], means["comfort_reward"], means["battery_penalty"], means["peak_penalty"],
                    means["total_scaled"], means["clipped_total"],
                    n,
                ])
        except PermissionError:
            print(f"[Error] Không thể ghi vào {self.csv_path} do file đang bị khóa.")

        # Clear buffer
        self._episode_rewards_per_env[env_idx] = []

        if self.verbose >= 1:
            print(f"[Logger] Env {env_idx} logged ({reason}) | steps: {n} | total_rew: {means['total_scaled']:.4f}")


