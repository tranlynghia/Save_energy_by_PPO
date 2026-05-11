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
    sau mỗi episode kết thúc.

    Ghi vào:
    - TensorBoard  (reward/eco, reward/comfort, reward/peak, ...)
    - CSV          (logs/reward_decomposition.csv)
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "reward_decomposition.csv")
        self._episode_rewards: list[dict] = []
        self._start_time = time.time()

        # Tạo header CSV
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "episode",
                "r_eco", "r_comfort", "r_peak", "r_task", "r_deg",
                "cost_reward", "comfort_reward", "battery_penalty", "peak_penalty",
                "total_scaled", "clipped_total", "ep_len",
            ])
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Vòng lặp chính: thu thập info từ mỗi step
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # self.locals["infos"] là list[dict] — mỗi phần tử cho 1 sub-env
        for i, info in enumerate(self.locals.get("infos", [])):
            if info is None:
                continue
            rb = info.get("reward_breakdown", None)
            if rb is None:
                continue
            # Lưu tạm nếu episode chưa kết thúc
            self._episode_rewards.append({
                "r_eco":            rb.get("r_eco",            0.0),
                "r_comfort":        rb.get("r_comfort",        0.0),
                "r_peak":           rb.get("r_peak",           0.0),
                "r_task":           rb.get("r_task",           0.0),
                "r_deg":            rb.get("r_deg",            0.0),
                # Reward decomposition mới (chưa có trọng số)
                "cost_reward":      rb.get("cost_reward",      0.0),
                "comfort_reward":   rb.get("comfort_reward",   0.0),
                "battery_penalty":  rb.get("battery_penalty",  0.0),
                "peak_penalty":     rb.get("peak_penalty",     0.0),
                "total_scaled":     rb.get("scaled_total",     0.0),
                "clipped_total":    rb.get("clipped_total",    0.0),
            })

            # Khi episode kết thúc → ghi log
            done = self.locals["dones"][i]
            if done and self._episode_rewards:
                self._log_episode()

        return True  # True = tiếp tục training

    def _log_episode(self):
        """Tổng hợp và ghi log một episode."""
        self._episode_count += 1
        n = len(self._episode_rewards)
        if n == 0:
            return

        # Tính trung bình các thành phần reward trong episode
        keys = [
            "r_eco", "r_comfort", "r_peak", "r_task", "r_deg",
            "cost_reward", "comfort_reward", "battery_penalty", "peak_penalty",
            "total_scaled", "clipped_total",
        ]
        means = {k: float(np.mean([s[k] for s in self._episode_rewards])) for k in keys}

        # --- Ghi TensorBoard ---
        if self.logger is not None:
            for k, v in means.items():
                self.logger.record(f"reward/{k}", v)
            self.logger.dump(self.num_timesteps)

        # --- Ghi CSV ---
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                self._episode_count,
                means["r_eco"],
                means["r_comfort"],
                means["r_peak"],
                means["r_task"],
                means["r_deg"],
                means["cost_reward"],
                means["comfort_reward"],
                means["battery_penalty"],
                means["peak_penalty"],
                means["total_scaled"],
                means["clipped_total"],
                n,
            ])

        # Reset bộ nhớ tạm
        self._episode_rewards = []

        if self.verbose >= 1:
            print(
                f"[Logger] ep={self._episode_count:4d} | "
                f"eco={means['r_eco']:7.2f}  "
                f"comfort={means['r_comfort']:7.2f}  "
                f"peak={means['r_peak']:7.2f}  "
                f"deg={means['r_deg']:6.3f}  "
                f"total={means['total_scaled']:8.4f}"
            )
