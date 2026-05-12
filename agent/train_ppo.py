"""
train_ppo.py — PPO Training Script with Full Reward Logging
============================================================
v2.1: Tích hợp RewardDecompositionLogger để ghi log phân rã reward
ra CSV + TensorBoard trong suốt quá trình huấn luyện.
"""
import os
import sys
import random
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# --- sys.path setup ---
cwd = os.getcwd()
if os.path.basename(cwd) == "project":
    project_dir = cwd
elif os.path.exists(os.path.join(cwd, "project")):
    project_dir = os.path.join(cwd, "project")
else:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env.wrappers import make_stable_env
from utils.logger import RewardDecompositionLogger


def set_global_seed(seed: int) -> None:
    """Fix tất cả nguồn ngẫu nhiên để đảm bảo reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_env_fn(seed: int, rank: int):
    """
    Factory: tạo một sub-env trong VecEnv với seed riêng.
    """
    def _init():
        env = make_stable_env(
            max_episode_steps=96,
            random_start=True,
            seed=seed + rank,
        )
        return env
    return _init


def run_training():
    print("=" * 55)
    print("   SMART HOME HEMS — PPO TRAINING ENGINE (v3.0 Advanced)")
    print("=" * 55)

    SEED        = 42
    NUM_ENVS    = 4
    TOTAL_STEPS = 500_000

    set_global_seed(SEED)
    print(f"[0/3] Seed = {SEED}")

    # -----------------------------------------------------------------------
    # 1. Vectorized Environment with VecNormalize
    # -----------------------------------------------------------------------
    print(f"[1/3] Khởi tạo {NUM_ENVS} môi trường song song + VecNormalize...")
    vec_env = DummyVecEnv([make_env_fn(SEED, i) for i in range(NUM_ENVS)])
    vec_env = VecMonitor(vec_env)
    
    # VecNormalize: Chuẩn hóa Obs và Reward online
    # clip_obs=10.0, clip_reward=10.0 để tránh outlier phá hỏng gradient
    vec_env = VecNormalize(
        vec_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0, 
        clip_reward=10.0
    )

    # -----------------------------------------------------------------------
    # 2. PPO Agent
    # -----------------------------------------------------------------------
    print("[2/3] Khởi tạo PPO...")
    log_dir   = os.path.join(project_dir, "logs")
    model_dir = os.path.join(project_dir, "models")
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,          # Đẩy mạnh Entropy để tránh Policy Collapse
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=log_dir,
        seed=SEED,
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # 3. Callbacks
    # -----------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000 // NUM_ENVS,   # Lưu checkpoint mỗi 20k timesteps
        save_path=model_dir,
        name_prefix="ppo_hems",
        verbose=1,
    )

    # RewardDecompositionLogger: ghi CSV + TensorBoard mỗi episode
    # Hoạt động bằng cách đọc info["reward_breakdown"] từ mỗi step.
    # CSV sẽ được lưu tại: logs/reward_decomposition.csv
    reward_logger_cb = RewardDecompositionLogger(
        log_dir=log_dir,
        log_freq=1000,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, reward_logger_cb])

    # -----------------------------------------------------------------------
    # 4. Training
    # -----------------------------------------------------------------------
    print(f"[3/3] Huấn luyện {TOTAL_STEPS:,} timesteps...")
    print(f"      Monitor TensorBoard: tensorboard --logdir {log_dir}")
    print(f"      Monitor CSV:         {log_dir}\\reward_decomposition.csv")
    print("-" * 55)

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    final_path = os.path.join(model_dir, "ppo_smart_home_final")
    model.save(final_path)
    print(f"\n✓ Model đã lưu tại: {final_path}.zip")
    print(f"✓ Reward log CSV:   {log_dir}\\reward_decomposition.csv")


if __name__ == "__main__":
    run_training()
