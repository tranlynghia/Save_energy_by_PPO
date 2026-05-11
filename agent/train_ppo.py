import os
import sys
import random
import numpy as np

# Đảm bảo console Windows hỗ trợ in tiếng Việt
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục project vào sys.path (Có xử lý lỗi đường dẫn tiếng Việt của VS Code)
cwd = os.getcwd()
if os.path.basename(cwd) == "project":
    project_dir = cwd
elif os.path.exists(os.path.join(cwd, "project")):
    project_dir = os.path.join(cwd, "project")
else:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env.smart_home_env import SmartHomeEnv
from env.wrappers import FlattenActionSpaceWrapper
from agent.ppo_agent import SmartHomeAgent
from utils.logger import RewardDecompositionLogger
from configs.config_loader import ConfigLoader


def set_global_seed(seed: int) -> None:
    """
    Fix toàn bộ nguồn ngẫu nhiên để đảm bảo reproducibility:
    Python, NumPy, PyTorch (CPU + CUDA), CuDNN deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA kernels (có thể làm chậm ~10-15% nhưng cho kết quả ổn định)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_env(seed: int, rank: int):
    """Hàm khởi tạo môi trường để dùng cho VecEnv, với seed riêng cho mỗi sub-env."""
    def _init():
        base_env = SmartHomeEnv()
        env = FlattenActionSpaceWrapper(base_env)
        # Mỗi sub-env dùng seed = seed + rank để tránh correlation nhưng vẫn reproducible
        env.reset(seed=seed + rank)
        return env
    return _init


def run_training_engine():
    print("=====================================================")
    print("    KHỞI ĐỘNG TRÌNH MÔ PHỎNG VÀ ĐỘNG CƠ HUẤN LUYỆN    ")
    print("=====================================================")

    # -----------------------------------------------------------------------
    # 0. Reproducibility: Fix random seed toàn cục
    # -----------------------------------------------------------------------
    SEED = ConfigLoader.get("training.seed", 42)
    set_global_seed(SEED)
    print(f"[0/3] Đã fix random seed = {SEED} (Python / NumPy / PyTorch / CUDA)")

    # -----------------------------------------------------------------------
    # 1. Khởi tạo cơ chế song song (Vectorized Rollout)
    # -----------------------------------------------------------------------
    num_envs = ConfigLoader.get("training.num_envs", 8)
    print(f"[1/3] Đang khởi tạo DummyVecEnv với {num_envs} môi trường song song...")

    # Gom num_envs môi trường vào DummyVecEnv, mỗi env có seed riêng
    vec_env = DummyVecEnv([make_env(SEED, i) for i in range(num_envs)])

    # Monitor môi trường để ghi lại các chỉ số Episode Reward, Length
    vec_env = VecMonitor(vec_env)

    # -----------------------------------------------------------------------
    # 2. Khởi tạo Tác tử PPO
    # -----------------------------------------------------------------------
    print("[2/3] Đang khởi tạo kiến trúc Mạng Actor-Critic PPO...")
    log_dir   = os.path.join(project_dir, "logs")
    model_dir = os.path.join(project_dir, "models")
    agent = SmartHomeAgent(env=vec_env, log_dir=log_dir, model_dir=model_dir, seed=SEED)

    # -----------------------------------------------------------------------
    # 3. Khởi tạo Callbacks
    # -----------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=ConfigLoader.get("training.checkpoint_freq", 10000),
        save_path=model_dir,
        name_prefix="ppo_smart_home",
        verbose=1,
    )

    reward_logger_callback = RewardDecompositionLogger(
        log_dir=log_dir,
        verbose=1,
    )

    callback_list = CallbackList([checkpoint_callback, reward_logger_callback])

    # -----------------------------------------------------------------------
    # 4. Tiến hành Huấn luyện
    # -----------------------------------------------------------------------
    total_timesteps = ConfigLoader.get("training.total_timesteps", 100_000)
    print(f"[3/3] Bắt đầu quá trình huấn luyện với {total_timesteps} timesteps...")
    print("Lưu ý: Theo dõi quá trình trực quan qua TensorBoard bằng lệnh:")
    print("       tensorboard --logdir ./logs")
    print("-----------------------------------------------------")

    agent.train(total_timesteps=total_timesteps, callback=callback_list)

    print("\nQuá trình Training đã hoàn thành!")


if __name__ == "__main__":
    run_training_engine()
