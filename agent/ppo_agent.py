import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn

class SmartHomeAgent:
    """
    Tác tử PPO (Proximal Policy Optimization) triển khai qua Stable-Baselines3.
    Gồm kiến trúc Mạng Nơ-ron Actor-Critic.
    """
    def __init__(self, env, log_dir="./logs", model_dir="./models", seed: int = 42):
        self.env = env
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Thiết lập mô hình PPO
        # Vì môi trường sử dụng Observation là Box nên ta dùng "MlpPolicy"
        # clip_range = 0.2 tương ứng với siêu tham số epsilon giúp "gọt" tỷ lệ xác suất chính sách
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]) # Mạng Actor (pi) và Critic (vf) chạy song song
        )
        
        from configs.config_loader import ConfigLoader
        config = ConfigLoader.load_config()
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            clip_range=ConfigLoader.get("ppo.clip_range", 0.2),
            ent_coef=ConfigLoader.get("ppo.ent_coef", 0.01),
            learning_rate=ConfigLoader.get("ppo.learning_rate", 3e-4),
            n_steps=ConfigLoader.get("ppo.n_steps", 2048),
            batch_size=ConfigLoader.get("ppo.batch_size", 2048),
            n_epochs=10,
            gamma=ConfigLoader.get("ppo.gamma", 0.99),
            gae_lambda=0.95,                 # GAE lambda
            normalize_advantage=True,         # Advantage normalization → ổn định gradient
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.log_dir,
            seed=seed,                        # Reproducibility
            device="auto",                    # Tự động chọn GPU nếu có
        )

    def train(self, total_timesteps: int = 100_000, callback=None):
        """
        Huấn luyện mô hình.

        :param total_timesteps: Tổng số bước huấn luyện.
        :param callback:        SB3 callback (hoặc CallbackList) từ train_ppo.py.
                                Nếu None → chỉ dùng checkpoint mặc định.
        """
        print(f"Bắt đầu huấn luyện PPO với {total_timesteps} timesteps...")

        # Nếu không truyền callback nào thì tạo checkpoint mặc định
        if callback is None:
            callback = CheckpointCallback(
                save_freq=10_000,
                save_path=self.model_dir,
                name_prefix="ppo_smart_home",
            )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="PPO",
            reset_num_timesteps=True,
        )
        
        # Lưu mô hình cuối cùng
        final_model_path = os.path.join(self.model_dir, "ppo_smart_home_final")
        self.model.save(final_model_path)
        print(f"Hoàn thành huấn luyện. Đã lưu mô hình tại: {final_model_path}")

    def predict(self, obs):
        """Dự đoán hành động dựa trên trạng thái"""
        action, _states = self.model.predict(obs, deterministic=True)
        return action
