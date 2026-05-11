import sys
sys.stdout.reconfigure(encoding='utf-8')
from env.smart_home_env import SmartHomeEnv
from env.wrappers import FlattenActionSpaceWrapper
from agent.ppo_agent import SmartHomeAgent

def main():
    print("Khởi tạo môi trường SmartHomeEnv...")
    base_env = SmartHomeEnv()
    
    # Sử dụng Wrapper để chuyển đổi Dict Action Space -> Box Action Space cho Stable-Baselines3
    env = FlattenActionSpaceWrapper(base_env)
    
    print("\n[Không gian Trạng thái - Observation Space]")
    print(env.observation_space)
    
    print("\n[Không gian Hành động sau khi qua Wrapper - Action Space]")
    print(env.action_space)
    
    print("\nKhởi tạo Tác tử PPO (Mạng Actor-Critic)...")
    try:
        agent = SmartHomeAgent(env=env)
        
        # Chạy thử huấn luyện 2048 bước để test kiến trúc Mạng Actor-Critic
        print("\nTiến hành huấn luyện thử nghiệm (Test Training)...")
        agent.train(total_timesteps=2048)
        
        print("\nTest dự đoán (Predict) từ mô hình đã khởi tạo:")
        obs, _ = env.reset()
        action = agent.predict(obs)
        print(f"Trạng thái đầu vào: {obs}")
        print(f"Hành động đầu ra (Box của PPO): {action}")
        
        # Xem kết quả Action sau khi Environment xử lý (qua Wrapper)
        print("\nĐây là cách Wrapper dịch Output của PPO về Dict gốc:")
        print(env.action(action))
        
    except ImportError as e:
        print("\n[Lỗi Import] Có vẻ như thư viện chưa được cài đặt đầy đủ. Vui lòng chạy lệnh sau:")
        print("pip install stable-baselines3[extra]")
        print(f"Chi tiết lỗi: {e}")

if __name__ == "__main__":
    main()
