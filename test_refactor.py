import os
import sys

# Đảm bảo đường dẫn project đúng
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.smart_home_env import SmartHomeEnv

def main():
    print("Khởi tạo môi trường SmartHomeEnv...")
    env = SmartHomeEnv()
    
    print("Reset môi trường...")
    obs, info = env.reset()
    
    print("Lấy mẫu hành động ngẫu nhiên và step...")
    action = env.action_space.sample()
    print(f"Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\n[Kết quả Step]")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print("\n[Info keys]:", list(info.keys()))
    if 'physics' in info:
        print("[Physics keys]:", list(info['physics'].keys()))
    if 'reward_breakdown' in info:
        print("[Reward breakdown keys]:", list(info['reward_breakdown'].keys()))
        
    print("\nKiểm tra thành công. Không có lỗi khi step!")

if __name__ == "__main__":
    main()
