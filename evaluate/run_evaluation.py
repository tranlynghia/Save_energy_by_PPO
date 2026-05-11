import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

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

from env.smart_home_env import SmartHomeEnv
from env.wrappers import FlattenActionSpaceWrapper
from agent.rule_based_agent import RuleBasedAgent
from evaluate.plot_metrics import plot_evaluation_results

def run_evaluation_episode(env, agent, is_ppo=True, num_steps=672, seed=42):
    """
    Chạy 1 chu kỳ đánh giá (mặc định 7 ngày = 672 steps).
    Lưu lại toàn bộ dữ liệu để vẽ biểu đồ.
    """
    obs, _ = env.reset(seed=seed)
    
    logs = {
        "step": [],
        "indoor_temp": [],
        "outdoor_temp": [],
        "soc": [],
        "price": [],
        "total_cost": [],
        "battery_power": []
    }
    
    cumulative_cost = 0.0
    
    for step in range(num_steps):
        # Lưu vết các giá trị hiện tại (de-normalize từ state)
        real_price   = np.interp(obs[0], [-1.0, 1.0], [1000.0, 3500.0])
        outdoor_temp = np.interp(obs[1], [-1.0, 1.0], [10.0, 40.0])
        # obs[6] = Indoor Temp (có nhiễu IoT), obs[7] = SoC (có nhiễu IoT)
        indoor_temp  = np.interp(obs[6], [-1.0, 1.0], [10.0, 40.0])
        soc          = np.interp(obs[7], [-1.0, 1.0], [0.0, 1.0])
        
        # 1. Tác tử ra quyết định
        if is_ppo:
            # PPO Predict (deterministic=True để model không dùng tính ngẫu nhiên khám phá)
            action, _ = agent.predict(obs, deterministic=True)
            battery_power_action = action[0] # Box space: -1 to 1
        else:
            # Rule-based Agent predict (Nhận input từ Môi trường gốc chưa Flatten)
            # Vì RuleBasedAgent được thiết kế để xuất ra dạng Dict, ta truyền vào env gốc
            action, _ = agent.predict(obs)
            battery_power_action = action["continuous"][0]
        
        # 2. Tương tác với môi trường
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Tính toán chi phí thực tế (Tiền điện = Công suất lưới * Giá điện)
        # Thông tin total_power_kW có sẵn trong info dict (Môi trường đã tính)
        grid_power_kw = info.get("total_power_kW", 0.0)
        # Giả sử mỗi step là 15 phút (0.25h) -> Năng lượng = Công suất * 0.25 (kWh)
        timestep_hours = env.unwrapped.timestep_min / 60.0
        step_cost = grid_power_kw * timestep_hours * real_price
        cumulative_cost += step_cost
        
        # 4. Lưu Log
        logs["step"].append(step)
        logs["indoor_temp"].append(indoor_temp)
        logs["outdoor_temp"].append(outdoor_temp)
        logs["soc"].append(soc)
        logs["price"].append(real_price)
        logs["total_cost"].append(cumulative_cost)
        logs["battery_power"].append(battery_power_action)
        
        obs = next_obs
        
        if terminated or truncated:
            break
            
    return pd.DataFrame(logs)

def main():
    print("=====================================================")
    print("        KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ (EVALUATION)      ")
    print("=====================================================")
    
    # 1. Khởi tạo Môi trường
    # PPO dùng Flatten wrapper, Rule-based dùng Môi trường Dict gốc
    base_env_ppo = SmartHomeEnv()
    env_ppo = FlattenActionSpaceWrapper(base_env_ppo)
    
    env_rule = SmartHomeEnv()
    
    # 2. Nạp Mô hình PPO
    model_path = os.path.join(project_dir, "models", "ppo_smart_home_final.zip")
    # Dự phòng: Nếu user vừa train từ trong thư mục agent (do lỗi path cũ), copy path đó luôn
    fallback_model_path = os.path.join(project_dir, "agent", "models", "ppo_smart_home_final.zip")
    
    if not os.path.exists(model_path):
        if os.path.exists(fallback_model_path):
            model_path = fallback_model_path
        else:
            print(f"[LỖI] Không tìm thấy mô hình PPO tại {model_path}. Vui lòng chạy train_ppo.py trước.")
            return
        
    print("[1/3] Đang nạp mô hình PPO đã huấn luyện...")
    ppo_agent = PPO.load(model_path)
    
    # 3. Khởi tạo Rule-based Agent
    print("[2/3] Đang khởi tạo Tác tử cơ sở (Rule-based)...")
    rule_agent = RuleBasedAgent(env=env_rule)
    
    # 4. Chạy mô phỏng Đánh giá
    test_steps = 672 # 7 Ngày
    test_seed = 42   # Cố định seed
    
    print(f"[3/3] Bắt đầu chạy mô phỏng Đánh giá ({test_steps} steps, seed={test_seed})...")
    
    print(" -> Chạy PPO Agent...")
    df_ppo = run_evaluation_episode(env_ppo, ppo_agent, is_ppo=True, num_steps=test_steps, seed=test_seed)
    
    print(" -> Chạy Rule-based Agent...")
    df_rule = run_evaluation_episode(env_rule, rule_agent, is_ppo=False, num_steps=test_steps, seed=test_seed)
    
    # 5. Xuất Biểu đồ
    print("\nQuá trình chạy hoàn tất. Đang vẽ biểu đồ phân tích...")
    plot_evaluation_results(df_ppo, df_rule)
    
if __name__ == "__main__":
    main()
