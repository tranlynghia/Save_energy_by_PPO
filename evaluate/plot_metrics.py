import os
import matplotlib.pyplot as plt
import numpy as np

def plot_evaluation_results(df_ppo, df_rule, save_dir=None):
    """
    Vẽ các biểu đồ phân tích từ dữ liệu thu thập được.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    
    # Thiết lập style đồ thị cho "Ngầu" (Cyberpunk / Modern feel có thể tuỳ biến sau)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    steps = df_ppo["step"]
    
    # ---------------------------------------------------------
    # Biểu đồ 1: Chiến lược Quản lý Pin (SoC Profile)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(steps, df_ppo["soc"], label="PPO Agent", color="#2ecc71", linewidth=2)
    plt.plot(steps, df_rule["soc"], label="Rule-based Baseline", color="#e74c3c", linewidth=2, linestyle='--')
    plt.title("SoC (State of Charge) Profile over 7 Days", fontsize=14, fontweight='bold')
    plt.xlabel("Timesteps (1 step = 15 min)")
    plt.ylabel("SoC (0.0 to 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "soc_profile.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Biểu đồ 2: So sánh Tổng Chi phí (Total Cost Comparison)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    final_cost_ppo = df_ppo["total_cost"].iloc[-1]
    final_cost_rule = df_rule["total_cost"].iloc[-1]
    
    agents = ['PPO Agent', 'Rule-based Baseline']
    costs = [final_cost_ppo, final_cost_rule]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(agents, costs, color=colors, width=0.5)
    plt.title("Total Electricity Cost over 7 Days (VNĐ)", fontsize=14, fontweight='bold')
    plt.ylabel("Cost (VNĐ)")
    
    # Thêm text giá trị lên trên cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{int(yval):,} đ", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cost_comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Biểu đồ 3: Theo dõi Nhiệt độ & Tính Tiện nghi (Thermal Tracking)
    # ---------------------------------------------------------
    # Lấy dữ liệu 2 ngày đầu (192 steps) để dễ nhìn
    plot_steps = 192
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps[:plot_steps], df_ppo["outdoor_temp"][:plot_steps], label="Outdoor Temp", color="#f39c12", linewidth=1.5, alpha=0.7)
    ax.plot(steps[:plot_steps], df_ppo["indoor_temp"][:plot_steps], label="Indoor Temp (PPO)", color="#3498db", linewidth=2)
    ax.plot(steps[:plot_steps], df_rule["indoor_temp"][:plot_steps], label="Indoor Temp (Rule-based)", color="#9b59b6", linewidth=2, linestyle=':')
    
    # Vùng thoải mái (Comfort Band: 24 - 26 C)
    ax.axhspan(24, 26, color='#2ecc71', alpha=0.2, label='Comfort Band (24-26°C)')
    
    ax.set_title("Thermal Comfort Tracking (First 48 Hours)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_tracking.png"), dpi=300)
    plt.close()

    print(f"Đã lưu thành công 3 biểu đồ phân tích tại thư mục: {os.path.abspath(save_dir)}")
