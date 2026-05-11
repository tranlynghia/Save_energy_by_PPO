import numpy as np

class RuleBasedAgent:
    """
    Tác tử cơ sở (Baseline) sử dụng luật if/else truyền thống.
    Đóng vai trò làm thước đo so sánh độ thông minh với AI (PPO).
    """
    def __init__(self, env):
        self.env = env
        
    def predict(self, obs, deterministic=True):
        """
        Ra quyết định dựa trên quan sát (obs).
        obs là mảng Numpy đã chuẩn hóa [-1, 1].
        """
        # Giải mã (De-normalize) các thông số cần thiết
        price = np.interp(obs[0], [-1.0, 1.0], [1000.0, 3500.0])
        indoor_temp = np.interp(obs[6], [-1.0, 1.0], [10.0, 40.0]) # Trong Dữ liệu mới, nhiệt độ phòng ở index 6
        
        # 1. Chiến lược Pin (Battery)
        # Sạc khi giá rẻ (< 1500), Xả khi giá đắt (> 2500)
        battery_power = 0.0
        if price > 2500.0:
            battery_power = -1.0 # Xả tối đa
        elif price < 1500.0:
            battery_power = 1.0  # Sạc tối đa
            
        # 2. Chiến lược Điều hòa (HVAC)
        # Bật lạnh nhất nếu nhiệt độ phòng vượt 26 độ C
        ac_temp = 28.0
        if indoor_temp > 26.0:
            ac_temp = 24.0
            
        # 3. Đèn (Light)
        light_level = 0.5
        
        # 4. Các thiết bị Discrete
        washer_action = 1 # Bật
        fridge_action = 1 # Bật
        heater_action = 1 # Bật
        
        # Đóng gói hành động theo định dạng Dict của Action Space
        action = {
            "continuous": np.array([battery_power, ac_temp, light_level], dtype=np.float32),
            "discrete": np.array([washer_action, fridge_action, heater_action], dtype=np.int32)
        }
        
        # Trả về action cùng với 1 list trống (bắt chước format của SB3)
        return action, None
