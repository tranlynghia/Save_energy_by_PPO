import numpy as np

class RuleBasedAgent:
    """
    Tác tử cơ sở (Baseline) sử dụng luật if/else.
    Đã cập nhật để tương thích với FlattenActionSpaceWrapper và Quy mô Tòa nhà.
    """
    def __init__(self, env=None):
        self.env = env
        
    def predict(self, obs, deterministic=True):
        """
        Ra quyết định dựa trên quan sát (obs).
        obs là mảng Numpy đã chuẩn hóa [-1, 1].
        """
        # Giải mã (De-normalize) các thông số cần thiết
        price       = np.interp(obs[0], [-1.0, 1.0], [1000.0, 3500.0])
        indoor_temp = np.interp(obs[6], [-1.0, 1.0], [10.0, 40.0])
        
        # 1. Chiến lược Pin (Battery)
        battery_power = 0.0
        if price > 2800.0:
            battery_power = -1.0 # Xả tối đa giúp tòa nhà giảm peak
        elif price < 1300.0:
            battery_power = 1.0  # Sạc khi giá cực rẻ
            
        # 2. Chiến lược Điều hòa (HVAC Mode)
        # 0: Off, 1: Heat (không dùng), 2: Cool
        hvac_mode = 0
        if indoor_temp > 26.5:
            hvac_mode = 2 # Chế độ làm mát
        elif indoor_temp < 22.0:
            hvac_mode = 0 # Tắt để tiết kiệm
            
        # 3. Thiết bị rời rạc (Washer/Fridge)
        # Bật máy giặt nếu giá không quá đắt (< 3000)
        washer_request = 1 if price < 3000.0 else 0
        fridge_request = 1
        
        # Đóng gói hành động theo định dạng 6 chiều [-1, 1] để Wrapper xử lý
        # [0]: Pin, [1]: AC Setpoint (dummy), [2]: Reserved, [3]: HVAC Mode, [4]: Washer, [5]: Fridge
        action = np.zeros(6, dtype=np.float32)
        
        action[0] = battery_power
        action[1] = 0.0  # Setpoint mặc định
        action[2] = 0.0
        
        # Mapped discrete values to [-1, 1] for wrapper logic:
        # HVAC Mode 0: -1.0, Mode 1: 0.0, Mode 2: 1.0
        action[3] = -1.0 if hvac_mode == 0 else (1.0 if hvac_mode == 2 else 0.0)
        action[4] = 1.0 if washer_request == 1 else -1.0
        action[5] = 1.0 if fridge_request == 1 else -1.0
        
        return action, None
