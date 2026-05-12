## 🌟 Tính năng nổi bật (v4.0 Senior Refactor)

*   **Physically Consistent Power Balance**: Tính toán công suất tại AC Bus, tách bạch hao hụt sạc/xả của Pin.
*   **Centralized Multi-Objective Reward**: Hệ thống phần thưởng hợp nhất, chống Reward Hacking và lạm dụng thiết bị.
*   **Battery Health Protection**: Tích hợp hình phạt cho việc xả sâu, sạc quá đầy và đảo chiều sạc/xả liên tục (Anti-chattering).
*   **Quadratic Comfort Zone**: Duy trì nhiệt độ 22°C-26°C với hình phạt bậc hai và các ràng buộc cứng (Hard constraints).
*   **Deep Telemetry Logging**: Ghi nhận toàn bộ các dòng nhiệt (q_ext, q_cool), COP, và trạng thái Clipping của reward.

---

## 📂 Cấu trúc Reward (v4.0)

Hàm phần thưởng tổng quát được tính toán tập trung tại `utils/reward.py`:

$$Reward = r_{eco} + r_{comfort} + r_{deg} + r_{soc} + r_{smooth} + r_{switch} + r_{peak} + r_{terminal}$$

*   **r_eco**: Chi phí tiền điện thực tế (không dùng r_arb để tránh hacking).
*   **r_comfort**: Phạt bậc hai nếu $T_{in} \notin [22, 26]$. Phạt nặng nếu bão hòa nhiệt (>27°C, <21°C).
*   **r_soc**: Ép pin hoạt động trong vùng tối ưu (30% - 80%).
*   **r_switch**: Phạt đảo chiều sạc/xả liên tục (Chattering control).
*   **r_smooth**: Phạt độ dốc thay đổi hành động trên không gian chuẩn hóa.

---

## 📂 Cấu trúc thư mục

```text
project/
├── agent/                  # Thuật toán PPO và kịch bản huấn luyện
├── env/                    # Môi trường Gymnasium (Physics & Data)
│   ├── data/               # Quản lý và tiền xử lý dữ liệu thực tế
│   └── physics/            # Các mô hình vật lý (Thermal, Battery)
├── evaluate/               # Công cụ đánh giá và vẽ biểu đồ (v3.0)
├── logs/                   # TensorBoard và Reward CSV logs
├── models/                 # Lưu trữ các checkpoints (zip)
└── docs/                   # Tài liệu chi tiết về thuật toán & hệ thống
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Huấn luyện Agent
Sử dụng PPO với cấu hình Advanced (Entropy=0.05, Gamma=0.995):
```powershell
python agent/train_ppo.py
```
*Theo dõi tiến trình qua TensorBoard:* `tensorboard --logdir logs/`

### 2. Đánh giá & So sánh
So sánh PPO Agent với Rule-based Baseline trên tập dữ liệu kiểm thử:
```powershell
python evaluate/run_evaluation.py
```

---

## 📊 Hệ thống biểu đồ phân tích

Sau khi chạy evaluation, hệ thống tự động xuất 7 biểu đồ tại thư mục `evaluate/`:
1.  **SoC Profile**: Theo dõi trạng thái pin trong 7 ngày.
2.  **Cost Comparison**: So sánh tổng chi phí (VND) giữa PPO và Baseline.
3.  **Thermal Tracking**: Độ bám đuổi vùng thoải mái (22°C-26°C).
4.  **Power Balance**: Phân tách nguồn cung (PV, Grid, Battery) theo thời gian.
5.  **Action vs Price**: Minh chứng chiến lược sạc/xả theo biến động giá điện.
6.  **Appliance Schedule**: Biểu đồ Gantt hoạt động của HVAC và Pin.
7.  **Reward Breakdown**: Phân rã các thành phần r_eco, r_comfort, r_arb, r_deg.

---

## 🔬 Thông số kỹ thuật chính

| Thông số | Giá trị | Ý nghĩa |
| --- | --- | --- |
| **Observation** | 20 dims | Bao gồm dự báo giá (Price Forecast) |
| **Action** | 2 dims (Cont) | P_battery [-5, 5]kW, P_hvac [0, 4]kW |
| **Normalization** | VecNormalize | Chuẩn hóa Reward & Obs online |
| **Scaling** | 1/40 Factor | Chuyển dữ liệu Building → Smart Home |
