# Physics-Informed Smart Home HEMS using PPO

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)

Hệ thống quản lý năng lượng thông minh (HEMS) ứng dụng Học tăng cường sâu (Deep Reinforcement Learning) để tối ưu hóa chi phí điện năng, duy trì sự thoải mái nhiệt và kéo dài tuổi thọ pin lưu trữ.

---

## 🌟 Tính năng nổi bật (Research-Grade)

*   **Physics-Informed Environment**: Mô phỏng nhiệt động lực học tòa nhà (RC Model) với chỉ số COP biến thiên theo nhiệt độ môi trường thực tế.
*   **Battery Arbitrage Strategy**: Agent học được cách "buôn điện" — sạc khi giá rẻ (hoặc có nắng) và xả khi giá cao (Peak hours) để tối ưu hóa lợi nhuận.
*   **Advanced PPO Pipeline**: Tích hợp `VecNormalize` cho cả Observation và Reward, giúp hội tụ nhanh và ổn định trong môi trường đa mục tiêu.
*   **20-Dimensional State Space**: Bao gồm các dự báo (Forecast) về PV, phụ tải và **giá điện tương lai** (6h, 12h, 24h).
*   **Visual Analytics v3.0**: Hệ thống báo cáo tự động với 7 biểu đồ phân tích chuyên sâu (Power Balance, Action-Price correlation, Cumulative Reward Breakdown, v.v.)

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
