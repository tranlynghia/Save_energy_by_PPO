# HEMS PPO - Smart Home Energy Management System

Dự án áp dụng Học Tăng Cường (Reinforcement Learning - PPO) để điều khiển Hệ thống Quản lý Năng lượng Thông minh (Home Energy Management System - HEMS) cho một căn nhà. 
Mục tiêu là tối ưu hóa việc sử dụng năng lượng, cắt giảm đỉnh tải (Peak Shaving), tận dụng điện mặt trời (PV) và tiết kiệm chi phí hóa đơn điện dựa trên giá điện TOU (Time of Use) theo chuẩn thực tế của Việt Nam năm 2026, đồng thời phải đảm bảo tuyệt đối điều kiện tiện nghi nhiệt độ (Thermal Comfort) và tuổi thọ pin lưu trữ (Battery Health).

## 🚀 Đặc điểm Nổi bật
- **Mô hình Nhiệt động học (Thermal Model) Tiên tiến**: Sử dụng cấu trúc RC 1R1C mở rộng, có xét đến truyền nhiệt qua vỏ nhà (Envelope), bức xạ mặt trời qua kính (Solar Gain), tỏa nhiệt nội bộ (Internal Gain) và xâm nhập khí (Infiltration).
- **Mô phỏng Pin lưu trữ (Battery ESS) Chuẩn Vật lý**: Tách bạch công suất AC-side và DC-side (Cell-side), mô phỏng giới hạn SoC động, và cơ chế tự động giảm công suất (Thermal Derating) khi nhiệt độ môi trường khắc nghiệt.
- **Giá điện TOU Việt Nam 2026**: Tích hợp bảng giá điện sản xuất/kinh doanh dưới 6kV.
- **Hệ thống Reward "Research-Grade"**: Không còn hiện tượng "Reward toàn âm", tích hợp các cơ chế Bonus/Penalty khắt khe giúp Agent học được chiến lược tối ưu có ý nghĩa vật lý.
- **Evaluation Pipeline Trực quan**: Xuất file CSV chi tiết, Tensorboard monitoring thời gian thực, và 7 biểu đồ Diagnostic Plot để mổ xẻ "bệnh" của thuật toán.

## 📁 Cấu trúc Dự án

```
project/
│
├── agent/
│   ├── train_ppo.py          # Script huấn luyện chính cho PPO Agent
│   └── rule_based_agent.py   # Baseline Agent dựa trên tập luật (Heuristic)
│
├── configs/
│   └── config.yaml           # "Single Source of Truth": Toàn bộ thông số vật lý và reward
│
├── env/
│   ├── data/
│   │   └── data_manager.py   # Xử lý dữ liệu, tạo outdoor_temp hình sin, map giá TOU
│   ├── physics/
│   │   ├── battery_model.py  # Phương trình Vật lý Pin (ESS)
│   │   └── thermal_model.py  # Phương trình Vật lý Nhiệt (1R1C)
│   ├── smart_home_env.py     # Lõi Gymnasium Environment
│   └── wrappers.py           # Tiền xử lý Action/Observation Space (Normalization)
│
├── evaluate/
│   ├── run_evaluation.py     # Script so sánh PPO vs Rule-based
│   └── plot_metrics.py       # Vẽ các biểu đồ Diagnostics
│
├── utils/
│   ├── reward.py             # Hệ thống phân rã Reward đa mục tiêu
│   └── logger.py             # Tensorboard & CSV Callbacks
│
├── docs/                     # Tài liệu thiết kế toán học và hệ thống
└── data/                     # Data CityLearn và EPW (nếu có)
```

## 🧠 Cách thiết lập (Setup)

1. **Yêu cầu Hệ thống:**
   - Python 3.9+
   - Thư viện: `gymnasium`, `stable-baselines3`, `numpy`, `pandas`, `matplotlib`, `pyyaml`.

2. **Cách chạy Huấn luyện (Training):**
   Mở terminal tại thư mục `project/` và chạy:
   ```bash
   python agent/train_ppo.py
   ```
   *Tip: Có thể mở Tensorboard để theo dõi quá trình cày điểm: `tensorboard --logdir logs/`*

3. **Cách chạy Đánh giá (Evaluation):**
   Sau khi đã train xong (có file `models/ppo_hems_final.zip`), chạy:
   ```bash
   python evaluate/run_evaluation.py
   ```
   Kết quả sẽ sinh ra các file CSV và biểu đồ đồ thị so sánh trong thư mục `evaluate/`.

## 📜 Tài liệu Chi tiết
Vui lòng đọc kỹ tài liệu trong thư mục `docs/` để hiểu cách hệ thống vận hành bên dưới:
- `docs/PHYSICS_AND_ENVIRONMENT.md`: Các công thức nhiệt động học, suy hao dung lượng pin, tính toán điện năng, thiết lập phần thưởng (Reward/Penalty) và cơ chế nội suy Baseline Rule-based.
