# Smart Home Microgrid HEMS — Deep Reinforcement Learning

## 1. Tổng Quan Dự Án

Hệ thống **Quản lý Năng lượng Nhà thông minh (HEMS)** ứng dụng Deep RL (PPO) để:

- **Tối ưu hóa** hóa đơn điện theo biểu giá TOU của EVN
- **Điều khiển** sạc/xả pin, điều hòa, máy giặt, tủ lạnh
- **Đảm bảo** tiện nghi nhiệt độ và tuổi thọ thiết bị

**Input:** Dữ liệu thực tế CityLearn 2021 (Scale-down ×40, resample 60→15 phút) + giá EVN TOU  
**Output:** Tín hiệu điều khiển liên tục (Pin) và rời rạc (Thiết bị)

---

## 2. Kiến Trúc Hệ Thống

```text
[CityLearn Dataset] + [EVN TOU Pricing]
            │
            ▼
  [DataManager: Scale-down ×40, Resample 60→15 phút]
            │
            ▼
[ForecastEngine: LSTM (PV) | Transformer (Load) | GRU (Occupancy)]
            │  ← Forecast có noise (no data leakage)
            ▼
[SmartHomeEnv — 17-dim State] ◄── [Physics: Battery + Advanced Thermal]
            │
      State + Reward
            │
            ▼
   [PPO Agent — StableBaselines3]
            │
   Actions: Battery∈[-1,1], AC∈{0,1,2}, Washer∈{0,1}, Fridge∈{0,1}
            │
            ▼
  [TensorBoard + CSV Logger (reward decomposition)]
```

---

## 3. Những Gì Đã Làm Được

### ✅ Phase 1 — Foundation
| Feature | Files |
|:---|:---|
| Gym Environment cơ bản | `env/smart_home_env.py` |
| PPO Agent (SB3) | `agent/ppo_agent.py` |
| Rule-based Baseline | `agent/rule_based_agent.py` |

### ✅ Phase 2 — Real Data Pipeline
| Feature | Files |
|:---|:---|
| CityLearn 2021 Dataset | `env/data/data_manager.py` |
| Scale-down ×40 (Tòa nhà → Smart Home) | `env/data/data_manager.py` |
| Resample 60→15 phút (nội suy tuyến tính) | `env/data/data_manager.py` |
| Battery Physics Model + Degradation | `env/physics/battery_model.py` |
| Thermal Model (T_in RC equation) | `env/physics/thermal_model.py` |
| MultiDiscrete Action (AC/Washer/Fridge) | `env/smart_home_env.py` |
| Reward Scaling (÷1000, chống gradient explosion) | `utils/reward.py` |
| Evaluation + Biểu đồ (cost, SOC, temp) | `evaluate/` |

### ✅ Phase 3 — Research Baseline Stable
| Feature | Files |
|:---|:---|
| **Reproducibility**: fix seed Python/NumPy/PyTorch/CUDA | `agent/train_ppo.py` |
| **Deterministic training**: `cudnn.deterministic=True` | `agent/train_ppo.py` |
| **Advantage normalization** (`normalize_advantage=True`) | `agent/ppo_agent.py` |
| **Reward clipping** `[-10, +2]` chống outlier | `utils/reward.py` |
| **NaN guard** mọi đầu vào reward | `utils/reward.py` |
| **Reward decomposition**: cost/comfort/battery/peak | `utils/reward.py`, `utils/logger.py` |
| **Action clamping**: NaN→0, clip vào action bounds | `env/smart_home_env.py` |
| **Physics assertions**: `assert 0≤SOC≤1`, `assert not isnan(obs)` | `env/smart_home_env.py` |
| **TensorBoard + CSV logger** phân rã reward | `utils/logger.py` |
| **32 Unit Tests** (SOC, Temp, Reward, ActionBounds) | `tests/test_physics.py` |
| `__init__.py` cho tất cả packages | `agent/`, `env/`, `utils/`, `tests/` |

### ✅ Phase 4 — Advanced Thermal Modeling
| Feature | Files |
|:---|:---|
| **Solar Heat Gain**: `Q_solar = A_window × SHGC × Irradiance` | `env/physics/thermal_model.py` |
| **Occupancy Heat**: `Q_occ = n_people × 0.1 kW` | `env/physics/thermal_model.py` |
| **Nonlinear HVAC COP**: `COP_eff = COP × exp(-k × ΔT/T_ref)` | `env/physics/thermal_model.py` |
| **Humidity + Heat Index** (Rothfusz): ảnh hưởng comfort reward | `env/physics/thermal_model.py` |
| Observation space mở rộng **15 → 17 chiều** (+HeatIndex, +Occupancy) | `env/smart_home_env.py` |

### ✅ Phase 5 — Forecasting System (No Data Leakage)
| Feature | Files |
|:---|:---|
| **PV Forecast — LSTM** + MC-Dropout uncertainty | `forecast/pv_forecast.py` |
| **Load Forecast — Transformer** + sinusoidal time features | `forecast/load_forecast.py` |
| **Occupancy Forecast — GRU** + schedule-based fallback | `forecast/occ_forecast.py` |
| **ForecastEngine**: update TRƯỚC predict (no leakage) | `forecast/forecast_engine.py` |
| **Forecast Noise**: inject tại prediction time | `forecast/forecast_engine.py` |
| Statistical fallback khi chưa có trained model | `forecast/pv_forecast.py`, `load_forecast.py`, `occ_forecast.py` |

---

## 4. Observation Space — 17 Chiều

| Index | Ý nghĩa | Đơn vị gốc | Chuẩn hóa |
|:---|:---|:---|:---|
| `obs[0]` | Giá điện TOU hiện tại | VNĐ/kWh | [-1, 1] |
| `obs[1]` | Nhiệt độ ngoài trời | °C | [-1, 1] |
| `obs[2]` | Bức xạ mặt trời (PV thực) | W/kW | [-1, 1] |
| `obs[3]` | Tải thiết bị nền (Equipment) | kWh | [-1, 1] |
| `obs[4]` | Tải nước nóng (DHW) | kWh | [-1, 1] |
| `obs[5]` | Tải làm mát nền (Cooling) | kWh | [-1, 1] |
| `obs[6]` | **Nhiệt độ phòng** (có nhiễu IoT) | °C | [-1, 1] |
| `obs[7]` | **SoC Pin** (có nhiễu IoT) | 0–1 | [-1, 1] |
| `obs[8]` | **PV Forecast +6h** (LSTM/statistical) | W/kW | [-1, 1] |
| `obs[9]` | **PV Forecast +12h** | W/kW | [-1, 1] |
| `obs[10]` | **PV Forecast +24h** | W/kW | [-1, 1] |
| `obs[11]` | **Load Forecast +6h** (Transformer/statistical) | kW | [-1, 1] |
| `obs[12]` | **Occupancy Forecast** (GRU/schedule) | xác suất | [-1, 1] |
| `obs[13]` | **Heat Index** (nhiệt cảm giác có ẩm) | °C | [-1, 1] |
| `obs[14]` | Giờ trong ngày | 0–23 | [-1, 1] |
| `obs[15]` | Thứ trong tuần | 0–6 | [-1, 1] |
| `obs[16]` | Tháng | 1–12 | [-1, 1] |

---

## 5. Action Space

```python
action_space = Dict({
    "continuous": Box([-1.0, 24.0, 0.0], [1.0, 28.0, 1.0]),  # [Battery, AC_setpoint, reserved]
    "discrete":   MultiDiscrete([3, 2, 2])                     # [AC mode, Washer, Fridge]
})
```

| Thiết bị | Mã | Công suất |
|:---|:---|:---|
| Pin (Battery) | -1.0 → +1.0 | -3.0 kW (xả) → +3.0 kW (sạc) |
| Điều hòa | 0 = Tắt / 1 = Nhẹ / 2 = Mạnh | 0 / 1.0 / 2.0 kW |
| Máy giặt | 0 = Tắt / 1 = Bật | 0 / 0.5 kW |
| Tủ lạnh | 0 = Tắt / 1 = Bật | 0 / 0.1 kW |

---

## 6. Kiến Trúc Thư Mục

```text
project/
├── agent/
│   ├── __init__.py
│   ├── ppo_agent.py          # PPO Actor-Critic (SB3), seed, normalize_advantage
│   ├── train_ppo.py          # ENTRY POINT training + seed + CallbackList
│   └── rule_based_agent.py
├── configs/
│   └── config.yaml           # Tất cả hyperparams (physics, reward, training, forecast)
├── data/real_world/          # CityLearn dataset (.csv)
├── env/
│   ├── __init__.py
│   ├── smart_home_env.py     # 17-dim obs, Phase 4+5 tích hợp
│   ├── wrappers.py           # FlattenActionSpaceWrapper (SB3 compat)
│   ├── data/
│   │   └── data_manager.py   # DataManager: load, scale, resample
│   └── physics/
│       ├── battery_model.py  # SOC dynamics + TCVN temperature constraint
│       └── thermal_model.py  # Advanced RC model: solar gain, occupancy, nonlinear COP, heat index
├── forecast/                 # Phase 5 — No data leakage
│   ├── __init__.py
│   ├── pv_forecast.py        # LSTM + MC-Dropout + statistical fallback
│   ├── load_forecast.py      # Transformer + time features + statistical fallback
│   ├── occ_forecast.py       # GRU + schedule-based fallback
│   └── forecast_engine.py    # Unified manager: update → predict (no leakage)
├── evaluate/
│   ├── run_evaluation.py
│   └── plot_metrics.py
├── tests/
│   ├── __init__.py
│   └── test_physics.py       # 32 unit tests (SOC, Temp, Reward, ActionBounds)
├── utils/
│   ├── __init__.py
│   ├── reward.py             # Multi-objective reward + NaN guard + clipping + decomposition
│   └── logger.py             # RewardDecompositionLogger (TensorBoard + CSV)
├── models/                   # Checkpoints PPO (.zip)
├── logs/                     # TensorBoard logs + reward_decomposition.csv
├── run_tests.py              # Script chạy pytest từ bất kỳ thư mục nào
└── README.md
```

---

## 7. Cách Chạy

```bash
# Cài đặt
conda activate torch-gpu
pip install stable-baselines3 gymnasium pandas numpy matplotlib pyyaml torch

# Chạy unit tests (32 tests)
python run_tests.py

# Huấn luyện PPO
python agent/train_ppo.py

# Theo dõi training
tensorboard --logdir ./logs

# Đánh giá
python evaluate/run_evaluation.py
```

---

## 8. Physics Models

### Advanced Thermal (Phase 4)

```
T_in(t+1) = T_in(t) + (dt/C) × [
    (T_out - T_in) / R           # Trao đổi nhiệt tự nhiên
    + A_window × SHGC × Irr / 1000   # Solar Heat Gain
    + n_occ × 0.1                    # Occupancy Heat
    - P_hvac × COP_eff               # HVAC (nonlinear COP)
]

COP_eff = COP_rated × exp(-k × max(0, T_out - T_in) / T_ref)
```

**Heat Index** (Rothfusz formula) dùng để tính comfort reward thay cho nhiệt độ khô.

### Battery (TCVN 14499-4-3:2025)
- Disable sạc/xả khi nhiệt độ < 0°C hoặc > 45°C
- `0.1 ≤ SOC ≤ 1.0` được enforce cứng

---

## 9. Forecast Models (Phase 5)

| Task | Model | Fallback | Horizon |
|:---|:---|:---|:---|
| PV Generation | LSTM (2L, h=64) + MC-Dropout | Lag-96 seasonal | 24h |
| Electricity Load | Transformer (4-head, 2L) + time features | Lag-672 weekly | 24h |
| Occupancy | GRU (1L, h=32) | Schedule-based | 24h |

> **No Data Leakage**: `ForecastEngine.update()` nhận dữ liệu thực tế **bước t** → `predict()` dự báo **bước t+1 trở đi**.  
> Forecast noise được inject tại thời điểm predict để simulate real-world error.

---

## 10. Monitoring & Logging

| Metric (TensorBoard) | Ý nghĩa |
|:---|:---|
| `reward/r_eco` | Chi phí điện (có trọng số) |
| `reward/r_comfort` | Phạt vi phạm tiện nghi |
| `reward/r_peak` | Phạt công suất đỉnh |
| `reward/r_deg` | Phạt hao mòn pin |
| `reward/cost_reward` | Chi phí điện gốc (chưa nhân w) |
| `reward/comfort_reward` | Comfort gốc |
| `reward/battery_penalty` | Battery penalty gốc |
| `reward/peak_penalty` | Peak penalty gốc |
| `reward/clipped_total` | Reward sau clip [-10, +2] |

CSV: `logs/reward_decomposition.csv` — ghi sau mỗi episode.

---

## 11. Roadmap

- **Phase 1** ✅ Foundation (Gym Env + PPO)
- **Phase 2** ✅ Real Data Pipeline + Physics Models
- **Phase 3** ✅ Research Baseline (Seed, NaN guard, Reward stabilization, 32 tests)
- **Phase 4** ✅ Advanced Thermal (Solar gain, Occupancy, Nonlinear COP, Heat Index)
- **Phase 5** ✅ Forecasting System (LSTM/Transformer/GRU, no data leakage)
- **Phase 6** 🔲 Train & Evaluate Forecast Models (pre-train trên CityLearn data)
- **Phase 7** 🔲 Multi-agent RL (P2P Energy Trading)
- **Phase 8** 🔲 Edge AI Deployment (Raspberry Pi + MQTT + Home Assistant)

---

## 12. Known Risks

| Risk | Mô tả | Mitigation |
|:---|:---|:---|
| Reward Instability | `r_deg` quá cao → agent "sợ" sạc xả | Reward clipping [-10, +2] |
| Overfitting | Chỉ train trên 1 tòa nhà | Cần multi-building training |
| AC Cold Trap | HVAC làm lạnh quá nhanh → cycle on/off | Tuning R, C, COP params |
| Forecast Error Cascade | Forecast sai → agent chọn action sai | MC-Dropout uncertainty quantification |
