# Quy trình vận hành hệ thống (System Workflow)

Tài liệu này mô tả luồng dữ liệu và logic điều khiển vòng kín (Closed-loop) của dự án HEMS.

---

## 1. Sơ đồ luồng dữ liệu (Data Flow)

```mermaid
graph TD
    A[building_3_load.csv] --> B[DataManager]
    B -->|Rescale 1/40| C[Synthetic Env State]
    B -->|Look-ahead| D[Forecasts PV/Load/Price]
    
    C & D --> E[SmartHomeEnv]
    E -->|Normalized Obs| F[PPO Agent]
    
    F -->|Action [-1,1]| G[Action Wrapper]
    G -->|Physical Action kW| E
    
    E -->|Physics Update| H[RC Thermal Model]
    E -->|Physics Update| I[Battery Model]
    
    H & I --> J[Calculate Reward]
    J --> K[Update PPO Policy]
```

---

## 2. Chi tiết các bước thực hiện trong mỗi Timestep (15 phút)

### Bước 1: Thu thập dữ liệu (State Collection)
Môi trường lấy dữ liệu phụ tải và nắng tại bước thời gian $t$. Kết hợp với nhiệt độ trong nhà và SoC hiện tại để tạo ra vector trạng thái.

### Bước 2: Dự báo tương lai (Forecasting)
`DataManager` cung cấp các thông tin dự báo "Oracle" (PV, Phụ tải, Giá điện) cho 6h, 12h và 24h tới. Điều này cho phép Agent thực hiện chiến lược sạc/xả chủ động.

### Bước 3: Ra quyết định (Inference)
Mạng Nơ-ron (Actor) xử lý vector trạng thái và trả về 2 hành động liên tục:
1.  Công suất pin (Charge/Discharge).
2.  Công suất làm mát HVAC.

### Bước 4: Cập nhật vật lý (Physics Step)
*   **Thermal Update**: Tính toán nhiệt lượng rò rỉ và nhiệt lượng làm mát để cập nhật nhiệt độ phòng cho bước $t+1$.
*   **Battery Update**: Cập nhật SoC dựa trên hiệu suất sạc/xả ($\eta=0.95$).

### Bước 5: Phản hồi (Reward & Logging)
Hệ thống tính toán reward dựa trên sự đánh đổi giữa tiền điện và sự thoải mái. Toàn bộ dữ liệu được `RewardDecompositionLogger` ghi lại vào CSV để phục vụ phân tích v3.0.

---

## 3. Quy trình huấn luyện & Đánh giá (Training & Eval)

1.  **Huấn luyện**: Sử dụng `VecNormalize` để ổn định hóa dải phần thưởng. Huấn luyện trong 300,000 bước thời gian trên 4 môi trường song song.
2.  **Đánh giá**: Sử dụng bộ dữ liệu kiểm thử (672 bước - 1 tuần). So sánh các chỉ số:
    *   Tổng chi phí (VND).
    *   Tỷ lệ vi phạm vùng thoải mái (%).
    *   Độ mượt của hành động.
    *   Hiệu quả sạc/xả theo giá điện.
