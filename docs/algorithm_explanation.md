# Giải thích Thuật toán & Thiết kế Reward

Dự án sử dụng thuật toán **Proximal Policy Optimization (PPO)** — một thuật toán thuộc họ Policy Gradient được tin dùng nhất trong các bài toán điều khiển công suất năng lượng nhờ tính ổn định và khả năng xử lý không gian hành động liên tục.

---

## 1. Không gian quan sát (Observation Space - 20D)

Để Agent có thể đưa ra quyết định tối ưu cho tương lai, hệ thống cung cấp 20 thông số đầu vào đã được chuẩn hóa về khoảng `[-1, 1]`:

1.  **Trạng thái nội tại (3)**: Nhiệt độ trong nhà, SoC của pin, Trạng thái HVAC hiện tại.
2.  **Môi trường (3)**: Nhiệt độ ngoài trời, Phụ tải cơ sở (Base load), Giá điện hiện hành.
3.  **Dự báo (Forecast - 10)**:
    *   PV Generation (6h, 12h, 24h).
    *   Base Load (6h, 12h, 24h).
    *   Outdoor Temp (6h).
    *   **Electricity Price (6h, 12h, 24h)** — *Cực kỳ quan trọng cho chiến lược Arbitrage.*
4.  **Thời gian (2)**: Mã hóa Hour Sin/Cos (để Agent hiểu tính chu kỳ ngày/đêm).
5.  **Sai số (2)**: Lỗi nhiệt độ bám đuổi (Comfort error), Net load hiện tại.

---

## 2. Cấu trúc Reward (Multi-Objective)

Hàm thưởng được thiết kế để cân bằng giữa **Kinh tế**, **Sự thoải mái** và **Độ bền thiết bị**:

$$Reward = w_{eco} \cdot r_{eco} + w_{com} \cdot r_{com} + r_{arb} + r_{deg} + r_{smooth} + r_{term}$$

### Thành phần chính:
*   **$r_{eco}$ (Economic)**: Phạt dựa trên chi phí mua điện từ lưới.
*   **$r_{arb}$ (Arbitrage)**: Thưởng khi xả pin lúc giá cao và sạc lúc giá thấp/có nắng dư thừa.
*   **$r_{com}$ (Comfort)**: Phạt phi tuyến ($error^{1.5}$) khi nhiệt độ nằm ngoài dải 22°C-26°C.
*   **$r_{deg}$ (Degradation)**: Phạt dựa trên cường độ sử dụng pin (Battery throughput) và phạt nặng khi xả sâu (SoC < 20%).
*   **$r_{smooth}$**: Phạt các thay đổi đột ngột trong hành động để bảo vệ máy nén HVAC và bộ biến tần.
*   **$r_{term}$**: Phạt nếu SoC cuối ngày không đạt mức mục tiêu (0.5), buộc Agent phải quản lý năng lượng bền vững.

---

## 3. Kỹ thuật ổn định hội tụ (PPO Stability)

Hệ thống áp dụng 3 kỹ thuật quan trọng để tránh "PPO Collapse":

1.  **Reward Clipping**: Giới hạn reward trong khoảng `[-10, 2]` để tránh Gradient Explosion khi có các spike về chi phí hoặc nhiệt độ.
2.  **VecNormalize**: Chuẩn hóa trung bình và phương sai của cả Observation và Reward trong thời gian thực. Điều này giúp Critic học được Value Function chính xác hơn.
3.  **Action Scaling Wrapper**: Chuyển đổi output `[-1, 1]` của mạng Nơ-ron sang các đơn vị vật lý thực tế (kW, °C) một cách mượt mà.
