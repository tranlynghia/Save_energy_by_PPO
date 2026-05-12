# Smart Home HEMS Optimization using Proximal Policy Optimization (PPO)

Dự án này ứng dụng thuật toán Học Tăng Cường Sâu (Deep Reinforcement Learning - DRL), cụ thể là thuật toán **Proximal Policy Optimization (PPO)**, để giải quyết bài toán quản lý năng lượng tối ưu trong mô hình Nhà thông minh (Home Energy Management System - HEMS). 

Mục tiêu cốt lõi là tạo ra một tác tử (agent) thông minh có khả năng tự động học hỏi và ra quyết định nhằm:
1. Giảm thiểu chi phí điện năng (Energy Arbitrage / Peak Shaving).
2. Tối đa hóa tỷ lệ tự tiêu thụ năng lượng mặt trời (Self-Consumption).
3. Đảm bảo tiện nghi nhiệt độ cho người dùng (Thermal Comfort).

---

## 1. Thiết lập Môi trường (Environment Setup)

Môi trường mô phỏng (HEMS Environment) được thiết kế dựa trên tiêu chuẩn OpenAI Gymnasium, cung cấp một hệ thống động lực học phức tạp, phản ánh chân thực các yếu tố vật lý và kinh tế.

### Kiến trúc Observation Space
Mỗi bước thời gian (15 phút), agent nhận một vector trạng thái gồm 10 chiều đã được chuẩn hóa (normalized) về khoảng [-1, 1], bao gồm:
1. **Current State:** Nhiệt độ phòng (Indoor Temp), Trạng thái sạc pin (SoC).
2. **Current Disturbance:** Nhiệt độ ngoài trời (Outdoor Temp), Phụ tải cơ bản (Base Load), Công suất năng lượng mặt trời (PV Generation).
3. **Temporal Features:** Giờ trong ngày (dạng sine/cosine để biểu diễn tính tuần hoàn).
4. **Market & Forecasts:** Giá điện hiện tại (TOU Pricing), Dự báo PV và nhiệt độ cho bước thời gian tiếp theo.

### Kiến trúc Action Space
Action space gồm 2 giá trị liên tục trong khoảng [-1, 1], sau đó được wrapper chuyển đổi thành các đại lượng vật lý:
1. **Battery Dispatch (kW):** `[-5.0, 5.0]` - Sạc (giá trị dương) hoặc Xả (giá trị âm).
2. **HVAC Input (kW):** `[0.0, 4.0]` - Công suất điện cung cấp cho hệ thống điều hòa.

### Động lực học Vật lý (Physics Dynamics)
* **Battery Model:** Giới hạn công suất sạc/xả, bảo vệ giới hạn SoC (0.05 - 0.95), suy hao hiệu suất.
* **Thermal Model:** Mô hình RC 1R1C tích hợp các nguồn nhiệt nội (Internal Gains), bức xạ mặt trời (Solar Heat Gain), và rò rỉ nhiệt. Đặc biệt, thông số được tinh chỉnh mô phỏng thời tiết khắc nghiệt mùa hè Hà Nội (R=1.5, C=30.0). Hệ số hiệu quả năng lượng (COP) của HVAC thay đổi động theo chênh lệch nhiệt độ trong/ngoài nhà.
* **Energy Flow Routing:** Năng lượng mặt trời (PV) ưu tiên phục vụ tải trước, phần thừa ưu tiên sạc pin, phần còn dư mới bán ra lưới.

---

## 2. Chiến lược Reward (Reward Design)

Phần thưởng (Reward) là tín hiệu cốt lõi định hướng hành vi của PPO. Dự án sử dụng hệ thống reward tổng hợp, được chuẩn hóa chặt chẽ trong khoảng `[-5.0, 5.0]` để giữ ổn định Gradient.

**1. Tối ưu Kinh tế & Phân phối Năng lượng:**
* `r_pv_to_load = 1.0 * pv_to_load`: Khuyến khích dùng điện mặt trời trực tiếp.
* `r_battery_use = 1.5 * battery_to_load`: Thưởng mạnh cho việc xả pin phục vụ tải (đặc biệt quan trọng để chống việc "ôm" pin).
* `r_grid = -1.0 * grid_import`: Phạt việc mua điện từ lưới.
* `r_peak_shift`: Nếu mua điện giờ cao điểm (giá đắt), sẽ bị phạt nặng thêm `-1.5 * grid_import`. Ngược lại, nếu xả pin giờ cao điểm, thưởng `+2.0 * battery_to_load`.

**2. Self-Consumption (Tự tiêu thụ):**
* `r_self_consumption = 1.5 * ((pv_load + pv_batt) / pv_total)`: Thưởng tỷ lệ PV được giữ lại trong nhà (hạn chế lãng phí xuất lưới).

**3. Thermal Comfort (Tiện nghi nhiệt):**
* Dùng hàm phạt bậc hai (Quadratic Penalty): `r_comfort = -0.8 * distance^2`. Khi nhiệt độ phòng vượt khỏi vùng thoải mái (22-26°C), hình phạt sẽ tăng tốc rất nhanh, ép agent phải khởi động HVAC.

**4. Battery Health:**
* `r_empty_battery = -2.0 * max(0, 0.2 - soc)`: Hàm phạt liên tục khi pin tụt dưới 20%, giúp agent biết sạc kịp thời mà không bị "sốc" gradient.

---

## 3. Thuật toán PPO (Proximal Policy Optimization)

PPO là thuật toán Actor-Critic thuộc họ Policy Gradient, được lựa chọn nhờ sự cân bằng xuất sắc giữa tính ổn định (stability) và hiệu suất lấy mẫu (sample efficiency).

### Cơ chế Hoạt động
1. **Actor Network:** Dự đoán phân phối xác suất của hành động tiếp theo dựa trên quan sát hiện tại (Observation). Output là `mu` (giá trị trung bình) và `sigma` (độ lệch chuẩn).
2. **Critic Network:** Dự đoán giá trị kỳ vọng (Value Function) của trạng thái hiện tại, giúp tính toán lợi thế (Advantage).
3. **Clipping Objective:** PPO sử dụng hàm mục tiêu bị giới hạn (Clipped Surrogate Objective) để ngăn chặn các bản cập nhật chính sách quá lớn. Nếu một hành động tốt hơn kỳ vọng (Advantage > 0), thuật toán tăng xác suất hành động đó nhưng không vượt quá ngưỡng `1 + clip_range`.

### Cấu hình (Hyperparameters)
Dự án sử dụng thư viện `stable-baselines3` với các thiết lập:
* Khởi tạo **4 môi trường song song (SubprocVecEnv)** kết hợp `VecNormalize` để ổn định dữ liệu đầu vào và lợi tức (returns).
* `learning_rate`: 0.0003
* `gamma`: 0.99 (Định hướng tầm nhìn dài hạn)
* `n_steps`: 2048 (Thu thập dữ liệu dài trước mỗi lần cập nhật)
* `batch_size`: 2048
* Total Timesteps: 500,000

---

## 4. Phân tích Các Vấn đề Kỹ thuật Cốt lõi đã giải quyết

### A. Vấn đề "Reward Hacking" (Khai thác lỗ hổng phần thưởng)
**Vấn đề:** PPO rất thông minh trong việc tìm ra cách kiếm điểm dễ nhất. Ở các phiên bản trước, agent luôn sạc đầy pin 100% bằng PV và giữ nguyên (ôm pin) để hưởng điểm thưởng an toàn, đồng thời tắt luôn HVAC để khỏi tốn điện.
**Giải pháp:** Áp dụng Zero-Sum concept. Giảm điểm thưởng khi sạc pin, tăng mạnh điểm thưởng khi xả pin (`battery_to_load`). Áp dụng hình phạt *Quadratic* cho nhiệt độ để nỗi đau mất tiện nghi lớn hơn chi phí tiền điện, buộc agent phải dùng năng lượng.

### B. Vấn đề "Scale & Magnitude"
**Vấn đề:** Dữ liệu tải ban đầu lấy từ các tòa nhà thương mại lớn (CityLearn), công suất lên tới hàng chục kW. Nếu không thay đổi, phần thưởng chi phí (VND) quá khổng lồ, che lấp hoàn toàn phần thưởng tiện nghi nhiệt (vốn chỉ có giá trị nhỏ). PPO mất khả năng học đa mục tiêu.
**Giải pháp:** DataManager tích hợp `SCALE_FACTOR = 1/40` để thu nhỏ tải về đúng chuẩn hộ gia đình (smart home). Các reward được chuẩn hóa chặt chẽ về khoảng `[-5, 5]`.

### C. Vấn đề "Thời tiết phi logic"
**Vấn đề:** Biểu diễn mùa đông nhưng mong muốn test thuật toán bật điều hòa làm mát (HVAC Cooling). Nhiệt độ ngoài trời chỉ ~20°C, phòng luôn mát tự nhiên.
**Giải pháp:** Custom hóa `weather_loader.py`, đọc file EPW và offset dữ liệu trực tiếp đến khung giờ mùa hè (Tháng 6 - Hour 3624). Outdoor temp dao động chân thực 30-40°C, kích hoạt đúng động lực học của thuật toán làm mát.

### D. Kiến trúc Evaluation Pipeline
Xây dựng một benchmark độc lập `evaluate/run_evaluation.py` so sánh trực tiếp PPO Agent với Rule-Based Agent. Output sinh ra các file `.csv` chi tiết theo từng step và tự động render chuỗi biểu đồ chất lượng cao (Research-grade) bằng `matplotlib`:
* Gantt chart: Appliance Schedule (HVAC, Charge, Discharge).
* Stacked Area: Phân tách nguồn tải (PV, Batt, Grid).
* Dual-Axis: Tương tác giữa giá điện (TOU Price) và lượng điện lưới sử dụng.
* Bar Chart: So sánh tổng chi phí tích lũy và % tiết kiệm năng lượng.
