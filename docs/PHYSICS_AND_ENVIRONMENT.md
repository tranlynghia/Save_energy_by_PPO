# Vật lý Môi trường, Thuật toán và Cơ chế Reward

Tài liệu này mô tả chi tiết các lý thuyết vật lý, sự tương tác của môi trường (Environment Setup), nguyên lý phần thưởng (Reward System) cho thuật toán PPO, và nguyên lý hoạt động của Rule-based Baseline.

---

## 1. Môi trường (Environment Setup)

Hệ thống được mô hình hóa theo chuẩn `gymnasium.Env` phục vụ cho mô hình học tăng cường. Mỗi bước thời gian (timestep) `dt` tương ứng với 15 phút (0.25h).

### Dữ liệu (Data Pipeline)
Toàn bộ dữ liệu được load, scale và giả lập từ `DataManager`:
- **Base Load (Tải điện sinh hoạt)**: Scale từ tập dữ liệu tòa nhà thương mại xuống mức Smart Home bằng hệ số `1/40`.
- **Nhiệt độ ngoài trời (Outdoor Temperature)**: Giả lập quỹ đạo hình sin với công thức:  
  $T_{out} = 30 + 5 \sin \left( \frac{2\pi (h - 9)}{24} \right) + N(0, 0.3)$  
  (Chạm đáy 25°C vào mờ sáng và đỉnh 35°C lúc 15h chiều).
- **GHI & PV Power (Điện mặt trời)**: Quang thông `GHI` giả lập theo hình chuông từ 6h đến 18h. Công suất PV được tính toán dựa trên hệ số suy hao 0.85 cho dàn pin 5kWp.
- **Giá điện (TOU)**: Áp dụng bảng giá sản xuất kinh doanh < 6kV năm 2026:
  - Off-peak (00:00 - 06:00): 1.300 VNĐ
  - Peak (17:30 - 22:30): 3.640 VNĐ
  - Normal: 1.987 VNĐ

---

## 2. Mô hình Vật lý Nhiệt động (Thermal Physics)

Ứng dụng mô hình RC (Resistor-Capacitor) phiên bản 1R1C mở rộng. Sự thay đổi nhiệt độ trong phòng ($T_{in}$) phụ thuộc vào Tổng nhiệt lượng tích lũy:

$$ C \frac{dT_{in}}{dt} = Q_{envelope} + Q_{solar} + Q_{internal} + Q_{infiltration} - Q_{hvac} $$

- **$Q_{envelope}$**: Nhiệt truyền thụ động qua tường/mái $= \frac{T_{out} - T_{in}}{R}$.
- **$Q_{solar}$**: Nhiệt do bức xạ mặt trời lọt qua kính chắn $\propto GHI$.
- **$Q_{internal}$**: Nhiệt tỏa ra từ người và thiết bị (tỷ lệ thuận với Base Load).
- **$Q_{infiltration}$**: Rò rỉ không khí giữa ngoài và trong.
- **$Q_{hvac}$**: Năng lượng làm mát từ điều hòa $= P_{hvac\_in} \times COP$.

**Sự suy giảm COP (Coefficient of Performance):**  
COP của điều hòa không cố định. Khi trời càng nóng, hiệu suất sẽ giảm dần:  
$$ COP = \max(COP_{min}, COP_{nom} \times (1 - k_{deg} \times \max(0, T_{out} - T_{in}))) $$

---

## 3. Mô hình Vật lý Pin lưu trữ (Battery ESS)

Tách bạch rõ rệt giữa công suất trao đổi với lưới điện ($P_{grid}$) và công suất thực nạp vào lõi cell ($P_{cell}$). Quy ước: Dương (+) là sạc, Âm (-) là xả.

- **Khi Sạc ($P_{grid} > 0$)**: Mất mát năng lượng do Inverter và hóa năng: $P_{cell} = P_{grid} \times \eta_{charge}$
- **Khi Xả ($P_{grid} < 0$)**: Cần kéo nhiều điện từ cell hơn để bù hao hụt: $P_{cell} = P_{grid} / \eta_{discharge}$

**Bảo vệ SOC (State of Charge):**
SOC ở bước tiếp theo: $SOC_{next} = SOC + \frac{P_{cell} \times dt}{Capacity}$.  
Công suất $P_{grid}$ luôn bị "bóp" lại để đảm bảo $SOC_{next}$ không bao giờ vượt qua biên `[0.05, 0.95]`.

**Bảo vệ Nhiệt độ (Thermal Derating):**
Nếu $T_{out} > 40°C$, công suất tối đa của Inverter giảm một nửa (Derating 50%). Nếu $< 0°C$ hoặc $> 45°C$, vô hiệu hóa pin hoàn toàn.

---

## 4. Power Balance (Cân bằng năng lượng lưới)

Lưới điện đóng vai trò bù đắp phần thiếu hụt:
$$ Net\_Load = Base\_Load + P_{hvac} + P_{batt\_grid} - P_{pv} $$
$$ Grid\_Import = \max(0, Net\_Load) $$

Tổng chi phí điện của một bước là: $Cost = Grid\_Import \times dt \times Price$.

---

## 5. Cấu trúc Reward cho Thuật toán PPO

Để Agent học được cách quản lý thay vì chỉ thu penalty, hàm Reward đã được phân rã thành **Bonuses** (Khuyến khích) và **Penalties** (Trừng phạt).

**Positive Bonuses (Phần thưởng dương):**
1. **$r_{saving}$**: Tỷ lệ thuận với lượng tiền tiết kiệm được so với Baseline Rule-based ($Cost_{baseline} - Cost_{ppo}$).
2. **$r_{comfort\_bonus}$**: Thưởng cố định nếu $T_{in} \in [22, 26]$.
3. **$r_{soc\_bonus}$**: Thưởng cố định nếu mức pin nằm trong khoảng khỏe mạnh $[0.3, 0.8]$.

**Negative Penalties (Trừng phạt):**
1. **$r_{eco}$**: Phạt bằng chính chi phí tiền điện tiêu thụ (chuẩn hóa về tỷ lệ scale).
2. **$r_{comfort}$** và **$r_{severe}$**: Phạt parabol bậc hai $(T_{in} - 26)^2$. Nếu $T_{in} > 27°C$, $r_{severe}$ trừng phạt cực gắt để Agent sợ việc lạm dụng tắt điều hòa.
3. **$r_{peak}$**: Phạt parabol nếu Grid Import vượt quá Peak Limit (4 kW).
4. **$r_{deg}$**: Phạt hao mòn pin dựa trên Throughput (lượng điện luân chuyển).
5. **$r_{switch}$**: Phạt nếu dòng xả/sạc của pin đảo dấu liên tục (chống Jittering làm hỏng pin).
6. **$r_{smooth}$**: Phạt nếu Action (batt, hvac) thay đổi biên độ quá lớn giữa 2 frame liên tiếp.

---

## 6. Baseline Rule-based (Hệ cơ sở so sánh)

Để biết PPO có thực sự "thông minh" hay không, hệ thống có một Baseline dùng tập luật (Heuristics) cơ bản:

- **Điều hòa (HVAC)**:  
  Duy trì luật bật theo các ngưỡng nhiệt (ví dụ: $>26.5°C \rightarrow$ Boost, $>25.5°C \rightarrow$ Normal, $\le 24.5 \rightarrow$ Eco, tắt khi lạnh).
- **Lưu trữ (Battery)**:  
  Sạc khi giá điện nằm trong vùng Off-peak và đang có thừa PV. Xả mạnh khi giá điện rơi vào vùng Peak để né tiền điện cao.

Việc thiết kế $r_{saving}$ bằng cách nhúng ngầm logic "Bật điều hòa 2.5kW nếu nhà $> 26°C$" vào env step giúp PPO hiểu được cái mốc "trung bình" để từ đó leo lên tối ưu tốt hơn, mang lại Episode Return mang giá trị Dương vững chắc trong quá trình đào tạo.
