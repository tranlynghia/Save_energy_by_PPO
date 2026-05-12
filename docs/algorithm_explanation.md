# Tài liệu Phân tích Thuật toán PPO trong HEMS

## 1. Giới thiệu Kiến trúc Học Tăng Cường (RL Framework)

Hệ thống HEMS (Home Energy Management System) được mô hình hóa dưới dạng một Quy trình Quyết định Markov (Markov Decision Process - MDP) bao gồm:
* **State ($S_t$):** Các yếu tố môi trường nội bộ và bên ngoài (Nhiệt độ, SoC, Phụ tải, Giá điện, Giờ...).
* **Action ($A_t$):** Lệnh điều khiển pin (Sạc/Xả) và hệ thống HVAC.
* **Reward ($R_t$):** Hàm mục tiêu để đánh giá mức độ "tốt" của quyết định $A_t$ tại trạng thái $S_t$.
* **Transition Dynamics ($P$):** Động lực học vật lý (Physical constraints) chuyển đổi từ $S_t$ sang $S_{t+1}$ thông qua `env.step()`.

## 2. Proximal Policy Optimization (PPO)

PPO là cốt lõi của hệ thống AI trong dự án này. Đây là thuật toán Actor-Critic tiên tiến do OpenAI phát triển, được thiết kế để giải quyết bài toán không gian hành động liên tục (continuous action spaces) một cách ổn định.

### 2.1 Tại sao lại chọn PPO cho HEMS?
Trong quản lý năng lượng, các hành động (như đặt dòng điện sạc 2.45 kW) là những biến liên tục. Thuật toán Q-Learning truyền thống (như DQN) chỉ giải quyết được các hành động rời rạc (ví dụ: Tắt, Bật, Sạc nhanh). PPO giải quyết tốt không gian liên tục thông qua việc xấp xỉ một hàm phân phối (thường là Gaussian).
Hơn nữa, PPO có tính chất Trust Region Policy Optimization (TRPO) đơn giản hóa, ngăn không cho chính sách học (Policy) nhảy quá xa khỏi chính sách cũ, từ đó ngăn chặn sự sụp đổ mô hình (catastrophic forgetting).

### 2.2 Hàm Mục Tiêu PPO (The Surrogate Objective)
Hàm mục tiêu cốt lõi của PPO tập trung vào tỷ lệ xác suất (Probability Ratio) giữa chính sách mới và chính sách cũ:
$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

Hàm Clipped Surrogate Objective:
$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$

Trong đó:
* $\hat{A}_t$ là Advantage Estimate (Lợi thế). Nếu Advantage dương, action đó tốt hơn trung bình, và xác suất thực hiện action đó sẽ được tăng lên.
* $\epsilon$ là clip range (trong dự án dùng 0.2). Nó cắt bỏ các cập nhật làm cho tỷ lệ quá chênh lệch, ép mạng nơ-ron học hỏi từng bước nhỏ, an toàn.

## 3. Generalized Advantage Estimation (GAE)
Dự án sử dụng Critic Network để dự đoán State Value $V(s)$. Từ đó, GAE được tính toán để giảm phương sai (variance) của quá trình huấn luyện:
$\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots + (\gamma \lambda)^{T-t+1} \delta_{T-1}$
Với $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (TD-Error).

Điều này giúp agent không chỉ nhìn vào phần thưởng ngay lập tức, mà còn phân bổ phần thưởng về các bước quyết định tốt trong quá khứ (Credit Assignment), cực kỳ quan trọng cho chiến lược xả pin (Peak Shaving) - phải sạc pin trước vài giờ để dùng.

## 4. Kiến trúc Mạng Nơ-ron
Chúng ta sử dụng kiến trúc Multi-Layer Perceptron (MLP) cho cả Actor và Critic:
1. **Input Layer:** 10 nodes (Kích thước observation space, các biến đã được normalized).
2. **Hidden Layers:** 2 lớp chia sẻ (hoặc độc lập tùy config mặc định sb3) với 64 hoặc 128 nodes, dùng hàm kích hoạt Tanh.
3. **Actor Output:** 2 nodes đại diện cho $\mu$ (mean) của action Sạc/Xả và HVAC. Kèm theo một tham số độc lập để học $\sigma$ (độ lệch chuẩn, điều khiển tính khám phá - exploration).
4. **Critic Output:** 1 node đại diện cho Value Function.

## 5. Kỹ thuật Ổn định Môi trường HEMS

Dù thuật toán PPO có mạnh đến đâu, nếu môi trường "độc hại" (toxic environment), nó cũng không thể học được. Các kỹ thuật đã được áp dụng trong project:

* **Reward Normalization & Clipping:** PPO nhạy cảm với scale của reward. Nếu chi phí điện là -100.000 VND và tiện nghi là -1.0, agent sẽ bỏ mặc tiện nghi. Toàn bộ reward được co giãn (scale down) và ép chặt (clip) vào đoạn `[-5, 5]`.
* **VecNormalize:** Cả Observation và Reward được normalize online bằng thuật toán Moving Average & Variance của SB3. Khử hoàn toàn khác biệt biên độ giữa GHI (0-1000 W/m2) và Nhiệt độ phòng (20-30°C).
* **Smooth Continuous Penalty:** Các khoản phạt giật cục (nếu SoC < 10% trừ 10 điểm) tạo ra các bức tường dốc đứng trên hàm loss, làm PPO bị "bật ngửa" (Gradient explosion). Tất cả phạt dạng step-function đã được chuyển thành continuous/quadratic (nỗi đau lớn dần).
* **Sequential Energy Routing:** Trong một timestep, vật lý quy định PV dùng trực tiếp -> phần dư sạc pin -> phần dư bán lưới. Mô hình đã code đúng logic nguyên lý tuần tự này, tránh việc Agent tính toán sai dòng điện đi vào tải và pin.
