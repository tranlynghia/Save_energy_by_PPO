"""
forecast/pv_forecast.py
========================
PV Generation Forecast — LSTM-based model.

Kiến trúc:
  Input  : sequence of past PV values  [seq_len, input_size]
  Hidden : 2-layer LSTM, hidden_size=64
  Output : forecast for H horizons     [horizon_steps]
  + Uncertainty: dropout-based Monte-Carlo estimate

StatisticalPVForecast:
  Fallback khi chưa có model trained —
  dùng seasonal average + additive Gaussian noise.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------
class _LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 horizon=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: [B, seq_len, input_size]
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])   # last step
        return self.head(out)                   # [B, horizon]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
class PVForecastLSTM:
    """
    LSTM-based PV Forecast.
    - Dự báo H bước (mỗi bước = 15 phút) phía trước.
    - Trả về (mean_forecast, std_forecast) từ MC-Dropout (n_samples=20).

    Nếu model chưa được train → .predict() fallback về StatisticalPVForecast.
    """

    def __init__(
        self,
        seq_len: int     = 96,          # 24 giờ lịch sử @ 15 phút
        horizon: int     = 96,          # Dự báo 24 giờ tới
        hidden_size: int = 64,
        num_layers: int  = 2,
        dropout: float   = 0.2,
        mc_samples: int  = 20,
        device: str      = "cpu",
        model_path: str  = None,
    ):
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.mc_samples = mc_samples
        self.device     = torch.device(device)

        self.net = _LSTMNet(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
        ).to(self.device)

        self._trained = False
        if model_path is not None:
            self.load(model_path)

        # Ring-buffer lịch sử
        self._history: deque = deque(maxlen=seq_len)

    def update_history(self, pv_value: float):
        """Cập nhật ring-buffer với giá trị PV mới nhất (W/kW)."""
        self._history.append(float(pv_value))

    def predict(self, noise_sigma: float = 0.0) -> tuple:
        """
        Trả về (mean [horizon], std [horizon]) dự báo PV.
        noise_sigma: độ lệch chuẩn của nhiễu thêm vào để mô phỏng uncertainty.
        """
        if not self._trained or len(self._history) < self.seq_len:
            # Fallback: statistical
            return StatisticalPVForecast.predict_from_history(
                list(self._history), self.horizon, noise_sigma
            )

        x = torch.tensor(
            list(self._history), dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, seq_len, 1]

        # MC-Dropout: bật dropout khi inference
        self.net.train()
        with torch.no_grad():
            preds = torch.stack(
                [self.net(x).squeeze(0) for _ in range(self.mc_samples)]
            )  # [mc_samples, horizon]
        self.net.eval()

        mean = preds.mean(dim=0).cpu().numpy()  # [horizon]
        std  = preds.std(dim=0).cpu().numpy()

        # Thêm noise nhân tạo để simulate real-world forecast error
        if noise_sigma > 0:
            mean = mean + np.random.normal(0, noise_sigma, size=mean.shape)

        mean = np.clip(mean, 0.0, None)
        return mean, std

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()
        self._trained = True

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)


# ---------------------------------------------------------------------------
# Statistical Fallback
# ---------------------------------------------------------------------------
class StatisticalPVForecast:
    """
    Fallback không cần training: seasonal average + additive Gaussian noise.
    PV có tính chu kỳ ngày mạnh → lag-96 (cùng giờ hôm qua) là baseline tốt.
    """

    @staticmethod
    def predict_from_history(
        history: list,
        horizon: int,
        noise_sigma: float = 50.0,   # W/kW — sai số forecast điển hình
    ) -> tuple:
        n = len(history)
        mean_pred = np.zeros(horizon, dtype=np.float32)
        std_pred  = np.full(horizon, noise_sigma, dtype=np.float32)

        for h in range(horizon):
            lag = 96  # cùng giờ ngày hôm trước (@ 15 phút)
            idx = n - lag + h
            if 0 <= idx < n:
                mean_pred[h] = max(0.0, history[idx] + np.random.normal(0, noise_sigma))
            # else: giữ 0 (ban đêm hoặc không đủ lịch sử)

        return mean_pred, std_pred
