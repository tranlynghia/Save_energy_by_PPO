"""
forecast/occ_forecast.py
=========================
Occupancy Forecast — GRU-based model.

Kiến trúc:
  Input  : past occupancy binary/count + time [seq_len, input_size]
  Model  : 1-layer GRU, hidden_size=32 (nhẹ — occupancy đơn giản hơn)
  Output : probability of occupancy for H horizons [horizon_steps]
  + Uncertainty: MC-Dropout

StatisticalOccupancyForecast:
  Fallback — schedule-based (giả định pattern ngày làm việc vs cuối tuần).
  Pattern mặc định: người ở nhà 07:00–09:00 và 18:00–23:00 các ngày thường.
"""

import numpy as np
import math
import torch
import torch.nn as nn
from collections import deque


# ---------------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------------
class _GRUNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1,
                 horizon=4, dropout=0.2):
        super().__init__()
        self.gru     = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, horizon)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out    = self.dropout(out[:, -1, :])
        return self.sigmoid(self.head(out))  # [B, horizon] — prob ∈ [0,1]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
class OccupancyForecastGRU:
    """
    GRU-based Occupancy Forecast.
    Trả về (mean_prob [horizon], std_prob [horizon]) — xác suất có người ở nhà.
    """

    def __init__(
        self,
        seq_len: int     = 192,        # 2 ngày lịch sử
        horizon: int     = 96,         # 24 giờ tới
        hidden_size: int = 32,
        num_layers: int  = 1,
        dropout: float   = 0.2,
        mc_samples: int  = 20,
        device: str      = "cpu",
        model_path: str  = None,
    ):
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.mc_samples = mc_samples
        self.device     = torch.device(device)

        self.net = _GRUNet(
            input_size=3,              # occ + hour_sin + hour_cos
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
        ).to(self.device)

        self._trained = False
        if model_path is not None:
            self.load(model_path)

        self._history: deque   = deque(maxlen=seq_len)
        self._time_feat: deque = deque(maxlen=seq_len)

    def update_history(self, occupancy: float, hour: int, weekday: int = 0):
        """Cập nhật với giá trị occupancy (0/1 hoặc số người) và time features."""
        self._history.append(float(occupancy))
        self._time_feat.append((
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
        ))

    def predict(self, noise_sigma: float = 0.0) -> tuple:
        if not self._trained or len(self._history) < self.seq_len:
            # Fallback: schedule-based
            if self._time_feat:
                last_hour_sin, last_hour_cos = self._time_feat[-1]
                # Tính lại giờ hiện tại từ sin/cos
                hour = int(round(math.atan2(last_hour_sin, last_hour_cos) * 24 / (2 * math.pi))) % 24
            else:
                hour = 12
            return StatisticalOccupancyForecast.predict_schedule(hour, self.horizon, noise_sigma)

        occ_arr  = np.array(self._history, dtype=np.float32).reshape(-1, 1)
        feat_arr = np.array(self._time_feat, dtype=np.float32)  # [seq_len, 2]
        x_np     = np.concatenate([occ_arr, feat_arr], axis=-1)  # [seq_len, 3]
        x = torch.tensor(x_np).unsqueeze(0).to(self.device)

        self.net.train()
        with torch.no_grad():
            preds = torch.stack(
                [self.net(x).squeeze(0) for _ in range(self.mc_samples)]
            )
        self.net.eval()

        mean = preds.mean(dim=0).cpu().numpy()
        std  = preds.std(dim=0).cpu().numpy()

        if noise_sigma > 0:
            mean = np.clip(mean + np.random.normal(0, noise_sigma, size=mean.shape), 0, 1)

        return mean, std

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()
        self._trained = True

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)


# ---------------------------------------------------------------------------
# Statistical Fallback — Schedule-based
# ---------------------------------------------------------------------------
class StatisticalOccupancyForecast:
    """
    Dự báo occupancy từ schedule cố định (không cần training).
    Pattern ngày thường: có người 07–09 và 18–23.
    Pattern cuối tuần: có người 08–12 và 14–23.
    """

    # Giờ có người ở nhà (hour ∈ [0,23])
    WEEKDAY_HOME_HOURS  = set(range(7, 10)) | set(range(18, 24))
    WEEKEND_HOME_HOURS  = set(range(8, 13)) | set(range(14, 24))

    @staticmethod
    def predict_schedule(
        current_hour: int,
        horizon: int,
        noise_sigma: float = 0.05,
    ) -> tuple:
        mean_pred = np.zeros(horizon, dtype=np.float32)
        std_pred  = np.full(horizon, noise_sigma, dtype=np.float32)

        for h in range(horizon):
            future_hour = (current_hour + h // 4) % 24   # mỗi 4 bước = 1 giờ
            # Giả định weekday (đơn giản)
            p = 1.0 if future_hour in StatisticalOccupancyForecast.WEEKDAY_HOME_HOURS else 0.0
            mean_pred[h] = float(np.clip(p + np.random.normal(0, noise_sigma), 0, 1))

        return mean_pred, std_pred
