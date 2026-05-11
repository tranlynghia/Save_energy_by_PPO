"""
forecast/load_forecast.py
==========================
Load (Electricity Demand) Forecast — Transformer-based model.

Kiến trúc:
  Input  : past load sequence + time features [seq_len, input_size]
  Model  : Transformer Encoder (4 heads, 2 layers, d_model=64)
  Output : forecast for H horizons [horizon_steps]
  + Uncertainty: MC-Dropout

StatisticalLoadForecast:
  Fallback — lag-672 (cùng giờ tuần trước) + weighted average + noise.

Load có tính chu kỳ TUẦN mạnh (weekly pattern) → Transformer phù hợp hơn LSTM.
"""

import numpy as np
import math
import torch
import torch.nn as nn
from collections import deque


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder Forecaster
# ---------------------------------------------------------------------------
class _TransformerNet(nn.Module):
    def __init__(
        self,
        input_size: int = 4,    # load + hour_sin + hour_cos + weekday
        d_model: int    = 64,
        nhead: int      = 4,
        num_layers: int = 2,
        horizon: int    = 4,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout=dropout)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)
        self.head       = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: [B, seq_len, input_size]
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.head(x)  # [B, horizon]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
class LoadForecastTransformer:
    """
    Transformer-based Load Forecast với time features (sin/cos encoding).
    Trả về (mean_forecast, std_forecast).
    """

    def __init__(
        self,
        seq_len: int     = 672,         # 1 tuần lịch sử @ 15 phút
        horizon: int     = 96,          # Dự báo 24 giờ tới
        d_model: int     = 64,
        nhead: int       = 4,
        num_layers: int  = 2,
        dropout: float   = 0.1,
        mc_samples: int  = 20,
        device: str      = "cpu",
        model_path: str  = None,
    ):
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.mc_samples = mc_samples
        self.device     = torch.device(device)

        self.net = _TransformerNet(
            input_size=4,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
        ).to(self.device)

        self._trained = False
        if model_path is not None:
            self.load(model_path)

        self._history: deque = deque(maxlen=seq_len)
        self._time_feat: deque = deque(maxlen=seq_len)  # (hour_sin, hour_cos, weekday_norm)

    def update_history(self, load_kw: float, hour: int, weekday: int):
        """Cập nhật ring-buffer với tải điện và time features."""
        self._history.append(float(load_kw))
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        wd_norm  = weekday / 6.0
        self._time_feat.append((hour_sin, hour_cos, wd_norm))

    def predict(self, noise_sigma: float = 0.0) -> tuple:
        if not self._trained or len(self._history) < self.seq_len:
            return StatisticalLoadForecast.predict_from_history(
                list(self._history), self.horizon, noise_sigma
            )

        load_arr  = np.array(self._history, dtype=np.float32).reshape(-1, 1)
        feat_arr  = np.array(self._time_feat, dtype=np.float32)  # [seq_len, 3]
        x_np      = np.concatenate([load_arr, feat_arr], axis=-1)  # [seq_len, 4]
        x = torch.tensor(x_np).unsqueeze(0).to(self.device)        # [1, seq_len, 4]

        self.net.train()
        with torch.no_grad():
            preds = torch.stack(
                [self.net(x).squeeze(0) for _ in range(self.mc_samples)]
            )
        self.net.eval()

        mean = preds.mean(dim=0).cpu().numpy()
        std  = preds.std(dim=0).cpu().numpy()

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
class StatisticalLoadForecast:
    """
    Fallback — Same-hour-last-week (lag-672) + linear trend + noise.
    """

    @staticmethod
    def predict_from_history(
        history: list,
        horizon: int,
        noise_sigma: float = 0.3,   # kW
    ) -> tuple:
        n = len(history)
        mean_pred = np.zeros(horizon, dtype=np.float32)
        std_pred  = np.full(horizon, noise_sigma, dtype=np.float32)

        lag = 672   # 1 tuần @ 15 phút
        for h in range(horizon):
            idx = n - lag + h
            if 0 <= idx < n:
                base = history[idx]
            elif n > 0:
                base = history[-1]
            else:
                base = 0.0
            mean_pred[h] = max(0.0, base + np.random.normal(0, noise_sigma))

        return mean_pred, std_pred
