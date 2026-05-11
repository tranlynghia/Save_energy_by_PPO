"""
forecast/forecast_engine.py
============================
ForecastEngine — Lớp tổng hợp quản lý tất cả forecast models.

Nhiệm vụ:
  1. Nhận dữ liệu thực tế từ DataManager tại mỗi step.
  2. Cập nhật ring-buffer lịch sử cho từng model.
  3. Trả về forecast (mean, std) cho PV / Load / Occupancy.
  4. Thêm Forecast Noise (tuỳ chỉnh noise_sigma) để tránh data leakage.

Tích hợp:
  - ForecastEngine được khởi tạo trong SmartHomeEnv.__init__()
  - Được gọi mỗi step để thay thế "future lookup" cũ trong DataManager.

Forecast horizons (bước @ 15 phút):
  4  bước =  1 giờ
  24 bước =  6 giờ
  48 bước = 12 giờ
  96 bước = 24 giờ
"""

import numpy as np
from forecast.pv_forecast   import PVForecastLSTM,   StatisticalPVForecast
from forecast.load_forecast  import LoadForecastTransformer, StatisticalLoadForecast
from forecast.occ_forecast   import OccupancyForecastGRU,   StatisticalOccupancyForecast


class ForecastEngine:
    """
    Unified interface cho tất cả forecast models.

    Mặc định dùng Statistical fallback (không cần pre-training).
    Sau khi train các neural models, truyền model_paths vào để enable.
    """

    # Horizon indices (bước @ 15 phút)
    H_6H  = 24
    H_12H = 48
    H_24H = 96

    def __init__(
        self,
        pv_noise_sigma:   float = 80.0,    # W/kW
        load_noise_sigma: float = 0.4,     # kW
        occ_noise_sigma:  float = 0.05,    # xác suất
        pv_model_path:    str   = None,
        load_model_path:  str   = None,
        occ_model_path:   str   = None,
        device: str             = "cpu",
    ):
        self.pv_noise   = pv_noise_sigma
        self.load_noise = load_noise_sigma
        self.occ_noise  = occ_noise_sigma

        # Khởi tạo models (fallback nếu không có model_path)
        self.pv_model  = PVForecastLSTM(model_path=pv_model_path,   device=device)
        self.load_model = LoadForecastTransformer(model_path=load_model_path, device=device)
        self.occ_model  = OccupancyForecastGRU(model_path=occ_model_path,  device=device)

        # Cache forecast mới nhất
        self._last_pv_mean   = np.zeros(self.H_24H)
        self._last_pv_std    = np.zeros(self.H_24H)
        self._last_load_mean = np.zeros(self.H_24H)
        self._last_load_std  = np.zeros(self.H_24H)
        self._last_occ_mean  = np.zeros(self.H_24H)
        self._last_occ_std   = np.zeros(self.H_24H)

    def update(
        self,
        pv_actual_w_kw: float,
        load_actual_kw: float,
        occupancy: float,
        hour: int,
        weekday: int,
    ):
        """
        Cập nhật lịch sử với giá trị thực tế tại bước hiện tại.
        Phải gọi TRƯỚC khi predict() để tránh data leakage.

        :param pv_actual_w_kw: Sản lượng PV thực tế (W/kW)
        :param load_actual_kw: Tải điện thực tế (kW)
        :param occupancy:      Số người ở nhà (hoặc xác suất)
        :param hour:           Giờ hiện tại [0–23]
        :param weekday:        Thứ trong tuần [0=Mon, 6=Sun]
        """
        self.pv_model.update_history(pv_actual_w_kw)
        self.load_model.update_history(load_actual_kw, hour, weekday)
        self.occ_model.update_history(occupancy, hour, weekday)

    def predict_all(self) -> dict:
        """
        Chạy forecast cho tất cả models và trả về dict kết quả.
        Noise được inject TẠI ĐÂY để tách biệt khỏi model output gốc.

        Returns:
            {
              "pv":   {"mean": ndarray[96], "std": ndarray[96]},
              "load": {"mean": ndarray[96], "std": ndarray[96]},
              "occ":  {"mean": ndarray[96], "std": ndarray[96]},
              # Scalar lookups cho observation vector:
              "pv_6h": float, "pv_12h": float, "pv_24h": float,
              "load_6h": float, "load_12h": float,
              "occ_now": float,
            }
        """
        pv_mean,   pv_std   = self.pv_model.predict(noise_sigma=self.pv_noise)
        load_mean, load_std = self.load_model.predict(noise_sigma=self.load_noise)
        occ_mean,  occ_std  = self.occ_model.predict(noise_sigma=self.occ_noise)

        # Cache
        self._last_pv_mean   = pv_mean
        self._last_pv_std    = pv_std
        self._last_load_mean = load_mean
        self._last_load_std  = load_std
        self._last_occ_mean  = occ_mean
        self._last_occ_std   = occ_std

        def _safe_idx(arr, idx):
            """Lấy phần tử tại idx, fallback 0 nếu không đủ dài."""
            return float(arr[idx]) if idx < len(arr) else 0.0

        return {
            "pv":   {"mean": pv_mean,   "std": pv_std},
            "load": {"mean": load_mean, "std": load_std},
            "occ":  {"mean": occ_mean,  "std": occ_std},
            # Scalar values cho observation vector
            "pv_6h":   _safe_idx(pv_mean,   self.H_6H  - 1),
            "pv_12h":  _safe_idx(pv_mean,   self.H_12H - 1),
            "pv_24h":  _safe_idx(pv_mean,   self.H_24H - 1),
            "load_6h": _safe_idx(load_mean, self.H_6H  - 1),
            "load_12h":_safe_idx(load_mean, self.H_12H - 1),
            "occ_now": _safe_idx(occ_mean,  0),
        }

    def get_pv_forecast(self, horizon_steps: int) -> float:
        """Lấy giá trị PV forecast tại một horizon cụ thể (W/kW)."""
        idx = min(horizon_steps - 1, len(self._last_pv_mean) - 1)
        return float(self._last_pv_mean[idx]) if len(self._last_pv_mean) > 0 else 0.0

    def get_load_forecast(self, horizon_steps: int) -> float:
        """Lấy giá trị Load forecast tại một horizon cụ thể (kW)."""
        idx = min(horizon_steps - 1, len(self._last_load_mean) - 1)
        return float(self._last_load_mean[idx]) if len(self._last_load_mean) > 0 else 0.0
