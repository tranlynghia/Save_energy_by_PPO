"""
data_manager.py — PPO-Stable Smart Home Data Pipeline
=====================================================
Senior RL Engineering Note:
    This module is the foundation of the environment's stability.
    All design decisions here directly affect PPO convergence.

Key Design Principles:
1.  SCALE-DOWN (Critical): Building-scale data (CityLearn ~60kW) MUST be rescaled
    to smart home scale (~1-5kW). Without this, reward magnitudes are 40x too large,
    causing critic value_loss to explode and explained_variance → 0.

2.  SYNTHETIC OUTDOOR TEMPERATURE: CityLearn Building 3 has no reliable outdoor temp.
    We generate a sinusoidal daily cycle (25°C night → 35°C peak at 2PM). This is
    essential because HVAC COP physics depends on (T_out - T_in). Without outdoor
    thermal forcing, the thermal model degenerates to a trivial system.

3.  FORECAST WITHOUT LEAKAGE: Using DataFrame.shift(-k) creates look-ahead forecasts
    from the agent's perspective. The key invariant: forecast[t] = f(actual_data[t+k]).
    Boundary conditions at episode end MUST be handled (wrap-around or clamp),
    otherwise the agent sees phantom zeros that corrupt the temporal information.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional
from env.weather.weather_loader import EPWWeatherLoader
from env.physics.pv_model import calculate_pv_physics
import yaml

# Load config for PV model parameters
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

# ============================================================
# SCALE FACTOR — WHY 1/40?
# ============================================================
# CityLearn Building 3 represents a commercial building:
#   - Equipment load: 20-50 kWh/h
#   - Cooling load: 30-63 kWh/h
# A realistic smart home has:
#   - Total load: 0.5-4 kWh/h
# Factor = 1/40 brings building data into smart home range.
# WITHOUT THIS: cost = 63 * 3500 VND = 220,500 VND per 15min step
#               → r_eco ≈ -44, while r_comfort ≈ -0.1
#               → reward dominated by cost, PPO can't balance objectives
# WITH THIS:    cost ≈ 1.5 * 3500 / 4 = 1,312 VND per 15min step
#               → r_eco ≈ -0.26, nicely balanced with comfort penalty
SCALE_FACTOR: float = 1.0 / 40.0

# Smart home PV system: 5 kWp rooftop panels
# CityLearn provides Solar Generation in W per kW installed
# We assume 5kW installed capacity -> max ≈ 5 kW during peak solar
PV_CAPACITY_KW: float = 5.0

# Electricity pricing (VND/kWh) — Vietnamese TOU structure (2026)
PRICE_OFF_PEAK: float = 1300.0   # 00:00 - 06:00
PRICE_NORMAL:   float = 1987.0   # 06:00 - 17:30, 22:30 - 24:00
PRICE_PEAK:     float = 3640.0   # 17:30 - 22:30


class DataManager:
    """
    PPO-stable data pipeline for Smart Home HEMS environment.

    Responsibilities:
    -   Load and scale CityLearn Building 3 data to smart home magnitude.
    -   Extract meteorological data from EPW files (Hanoi IWEC).
    -   Build look-ahead forecasts (6h, 12h, 24h) without data leakage.
    -   Provide step-level data access with safe boundary handling.
    """

    # CSV column names (CityLearn Building 3 format)
    COL_MONTH    = "Month"
    COL_HOUR     = "Hour"
    COL_DAY_TYPE = "Day Type"
    COL_EQUIP    = "Equipment Electric Power [kWh]"
    COL_DHW      = "DHW Heating [kWh]"
    COL_COOLING  = "Cooling Load [kWh]"
    COL_HEATING  = "Heating Load [kWh]"
    COL_SOLAR    = "Solar Generation [W/kW]"
    COL_HUMIDITY = "Indoor Relative Humidity [%]"

    def __init__(
        self,
        csv_path: Optional[str] = None,
        epw_path: Optional[str] = None,
        scale_factor: float = SCALE_FACTOR,
    ) -> None:
        self.scale_factor = scale_factor

        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", ".."))

        # --- Locate Load Data ---
        if csv_path is None:
            csv_path = os.path.join(project_root, "data", "real_world", "building_3_load.csv")

        # --- Locate Weather Data ---
        if epw_path is None:
            epw_path = os.path.join(project_root, "data", "real_world", "VNM_Hanoi.488200_IWEC.epw")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Load data not found at {csv_path}")

        raw = pd.read_csv(csv_path)
        
        # Load Weather
        self.weather_loader = None
        if os.path.exists(epw_path):
            self.weather_loader = EPWWeatherLoader(epw_path)
        
        self._df = self._build_pipeline(raw)
        self.max_steps: int = len(self._df)

    def _build_pipeline(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        n = len(df)

        # 1. Scale Load Data
        df["equipment_load_kw"] = df[self.COL_EQUIP]   * self.scale_factor
        df["cooling_load_kw"]   = df[self.COL_COOLING] * self.scale_factor
        df["dhw_load_kw"]       = df[self.COL_DHW]     * self.scale_factor
        df["heating_load_kw"]   = df[self.COL_HEATING] * self.scale_factor
        df["base_load_kw"] = df["equipment_load_kw"] + df["dhw_load_kw"] + df["heating_load_kw"]

        for col in ["equipment_load_kw", "cooling_load_kw", "dhw_load_kw", "base_load_kw"]:
            df[col] = df[col].clip(lower=0.0)

        # 2. Integrate Weather Data
        if self.weather_loader:
            weather_df = self.weather_loader.get_weather_series(n)
            df["outdoor_temp"] = weather_df["outdoor_temp"].values
            df["ghi_w_m2"] = weather_df["ghi"].values
            df["dni_w_m2"] = weather_df["dni"].values
            df["dhi_w_m2"] = weather_df["dhi"].values
            df["humidity"] = weather_df["humidity"].values
            df["wind_speed"] = weather_df["wind_speed"].values
            df["weather_source"] = "Hanoi EPW Real Weather"
        else:
            # Fallback to synthetic
            hours = np.arange(n) % 24
            noise = np.random.normal(0, 0.3, n)
            df["outdoor_temp"] = 30.0 + 5.0 * np.sin(2 * np.pi * (hours - 9.0) / 24.0) + noise
            df["ghi_w_m2"] = 900.0 * np.sin(np.pi * (np.clip(hours, 6, 18) - 6.0) / 12.0)
            df["dni_w_m2"] = df["ghi_w_m2"] * 0.8
            df["dhi_w_m2"] = df["ghi_w_m2"] * 0.2
            df["humidity"] = 70.0 + 10.0 * np.sin(2 * np.pi * hours / 24.0)
            df["wind_speed"] = 2.0 + np.random.normal(0, 0.5, n)
            df["weather_source"] = "Demo weather profile"

        # 3. PV Model (Physical)
        pv_ac_list = []
        pv_cell_temp_list = []
        
        for idx in range(n):
            ghi = df["ghi_w_m2"].iloc[idx]
            out_t = df["outdoor_temp"].iloc[idx]
            pv_info = calculate_pv_physics(ghi, out_t, _CONFIG)
            pv_ac_list.append(pv_info["pv_ac_kw"])
            pv_cell_temp_list.append(pv_info["pv_cell_temp"])
            
        df["pv_kw"] = np.array(pv_ac_list, dtype=np.float32)
        df["pv_cell_temp"] = np.array(pv_cell_temp_list, dtype=np.float32)

        # 4. Electricity Pricing
        hours = np.arange(n) % 24
        price = np.full(n, PRICE_NORMAL)
        price[(hours >= 0) & (hours < 6)] = PRICE_OFF_PEAK
        price[(hours >= 18) & (hours <= 22)] = PRICE_PEAK
        df["price_vnd_kwh"] = price

        # 5. Time Features
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["hour"] = hours

        # 6. Forecasts
        for h in [1, 6, 12, 24]:
            df[f"pv_fc_{h}h"]    = df["pv_kw"].shift(-h).bfill().astype(np.float32)
            df[f"load_fc_{h}h"]  = df["base_load_kw"].shift(-h).bfill().astype(np.float32)
            df[f"temp_fc_{h}h"]  = df["outdoor_temp"].shift(-h).bfill().astype(np.float32)
            df[f"price_fc_{h}h"] = df["price_vnd_kwh"].shift(-h).bfill().astype(np.float32)

        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype(np.float32)

        return df.reset_index(drop=True)

    def get_step_data(self, step: int) -> Dict[str, float]:
        idx = int(np.clip(step, 0, self.max_steps - 1))
        row = self._df.iloc[idx]

        data = {
            "equipment_load_kw": float(row["equipment_load_kw"]),
            "cooling_load_kw":   float(row["cooling_load_kw"]),
            "dhw_load_kw":       float(row["dhw_load_kw"]),
            "base_load_kw":      float(row["base_load_kw"]),
            "pv_kw":             float(row["pv_kw"]),
            "pv_cell_temp":      float(row.get("pv_cell_temp", 25.0)),
            "outdoor_temp":      float(row["outdoor_temp"]),
            "ghi_w_m2":          float(row["ghi_w_m2"]),
            "dni_w_m2":          float(row.get("dni_w_m2", 0.0)),
            "dhi_w_m2":          float(row.get("dhi_w_m2", 0.0)),
            "humidity":          float(row.get("humidity", 60.0)),
            "wind_speed":        float(row.get("wind_speed", 2.0)),
            "price_vnd_kwh":     float(row["price_vnd_kwh"]),
            "hour":              int(row["hour"]),
            "hour_sin":          float(row["hour_sin"]),
            "hour_cos":          float(row["hour_cos"]),
            "weather_source":    row["weather_source"],
        }
        
        # Add forecasts
        for h in [1, 6, 12, 24]:
            data[f"pv_fc_{h}h"] = float(row[f"pv_fc_{h}h"])
            data[f"load_fc_{h}h"] = float(row[f"load_fc_{h}h"])
            data[f"temp_fc_{h}h"] = float(row[f"temp_fc_{h}h"])
            data[f"price_fc_{h}h"] = float(row[f"price_fc_{h}h"])
            
        return data
