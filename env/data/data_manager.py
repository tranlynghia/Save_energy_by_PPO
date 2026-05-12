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
    -   Generate realistic outdoor temperature via sinusoidal model.
    -   Build look-ahead forecasts (6h, 12h, 24h) without data leakage.
    -   Provide step-level data access with safe boundary handling.

    Args:
        csv_path: Absolute path to building_3_load.csv.
                  If None, searches relative to this file's location.
        scale_factor: Physical scaling coefficient. Default = 1/40.
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
        scale_factor: float = SCALE_FACTOR,
    ) -> None:
        self.scale_factor = scale_factor

        # --- Locate the CSV file ---
        if csv_path is None:
            # __file__ = project/env/data/data_manager.py
            # Go up 2 levels: env/data/ → env/ → project/
            here = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(here, "..", ".."))
            csv_path = os.path.join(
                project_root, "data", "real_world", "building_3_load.csv"
            )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[DataManager] Dataset not found at: {csv_path}\n"
                "Place building_3_load.csv in project/data/real_world/"
            )

        raw = pd.read_csv(csv_path)
        self._df = self._build_pipeline(raw)
        self.max_steps: int = len(self._df)

    # ------------------------------------------------------------------
    # PRIVATE: Build the full processed DataFrame once at init
    # ------------------------------------------------------------------
    def _build_pipeline(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        n = len(df)

        # ============================================================
        # STEP 1: PHYSICAL SCALE-DOWN
        # ============================================================
        df["equipment_load_kw"] = df[self.COL_EQUIP]   * self.scale_factor
        df["cooling_load_kw"]   = df[self.COL_COOLING] * self.scale_factor
        df["dhw_load_kw"]       = df[self.COL_DHW]     * self.scale_factor
        df["heating_load_kw"]   = df[self.COL_HEATING] * self.scale_factor

        df["base_load_kw"] = (
            df["equipment_load_kw"]
            + df["dhw_load_kw"]
            + df["heating_load_kw"]
        )

        for col in ["equipment_load_kw", "cooling_load_kw", "dhw_load_kw", "base_load_kw"]:
            df[col] = df[col].clip(lower=0.0)

        # ============================================================
        # STEP 2: OUTDOOR TEMPERATURE & GHI (REAL EPW OR SYNTHETIC)
        # ============================================================
        hours = np.arange(n) % 24  # 0-23 cycling hourly
        
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", ".."))
        epw_path = os.path.join(project_root, "data", "real_world", "VNM_Hanoi.488200_IWEC.epw")
        
        has_epw = False
        if os.path.exists(epw_path):
            try:
                # EPW data starts at line 9 (index 8)
                epw_df = pd.read_csv(epw_path, skiprows=8, header=None)
                if len(epw_df) >= n:
                    # Col 6: Dry Bulb Temp, Col 13: Global Horizontal Radiation
                    df["outdoor_temp"] = epw_df.iloc[:n, 6].values.astype(float)
                    df["ghi_w_m2"] = epw_df.iloc[:n, 13].values.astype(float)
                    df["weather_source"] = "Hanoi EPW Real Weather"
                    has_epw = True
            except Exception as e:
                print(f"[DataManager] Warning: Could not load EPW file ({e}).")
                
        if not has_epw:
            noise = np.random.normal(0, 0.3, n)
            outdoor_temp = 30.0 + 5.0 * np.sin(2 * np.pi * (hours - 9.0) / 24.0) + noise
            df["outdoor_temp"] = np.clip(outdoor_temp, 22.0, 38.0)
            
            ghi = np.zeros(n)
            daylight_mask = (hours >= 6) & (hours <= 18)
            ghi[daylight_mask] = 900.0 * np.sin(np.pi * (hours[daylight_mask] - 6.0) / 12.0)
            df["ghi_w_m2"] = ghi
            df["weather_source"] = "Demo weather profile"

        # ============================================================
        # STEP 2.5: PV MODEL
        # ============================================================
        pv_capacity_kw = 5.0
        pv_derate_factor = 0.85
        pv_kw = pv_capacity_kw * (df["ghi_w_m2"] / 1000.0) * pv_derate_factor
        df["pv_kw"] = pv_kw.clip(lower=0.0, upper=pv_capacity_kw)

        # ============================================================
        # STEP 3: ELECTRICITY PRICING (TIME-OF-USE)
        # ============================================================
        price = np.full(n, PRICE_NORMAL)
        # Off-peak: 00:00 - 06:00 (hours 0 to 5)
        price[(hours >= 0) & (hours < 6)] = PRICE_OFF_PEAK
        # Peak: 17:30 - 22:30 -> Approx: hours 18 to 22
        price[(hours >= 18) & (hours <= 22)] = PRICE_PEAK
        df["price_vnd_kwh"] = price

        # ============================================================
        # STEP 4: TIME FEATURES (PERIODIC ENCODING FOR PPO)
        # ============================================================
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["hour"] = hours
        
        # ============================================================
        # STEP 5: LOOK-AHEAD FORECASTS (NO LEAKAGE)
        # ============================================================
        for h in [1, 6, 12, 24]:
            df[f"pv_fc_{h}h"]    = df["pv_kw"].shift(-h).bfill().astype(np.float32)
            df[f"load_fc_{h}h"]  = df["base_load_kw"].shift(-h).bfill().astype(np.float32)
            df[f"outdoor_temp_fc_{h}h"]  = df["outdoor_temp"].shift(-h).bfill().astype(np.float32)
            df[f"price_fc_{h}h"] = df["price_vnd_kwh"].shift(-h).bfill().astype(np.float32)

        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype(np.float32)

        return df.reset_index(drop=True)

    def get_step_data(self, step: int) -> Dict[str, float]:
        """
        Return all environment data for a given timestep.
        """
        idx = int(np.clip(step, 0, self.max_steps - 1))
        row = self._df.iloc[idx]

        return {
            "equipment_load_kw": float(row["equipment_load_kw"]),
            "cooling_load_kw":   float(row["cooling_load_kw"]),
            "dhw_load_kw":       float(row["dhw_load_kw"]),
            "base_load_kw":      float(row["base_load_kw"]),
            "pv_kw":             float(row["pv_kw"]),
            "outdoor_temp":      float(row["outdoor_temp"]),
            "ghi_w_m2":          float(row["ghi_w_m2"]),
            "indoor_humidity":   float(row.get(self.COL_HUMIDITY, 60.0)),
            "price_vnd_kwh":     float(row["price_vnd_kwh"]),
            "hour":              int(row["hour"]),
            "hour_sin":          float(row["hour_sin"]),
            "hour_cos":          float(row["hour_cos"]),
            "month":             int(row.get(self.COL_MONTH, 1)),
            "weather_source":    row["weather_source"],
            "pv_fc_6h":          float(row["pv_fc_6h"]),
            "pv_fc_12h":         float(row["pv_fc_12h"]),
            "pv_fc_24h":         float(row["pv_fc_24h"]),
            "load_fc_6h":        float(row["load_fc_6h"]),
            "load_fc_12h":       float(row["load_fc_12h"]),
            "load_fc_24h":       float(row["load_fc_24h"]),
            "outdoor_temp_fc_1h":float(row["outdoor_temp_fc_1h"]),
            "outdoor_temp_fc_6h":float(row["outdoor_temp_fc_6h"]),
            "outdoor_temp_fc_12h":float(row["outdoor_temp_fc_12h"]),
            "outdoor_temp_fc_24h":float(row["outdoor_temp_fc_24h"]),
            "price_fc_6h":       float(row["price_fc_6h"]),
            "price_fc_12h":      float(row["price_fc_12h"]),
            "price_fc_24h":      float(row["price_fc_24h"]),
        }
