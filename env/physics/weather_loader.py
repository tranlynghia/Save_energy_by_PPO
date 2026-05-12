import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

class EPWWeatherLoader:
    """
    Loader for EnergyPlus Weather (EPW) files.
    Extracts key meteorological data for HEMS simulation.
    """
    
    # EPW Column Mapping (0-indexed)
    COL_YEAR = 0
    COL_MONTH = 1
    COL_DAY = 2
    COL_HOUR = 3
    COL_DRY_BULB = 6
    COL_DEW_POINT = 7
    COL_REL_HUMIDITY = 8
    COL_PRESSURE = 9
    COL_GHI = 13
    COL_DNI = 14
    COL_DHI = 15
    COL_WIND_SPEED = 21

    def __init__(self, epw_path: str):
        self.epw_path = epw_path
        if not os.path.exists(epw_path):
            raise FileNotFoundError(f"EPW file not found at {epw_path}")
        
        self.df = self._load_epw()

    def _load_epw(self) -> pd.DataFrame:
        """Reads EPW file and returns a cleaned DataFrame."""
        # EPW files skip the first 8 lines of metadata
        df = pd.read_csv(self.epw_path, skiprows=8, header=None)
        
        processed_df = pd.DataFrame({
            "year": df[self.COL_YEAR],
            "month": df[self.COL_MONTH],
            "day": df[self.COL_DAY],
            "hour": df[self.COL_HOUR] - 1,  # EPW uses 1-24, we use 0-23
            "outdoor_temp": df[self.COL_DRY_BULB],
            "humidity": df[self.COL_REL_HUMIDITY],
            "ghi": df[self.COL_GHI],
            "dni": df[self.COL_DNI],
            "dhi": df[self.COL_DHI],
            "wind_speed": df[self.COL_WIND_SPEED]
        })
        
        # EPW data is usually 8760 hours. 
        # If we need higher resolution (e.g. 15min), we would interpolate here.
        # For now, we assume 1-hour resolution data.
        return processed_df

    def get_weather_series(self, length: int) -> pd.DataFrame:
        """Returns weather data for a specific duration, handling wrapping."""
        n_available = len(self.df)
        
        # Offset to June 1st for summer Hanoi weather
        start_idx = 3624
        
        if start_idx + length <= n_available:
            return self.df.iloc[start_idx:start_idx+length].copy().reset_index(drop=True)
        else:
            # Wrap around for multi-year simulations if needed
            repeats = ((start_idx + length) // n_available) + 1
            full_df = pd.concat([self.df] * repeats, ignore_index=True)
            return full_df.iloc[start_idx:start_idx+length].copy().reset_index(drop=True)
