"""
env/physics/pv_model.py
=======================
Physical model for Photovoltaic (PV) power generation.
Calculates electrical output based on irradiance, ambient temperature, 
and cell-level physics (NOCT model, temperature derating).
"""

import numpy as np
from typing import Dict, Any

def calculate_pv_physics(
    ghi_w_m2: float,
    outdoor_temp: float,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculates PV AC power output using a physical derating model.
    
    Formula:
    1. T_cell = T_ambient + (NOCT - 20)/800 * GHI
    2. eta_temp = 1 - beta_temp * (T_cell - 25)
    3. P_dc = P_nom * (GHI / 1000) * eta_temp * shading_factor
    4. P_ac = P_dc * eta_inverter
    
    Args:
        ghi_w_m2: Global Horizontal Irradiance (W/m2)
        outdoor_temp: Ambient temperature (°C)
        config: Dictionary containing physics parameters.
        
    Returns:
        Dict containing:
            - pv_ac_kw: Final AC output (kW)
            - pv_dc_kw: DC output before inverter (kW)
            - pv_cell_temp: Calculated cell temperature (°C)
            - pv_efficiency_factor: Temperature derating factor
    """
    
    # 1. Configuration Parameters
    pv_params = config.get("physics", {}).get("pv", {})
    pv_capacity_kw = pv_params.get("capacity_kw", 5.0)
    noct = pv_params.get("noct", 45.0)  # Nominal Operating Cell Temp
    beta_temp = pv_params.get("temp_coeff", 0.004)  # efficiency loss per °C
    eta_inverter = pv_params.get("inverter_efficiency", 0.96)
    shading_factor = pv_params.get("shading_factor", 0.95) # 5% loss from dust/shading
    
    # 2. PV Cell Temperature Model (NOCT)
    # T_cell increases linearly with irradiance
    pv_cell_temp = outdoor_temp + ((noct - 20) / 800.0) * ghi_w_m2
    
    # 3. Temperature Derating Factor
    # Efficiency is normalized at 25°C. Above this, efficiency drops.
    pv_efficiency_factor = 1.0 - beta_temp * (pv_cell_temp - 25.0)
    pv_efficiency_factor = max(0.1, pv_efficiency_factor) # Sanity floor
    
    # 4. DC Power Calculation
    # Note: Using GHI for simple horizontal rooftop model. 
    # For tilted panels, one would use Plane of Array (POA) irradiance.
    pv_dc_kw = pv_capacity_kw * (ghi_w_m2 / 1000.0) * pv_efficiency_factor * shading_factor
    pv_dc_kw = max(0.0, pv_dc_kw)
    
    # 5. Inverter and AC Output
    pv_ac_kw = pv_dc_kw * eta_inverter
    
    # 6. Clipping to capacity (Inverter clipping)
    pv_ac_kw = min(pv_ac_kw, pv_capacity_kw)
    
    return {
        "pv_ac_kw": float(pv_ac_kw),
        "pv_dc_kw": float(pv_dc_kw),
        "pv_cell_temp": float(pv_cell_temp),
        "pv_efficiency_factor": float(pv_efficiency_factor),
        "inverter_efficiency": float(eta_inverter)
    }
