"""
evaluate/plot_metrics.py
========================
Research-Grade Visualization Suite for PPO HEMS Evaluation.
Designed for professional reports and research publications.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Professional Color Palette ---
COLORS = {
    "ppo":          "#1f77b4",  # Professional Blue
    "rule":         "#d62728",  # Professional Red
    "outdoor":      "#ff7f0e",  # Orange
    "pv":           "#bcbd22",  # Muted Olive/Yellow
    "grid":         "#7f7f7f",  # Gray
    "battery":      "#2ca02c",  # Green
    "load":         "#9467bd",  # Purple
    "hvac":         "#8c564b",  # Brown
    "comfort_zone": "#e1f5fe",  # Very Light Blue
    "charge":       "#2ecc71",
    "discharge":    "#e74c3c",
    "price":        "#f1c40f"
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def _hours(indices): return indices * 0.25

def _save(fig, path, title):
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Output] {title:<40} -> {os.path.basename(path)}")

def _add_annotations(ax, df, behavior_type):
    """Automatically detects and adds behavioral annotations to plots."""
    notes = []
    if behavior_type == "hvac":
        if (df["hvac_power_kw"] < 0.1).all():
            notes.append("HVAC Remained OFF")
        if (df["indoor_temp"] < 20).any():
            notes.append("Indoor Temp Collapsed")
    elif behavior_type == "battery":
        if (df["batt_power_kw"].abs() < 0.05).all():
            notes.append("Battery Frozen (No Cycling)")
        if df["soc"].max() < 0.2:
            notes.append("Low SoC Utilization")
    elif behavior_type == "pv":
        if (df.get("r_pv_export", 0) != 0).any() if "r_pv_export" in df.columns else False:
            notes.append("Significant PV Waste/Export")
            
    if notes:
        text = "\n".join(notes)
        ax.text(0.02, 0.95, text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9, color='red')

# 1. HVAC & Thermal Behavior (Dual Axis)
def plot_hvac_behavior(df_ppo, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    n = min(192, len(df_ppo)) # Show 2 days
    time = _hours(np.arange(n))
    
    # Left Axis: Temperature
    ax1.plot(time, df_ppo["outdoor_temp"].values[:n], color=COLORS["outdoor"], lw=1.5, ls="--", label="Outdoor Temp")
    ax1.plot(time, df_ppo["indoor_temp"].values[:n], color=COLORS["ppo"], lw=2, label="PPO Indoor Temp")
    
    # Comfort Zone Shading
    ax1.fill_between(time, 22, 26, color=COLORS["battery"], alpha=0.1, label="Target Comfort (22-26°C)")
    ax1.axhline(30, color="red", lw=1, ls=":", alpha=0.5, label="Overheat Limit")
    ax1.axhline(20, color="blue", lw=1, ls=":", alpha=0.5, label="Overcool Limit")
    
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylim(15, 40)
    
    # Right Axis: HVAC Power
    ax2 = ax1.twinx()
    ax2.fill_between(time, df_ppo["hvac_power_kw"].values[:n], 0, color="gray", alpha=0.3, label="HVAC Input (kW)")
    ax2.set_ylabel("HVAC Power (kW)")
    ax2.set_ylim(0, 5)
    
    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9)
    
    _add_annotations(ax1, df_ppo[:n], "hvac")
    ax1.set_title("HVAC Operation & Thermal Comfort Timeline")
    _save(fig, os.path.join(out_dir, "behavior_hvac_thermal.png"), "HVAC Behavior Analysis")

# 2. PV Flow & Utilization (Stacked Area)
def plot_pv_flow(df_ppo, out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    n = min(192, len(df_ppo))
    time = _hours(np.arange(n))
    
    def _safe(col): return df_ppo[col].values[:n] if col in df_ppo.columns else np.zeros(n)
    
    pv_gen = _safe("pv_kw")
    pv_to_load = _safe("pv_to_load")
    pv_to_batt = _safe("pv_to_battery")
    pv_export = _safe("pv_export")
    
    ax.stackplot(time, pv_to_load, pv_to_batt, pv_export,
                 labels=["PV -> Load", "PV -> Battery", "PV Export (Waste)"],
                 colors=[COLORS["pv"], COLORS["battery"], "#e0e0e0"], alpha=0.8)
    
    ax.plot(time, pv_gen, color="black", lw=1, ls="--", label="Total PV Available")
    
    ax.set_title("PV Generation & Utilization Flow")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time (hours)")
    ax.legend(loc="upper right")
    
    _add_annotations(ax, df_ppo[:n], "pv")
    _save(fig, os.path.join(out_dir, "behavior_pv_utilization.png"), "PV Flow Analysis")

# 3. Battery Dispatch Behavior
def plot_battery_behavior(df_ppo, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    n = min(192, len(df_ppo))
    time = _hours(np.arange(n))
    
    # Top: SoC
    ax1.plot(time, df_ppo["soc"].values[:n] * 100, color=COLORS["battery"], lw=2, label="Battery SoC")
    ax1.fill_between(time, df_ppo["soc"].values[:n] * 100, 0, color=COLORS["battery"], alpha=0.1)
    ax1.set_ylabel("SoC (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right")
    
    # Bottom: Dispatch Power
    batt_kw = df_ppo["batt_power_kw"].values[:n]
    charge = np.maximum(0, batt_kw)
    discharge = np.maximum(0, -batt_kw)
    
    ax2.bar(time, charge, width=0.2, color=COLORS["charge"], label="Charging (+)", alpha=0.8)
    ax2.bar(time, -discharge, width=0.2, color=COLORS["discharge"], label="Discharging (-)", alpha=0.8)
    
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Power (kW)")
    ax2.set_xlabel("Time (hours)")
    ax2.legend(loc="upper right")
    
    _add_annotations(ax2, df_ppo[:n], "battery")
    fig.suptitle("Battery Storage Dispatch Behavior")
    _save(fig, os.path.join(out_dir, "behavior_battery_dispatch.png"), "Battery Behavior Analysis")

# 4. Grid Dependency & Price Awareness
def plot_grid_analysis(df_ppo, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    n = min(192, len(df_ppo))
    time = _hours(np.arange(n))
    
    def _safe(col): return df_ppo[col].values[:n] if col in df_ppo.columns else np.zeros(n)
    
    grid_import = _safe("grid_import_kw")
    batt_to_load = _safe("battery_to_load")
    pv_to_load = _safe("pv_to_load")
    
    ax1.stackplot(time, pv_to_load, batt_to_load, grid_import,
                 labels=["PV Direct", "Battery Support", "Grid Import"],
                 colors=[COLORS["pv"], COLORS["battery"], COLORS["grid"]], alpha=0.7)
    
    ax1.set_ylabel("Load Supply Mix (kW)")
    ax1.set_xlabel("Time (hours)")
    
    # Pricing Overlay
    ax2 = ax1.twinx()
    if "price_vnd_kwh" in df_ppo.columns:
        price = df_ppo["price_vnd_kwh"].values[:n]
    else:
        h = (np.arange(n) % 96) / 4
        price = np.where((h >= 17) & (h <= 21), 3500, np.where((h <= 6), 1200, 2000))
        
    ax2.step(time, price, color=COLORS["price"], lw=1.5, where='post', label="Electricity Price (VND/kWh)")
    ax2.set_ylabel("Price (VND/kWh)")
    
    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    ax1.set_title("Grid Dependency & Price-Aware Dispatch Analysis")
    _save(fig, os.path.join(out_dir, "behavior_grid_dependency.png"), "Grid Dependency Analysis")

# 5. Reward & Penalty Flow
def plot_reward_flow(df_ppo, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    n = min(192, len(df_ppo))
    time = _hours(np.arange(n))
    
    def _safe(col): return df_ppo[col].values[:n] if col in df_ppo.columns else np.zeros(n)
    
    # Positive Rewards
    r_pv_to_batt = _safe("r_pv_to_battery")
    r_batt_use = _safe("r_battery_use")
    r_peak_shift = _safe("r_peak_shift")
    r_self_cons = _safe("r_self_consumption")
    r_pv_to_load = _safe("r_pv_to_load")
    
    ax1.stackplot(time, r_pv_to_load, r_pv_to_batt, r_batt_use, r_peak_shift, r_self_cons,
                 labels=["PV-to-Load", "PV-to-Batt", "Batt Support", "Peak Shift", "Self-Cons"],
                 colors=["#ff7f0e", "#bcbd22", "#17a2b8", "#2ca02c", "#2980b9"], alpha=0.8)
    ax1.plot(time, df_ppo["reward"].values[:n], color="black", lw=1.5, label="Net Reward")
    ax1.set_title("Reward Flow Analysis (Strategic Drivers)")
    ax1.set_ylabel("Reward Value")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    # Penalties (Absolute)
    p_grid = np.abs(_safe("r_grid"))
    p_com = np.abs(_safe("r_comfort"))
    p_pv_waste = np.abs(_safe("r_pv_waste"))
    p_empty_batt = np.abs(_safe("r_empty_battery"))
    
    ax2.stackplot(time, p_grid, p_com, p_pv_waste, p_empty_batt,
                 labels=["Grid Cost", "Comfort", "PV Waste", "Empty Batt"],
                 colors=["#e67e22", "#c0392b", "#7f8c8d", "#d3d3d3"], alpha=0.8)
    ax2.set_title("Penalty Flow Analysis (Strategic Inhibitors)")
    ax2.set_ylabel("Penalty Magnitude")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    _save(fig, os.path.join(out_dir, "behavior_reward_decomposition.png"), "Reward Flow Analysis")

# 6. Appliance & Battery Schedule Timeline
def plot_appliance_schedule(df_ppo, out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(192, len(df_ppo)) # Show up to 48h
    time = _hours(np.arange(n))
    
    # Extract data
    hvac = df_ppo["hvac_power_kw"].values[:n]
    batt = df_ppo["batt_power_kw"].values[:n]
    
    # 1. HVAC Cooling (Purple)
    hvac_active = hvac > 0.1
    ax.fill_between(time, 2.1, 2.9, where=hvac_active, color=COLORS["hvac"], alpha=0.8, label="HVAC Cooling")
    
    # 2. Battery Charging (Blue/Green)
    batt_charge = batt > 0.05
    ax.fill_between(time, 1.1, 1.9, where=batt_charge, color=COLORS["charge"], alpha=0.8, label="Battery Charge")
    
    # 3. Battery Discharging (Red)
    batt_discharge = batt < -0.05
    ax.fill_between(time, 0.1, 0.9, where=batt_discharge, color=COLORS["discharge"], alpha=0.8, label="Battery Discharge")
    
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Batt Discharge", "Batt Charge", "HVAC Cooling"])
    ax.set_xlabel("Time (hours)")
    ax.set_title("Appliance & Battery Schedule Timeline")
    ax.set_ylim(0, 3)
    ax.set_xlim(0, max(time) if n > 0 else 48)
    
    # Annotations
    notes = []
    if not np.any(hvac_active):
        notes.append("Warning: HVAC remained OFF")
    if not np.any(batt_discharge):
        notes.append("Warning: Battery never discharged")
        
    if notes:
        text = "\n".join(notes)
        ax.text(0.02, 0.90, text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10, color='red', fontweight='bold')
                
    _save(fig, os.path.join(out_dir, "behavior_schedule_timeline.png"), "Schedule Timeline Analysis")

# 7. Total Cost (Bar)
def plot_cost_bar(df_ppo, df_rule, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    final_ppo = df_ppo["electricity_cost"].sum()
    final_rule = df_rule["electricity_cost"].sum()
    
    bars = ax.bar(["PPO Agent", "Rule-based"], [final_ppo, final_rule], 
                  color=[COLORS["ppo"], COLORS["rule"]], alpha=0.8, width=0.6)
    
    ax.set_title("Total Cumulative Electricity Cost Comparison")
    ax.set_ylabel("Cost (VND)")
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + max(100, h*0.02), f'{h:,.0f} VND', ha='center', va='bottom', fontweight='bold')
        
    # Calculate savings
    saving_abs = final_rule - final_ppo
    saving_pct = (saving_abs / final_rule * 100) if final_rule > 0 else 0
    
    subtitle = f"Cost Reduction: {saving_pct:.1f}% ({saving_abs:,.0f} VND saved)"
    ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    # Check for comfort exploitation
    ppo_comfort_vio = (df_ppo["comfort_violation"] > 0).mean()
    rule_comfort_vio = (df_rule["comfort_violation"] > 0).mean()
    
    if ppo_comfort_vio > rule_comfort_vio + 0.05 or (df_ppo["hvac_power_kw"] < 0.1).all():
        ax.text(0.5, 0.85, "Warning: savings achieved with reduced thermal comfort", 
                transform=ax.transAxes, ha='center', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.8))
                
    # Lower ylim to make space for text
    ax.set_ylim(0, max(final_ppo, final_rule) * 1.25)
    
    _save(fig, os.path.join(out_dir, "research_cost_bar.png"), "Final Cost Comparison")

def plot_evaluation_results(df_ppo, df_rule, save_dir=None):
    """Main entry point for behavior-driven research visualization."""
    if save_dir is None: save_dir = "evaluate/"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[Visualization] Generating Behavior Analysis Suite...")
    
    plot_hvac_behavior(df_ppo, save_dir)
    plot_pv_flow(df_ppo, save_dir)
    plot_battery_behavior(df_ppo, save_dir)
    plot_grid_analysis(df_ppo, save_dir)
    plot_reward_flow(df_ppo, save_dir)
    plot_appliance_schedule(df_ppo, save_dir)
    plot_cost_bar(df_ppo, df_rule, save_dir)
    
    print("[Visualization] Done. All behavior reports saved to:", save_dir)
