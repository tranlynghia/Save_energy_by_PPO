"""
evaluate/plot_metrics.py
========================
v3.0: Clean Visuals & Research-Grade Aesthetics
-----------------------------------------------
Các quy tắc thiết kế áp dụng:
1. Phân cấp Line Width (3.0 PPO, 2.5 Baseline, 1.8 Secondary).
2. Tách Power Balance thành 2 subplots để tránh rối.
3. Chuyển Action vs Price sang Step-plot để nhìn rõ tương quan.
4. Tối giản Legend cho biểu đồ Reward.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style & Colors ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "ppo":      "#2ecc71",    # xanh lá
    "rule":     "#e74c3c",    # đỏ
    "pv":       "#f1c40f",    # vàng
    "batt":     "#27ae60",    # xanh đậm (discharge)
    "grid":     "#c0392b",    # đỏ đậm (import)
    "charge":   "#3498db",    # xanh dương (charge)
    "hvac":     "#8e44ad",    # tím
    "price":    "#e67e22",    # cam
    "load":     "#7f8c8d",    # xám
    "total":    "#ecf0f1",    # trắng đục
}

LW_MAIN   = 3.0
LW_BASE   = 2.5
LW_SEC    = 1.8
LW_REF    = 1.2

FONT_TITLE = dict(fontsize=13, fontweight="bold")
FONT_LABEL = dict(fontsize=10)

def _save(fig, path: str, title_for_log: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {title_for_log:45s} → {os.path.basename(path)}")

def _hours(steps: np.ndarray) -> np.ndarray:
    return steps * 0.25

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 1 — SoC Profile (Cleaned)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_soc(df_ppo, df_rule, out_dir: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    hours = _hours(df_ppo["step"].values)

    ax.plot(hours, df_ppo["soc"],  label="PPO Agent", color=COLORS["ppo"], lw=LW_MAIN)
    ax.plot(hours, df_rule["soc"], label="Rule-based", color=COLORS["rule"], lw=LW_BASE, ls="--")
    
    ax.axhline(0.9, color="gray", lw=LW_REF, ls=":", alpha=0.5)
    ax.axhline(0.1, color="gray", lw=LW_REF, ls=":", alpha=0.5)

    ax.set_title("Battery SoC Profile (7-Day Simulation)", **FONT_TITLE)
    ax.set_ylabel("State of Charge", **FONT_LABEL)
    ax.legend(loc="upper right", frameon=True)
    _save(fig, os.path.join(out_dir, "chart1_soc_profile.png"), "Chart 1 — SoC Profile")

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 3 — Temperature (Cleaned)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_thermal(df_ppo, df_rule, out_dir: str):
    n = min(192, len(df_ppo)) # 48h
    hours = _hours(df_ppo["step"].values[:n])

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(hours, df_ppo["outdoor_temp"].values[:n], label="Outdoor", color=COLORS["price"], lw=LW_SEC, alpha=0.5)
    ax.plot(hours, df_ppo["indoor_temp"].values[:n],  label="PPO Indoor", color=COLORS["ppo"], lw=LW_MAIN)
    ax.plot(hours, df_rule["indoor_temp"].values[:n], label="Rule Indoor", color=COLORS["rule"], lw=LW_BASE, ls=":")
    
    ax.axhspan(22, 26, color=COLORS["ppo"], alpha=0.1, label="Comfort Zone")
    ax.set_title("Thermal Comfort Tracking (First 48 Hours)", **FONT_TITLE)
    ax.set_ylabel("Temp (°C)", **FONT_LABEL)
    ax.legend(loc="upper right", ncol=2)
    _save(fig, os.path.join(out_dir, "chart3_temperature_tracking.png"), "Chart 3 — Temperature")

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 4 — Power Balance (REDESIGN: Split Subplots)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_power_balance(df_ppo, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    n = min(192, len(df_ppo))
    hours = _hours(df_ppo["step"].values[:n])

    # Top: Energy Supplies
    pv    = df_ppo["pv_kw"].values[:n]
    grid  = df_ppo["grid_import_kw"].values[:n]
    batt_dis = np.clip(df_ppo["batt_power_kw"].values[:n], 0, None)
    
    ax1.stackplot(hours, pv, batt_dis, grid, 
                 labels=["PV Supply", "Battery Discharge", "Grid Import"],
                 colors=[COLORS["pv"], COLORS["batt"], COLORS["grid"]], alpha=0.8)
    ax1.set_title("System Energy Supply Mix (48h)", **FONT_TITLE)
    ax1.set_ylabel("Power (kW)", **FONT_LABEL)
    ax1.legend(loc="upper right")

    # Bottom: Battery Dynamics
    batt_p = df_ppo["batt_power_kw"].values[:n]
    soc = df_ppo["soc"].values[:n]
    
    ax2.step(hours, batt_p, where="post", color=COLORS["charge"], lw=LW_SEC, label="Battery Power (kW)")
    ax2.fill_between(hours, batt_p, step="post", alpha=0.2, color=COLORS["charge"])
    ax2.axhline(0, color="white", lw=LW_REF, alpha=0.3)
    
    ax2_soc = ax2.twinx()
    ax2_soc.plot(hours, soc, color=COLORS["ppo"], lw=LW_MAIN, label="SoC")
    ax2_soc.set_ylabel("SoC", color=COLORS["ppo"])
    ax2_soc.set_ylim(0, 1)

    ax2.set_title("Battery Power & State of Charge", **FONT_TITLE)
    ax2.set_ylabel("Power (kW)", **FONT_LABEL)
    ax2.set_xlabel("Time (hours)", **FONT_LABEL)
    _save(fig, os.path.join(out_dir, "chart4_power_balance.png"), "Chart 4 — Power Balance")

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 5 — Action vs Price (REDESIGN: Step Lines)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_action_vs_price(df_ppo, out_dir: str):
    n = min(192, len(df_ppo))
    hours = _hours(df_ppo["step"].values[:n])
    
    # Battery action line
    batt = df_ppo["action_batt"].values[:n]
    pv   = df_ppo["pv_kw"].values[:n]
    
    # Reconstruct Price
    grid_kw = np.clip(df_ppo["grid_import_kw"].values[:n], 0.01, None)
    price   = np.clip(df_ppo["electricity_cost"].values[:n] / (grid_kw * 0.25 + 1e-6), 0, 4000)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # PV Background
    ax1.fill_between(hours, 0, pv, alpha=0.25, color=COLORS["pv"], label="PV Power")

    # Battery Action (Line instead of Bar)
    ax1.plot(hours, batt, color=COLORS["batt"], lw=LW_MAIN, label="Battery Action (kW)")
    ax1.axhline(0, color="white", lw=LW_REF, alpha=0.3)
    ax1.set_ylabel("Power (kW)", **FONT_LABEL)

    # Price (Step Line)
    ax2 = ax1.twinx()
    ax2.step(hours, price, where="post", color=COLORS["price"], lw=LW_SEC, ls="--", label="TOU Price")
    ax2.set_ylabel("Price (VNĐ/kWh)", color=COLORS["price"])

    ax1.set_title("Battery Action vs Electricity Price (Step-Analysis)", **FONT_TITLE)
    ax1.set_xlabel("Time (hours)", **FONT_LABEL)
    
    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=3)
    
    _save(fig, os.path.join(out_dir, "chart5_action_vs_price.png"), "Chart 5 — Action vs Price")

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 7 — Cumulative Reward (REDESIGN: No Overload)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_cumulative_reward(df_ppo, out_dir: str):
    n = min(96, len(df_ppo)) # 24h
    hours = _hours(np.arange(n))

    r_eco    = df_ppo["r_eco"].values[:n].cumsum()
    r_com    = df_ppo["r_comfort"].values[:n].cumsum()
    r_deg    = df_ppo["r_deg"].values[:n].cumsum()
    r_arb    = df_ppo["r_arb"].values[:n].cumsum()
    r_total  = (df_ppo["r_eco"].values[:n] + df_ppo["r_comfort"].values[:n] + 
                df_ppo["r_deg"].values[:n] + df_ppo["r_arb"].values[:n]).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Chỉ giữ 4 đường chính (Economic giờ là r_eco + r_arb)
    ax.plot(hours, r_eco,   color=COLORS["grid"],  lw=LW_SEC, label="Grid Cost (r_eco)")
    ax.plot(hours, r_arb,   color=COLORS["pv"],    lw=LW_SEC, label="Arbitrage (r_arb)")
    ax.plot(hours, r_com,   color=COLORS["hvac"],  lw=LW_SEC, label="Comfort (r_comfort)")
    ax.plot(hours, r_total, color=COLORS["ppo"],   lw=LW_MAIN, label="TOTAL REWARD")

    ax.set_title("Cumulative Reward Decomposition (24h Window)", **FONT_TITLE)
    ax.set_ylabel("Cumulative Sum", **FONT_LABEL)
    ax.set_xlabel("Time (hours)", **FONT_LABEL)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    
    _save(fig, os.path.join(out_dir, "chart7_cumulative_reward.png"), "Chart 7 — Cumulative Reward")

# ─────────────────────────────────────────────────────────────────────────────
# Các biểu đồ còn lại (giữ nguyên logic nhưng áp dụng LW mới)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_cost_bar(df_ppo, df_rule, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ppo_c  = df_ppo["total_cost"].iloc[-1]
    rule_c = df_rule["total_cost"].iloc[-1]
    ax.bar(["PPO", "Rule"], [ppo_c, rule_c], color=[COLORS["ppo"], COLORS["rule"]], width=0.5)
    ax.set_title(f"Total Cost Comparison (PPO saves {((rule_c-ppo_c)/rule_c)*100:.1f}%)", **FONT_TITLE)
    _save(fig, os.path.join(out_dir, "chart2_cost_comparison.png"), "Chart 2 — Cost Comparison")

def _plot_appliance_schedule(df_ppo, out_dir: str):
    n = min(96, len(df_ppo)) # 24h
    hours = _hours(df_ppo["step"].values[:n])
    hvac  = df_ppo["hvac_power_kw"].values[:n]
    batt  = df_ppo["batt_power_kw"].values[:n]

    hvac_on  = hvac > 0.1
    batt_chg = batt > 0.1
    batt_dis = batt < -0.1

    fig, ax = plt.subplots(figsize=(14, 4))
    
    DEVICES = [
        ("HVAC Cooling", hvac_on,  COLORS["hvac"],   2.5),
        ("Batt Charge",  batt_chg, COLORS["charge"], 1.5),
        ("Batt Discharge", batt_dis, COLORS["batt"],  0.5),
    ]

    yticks, ylabels = [], []
    for label, mask, color, y in DEVICES:
        on_start = None
        for i, (h, m) in enumerate(zip(hours, mask)):
            if m and on_start is None: on_start = h
            elif not m and on_start is not None:
                ax.barh(y, h - on_start, left=on_start, height=0.6, color=color, alpha=0.8)
                on_start = None
        if on_start is not None:
            ax.barh(y, hours[-1] - on_start, left=on_start, height=0.6, color=color, alpha=0.8)
        yticks.append(y)
        ylabels.append(label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title("Appliance & Battery Schedule (24h Window)", **FONT_TITLE)
    ax.set_xlabel("Time (hours)", **FONT_LABEL)
    ax.set_ylim(0, 3.2)
    _save(fig, os.path.join(out_dir, "chart6_appliance_schedule.png"), "Chart 6 — Schedule")

def plot_evaluation_results(df_ppo, df_rule, save_dir=None):
    if save_dir is None: save_dir = "evaluate/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure columns
    for df in [df_ppo, df_rule]:
        if "total_cost" not in df.columns: df["total_cost"] = df["electricity_cost"].cumsum()

    _plot_soc(df_ppo, df_rule, save_dir)
    _plot_cost_bar(df_ppo, df_rule, save_dir)
    _plot_thermal(df_ppo, df_rule, save_dir)
    _plot_power_balance(df_ppo, save_dir)
    _plot_action_vs_price(df_ppo, save_dir)
    _plot_appliance_schedule(df_ppo, save_dir)
    _plot_cumulative_reward(df_ppo, save_dir)
