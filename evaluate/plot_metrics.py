"""
evaluate/plot_metrics.py
========================
v4.0: Research-Grade Diagnostics & Realism Verification
-------------------------------------------------------
Bổ sung các biểu đồ kiểm soát "bệnh lý" RL:
1. Action Distribution (Histogram)
2. Comfort Violation Stats
3. Battery Cycle Analysis
4. Detailed Reward decomposition
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style & Colors ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "ppo":      "#2ecc71",
    "rule":     "#e74c3c",
    "pv":       "#f1c40f",
    "batt":     "#27ae60",
    "grid":     "#c0392b",
    "charge":   "#3498db",
    "hvac":     "#8e44ad",
    "price":    "#e67e22",
    "total":    "#ecf0f1",
}
LW_MAIN = 3.0
LW_SEC  = 1.8
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
# NEW: Action Distribution (Histogram) — Kiểm tra bão hòa hành động
# ─────────────────────────────────────────────────────────────────────────────
def _plot_action_distribution(df_ppo, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Battery Action Histogram
    ax1.hist(df_ppo["action_batt"], bins=30, color=COLORS["batt"], alpha=0.7, edgecolor="white")
    ax1.set_title("Battery Action Distribution (kW)", **FONT_TITLE)
    ax1.set_xlabel("Charge (+) / Discharge (-) [kW]")
    ax1.axvline(0, color="black", lw=1, ls="--")

    # HVAC Action Histogram
    ax2.hist(df_ppo["action_hvac"], bins=30, color=COLORS["hvac"], alpha=0.7, edgecolor="white")
    ax2.set_title("HVAC Input Distribution (kW)", **FONT_TITLE)
    ax2.set_xlabel("Electrical Input [kW]")

    _save(fig, os.path.join(out_dir, "diag_action_distribution.png"), "Diag — Action Distribution")

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Comfort & Battery Health Stats (Table/Text)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_health_stats(df_ppo, out_dir: str):
    # 1. Comfort Violations
    temp = df_ppo["indoor_temp"].values
    vio_hot  = np.sum(temp > 26.0) * 0.25  # hours
    vio_cold = np.sum(temp < 22.0) * 0.25
    total_hours = len(temp) * 0.25
    pct_vio = (vio_hot + vio_cold) / total_hours * 100

    # 2. Battery Stress (Equivalent Cycles)
    # Total energy throughput / (2 * capacity)
    throughput = np.sum(np.abs(df_ppo["batt_power_kw"].values)) * 0.25
    capacity = 10.0 # kWh
    eq_cycles = throughput / (2 * capacity)

    # 3. Control Smoothness (Mean Delta Action)
    smoothness = np.mean(np.abs(np.diff(df_ppo["action_batt"].values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    stats_text = (
        f"🔎 ENVIRONMENT REALISM & RELIABILITY REPORT\n"
        f"-------------------------------------------\n\n"
        f"🔴 THERMAL COMFORT:\n"
        f"   - Total Violation Time: {vio_hot+vio_cold:.1f} hours\n"
        f"   - Violation Percentage: {pct_vio:.1f}%\n"
        f"   - Severe Overheating (>27°C): {np.sum(temp > 27.0)*0.25:.1f}h\n\n"
        f"🔋 BATTERY HEALTH:\n"
        f"   - 7-Day Energy Throughput: {throughput:.1f} kWh\n"
        f"   - Equivalent Full Cycles: {eq_cycles:.2f} cycles/week\n"
        f"   - Estimated Annual Stress: {eq_cycles * 52:.0f} cycles/year\n\n"
        f"📉 CONTROL STABILITY:\n"
        f"   - Battery Jitter (Mean ΔAction): {smoothness:.3f} kW/step\n"
    )
    ax.text(0.1, 0.5, stats_text, family="monospace", fontsize=11, va="center", bbox=dict(facecolor='white', alpha=0.5))
    _save(fig, os.path.join(out_dir, "diag_reliability_report.png"), "Diag — Reliability Report")

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 4 — Power Balance (Split)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_power_balance(df_ppo, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    n = min(192, len(df_ppo))
    hours = _hours(df_ppo["step"].values[:n])

    pv    = df_ppo["pv_kw"].values[:n]
    grid  = df_ppo["grid_import_kw"].values[:n]
    batt_dis = np.clip(df_ppo["batt_power_kw"].values[:n], 0, None)
    
    ax1.stackplot(hours, pv, batt_dis, grid, labels=["PV", "Batt Discharge", "Grid"], colors=[COLORS["pv"], COLORS["batt"], COLORS["grid"]], alpha=0.8)
    ax1.set_title("System Energy Mix (Supplies)", **FONT_TITLE)
    ax1.legend(loc="upper right")

    batt_p = df_ppo["batt_power_kw"].values[:n]
    soc = df_ppo["soc"].values[:n]
    ax2.step(hours, batt_p, where="post", color=COLORS["charge"], lw=LW_SEC, label="Battery Power (kW)")
    ax2.fill_between(hours, batt_p, step="post", alpha=0.2, color=COLORS["charge"])
    
    ax2_soc = ax2.twinx()
    ax2_soc.plot(hours, soc, color=COLORS["ppo"], lw=LW_MAIN, label="SoC")
    ax2_soc.set_ylim(0, 1)
    ax2.set_title("Battery Power & SoC (Control Stability)", **FONT_TITLE)
    _save(fig, os.path.join(out_dir, "chart4_power_balance.png"), "Chart 4 — Power Balance")

# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 7 — Cumulative Reward (Decomposed)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_cumulative_reward(df_ppo, out_dir: str):
    n = min(96, len(df_ppo))
    hours = _hours(np.arange(n))

    r_eco   = df_ppo["r_eco"].values[:n].cumsum()
    r_com   = df_ppo["r_comfort"].values[:n].cumsum()
    r_deg   = df_ppo["r_deg"].values[:n].cumsum()
    r_arb   = df_ppo["r_arb"].values[:n].cumsum()
    r_total = (df_ppo["r_eco"].values[:n] + df_ppo["r_comfort"].values[:n] + 
               df_ppo["r_deg"].values[:n] + df_ppo["r_arb"].values[:n]).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, r_eco, color=COLORS["grid"], lw=LW_SEC, label="Grid Cost (r_eco)")
    ax.plot(hours, r_arb, color=COLORS["pv"],   lw=LW_SEC, label="Arbitrage (r_arb)")
    ax.plot(hours, r_com, color=COLORS["hvac"], lw=LW_SEC, label="Comfort (r_comfort)")
    ax.plot(hours, r_deg, color="#95a5a6",      lw=LW_SEC, label="Battery Health (r_deg)")
    ax.plot(hours, r_total, color=COLORS["ppo"], lw=LW_MAIN, label="TOTAL REWARD")

    ax.set_title("Cumulative Reward Decomposition (Loophole Check)", **FONT_TITLE)
    ax.legend(loc="lower left", ncol=2)
    _save(fig, os.path.join(out_dir, "chart7_cumulative_reward.png"), "Chart 7 — Cumulative Reward")

# ─────────────────────────────────────────────────────────────────────────────
# Standard Plots
# ─────────────────────────────────────────────────────────────────────────────
def _plot_soc(df_ppo, df_rule, out_dir: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    hours = _hours(df_ppo["step"].values)
    ax.plot(hours, df_ppo["soc"], label="PPO Agent", color=COLORS["ppo"], lw=LW_MAIN)
    ax.plot(hours, df_rule["soc"], label="Rule-based", color=COLORS["rule"], lw=LW_SEC, ls="--")
    ax.set_title("SoC Profile (7 Days)", **FONT_TITLE)
    ax.legend()
    _save(fig, os.path.join(out_dir, "chart1_soc_profile.png"), "Chart 1 — SoC Profile")

def _plot_thermal(df_ppo, df_rule, out_dir: str):
    n = min(192, len(df_ppo))
    hours = _hours(df_ppo["step"].values[:n])
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(hours, df_ppo["indoor_temp"].values[:n], label="PPO Indoor", color=COLORS["ppo"], lw=LW_MAIN)
    ax.plot(hours, df_rule["indoor_temp"].values[:n], label="Rule Indoor", color=COLORS["rule"], lw=LW_SEC, ls=":")
    ax.axhspan(22, 26, color=COLORS["ppo"], alpha=0.1, label="Comfort Zone")
    ax.set_title("Thermal Comfort Tracking (48h)", **FONT_TITLE)
    ax.legend()
    _save(fig, os.path.join(out_dir, "chart3_temperature_tracking.png"), "Chart 3 — Temperature")

def plot_evaluation_results(df_ppo, df_rule, save_dir=None):
    if save_dir is None: save_dir = "evaluate/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Core Plots
    _plot_soc(df_ppo, df_rule, save_dir)
    _plot_thermal(df_ppo, df_rule, save_dir)
    _plot_power_balance(df_ppo, save_dir)
    _plot_cumulative_reward(df_ppo, save_dir)
    
    # Diagnostic Plots (v4.0)
    _plot_action_distribution(df_ppo, save_dir)
    _plot_health_stats(df_ppo, save_dir)
