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
# NEW: Reward Decomposition & Scoring (v5.0)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_step_reward_decomposition(df_ppo, out_dir: str):
    n = min(96, len(df_ppo))
    hours = _hours(np.arange(n))

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Bonuses
    r_saving = df_ppo.get("r_saving", np.zeros(n)).values[:n]
    r_comfort_bonus = df_ppo.get("r_comfort_bonus", np.zeros(n)).values[:n]
    r_soc_bonus = df_ppo.get("r_soc_bonus", np.zeros(n)).values[:n]
    
    # Penalties
    r_eco = df_ppo.get("r_eco", np.zeros(n)).values[:n]
    r_comfort = df_ppo.get("r_comfort", np.zeros(n)).values[:n]
    r_severe = df_ppo.get("r_severe", np.zeros(n)).values[:n]
    r_deg = df_ppo.get("r_deg", np.zeros(n)).values[:n]
    r_soc = df_ppo.get("r_soc", np.zeros(n)).values[:n]
    r_smooth = df_ppo.get("r_smooth", np.zeros(n)).values[:n]
    r_switch = df_ppo.get("r_switch", np.zeros(n)).values[:n]
    
    raw_reward = df_ppo.get("raw_reward", np.zeros(n)).values[:n]

    ax.plot(hours, r_saving, label="Saving Bonus", color="green", lw=1.5)
    ax.plot(hours, r_comfort_bonus, label="Comfort Bonus", color="blue", lw=1.5)
    ax.plot(hours, r_soc_bonus, label="SoC Bonus", color="cyan", lw=1.5)
    
    ax.plot(hours, r_eco, label="Eco Penalty", color="orange", lw=1.5, ls="--")
    ax.plot(hours, r_comfort, label="Comfort Penalty", color="red", lw=1.5, ls="--")
    ax.plot(hours, r_severe, label="Severe Penalty", color="darkred", lw=1.5, ls=":")
    ax.plot(hours, r_soc, label="SoC Penalty", color="purple", lw=1.5, ls="--")
    ax.plot(hours, r_deg, label="Degradation", color="gray", lw=1.5, ls="--")
    
    ax.plot(hours, raw_reward, label="Total Step Reward", color="black", lw=2.5)

    ax.set_title("Step Reward Decomposition (Bonuses vs Penalties)", **FONT_TITLE)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reward Value")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, os.path.join(out_dir, "reward_step_decomposition.png"), "Reward Step Decomposition")

def _plot_cumulative_penalty_decomposition(df_ppo, out_dir: str):
    n = min(96, len(df_ppo))
    hours = _hours(np.arange(n))

    r_eco = df_ppo.get("r_eco", np.zeros(n)).values[:n].cumsum()
    r_com = df_ppo.get("r_comfort", np.zeros(n)).values[:n].cumsum()
    r_sev = df_ppo.get("r_severe", np.zeros(n)).values[:n].cumsum()
    r_deg = df_ppo.get("r_deg", np.zeros(n)).values[:n].cumsum()
    r_soc = df_ppo.get("r_soc", np.zeros(n)).values[:n].cumsum()
    r_smooth = df_ppo.get("r_smooth", np.zeros(n)).values[:n].cumsum()
    r_switch = df_ppo.get("r_switch", np.zeros(n)).values[:n].cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, r_eco, color="orange", lw=LW_SEC, label="Grid Cost (r_eco)")
    ax.plot(hours, r_com, color="red", lw=LW_SEC, label="Comfort (r_comfort)")
    ax.plot(hours, r_sev, color="darkred", lw=LW_SEC, label="Severe Overheat (r_severe)")
    ax.plot(hours, r_deg, color="gray", lw=LW_SEC, label="Battery Degradation (r_deg)")
    ax.plot(hours, r_soc, color="purple", lw=LW_SEC, label="SoC Health (r_soc)")
    ax.plot(hours, r_switch, color="brown", lw=LW_SEC, label="Battery Switch (r_switch)")

    ax.set_title("Cumulative Penalty Decomposition", **FONT_TITLE)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Penalty")
    ax.legend(loc="lower left", ncol=2)
    _save(fig, os.path.join(out_dir, "reward_cumulative_penalty.png"), "Cumulative Penalty Decomposition")

def _plot_demo_score_breakdown(df_ppo, out_dir: str):
    n = len(df_ppo)
    
    # 1. Comfort Score
    vio_rate = np.mean(df_ppo.get("comfort_violation", np.zeros(n)).values > 0)
    comfort_score = 100 * (1 - vio_rate)
    
    # 2. Cost Score (Saving Ratio)
    total_cost = np.sum(df_ppo.get("electricity_cost", np.zeros(n)).values)
    total_baseline = np.sum(df_ppo.get("baseline_cost", np.full(n, total_cost + 1e-6)).values)
    saving_ratio = (total_baseline - total_cost) / (total_baseline + 1e-6)
    cost_score = np.clip(100 * saving_ratio, 0, 100) if saving_ratio > 0 else 0
    
    # 3. Battery Score
    avg_soc = np.mean(df_ppo.get("soc", np.full(n, 0.5)).values)
    is_healthy = 0.3 <= avg_soc <= 0.8
    battery_score = 100 if is_healthy else 0
    
    # 4. Smooth Score
    switch_count = np.sum(df_ppo.get("battery_switch", np.zeros(n)).values)
    norm_switch = np.clip(switch_count / max(1, n * 0.1), 0, 1) # max acceptable 10% switches
    smooth_score = 100 * (1 - norm_switch)
    
    # Final Score
    final_score = 0.4 * comfort_score + 0.3 * cost_score + 0.2 * battery_score + 0.1 * smooth_score
    
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Comfort Score', 'Cost Score', 'Battery Score', 'Smooth Score']
    scores = [comfort_score, cost_score, battery_score, smooth_score]
    
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, scores, align='center', color=['blue', 'green', 'purple', 'orange'])
    ax.set_yticks(y_pos, labels=categories)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Score (0-100)')
    ax.set_title(f"Evaluation Score: {final_score:.1f}/100", **FONT_TITLE)
    
    for bar in bars:
        width = bar.get_width()
        label_y = bar.get_y() + bar.get_height() / 2
        ax.text(width + 2, label_y, s=f'{width:.1f}', va='center', fontweight='bold')
        
    _save(fig, os.path.join(out_dir, "demo_score_breakdown.png"), "Demo Score Breakdown")


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

def plot_thermal_comfort_with_weather(df, save_path):
    time = df["time_hours"].values
    ppo_indoor = df["ppo_indoor_temp"].values
    rule_indoor = df["rule_indoor_temp"].values
    outdoor_temp = df["outdoor_temp"].values

    vio_hot = np.sum(ppo_indoor > 26)
    vio_cold = np.sum(ppo_indoor < 22)
    pct_vio = (vio_hot + vio_cold) / len(ppo_indoor) * 100
    
    dt = time[1] - time[0] if len(time) > 1 else 0.25
    sev_hot_hours = np.sum(ppo_indoor > 27) * dt
    max_ppo = np.max(ppo_indoor)
    avg_out = np.mean(outdoor_temp)
    max_out = np.max(outdoor_temp)
    
    source_val = df["weather_source"].iloc[0] if "weather_source" in df.columns else "Demo profile"

    title = "Thermal Comfort Tracking with Outdoor Temperature — Demo Evaluation"
    if pct_vio > 50:
        title = "Thermal Comfort Failure with Outdoor Temperature — PPO Overheats"

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title(title, **FONT_TITLE)
    ax1.set_xlabel("Time (hours)", **FONT_LABEL)
    ax1.set_ylabel("Indoor Temperature (°C)", **FONT_LABEL)

    l1 = ax1.plot(time, ppo_indoor, color="#e74c3c", linewidth=2.8, label="PPO Indoor Temp")[0]
    l2 = ax1.plot(time, rule_indoor, color="#2ecc71", linestyle="--", linewidth=2.3, label="Rule-based Temp")[0]

    ax2 = ax1.twinx()
    l3 = ax2.plot(time, outdoor_temp, color="orange", linestyle="-.", linewidth=1.8, alpha=0.8, label="Outdoor Temp")[0]
    ax2.set_ylabel("Outdoor Temperature (°C)", **FONT_LABEL)

    # Comfort Zone
    ax1.axhspan(22, 26, color="lightblue", alpha=0.3, label="Comfort Zone 22–26°C")
    ax1.axhline(22, color="blue", linestyle="--", alpha=0.5)
    ax1.axhline(26, color="darkblue", linestyle="--", alpha=0.8)
    l4 = ax1.axhline(27, color="red", linestyle="--", alpha=0.7, label="Severe Overheating >27°C")

    # Highlight vi phạm
    ax1.fill_between(time, 26, ppo_indoor, where=(ppo_indoor > 26), color="salmon", alpha=0.3, label="PPO Comfort Violation")
    
    if np.any(ppo_indoor > 27):
        ax1.fill_between(time, 27, ppo_indoor, where=(ppo_indoor > 27), color="darkred", alpha=0.4, hatch="///")

    y_min = min(21, np.min(rule_indoor), np.min(ppo_indoor)) - 0.5
    y_max = max(30, np.max(ppo_indoor), 27) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max)+1, 1))

    # Legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    comfort_patch = mpatches.Patch(color="lightblue", alpha=0.3, label="Comfort Zone 22–26°C")
    vio_patch = mpatches.Patch(color="salmon", alpha=0.3, label="PPO Comfort Violation")
    
    ax1.legend(lines + [comfort_patch, vio_patch], labels + [comfort_patch.get_label(), vio_patch.get_label()], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Annotation
    anno_text = (f"Violation: {pct_vio:.1f}%\n"
                 f"Severe Overheat: {sev_hot_hours:.1f}h\n"
                 f"Max PPO Temp: {max_ppo:.1f}°C\n"
                 f"Avg Outdoor: {avg_out:.1f}°C\n"
                 f"Max Outdoor: {max_out:.1f}°C\n"
                 f"Weather: {source_val}")
    ax1.text(1.02, 0.5, anno_text, transform=ax1.transAxes, verticalalignment='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Chart 3 (v2) → {save_path}")

def plot_evaluation_results(df_ppo, df_rule, save_dir=None):
    if save_dir is None: save_dir = "evaluate/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Core Plots
    _plot_soc(df_ppo, df_rule, save_dir)
    
    import pandas as pd
    n = min(192, len(df_ppo), len(df_rule))
    df_combined = pd.DataFrame({
        "time_hours": _hours(df_ppo["step"].values[:n]),
        "ppo_indoor_temp": df_ppo["indoor_temp"].values[:n],
        "rule_indoor_temp": df_rule["indoor_temp"].values[:n],
        "outdoor_temp": df_ppo["outdoor_temp"].values[:n],
        "weather_source": ["Evaluation Log"] * n
    })
    v2_path = os.path.join(save_dir, "chart3_temperature_tracking_v2.png")
    plot_thermal_comfort_with_weather(df_combined, v2_path)
    
    _plot_power_balance(df_ppo, save_dir)
    
    # Reward & Scores Plots (v5.0)
    _plot_step_reward_decomposition(df_ppo, save_dir)
    _plot_cumulative_penalty_decomposition(df_ppo, save_dir)
    _plot_demo_score_breakdown(df_ppo, save_dir)
    
    # Diagnostic Plots (v4.0)
    _plot_action_distribution(df_ppo, save_dir)
    _plot_health_stats(df_ppo, save_dir)
