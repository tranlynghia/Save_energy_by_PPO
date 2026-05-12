"""
run_evaluation.py — Evaluation & Comparison Script (v2.0)
=========================================================
So sánh hiệu năng: PPO Agent vs Rule-based Baseline Agent
trên cùng một môi trường Smart Home HEMS đã được chuẩn hóa.
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# --- sys.path setup ---
cwd = os.getcwd()
if os.path.basename(cwd) == "project":
    project_dir = cwd
elif os.path.basename(cwd) == "evaluate":
    project_dir = os.path.abspath(os.path.join(cwd, ".."))
elif os.path.exists(os.path.join(cwd, "project")):
    project_dir = os.path.join(cwd, "project")
else:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from env.wrappers import make_stable_env
from agent.rule_based_agent import RuleBasedAgent
try:
    from evaluate.plot_metrics import plot_evaluation_results
except ImportError:
    plot_evaluation_results = None


# ============================================================
# RULE-BASED BASELINE (updated for new obs format)
# ============================================================
class HEMSRuleBasedAgent:
    """
    Rule-based heuristic for Smart Home HEMS.
    Compatible with new normalized observation format.

    Obs indices (new format):
        [0] indoor_temp [-1,1] → [15,40]°C
        [1] outdoor_temp
        [2] soc [-1,1] → [0,1]
        [3] base_load [-1,1] → [0,5]kW
        [4] pv_kw [-1,1] → [0,5]kW
        [5] price [-1,1] → [1000,3500] VND

    Action format (physical units via wrapper):
        [0] battery_power ∈ [-5, 5] kW
        [1] hvac_kw ∈ {0, 1, 2.5, 4} kW
    """

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        soc       = float(obs[2]) * 0.5 + 0.5        # [0, 1]
        pv        = float(obs[4]) * 2.5 + 2.5        # [0, 5] kW
        price_n   = float(obs[5])                    # normalized price
        indoor_t  = float(obs[0]) * 12.5 + 27.5     # [15, 40] °C

        # Battery strategy: charge when cheap+solar, discharge when expensive
        if price_n < -0.3 and pv > 1.5 and soc < 0.85:
            batt_p = 3.0   # Charge from solar when price is low
        elif price_n > 0.5 and soc > 0.3:
            batt_p = -3.0  # Discharge to avoid peak price
        else:
            batt_p = 0.0

        # HVAC strategy: cool only if too hot
        if indoor_t > 26.5:
            hvac_raw = 0.6   # → BOOST mode
        elif indoor_t > 25.5:
            hvac_raw = 0.2   # → NORMAL mode
        elif indoor_t > 24.5:
            hvac_raw = -0.2  # → ECO mode
        else:
            hvac_raw = -1.0  # → OFF

        # Return in wrapper-input format ([-1,1] for both actions)
        # The wrapper will scale these to physical units
        batt_norm = float(np.clip(batt_p / 5.0, -1.0, 1.0))
        action = np.array([batt_norm, hvac_raw], dtype=np.float32)
        return action, None


def run_episode(env, agent, num_steps: int = 672, seed: int = 42) -> pd.DataFrame:
    """
    Run one evaluation episode and collect metrics.

    Returns DataFrame with per-step data.
    """
    obs, _ = env.reset(seed=seed)
    records = []

    for step in range(num_steps):
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        grid_import_kw = info.get("grid_import_kw", 0.0)
        grid_export_kw = info.get("grid_export_kw", 0.0)
        net_load       = info.get("net_load_kw", 0.0)
        base_load      = info.get("base_load_kw", 0.0)
        
        outdoor_temp = info.get("outdoor_temp")
        if outdoor_temp is None:
            try:
                data_step = env.unwrapped._data_step - 1
                outdoor_temp = env.unwrapped.data_manager.get_step_data(data_step)["outdoor_temp"]
            except Exception:
                outdoor_temp = 25.0
                
        records.append({
            "step":             step,
            "reward":           reward,
            "indoor_temp":      info.get("indoor_temp", float("nan")),
            "outdoor_temp":     outdoor_temp,
            "soc":              info.get("soc", float("nan")),
            "grid_import_kw":   grid_import_kw,
            "grid_export_kw":   grid_export_kw,
            "pv_kw":            info.get("pv_kw", 0.0),
            "pv_to_load":       info.get("pv_to_load", 0.0),
            "pv_to_battery":    info.get("pv_to_battery", 0.0),
            "battery_to_load":  info.get("battery_to_load", 0.0),
            "pv_export":        info.get("grid_export_kw", 0.0),
            "hvac_power_kw":    info.get("hvac_input_kw", 0.0),
            "batt_power_kw":    info.get("batt_power_kw", 0.0),
            "base_load_kw":     base_load,
            "net_load_kw":      net_load,
            "electricity_cost": info.get("electricity_cost", 0.0),
            "baseline_cost":    info.get("baseline_cost", 0.0),
            "r_grid":           info.get("r_grid", 0.0),
            "r_saving":         info.get("r_saving", 0.0),
            "r_pv_use":         info.get("r_pv_use", 0.0),
            "r_pv_to_battery":  info.get("r_pv_to_battery", 0.0),
            "r_battery_use":    info.get("r_battery_use", 0.0),
            "r_pv_export":      info.get("r_pv_export", 0.0),
            "r_comfort":        info.get("r_comfort", 0.0),
            "r_comfort_bonus":  info.get("r_comfort_bonus", 0.0),
            "r_severe_hot":     info.get("r_severe_hot", 0.0),
            "r_severe_cold":    info.get("r_severe_cold", 0.0),
            "r_peak":           info.get("r_peak", 0.0),
            "battery_switch":   info.get("battery_switch", 0.0),
            "comfort_violation":info.get("comfort_violation", 0.0),
            "ghi_w_m2":         info.get("ghi_w_m2", 0.0),
            "weather_source":   info.get("weather_source", "unknown"),
        })

        obs = next_obs
        if terminated or truncated:
            break

    df = pd.DataFrame(records)
    df["total_cost"] = df["electricity_cost"].cumsum()
    return df

def print_research_summary(df_ppo, df_rule):
    """Prints a research-grade comparison table."""
    def compute_metrics(df):
        n_days = (len(df) * 0.25) / 24.0
        grid_import_kwh_day = (df["grid_import_kw"].sum() * 0.25) / n_days
        comfort_vio_rate = (df["comfort_violation"] > 0).mean() * 100
        avg_temp = df["indoor_temp"].mean()
        batt_throughput = (df["batt_power_kw"].abs().sum() * 0.25) / n_days
        
        pv_sum = df["pv_kw"].sum()
        export_sum = df["grid_export_kw"].sum()
        pv_self_ratio = 100 * (1 - export_sum / (pv_sum + 1e-6))
        
        total_cost = df["electricity_cost"].sum()
        
        return {
            "Total Cost": f"{total_cost:,.0f} VND",
            "Grid Import": f"{grid_import_kwh_day:.2f} kWh/day",
            "Comfort Vio": f"{comfort_vio_rate:.1f} %",
            "Avg Temp": f"{avg_temp:.2f} °C",
            "Batt Throughput": f"{batt_throughput:.2f} kWh/day",
            "PV Self-Cons": f"{pv_self_ratio:.1f} %"
        }

    ppo_m = compute_metrics(df_ppo)
    rule_m = compute_metrics(df_rule)

    print("\n" + "="*70)
    print(f"{'METRIC':<30} | {'PPO AGENT':<15} | {'RULE-BASED':<15}")
    print("-" * 70)
    for k in ppo_m.keys():
        print(f"{k:<30} | {ppo_m[k]:<15} | {rule_m[k]:<15}")
    print("="*70 + "\n")

def main():
    print("=" * 60)
    print("   HEMS PPO RESEARCH EVALUATION — PERFORMANCE BENCHMARK")
    print("=" * 60)

    TEST_STEPS = 672  # 7 days
    TEST_SEED  = 42

    env_ppo  = make_stable_env(max_episode_steps=TEST_STEPS, random_start=False, seed=TEST_SEED)
    env_rule = make_stable_env(max_episode_steps=TEST_STEPS, random_start=False, seed=TEST_SEED)

    model_path = os.path.join(project_dir, "models", "ppo_hems_final.zip")
    if not os.path.exists(model_path):
        # Check alternative
        alt = os.path.join(project_dir, "models", "ppo_smart_home_final.zip")
        if os.path.exists(alt): model_path = alt
        else:
            print("[Error] Model not found. Run training first.")
            return

    print(f"[*] Loading PPO Model: {os.path.basename(model_path)}")
    ppo_agent = PPO.load(model_path, env=env_ppo)
    rule_agent = HEMSRuleBasedAgent()

    print(f"[*] Running Evaluation ({TEST_STEPS} steps)...")
    df_ppo = run_episode(env_ppo, ppo_agent, num_steps=TEST_STEPS, seed=TEST_SEED)
    df_rule = run_episode(env_rule, rule_agent, num_steps=TEST_STEPS, seed=TEST_SEED)

    # Summary table
    print_research_summary(df_ppo, df_rule)

    # Save and Plot
    out_dir = os.path.join(project_dir, "evaluate")
    os.makedirs(out_dir, exist_ok=True)
    df_ppo.to_csv(os.path.join(out_dir, "eval_ppo.csv"), index=False)
    df_rule.to_csv(os.path.join(out_dir, "eval_rule.csv"), index=False)

    if plot_evaluation_results:
        plot_evaluation_results(df_ppo, df_rule, save_dir=out_dir)

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
