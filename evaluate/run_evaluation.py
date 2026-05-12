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

        grid_kw     = info.get("grid_import_kw", 0.0)
        net_load    = info.get("net_load", 0.0)
        
        outdoor_temp = info.get("outdoor_temp")
        if outdoor_temp is None:
            # Fallback to unwrapped data_manager
            # Note: _data_step has already advanced by 1 in step(), so use _data_step - 1
            # But the user instruction says get_step_data(env.unwrapped._data_step). Let's be safe.
            try:
                data_step = env.unwrapped._data_step - 1
                outdoor_temp = env.unwrapped.data_manager.get_step_data(data_step)["outdoor_temp"]
            except Exception:
                outdoor_temp = 25.0 + 5.0 * np.sin(step * np.pi / 48) # fallback demo weather
                
        physics = info.get("physics", {})

        records.append({
            "step":             step,
            "reward":           reward,
            "indoor_temp":      info.get("indoor_temp", float("nan")),
            "outdoor_temp":     outdoor_temp,
            "soc":              info.get("soc", float("nan")),
            "grid_import_kw":   grid_kw,
            "pv_kw":            info.get("pv_kw", 0.0),
            "hvac_power_kw":    info.get("hvac_input_kw", 0.0),
            "batt_power_kw":    info.get("batt_power_kw", 0.0),
            "net_load_kw":      net_load,
            "electricity_cost": info.get("electricity_cost", 0.0),
            "baseline_cost":    info.get("baseline_cost", 0.0),
            "r_eco":            info.get("r_eco", 0.0),
            "r_saving":         info.get("r_saving", 0.0),
            "r_comfort_bonus":  info.get("r_comfort_bonus", 0.0),
            "r_comfort":        info.get("r_comfort", 0.0),
            "r_severe":         info.get("r_severe", 0.0),
            "r_soc_bonus":      info.get("r_soc_bonus", 0.0),
            "r_soc":            info.get("r_soc", 0.0),
            "r_deg":            info.get("r_deg", 0.0),
            "r_peak":           info.get("r_peak", 0.0),
            "r_smooth":         info.get("r_smooth", 0.0),
            "r_switch":         info.get("r_switch", 0.0),
            "raw_reward":       info.get("raw_reward", 0.0),
            "clipped_reward":   info.get("clipped_reward", 0.0),
            "battery_switch":   info.get("battery_switch", 0.0),
            "action_batt":      float(action[0]),
            "action_hvac":      float(action[1]),
            "comfort_violation":info.get("comfort_violation", 0.0),
            "cop":              physics.get("cop", float("nan")),
            "q_ext_leakage":    physics.get("q_ext_leakage", float("nan")),
            "q_cool_thermal":   physics.get("q_cool_thermal", float("nan"))
        })

        obs = next_obs
        if terminated or truncated:
            break

    df = pd.DataFrame(records)
    df["total_cost"] = df["electricity_cost"].cumsum()
    return df


def main():
    print("=" * 55)
    print("   SMART HOME HEMS — EVALUATION (PPO vs Rule-based)")
    print("=" * 55)

    TEST_STEPS = 672  # 7 days
    TEST_SEED  = 42

    # --- Environments (same wrapper stack as training) ---
    env_ppo  = make_stable_env(max_episode_steps=TEST_STEPS, random_start=False, seed=TEST_SEED)
    env_rule = make_stable_env(max_episode_steps=TEST_STEPS, random_start=False, seed=TEST_SEED)

    # --- Load PPO model ---
    model_candidates = [
        os.path.join(project_dir, "models", "ppo_hems_final.zip"),
        os.path.join(project_dir, "models", "ppo_smart_home_final.zip"),
    ]
    model_path = None
    for p in model_candidates:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("[LỖI] Không tìm thấy model. Chạy train_ppo.py trước.")
        return

    print(f"[1/3] Nạp model PPO từ: {model_path}")
    ppo_agent = PPO.load(model_path, env=env_ppo)

    # --- Rule-based agent ---
    print("[2/3] Khởi tạo Rule-based Agent...")
    rule_agent = HEMSRuleBasedAgent()

    # --- Run evaluations ---
    print(f"[3/3] Chạy đánh giá ({TEST_STEPS} steps, seed={TEST_SEED})...")

    print("  → PPO Agent...")
    df_ppo = run_episode(env_ppo, ppo_agent, num_steps=TEST_STEPS, seed=TEST_SEED)

    print("  → Rule-based Agent...")
    df_rule = run_episode(env_rule, rule_agent, num_steps=TEST_STEPS, seed=TEST_SEED)

    # --- Summary statistics ---
    print("\n=== KẾT QUẢ SO SÁNH ===")
    for name, df in [("PPO", df_ppo), ("Rule-based", df_rule)]:
        total_cost = df["electricity_cost"].sum()
        mean_temp  = df["indoor_temp"].mean()
        mean_soc   = df["soc"].mean()
        mean_reward = df["reward"].mean()
        print(f"\n[{name}]")
        print(f"  Tổng chi phí điện : {total_cost:,.0f} VND")
        print(f"  Nhiệt độ TB trong : {mean_temp:.2f} °C")
        print(f"  SoC TB            : {mean_soc:.2f}")
        print(f"  Reward TB/step    : {mean_reward:.4f}")

    # --- Save CSVs ---
    out_dir = os.path.join(project_dir, "evaluate")
    df_ppo.to_csv(os.path.join(out_dir, "eval_ppo.csv"), index=False)
    df_rule.to_csv(os.path.join(out_dir, "eval_rule.csv"), index=False)
    print(f"\nĐã lưu CSV tại: {out_dir}")

    # --- Plots ---
    if plot_evaluation_results is not None:
        print("Đang vẽ biểu đồ phân tích...")
        plot_evaluation_results(df_ppo, df_rule)
    else:
        print("[Bỏ qua biểu đồ] Không tìm thấy plot_metrics.py")


if __name__ == "__main__":
    main()
