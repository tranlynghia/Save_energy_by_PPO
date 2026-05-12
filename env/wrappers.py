"""
wrappers.py — Action Scaling & Stabilization Wrappers for PPO
=============================================================
Senior RL Engineering Note:

This file solves the most critical PPO training problem in the current codebase:
ACTION SCALE MISMATCH.

The Problem:
    PPO with a tanh output layer produces actions in [-1, 1].
    If the environment interprets [-1, 1] directly as [-1kW, +1kW] for a 10kWh
    battery with 5kW max power, then 80% of the action space is wasted.
    The agent physically CAN charge/discharge at 5kW but its output only moves
    the actual power by 1kW. This is equivalent to asking a driver to steer a
    car by moving the wheel 1mm. It causes:
      - Policy saturation: agent learns to push output to ±1 always
      - Dead actions: large portions of action space have no effect
      - Slow convergence: gradient signal is too weak to differentiate actions

The Fix (ActionScalingWrapper):
    Map PPO output [-1, 1] → Physical range [-5kW, +5kW] for battery.
    This means every unit of PPO output corresponds to real physical effect.

WHY WRAPPER-SIDE SCALING STABILIZES PPO:
    1. Keeps PPO action distribution symmetric around 0 (matches Gaussian prior)
    2. Avoids modifying the core environment (keeps physics clean)
    3. Allows easy hyperparameter changes without touching environment code
    4. SB3 VecEnv handles wrappers correctly, including normalization compatibility

HVAC Discretization:
    PPO continuous output → 4 discrete HVAC modes.
    This prevents the "HVAC shimmer" problem where the agent oscillates
    between 2.0kW and 2.1kW every step, wasting compressor cycles.
    Discrete modes create a cleaner credit assignment problem.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional


# HVAC Operating Modes (power consumed by compressor, in kW)
# Mode selection: act[1] ∈ [-1, 1] → quantized into 4 regions
HVAC_MODES = {
    "OFF":    0.0,   # [-1.00, -0.50) → No cooling
    "ECO":    1.0,   # [-0.50,  0.00) → Light cooling (COP-efficient)
    "NORMAL": 2.5,   # [ 0.00,  0.50) → Standard comfort
    "BOOST":  4.0,   # [ 0.50,  1.00] → Maximum cooling
}


class ActionScalingWrapper(gym.ActionWrapper):
    """
    Translates PPO's normalized output [-1, 1] into physical control signals.

    PPO (after ActionScalingWrapper):
        act[0] ∈ [-1, 1] → battery_power ∈ [-5.0, +5.0] kW
        act[1] ∈ [-1, 1] → hvac_cooling_kw ∈ {0.0, 1.0, 2.5, 4.0} kW

    The wrapper's action_space advertises [-1, 1]² to PPO.
    The underlying env receives physical units.

    WHY DISCRETIZE HVAC HERE (not in env):
        - Keeps environment physics pure and continuous
        - Allows changing discretization scheme without modifying physics
        - Reduces the number of learning dimensions for PPO (4 modes vs continuous)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # PPO sees a normalized action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Physical action limits (must match what env expects)
        self._batt_scale = 5.0   # kW — full-power = 5kW
        self._hvac_thresholds = [-0.5, 0.0, 0.5]  # boundaries between 4 modes

    def action(self, act: np.ndarray) -> np.ndarray:
        """
        Convert PPO output to physical control signals.

        Args:
            act: PPO action in [-1, 1]² (after tanh/clipping)

        Returns:
            Physical action in env's expected units:
            [battery_power_kw, hvac_cooling_kw]
        """
        act = np.asarray(act, dtype=np.float32)

        # --- Battery scaling ---
        # [-1, 1] → [-5.0, +5.0] kW
        # Linear scaling preserves the symmetry of the action distribution.
        # This is critical because PPO uses a Gaussian policy centered at 0.
        # If we used a non-symmetric mapping, the policy would be biased.
        battery_kw = float(act[0]) * self._batt_scale

        # --- HVAC discretization ---
        # WHY 4 MODES: Empirically, a continuous HVAC actuator leads to
        # "action shimmer" (oscillating between 2.0kW and 2.05kW every step).
        # This creates non-stationary transitions that confuse the critic.
        # Discrete modes provide clean, repeatable transitions.
        hvac_raw = float(act[1])
        if hvac_raw < self._hvac_thresholds[0]:
            hvac_kw = HVAC_MODES["OFF"]
        elif hvac_raw < self._hvac_thresholds[1]:
            hvac_kw = HVAC_MODES["ECO"]
        elif hvac_raw < self._hvac_thresholds[2]:
            hvac_kw = HVAC_MODES["NORMAL"]
        else:
            hvac_kw = HVAC_MODES["BOOST"]

        return np.array([battery_kw, hvac_kw], dtype=np.float32)

    def reverse_action(self, physical_action: np.ndarray) -> np.ndarray:
        """Inverse mapping for logging/debugging purposes."""
        batt_norm = physical_action[0] / self._batt_scale
        # HVAC: find which mode and return its threshold midpoint
        hvac_kw = physical_action[1]
        mode_kws = list(HVAC_MODES.values())
        thresholds = [-0.75, -0.25, 0.25, 0.75]
        hvac_norm = thresholds[min(mode_kws, key=lambda x: abs(x - hvac_kw))]
        return np.array([batt_norm, hvac_norm], dtype=np.float32)


class ObservationNormalizationWrapper(gym.ObservationWrapper):
    """
    Final-layer normalization guard.

    Ensures observations never exceed [-1.2, 1.2] even if the internal
    environment produces slightly out-of-range values due to physics edge cases.

    This is a safety net — the core environment should already produce
    well-normalized observations. This wrapper catches any remaining issues.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Clip and sanitize observation."""
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(obs, -1.2, 1.2)


class RewardClippingWrapper(gym.RewardWrapper):
    """
    Hard clip on reward as a final guard against reward spikes.

    Use this ONLY as a safety net. The core environment should already
    clip rewards internally. Double-clipping has no negative effect.
    """

    def __init__(self, env: gym.Env, r_min: float = -10.0, r_max: float = 2.0) -> None:
        super().__init__(env)
        self.r_min = r_min
        self.r_max = r_max

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.r_min, self.r_max))


def make_stable_env(
    data_manager=None,
    max_episode_steps: int = 96,
    random_start: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Factory function that assembles the full PPO-ready environment stack.

    Stack (innermost → outermost):
        SmartHomeEnv         ← Core physics
        ActionScalingWrapper ← Maps [-1,1] → physical units
        ObsNorm              ← Safety clamp on observations

    Usage:
        env = make_stable_env(seed=42)
        obs, _ = env.reset()
        action = env.action_space.sample()  # agent sees [-1,1]²
        obs, reward, done, truncated, info = env.step(action)

    Args:
        data_manager: Optional pre-built DataManager (for shared datasets).
        max_episode_steps: Number of 15-min steps per episode (96=24h).
        random_start: Whether to randomize the starting position in dataset.
        seed: Random seed for reproducibility.

    Returns:
        Wrapped environment compatible with SB3 PPO.
    """
    from env.smart_home_env import SmartHomeEnv

    env = SmartHomeEnv(
        data_manager=data_manager,
        max_episode_steps=max_episode_steps,
        random_start=random_start,
        seed=seed,
    )
    env = ActionScalingWrapper(env)
    env = ObservationNormalizationWrapper(env)
    return env
