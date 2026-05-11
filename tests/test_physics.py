"""
tests/test_physics.py
=====================
Unit tests cho các mô-đun vật lý (Physics Validation):
  - SOC bounds (battery_model)
  - Temperature bounds (thermal_model)
  - Reward decomposition
  - Action bounds (smart_home_env)
  - NaN observation guard

Chạy bằng:
    cd project
    python -m pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np

# --- Đảm bảo import đúng project path ---
_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from env.physics.battery_model import calculate_next_soc
from env.physics.thermal_model import calculate_next_indoor_temp
from utils.reward import calculate_multi_objective_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def config():
    """Config object tối thiểu dùng cho physics tests."""
    from configs.config_loader import ConfigLoader
    return ConfigLoader.load_config()


@pytest.fixture
def default_env_params():
    """env_params mặc định hợp lệ cho reward tests."""
    return {
        "w_eco":               1.0,
        "w_peak":              2.0,
        "w_comfort":           10.0,
        "w_task":              10.0,
        "current_price":       2.0,
        "power_limit":         7.0,
        "current_indoor_temp": 25.0,
        "target_temp":         25.0,
        "current_fridge_temp": 4.0,
        "washer_deadline":     10,
        "washer_completed":    False,
        "battery_power":       0.0,
        "current_soc":         0.5,
    }


# ===========================================================================
# 1. SOC TESTS (Battery Model)
# ===========================================================================

class TestSOCBounds:
    """Kiểm tra SOC luôn nằm trong [min_soc, max_soc]."""

    def test_soc_stays_in_range_on_charge(self, config):
        """Sạc bình thường → SOC tăng nhưng không vượt max_soc."""
        soc, _ = calculate_next_soc(0.5, 3.0, 25.0, config)
        assert 0.0 <= soc <= 1.0, f"SOC out of range: {soc}"

    def test_soc_stays_in_range_on_discharge(self, config):
        """Xả bình thường → SOC giảm nhưng không dưới min_soc."""
        soc, _ = calculate_next_soc(0.5, -3.0, 25.0, config)
        assert 0.0 <= soc <= 1.0, f"SOC out of range: {soc}"

    def test_soc_does_not_exceed_max_when_full(self, config):
        """Khi pin đầy (SOC = max), thêm charge không làm SOC > max."""
        max_soc = config.get("physics.battery.max_soc", 1.0)
        soc, _ = calculate_next_soc(max_soc, 3.0, 25.0, config)
        assert soc <= max_soc, f"SOC vượt max_soc: {soc} > {max_soc}"

    def test_soc_does_not_go_below_min_when_empty(self, config):
        """Khi pin cạn (SOC = min), discharge không làm SOC < min."""
        min_soc = config.get("physics.battery.min_soc", 0.1)
        soc, _ = calculate_next_soc(min_soc, -3.0, 25.0, config)
        assert soc >= min_soc, f"SOC dưới min_soc: {soc} < {min_soc}"

    def test_soc_disabled_in_extreme_temperature(self, config):
        """Nhiệt độ cực đoan → battery bị vô hiệu hóa, SOC không đổi."""
        initial_soc = 0.5
        soc, actual_power = calculate_next_soc(initial_soc, 3.0, 50.0, config)  # 50°C > 45°C
        assert actual_power == 0.0, "Battery phải bị vô hiệu hóa ở nhiệt độ cực đoan"
        assert soc == initial_soc, f"SOC không được thay đổi khi nhiệt độ cực đoan: {soc}"

    def test_soc_is_float(self, config):
        """SOC trả về phải là float."""
        soc, _ = calculate_next_soc(0.5, 1.0, 25.0, config)
        assert isinstance(soc, float), f"SOC phải là float, nhận {type(soc)}"

    @pytest.mark.parametrize("power", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_soc_valid_for_various_powers(self, config, power):
        """SOC hợp lệ với nhiều mức công suất sạc/xả."""
        soc, _ = calculate_next_soc(0.5, power, 25.0, config)
        assert 0.0 <= soc <= 1.0, f"SOC={soc} không hợp lệ với power={power}"


# ===========================================================================
# 2. TEMPERATURE TESTS (Thermal Model)
# ===========================================================================

class TestTemperatureBounds:
    """Kiểm tra mô hình nhiệt động học."""

    def test_indoor_temp_drops_with_hvac(self, config):
        """Bật điều hòa → nhiệt độ phòng giảm (khi nhà đang nóng)."""
        hot_start = 35.0
        outdoor   = 38.0
        temp_without_hvac = calculate_next_indoor_temp(hot_start, outdoor, 0.0,  config)
        temp_with_hvac    = calculate_next_indoor_temp(hot_start, outdoor, 2.0,  config)
        assert temp_with_hvac < temp_without_hvac, \
            f"HVAC bật phải làm giảm nhiệt độ: {temp_with_hvac} vs {temp_without_hvac}"

    def test_indoor_temp_rises_without_hvac_in_hot_outdoor(self, config):
        """Không HVAC, ngoài nóng → nhiệt độ phòng tăng."""
        start = 24.0
        outdoor = 38.0
        temp_next = calculate_next_indoor_temp(start, outdoor, 0.0, config)
        assert temp_next > start, \
            f"Nhà phải nóng lên khi outdoor={outdoor} và không có HVAC: {temp_next}"

    def test_indoor_temp_is_float(self, config):
        """Nhiệt độ trả về phải là float."""
        temp = calculate_next_indoor_temp(25.0, 30.0, 1.0, config)
        assert isinstance(temp, float), f"Temp phải là float, nhận {type(temp)}"

    def test_indoor_temp_no_nan(self, config):
        """Không được có NaN trong kết quả nhiệt độ."""
        temp = calculate_next_indoor_temp(25.0, 30.0, 1.0, config)
        assert np.isfinite(temp), f"Temp chứa NaN/Inf: {temp}"

    @pytest.mark.parametrize("outdoor", [5.0, 15.0, 25.0, 35.0, 43.0])
    def test_temp_finite_across_outdoor_range(self, config, outdoor):
        """Nhiệt độ hợp lệ với nhiều mức nhiệt độ ngoài trời."""
        temp = calculate_next_indoor_temp(25.0, outdoor, 1.0, config)
        assert np.isfinite(temp), f"Temp={temp} không finite với outdoor={outdoor}"


# ===========================================================================
# 3. REWARD TESTS
# ===========================================================================

class TestReward:
    """Kiểm tra hàm reward và các thành phần."""

    def test_reward_returns_tuple(self, default_env_params):
        """Reward phải trả về tuple (float, dict)."""
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        result = calculate_multi_objective_reward(dummy_state, dummy_action, 1.0, default_env_params)
        assert isinstance(result, tuple) and len(result) == 2

    def test_reward_is_clipped(self, default_env_params):
        """Reward phải nằm trong [CLIP_MIN, CLIP_MAX]."""
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        reward, _ = calculate_multi_objective_reward(dummy_state, dummy_action, 1.0, default_env_params)
        assert REWARD_CLIP_MIN <= reward <= REWARD_CLIP_MAX, \
            f"Reward {reward} nằm ngoài clip range [{REWARD_CLIP_MIN}, {REWARD_CLIP_MAX}]"

    def test_reward_is_finite(self, default_env_params):
        """Reward không được là NaN hay Inf."""
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        reward, _ = calculate_multi_objective_reward(dummy_state, dummy_action, 1.0, default_env_params)
        assert np.isfinite(reward), f"Reward phải finite, nhận {reward}"

    def test_reward_decomposition_keys(self, default_env_params):
        """Dict reward_info phải có đủ các keys cần thiết."""
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        _, info = calculate_multi_objective_reward(dummy_state, dummy_action, 1.0, default_env_params)
        required_keys = {"cost_reward", "comfort_reward", "battery_penalty", "peak_penalty",
                         "r_eco", "r_peak", "r_comfort", "r_task", "r_deg",
                         "total", "scaled_total", "clipped_total"}
        missing = required_keys - set(info.keys())
        assert not missing, f"Thiếu keys trong reward_info: {missing}"

    def test_peak_penalty_triggered_when_over_limit(self, default_env_params):
        """Khi công suất vượt giới hạn → peak_penalty phải âm."""
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        # power_limit = 7.0; power_consumed = 10.0 → excess = 3kW
        _, info = calculate_multi_objective_reward(dummy_state, dummy_action, 10.0, default_env_params)
        assert info["peak_penalty"] < 0, f"Peak penalty phải âm khi vượt giới hạn: {info['peak_penalty']}"

    def test_comfort_penalty_triggered_when_too_hot(self, default_env_params):
        """Khi nhiệt độ phòng > 26°C → comfort_reward phải âm."""
        params = dict(default_env_params)
        params["current_indoor_temp"] = 35.0  # Rất nóng
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        _, info = calculate_multi_objective_reward(dummy_state, dummy_action, 1.0, params)
        assert info["comfort_reward"] < 0, f"Comfort reward phải âm khi quá nóng: {info['comfort_reward']}"

    def test_nan_power_handled_gracefully(self, default_env_params):
        """Đầu vào NaN phải được xử lý mà không raise exception."""
        dummy_state  = np.zeros(17, dtype=np.float32)
        dummy_action = {"continuous": np.array([0.0, 25.0, 0.5]), "discrete": np.array([0, 1, 1])}
        reward, _ = calculate_multi_objective_reward(dummy_state, dummy_action, float("nan"), default_env_params)
        assert np.isfinite(reward), f"Reward phải finite dù đầu vào NaN: {reward}"


# ===========================================================================
# 4. ACTION BOUNDS TESTS (SmartHomeEnv)
# ===========================================================================

class TestActionBounds:
    """Kiểm tra action clamping trong SmartHomeEnv."""

    @pytest.fixture(scope="class")
    def env(self):
        from env.smart_home_env import SmartHomeEnv
        from env.wrappers import FlattenActionSpaceWrapper
        base = SmartHomeEnv()
        wrapped = FlattenActionSpaceWrapper(base)
        wrapped.reset(seed=42)
        yield wrapped
        wrapped.close()

    def test_valid_action_does_not_raise(self, env):
        """Action hợp lệ → không raise exception."""
        valid_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(valid_action)
        assert obs is not None

    def test_observation_no_nan_after_step(self, env):
        """Observation sau step() không chứa NaN."""
        action = env.action_space.sample()
        obs, *_ = env.step(action)
        assert not np.isnan(obs).any(), f"Observation chứa NaN: {obs}"

    def test_observation_within_bounds(self, env):
        """Observation nằm trong [-1, 1] (sau normalize)."""
        action = env.action_space.sample()
        obs, *_ = env.step(action)
        assert obs.min() >= -1.1 and obs.max() <= 1.1, \
            f"Observation vượt bounds: min={obs.min()}, max={obs.max()}"

    def test_reward_finite_after_random_actions(self, env):
        """Reward finite sau 10 bước random."""
        env.reset(seed=123)
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), f"Reward không finite: {reward}"
            if terminated or truncated:
                env.reset()

    def test_episode_runs_without_crash(self, env):
        """Một episode đầy đủ (hoặc 100 bước) không crash."""
        env.reset(seed=7)
        for _ in range(min(100, env.unwrapped.max_steps - 1)):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
