"""Tests for BanditTestEnvironment."""

import pytest
import torch
import numpy as np

from learnrl.utils import BanditTestEnvironment


class TestBanditTestEnvironment:
    """Test cases for BanditTestEnvironment class."""

    def test_initialization_default(self):
        """Test default initialization."""
        env = BanditTestEnvironment()

        assert env.k == 10
        assert env.true_value_mean == 0.0
        assert env.true_value_std == 1.0
        assert env.reward_std == 1.0
        assert env.device == torch.device("cpu")
        assert len(env.true_values) == 10
        assert 0 <= env.optimal_action < 10
        assert isinstance(env.optimal_action, int)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        env = BanditTestEnvironment(
            k=5, true_value_mean=2.0, true_value_std=0.5, reward_std=0.8, seed=42
        )

        assert env.k == 5
        assert env.true_value_mean == 2.0
        assert env.true_value_std == 0.5
        assert env.reward_std == 0.8
        assert len(env.true_values) == 5
        assert 0 <= env.optimal_action < 5

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        device = torch.device("cpu")
        env = BanditTestEnvironment(k=3, device=device)

        assert env.device == device
        assert env.true_values.device == device

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical environments."""
        env1 = BanditTestEnvironment(k=5, seed=123)
        env2 = BanditTestEnvironment(k=5, seed=123)

        assert torch.allclose(env1.true_values, env2.true_values)
        assert env1.optimal_action == env2.optimal_action

    def test_different_seeds_produce_different_envs(self):
        """Test that different seeds produce different environments."""
        env1 = BanditTestEnvironment(k=5, seed=123)
        env2 = BanditTestEnvironment(k=5, seed=456)

        # Very unlikely to be identical with different seeds
        assert not torch.allclose(env1.true_values, env2.true_values)

    def test_get_reward_valid_actions(self):
        """Test reward generation for valid actions."""
        env = BanditTestEnvironment(k=5, seed=42)

        # Test all valid actions
        for action in range(5):
            reward = env.get_reward(action)
            assert isinstance(reward, float)
            # Reward should be reasonably close to true value (within a few std devs)
            true_value = env.true_values[action].item()
            assert abs(reward - true_value) < 5 * env.reward_std

    def test_get_reward_invalid_actions(self):
        """Test that invalid actions raise ValueError."""
        env = BanditTestEnvironment(k=5)

        with pytest.raises(ValueError, match="not in valid range"):
            env.get_reward(-1)

        with pytest.raises(ValueError, match="not in valid range"):
            env.get_reward(5)

        with pytest.raises(ValueError, match="not in valid range"):
            env.get_reward(10)

    def test_reward_distribution(self):
        """Test that rewards follow expected distribution."""
        env = BanditTestEnvironment(k=3, reward_std=1.0, seed=42)
        action = 0
        true_value = env.true_values[action].item()

        # Collect many reward samples
        rewards = [env.get_reward(action) for _ in range(1000)]

        # Check empirical mean is close to true value
        empirical_mean = np.mean(rewards)
        assert abs(empirical_mean - true_value) < 0.1

        # Check empirical std is close to expected
        empirical_std = np.std(rewards, ddof=1)
        assert abs(empirical_std - env.reward_std) < 0.1

    def test_is_optimal_action(self):
        """Test optimal action identification."""
        env = BanditTestEnvironment(k=5, seed=42)

        # Optimal action should return True
        assert env.is_optimal_action(env.optimal_action) is True

        # Non-optimal actions should return False
        for action in range(5):
            if action != env.optimal_action:
                assert env.is_optimal_action(action) is False

    def test_get_optimal_value(self):
        """Test getting optimal action value."""
        env = BanditTestEnvironment(k=5, seed=42)

        optimal_value = env.get_optimal_value()
        expected_value = env.true_values[env.optimal_action].item()

        assert isinstance(optimal_value, float)
        assert optimal_value == expected_value
        assert optimal_value == max(env.true_values).item()

    def test_reset_functionality(self):
        """Test environment reset with new random values."""
        env = BanditTestEnvironment(k=5, seed=42)

        # Store initial state
        initial_values = env.true_values.clone()
        initial_optimal = env.optimal_action

        # Reset without seed should change values
        env.reset()

        # Values should be different (very unlikely to be identical)
        assert not torch.allclose(env.true_values, initial_values)
        assert len(env.true_values) == 5  # k should remain same
        assert 0 <= env.optimal_action < 5

    def test_reset_with_seed(self):
        """Test reset with specific seed for reproducibility."""
        env = BanditTestEnvironment(k=5, seed=42)

        # Reset with same seed
        env.reset(seed=42)
        values_first = env.true_values.clone()
        optimal_first = env.optimal_action

        # Reset with same seed again
        env.reset(seed=42)

        # Should be identical
        assert torch.allclose(env.true_values, values_first)
        assert env.optimal_action == optimal_first

    def test_repr_string(self):
        """Test string representation."""
        env = BanditTestEnvironment(k=5, seed=42)
        repr_str = repr(env)

        assert "BanditTestEnvironment" in repr_str
        assert "k=5" in repr_str
        assert f"optimal_action={env.optimal_action}" in repr_str
        assert "optimal_value=" in repr_str

    def test_true_values_distribution(self):
        """Test that true values follow expected distribution."""
        mean, std = 1.0, 0.5
        env = BanditTestEnvironment(
            k=1000,  # Large k for good statistics
            true_value_mean=mean,
            true_value_std=std,
            seed=42,
        )

        # Check empirical distribution
        values = env.true_values.numpy()
        empirical_mean = np.mean(values)
        empirical_std = np.std(values, ddof=1)

        # Should be close to specified parameters
        assert abs(empirical_mean - mean) < 0.1
        assert abs(empirical_std - std) < 0.1

    def test_optimal_action_correctness(self):
        """Test that optimal action is indeed the one with highest true value."""
        env = BanditTestEnvironment(k=10, seed=42)

        optimal_action = env.optimal_action
        optimal_value = env.true_values[optimal_action]

        # Check that this is indeed the maximum
        max_value = torch.max(env.true_values)
        assert torch.isclose(optimal_value, max_value)

        # Check that no other action has higher value
        for i, value in enumerate(env.true_values):
            if i != optimal_action:
                assert value <= optimal_value

    @pytest.mark.parametrize("k", [1, 2, 5, 10, 20])
    def test_different_k_values(self, k):
        """Test various numbers of arms."""
        env = BanditTestEnvironment(k=k, seed=42)

        assert env.k == k
        assert len(env.true_values) == k
        assert 0 <= env.optimal_action < k

        # Should be able to get rewards for all actions
        for action in range(k):
            reward = env.get_reward(action)
            assert isinstance(reward, float)

    @pytest.mark.parametrize("reward_std", [0.1, 0.5, 1.0, 2.0])
    def test_different_reward_std(self, reward_std):
        """Test various reward standard deviations."""
        env = BanditTestEnvironment(k=5, reward_std=reward_std, seed=42)

        assert env.reward_std == reward_std

        # Collect rewards and check variance
        action = 0
        rewards = [env.get_reward(action) for _ in range(500)]
        empirical_std = np.std(rewards, ddof=1)

        # Should be reasonably close (within 20%)
        assert abs(empirical_std - reward_std) < 0.2 * reward_std

    def test_edge_case_single_arm(self):
        """Test edge case with single arm bandit."""
        env = BanditTestEnvironment(k=1, seed=42)

        assert env.k == 1
        assert env.optimal_action == 0
        assert env.is_optimal_action(0) is True

        reward = env.get_reward(0)
        assert isinstance(reward, float)
