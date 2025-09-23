"""Tests for BanditTestEnvironment."""

import pytest
import torch
import numpy as np

from learnrl.utils import BanditTestEnvironment


def test_bandit_env_basic():
    """Test basic BanditTestEnvironment functionality."""
    env = BanditTestEnvironment(k=5, seed=42)

    assert env.k == 5
    assert len(env.true_values) == 5
    assert 0 <= env.optimal_action < 5

    # Test reward generation
    reward = env.get_reward(0)
    assert isinstance(reward, float)

    # Test optimal action identification
    is_optimal = env.is_optimal_action(env.optimal_action)
    assert is_optimal is True


def test_bandit_env_reproducibility():
    """Test that same seed produces identical environments."""
    env1 = BanditTestEnvironment(k=5, seed=123)
    env2 = BanditTestEnvironment(k=5, seed=123)

    assert torch.allclose(env1.true_values, env2.true_values)
    assert env1.optimal_action == env2.optimal_action


def test_bandit_env_invalid_action():
    """Test that invalid actions raise ValueError."""
    env = BanditTestEnvironment(k=5)

    with pytest.raises(ValueError):
        env.get_reward(-1)

    with pytest.raises(ValueError):
        env.get_reward(5)
