"""Test configuration and fixtures for LearnRL tests."""

import pytest
import torch
import numpy as np

from learnrl.utils import BanditTestEnvironment
from learnrl.bandits import EpsilonGreedyBandit


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def device():
    """Provide a torch device for tests."""
    return torch.device("cpu")


@pytest.fixture
def simple_bandit_params():
    """Common parameters for bandit tests."""
    return {"k": 5, "epsilon": 0.1, "initial_values": 0.0, "step_size": None}


@pytest.fixture
def fixed_step_bandit_params():
    """Parameters for bandit with fixed step size."""
    return {"k": 3, "epsilon": 0.2, "initial_values": 1.0, "step_size": 0.1}


class TestEnvironment:
    """Simple test environment for bandit problems."""

    def __init__(self, k: int, true_values: torch.Tensor = None):
        """
        Initialize test environment.

        Args:
            k: Number of arms
            true_values: True action values. If None, uses standard normal.
        """
        self.k = k
        if true_values is None:
            self.true_values = torch.randn(k)
        else:
            self.true_values = true_values

    def get_reward(self, action: int) -> float:
        """
        Get reward for an action.

        Args:
            action: Action index

        Returns:
            Reward (true value + noise)
        """
        return self.true_values[action].item() + torch.randn(1).item()

    def get_optimal_action(self) -> int:
        """Get the optimal action (highest true value)."""
        return torch.argmax(self.true_values).item()


@pytest.fixture
def test_env():
    """Provide a test environment."""
    return TestEnvironment(k=5)


@pytest.fixture
def deterministic_env():
    """Provide a deterministic test environment."""
    true_values = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])
    return TestEnvironment(k=5, true_values=true_values)


# New fixtures for BanditTestEnvironment


@pytest.fixture
def bandit_test_env():
    """Provide a BanditTestEnvironment for testing."""
    return BanditTestEnvironment(k=5, seed=42)


@pytest.fixture
def bandit_test_env_params():
    """Common parameters for BanditTestEnvironment."""
    return {
        "k": 5,
        "true_value_mean": 0.0,
        "true_value_std": 1.0,
        "reward_std": 1.0,
        "seed": 42,
    }


@pytest.fixture
def experiment_params():
    """Parameters for small-scale experiment testing."""
    return {"k": 5, "num_problems": 5, "num_steps": 10, "seed": 42}


@pytest.fixture
def epsilon_greedy_agent():
    """Provide a basic epsilon-greedy agent."""
    return EpsilonGreedyBandit(k=5, epsilon=0.1)


@pytest.fixture
def greedy_agent():
    """Provide a greedy agent (epsilon=0)."""
    return EpsilonGreedyBandit(k=5, epsilon=0.0)


@pytest.fixture
def algorithm_specs():
    """Provide algorithm specifications for comparison tests."""
    return [
        {
            "class": EpsilonGreedyBandit,
            "kwargs": {"epsilon": 0.0, "initial_values": 0.0},
            "name": "greedy",
        },
        {
            "class": EpsilonGreedyBandit,
            "kwargs": {"epsilon": 0.1, "initial_values": 0.0},
            "name": "epsilon-greedy",
        },
    ]
