"""Tests for EpsilonGreedyBandit."""

import pytest
import torch
from unittest.mock import patch

from learnrl.bandits import EpsilonGreedyBandit


class TestEpsilonGreedyBandit:
    """Test cases for EpsilonGreedyBandit class."""

    def test_initialization(self):
        """Test proper initialization of the bandit."""
        bandit = EpsilonGreedyBandit(k=5, epsilon=0.2, initial_values=1.0)

        assert bandit.k == 5
        assert bandit.epsilon == 0.2
        assert bandit.step_size is None
        assert torch.allclose(bandit.q_values, torch.ones(5))
        assert torch.allclose(bandit.action_counts, torch.zeros(5, dtype=torch.long))
        assert bandit.t == 0

    def test_initialization_with_step_size(self):
        """Test initialization with fixed step size."""
        bandit = EpsilonGreedyBandit(k=3, epsilon=0.1, step_size=0.5)

        assert bandit.step_size == 0.5

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        device = torch.device("cpu")
        bandit = EpsilonGreedyBandit(k=4, device=device)

        assert bandit.q_values.device == device
        assert bandit.action_counts.device == device

    def test_select_action_pure_exploration(self):
        """Test action selection with epsilon=1.0 (pure exploration)."""
        bandit = EpsilonGreedyBandit(k=5, epsilon=1.0)

        # With epsilon=1.0, should always explore (random action)
        actions = [bandit.select_action() for _ in range(100)]

        # Check that actions are in valid range
        assert all(0 <= action < 5 for action in actions)

        # With enough samples, should see multiple different actions
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_select_action_pure_exploitation(self):
        """Test action selection with epsilon=0.0 (pure exploitation)."""
        bandit = EpsilonGreedyBandit(k=5, epsilon=0.0)

        # Set different Q-values to make action 2 clearly the best
        bandit.q_values = torch.tensor([1.0, 2.0, 5.0, 1.5, 0.5])

        # With epsilon=0.0, should always select action 2 (highest Q-value)
        actions = [bandit.select_action() for _ in range(50)]
        assert all(action == 2 for action in actions)

    def test_select_action_tie_breaking(self):
        """Test that ties are broken randomly."""
        bandit = EpsilonGreedyBandit(k=4, epsilon=0.0)

        # Set Q-values so actions 1 and 3 are tied for best
        bandit.q_values = torch.tensor([1.0, 3.0, 2.0, 3.0])

        actions = [bandit.select_action() for _ in range(100)]

        # Should only select actions 1 or 3
        assert all(action in [1, 3] for action in actions)

        # With enough samples, should see both actions
        unique_actions = set(actions)
        assert len(unique_actions) == 2

    @patch("torch.rand")
    def test_select_action_epsilon_boundary(self, mock_rand):
        """Test epsilon boundary conditions."""
        bandit = EpsilonGreedyBandit(k=3, epsilon=0.3)
        bandit.q_values = torch.tensor([1.0, 5.0, 2.0])  # Action 1 is best

        # Test exploration case (rand < epsilon)
        mock_rand.return_value = torch.tensor([0.2])  # 0.2 < 0.3
        with patch("torch.randint", return_value=torch.tensor([2])):
            action = bandit.select_action()
            assert action == 2  # Should explore

        # Test exploitation case (rand >= epsilon)
        mock_rand.return_value = torch.tensor([0.5])  # 0.5 >= 0.3
        action = bandit.select_action()
        assert action == 1  # Should exploit (best action)

    def test_update_sample_averaging(self):
        """Test Q-value updates with sample averaging."""
        bandit = EpsilonGreedyBandit(k=3, epsilon=0.1)

        # Update action 0 multiple times
        bandit.update(0, 1.0)
        assert torch.isclose(bandit.q_values[0], torch.tensor(1.0))
        assert bandit.action_counts[0] == 1
        assert bandit.t == 1

        bandit.update(0, 3.0)
        expected_q = (1.0 + 3.0) / 2  # Average of rewards
        assert torch.isclose(bandit.q_values[0], torch.tensor(expected_q))
        assert bandit.action_counts[0] == 2
        assert bandit.t == 2

        bandit.update(0, 2.0)
        expected_q = (1.0 + 3.0 + 2.0) / 3
        assert torch.isclose(bandit.q_values[0], torch.tensor(expected_q))
        assert bandit.action_counts[0] == 3

    def test_update_fixed_step_size(self):
        """Test Q-value updates with fixed step size."""
        bandit = EpsilonGreedyBandit(k=3, epsilon=0.1, step_size=0.1)

        initial_q = bandit.q_values[0].item()
        bandit.update(0, 5.0)

        # Q(a) ← Q(a) + α[R - Q(a)]
        expected_q = initial_q + 0.1 * (5.0 - initial_q)
        assert torch.isclose(bandit.q_values[0], torch.tensor(expected_q))
        assert bandit.action_counts[0] == 1

    def test_reset(self):
        """Test reset functionality."""
        bandit = EpsilonGreedyBandit(k=4, epsilon=0.2)

        # Make some updates
        bandit.update(0, 1.0)
        bandit.update(1, 2.0)
        bandit.update(2, 3.0)

        # Reset with new initial values
        bandit.reset(initial_values=2.5)

        assert torch.allclose(bandit.q_values, torch.full((4,), 2.5))
        assert torch.allclose(bandit.action_counts, torch.zeros(4, dtype=torch.long))
        assert bandit.t == 0

    def test_get_action_probabilities(self):
        """Test action probability calculation."""
        bandit = EpsilonGreedyBandit(k=4, epsilon=0.2)
        bandit.q_values = torch.tensor(
            [1.0, 3.0, 2.0, 3.0]
        )  # Actions 1,3 tied for best

        probs = bandit.get_action_probabilities()

        # Expected probabilities:
        # - Base exploration: 0.2/4 = 0.05 for each action
        # - Greedy probability: (1-0.2)/2 = 0.4 shared between actions 1,3
        expected_probs = torch.tensor([0.05, 0.45, 0.05, 0.45])

        assert torch.allclose(probs, expected_probs)
        assert torch.isclose(torch.sum(probs), torch.tensor(1.0))

    def test_get_greedy_action(self):
        """Test greedy action selection."""
        bandit = EpsilonGreedyBandit(k=5, epsilon=0.3)
        bandit.q_values = torch.tensor([1.0, 4.0, 2.0, 4.0, 1.5])

        # Actions 1 and 3 are tied for best (Q=4.0)
        greedy_actions = [bandit.get_greedy_action() for _ in range(100)]

        # Should only return actions 1 or 3
        assert all(action in [1, 3] for action in greedy_actions)

        # With enough samples, should see both actions
        unique_actions = set(greedy_actions)
        assert len(unique_actions) == 2

    def test_repr(self):
        """Test string representation."""
        bandit = EpsilonGreedyBandit(k=5, epsilon=0.1, step_size=0.2)
        bandit.t = 10

        expected = "EpsilonGreedyBandit(k=5, epsilon=0.1, step_size=0.2, t=10)"
        assert repr(bandit) == expected

    def test_multiple_actions_update(self):
        """Test updating multiple different actions."""
        bandit = EpsilonGreedyBandit(k=3, epsilon=0.1)

        bandit.update(0, 1.0)
        bandit.update(1, 2.0)
        bandit.update(2, 3.0)
        bandit.update(0, 4.0)

        # Check that each action was updated correctly
        assert torch.isclose(bandit.q_values[0], torch.tensor(2.5))  # (1+4)/2
        assert torch.isclose(bandit.q_values[1], torch.tensor(2.0))  # 2/1
        assert torch.isclose(bandit.q_values[2], torch.tensor(3.0))  # 3/1

        assert bandit.action_counts[0] == 2
        assert bandit.action_counts[1] == 1
        assert bandit.action_counts[2] == 1
        assert bandit.t == 4

    def test_invalid_action_bounds(self):
        """Test behavior with edge case inputs."""
        bandit = EpsilonGreedyBandit(k=3, epsilon=0.1)

        # Valid actions should work
        bandit.update(0, 1.0)
        bandit.update(2, 2.0)

        # Test that action selection stays in bounds
        for _ in range(100):
            action = bandit.select_action()
            assert 0 <= action < 3

    @pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5, 1.0])
    def test_epsilon_values(self, epsilon):
        """Test various epsilon values."""
        bandit = EpsilonGreedyBandit(k=5, epsilon=epsilon)
        assert bandit.epsilon == epsilon

        # Should be able to select actions regardless of epsilon
        action = bandit.select_action()
        assert 0 <= action < 5

    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_different_k_values(self, k):
        """Test various numbers of arms."""
        bandit = EpsilonGreedyBandit(k=k, epsilon=0.1)
        assert bandit.k == k
        assert len(bandit.q_values) == k
        assert len(bandit.action_counts) == k

        action = bandit.select_action()
        assert 0 <= action < k
