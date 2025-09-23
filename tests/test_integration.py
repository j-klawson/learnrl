"""Integration tests for bandit experiments and comparisons."""

import pytest
import torch
import numpy as np
import os
import sys

# Add examples directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from learnrl.bandits import EpsilonGreedyBandit
from learnrl.utils import BanditTestEnvironment

# Import from examples
from sutton_barto_bandit_comparison import (
    run_single_algorithm,
    compare_algorithms,
    run_sutton_barto_experiment,
    ExperimentResult,
)


class TestBanditExperimentIntegration:
    """Integration tests for bandit experiment functionality."""

    def test_single_algorithm_run(self):
        """Test running experiment with single algorithm."""
        result = run_single_algorithm(
            agent_class=EpsilonGreedyBandit,
            agent_kwargs={"epsilon": 0.1, "initial_values": 0.0},
            algorithm_name="test-epsilon-greedy",
            k=5,
            num_problems=10,
            num_steps=20,
            seed=42,
        )

        # Check result structure
        assert isinstance(result, ExperimentResult)
        assert result.algorithm_name == "test-epsilon-greedy"
        assert len(result.average_rewards) == 20
        assert len(result.optimal_action_rates) == 20
        assert len(result.final_q_values) == 10

        # Check data types and ranges
        assert all(isinstance(r, (float, np.floating)) for r in result.average_rewards)
        assert all(0 <= rate <= 1 for rate in result.optimal_action_rates)
        assert all(len(q_vals) == 5 for q_vals in result.final_q_values)

    def test_compare_algorithms_basic(self):
        """Test basic algorithm comparison."""
        algorithms = [
            {
                "class": EpsilonGreedyBandit,
                "kwargs": {"epsilon": 0.0},  # greedy
                "name": "greedy",
            },
            {
                "class": EpsilonGreedyBandit,
                "kwargs": {"epsilon": 0.1},  # epsilon-greedy
                "name": "epsilon-greedy",
            },
        ]

        results = compare_algorithms(
            algorithms=algorithms, k=5, num_problems=5, num_steps=10, seed=42
        )

        assert len(results) == 2
        assert results[0].algorithm_name == "greedy"
        assert results[1].algorithm_name == "epsilon-greedy"

        # Both should have same structure
        for result in results:
            assert len(result.average_rewards) == 10
            assert len(result.optimal_action_rates) == 10
            assert len(result.final_q_values) == 5

    def test_sutton_barto_experiment_small(self):
        """Test the full Sutton & Barto experiment with small parameters."""
        results = run_sutton_barto_experiment(
            epsilon_values=[0.0, 0.1],
            k=5,
            num_problems=5,
            num_steps=10,
            seed=42,
            plot=False,  # Don't show plots in tests
        )

        assert len(results) == 2
        assert results[0].algorithm_name == "greedy"
        assert results[1].algorithm_name == "ε-greedy (ε=0.1)"

        # Verify results structure
        for result in results:
            assert isinstance(result, ExperimentResult)
            assert len(result.average_rewards) == 10
            assert len(result.optimal_action_rates) == 10
            assert len(result.final_q_values) == 5

    def test_experiment_reproducibility(self):
        """Test that experiments are reproducible with same seed."""
        algorithms = [
            {"class": EpsilonGreedyBandit, "kwargs": {"epsilon": 0.1}, "name": "test"}
        ]

        # Run same experiment twice with same seed
        results1 = compare_algorithms(
            algorithms=algorithms, k=3, num_problems=3, num_steps=5, seed=123
        )

        results2 = compare_algorithms(
            algorithms=algorithms, k=3, num_problems=3, num_steps=5, seed=123
        )

        # Results should be identical
        assert len(results1) == len(results2) == 1
        result1, result2 = results1[0], results2[0]

        assert np.allclose(result1.average_rewards, result2.average_rewards)
        assert np.allclose(result1.optimal_action_rates, result2.optimal_action_rates)

    def test_experiment_different_seeds(self):
        """Test that different seeds produce different results."""
        algorithms = [
            {"class": EpsilonGreedyBandit, "kwargs": {"epsilon": 0.1}, "name": "test"}
        ]

        # Run with different seeds
        results1 = compare_algorithms(
            algorithms, k=5, num_problems=10, num_steps=10, seed=123
        )
        results2 = compare_algorithms(
            algorithms, k=5, num_problems=10, num_steps=10, seed=456
        )

        result1, result2 = results1[0], results2[0]

        # Results should be different (very unlikely to be identical)
        assert not np.allclose(result1.average_rewards, result2.average_rewards)

    def test_epsilon_greedy_vs_greedy_performance(self):
        """Test that epsilon-greedy outperforms greedy in the long run."""
        results = run_sutton_barto_experiment(
            epsilon_values=[0.0, 0.1],
            k=10,
            num_problems=50,  # Enough to see difference
            num_steps=100,
            seed=42,
            plot=False,
        )

        greedy_result = next(r for r in results if r.algorithm_name == "greedy")
        eps_greedy_result = next(r for r in results if "ε-greedy" in r.algorithm_name)

        # In later steps, epsilon-greedy should generally perform better
        final_steps = slice(-20, None)  # Last 20 steps
        greedy_final_optimal = np.mean(greedy_result.optimal_action_rates[final_steps])
        eps_greedy_final_optimal = np.mean(
            eps_greedy_result.optimal_action_rates[final_steps]
        )

        # Epsilon-greedy should have higher optimal action rate in the long run
        assert eps_greedy_final_optimal > greedy_final_optimal

    def test_bandit_environment_integration(self):
        """Test integration between algorithms and environment."""
        env = BanditTestEnvironment(k=5, seed=42)
        agent = EpsilonGreedyBandit(k=5, epsilon=0.1, device=env.device)

        # Run a short interaction
        total_optimal = 0
        total_reward = 0.0
        num_steps = 100

        for step in range(num_steps):
            action = agent.select_action()
            reward = env.get_reward(action)
            agent.update(action, reward)

            total_optimal += env.is_optimal_action(action)
            total_reward += reward

        # Check that learning occurred
        optimal_rate = total_optimal / num_steps
        avg_reward = total_reward / num_steps

        # With some exploration, should have some optimal actions
        assert 0 < optimal_rate < 1
        assert isinstance(avg_reward, float)

        # Agent should have learned something (Q-values not all zeros)
        assert not torch.allclose(agent.q_values, torch.zeros_like(agent.q_values))

    def test_multiple_epsilon_values(self):
        """Test experiment with multiple epsilon values."""
        epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.2]

        results = run_sutton_barto_experiment(
            epsilon_values=epsilon_values,
            k=5,
            num_problems=10,
            num_steps=20,
            seed=42,
            plot=False,
        )

        assert len(results) == len(epsilon_values)

        # Check that each algorithm has correct name
        expected_names = [
            "greedy",
            "ε-greedy (ε=0.01)",
            "ε-greedy (ε=0.05)",
            "ε-greedy (ε=0.1)",
            "ε-greedy (ε=0.2)",
        ]

        actual_names = [r.algorithm_name for r in results]
        assert actual_names == expected_names

    def test_experiment_result_data_consistency(self):
        """Test that experiment results are internally consistent."""
        result = run_single_algorithm(
            agent_class=EpsilonGreedyBandit,
            agent_kwargs={"epsilon": 0.1},
            algorithm_name="test",
            k=5,
            num_problems=10,
            num_steps=15,
            seed=42,
        )

        # Check array lengths match num_steps
        assert len(result.average_rewards) == 15
        assert len(result.optimal_action_rates) == 15

        # Check number of final Q-values matches num_problems
        assert len(result.final_q_values) == 10

        # Check Q-values have correct shape
        for q_vals in result.final_q_values:
            assert q_vals.shape == (5,)  # k=5

        # Check optimal action rates are valid probabilities
        assert all(0 <= rate <= 1 for rate in result.optimal_action_rates)

    @pytest.mark.parametrize("k", [2, 5, 10])
    def test_experiment_different_k_values(self, k):
        """Test experiment with different numbers of arms."""
        result = run_single_algorithm(
            agent_class=EpsilonGreedyBandit,
            agent_kwargs={"epsilon": 0.1},
            algorithm_name="test",
            k=k,
            num_problems=5,
            num_steps=10,
            seed=42,
        )

        # Q-values should have correct dimensionality
        for q_vals in result.final_q_values:
            assert q_vals.shape == (k,)
