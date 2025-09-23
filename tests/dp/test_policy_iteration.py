"""Tests for PolicyIteration algorithm."""

import pytest
import torch
import numpy as np

from learnrl.dp.policy_iteration import PolicyIteration
from learnrl.utils.gridworld_env import GridWorldEnv, GridWorldConfig, create_simple_gridworld

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class TestPolicyIteration:
    """Test cases for PolicyIteration algorithm."""

    def test_basic_initialization(self):
        """Test basic initialization with integer spaces."""
        n_states = 16
        n_actions = 4
        P = torch.zeros((n_states, n_actions, n_states))
        R = torch.zeros((n_states, n_actions, n_states))

        # Make valid transition model (identity for simplicity)
        for s in range(n_states):
            for a in range(n_actions):
                P[s, a, s] = 1.0

        pi = PolicyIteration(n_states, n_actions, P, R)

        assert pi.n_states == 16
        assert pi.n_actions == 4
        assert pi.gamma == 0.9
        assert pi.theta == 1e-6
        assert pi.policy.shape == (16, 4)
        assert torch.allclose(pi.policy, torch.ones(16, 4) / 4)
        assert pi.converged is False

    @pytest.mark.skipif(not HAS_GYMNASIUM, reason="Gymnasium not available")
    def test_gymnasium_spaces_initialization(self):
        """Test initialization with Gymnasium spaces."""
        obs_space = spaces.Discrete(10)
        action_space = spaces.Discrete(3)
        P = torch.zeros((10, 3, 10))
        R = torch.zeros((10, 3, 10))

        # Make valid transitions
        for s in range(10):
            for a in range(3):
                P[s, a, s] = 1.0

        pi = PolicyIteration(obs_space, action_space, P, R)

        assert pi.n_states == 10
        assert pi.n_actions == 3

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        n_states, n_actions = 5, 2
        P = torch.eye(n_states).unsqueeze(1).repeat(1, n_actions, 1)
        R = torch.zeros((n_states, n_actions, n_states))

        pi = PolicyIteration(
            n_states, n_actions, P, R,
            gamma=0.95, theta=1e-8,
            device=torch.device('cpu')
        )

        assert pi.gamma == 0.95
        assert pi.theta == 1e-8
        assert pi.device == torch.device('cpu')

    def test_invalid_spaces(self):
        """Test error handling for invalid spaces."""
        P = torch.zeros((4, 2, 4))
        R = torch.zeros((4, 2, 4))

        with pytest.raises(ValueError, match="Unsupported observation space type"):
            PolicyIteration("invalid", 2, P, R)

        with pytest.raises(ValueError, match="Unsupported action space type"):
            PolicyIteration(4, "invalid", P, R)

    def test_tensor_shape_validation(self):
        """Test validation of tensor shapes."""
        n_states, n_actions = 4, 2

        # Wrong transition probabilities shape
        P_wrong = torch.zeros((3, 2, 4))
        R = torch.zeros((4, 2, 4))

        with pytest.raises(ValueError, match="Transition probs shape"):
            PolicyIteration(n_states, n_actions, P_wrong, R)

        # Wrong rewards shape
        P = torch.zeros((4, 2, 4))
        R_wrong = torch.zeros((4, 2, 3))

        with pytest.raises(ValueError, match="Rewards shape"):
            PolicyIteration(n_states, n_actions, P, R_wrong)

    def test_policy_evaluation_convergence(self):
        """Test policy evaluation convergence."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-4
        )

        iterations = pi.policy_evaluation(max_iterations=1000)

        assert iterations > 0
        assert iterations < 1000  # Should converge before max
        assert pi.n_policy_evaluations == 1

    def test_policy_improvement(self):
        """Test policy improvement step."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        # Evaluate initial policy
        pi.policy_evaluation()
        initial_policy = pi.policy.clone()

        # Improve policy
        policy_stable = pi.policy_improvement()

        assert pi.n_policy_improvements == 1
        assert not torch.allclose(initial_policy, pi.policy)
        assert len(pi.policy_stable_history) == 1

    def test_full_algorithm_convergence(self):
        """Test complete policy iteration algorithm."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-6
        )

        result = pi.solve(max_iterations=50)

        assert result['converged'] is True
        assert result['iterations'] > 0
        assert result['policy_evaluations'] > 0
        assert result['policy_improvements'] > 0
        assert len(result['evaluation_history']) == result['policy_evaluations']
        assert len(result['policy_stable_history']) == result['policy_improvements']

    def test_max_iterations_limit(self):
        """Test behavior when max iterations reached."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-12  # Very strict convergence
        )

        result = pi.solve(max_iterations=3)

        assert result['iterations'] == 3
        # May or may not have converged with only 3 iterations

    def test_policy_extraction(self):
        """Test policy extraction methods."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        pi.solve()

        # Test get_policy for all states
        full_policy = pi.get_policy()
        assert full_policy.shape == (env.n_states, env.n_actions)
        assert torch.allclose(torch.sum(full_policy, dim=1), torch.ones(env.n_states))

        # Test get_policy for specific state
        state_policy = pi.get_policy(state=0)
        assert state_policy.shape == (env.n_actions,)
        assert torch.isclose(torch.sum(state_policy), torch.tensor(1.0))

        # Test get_action
        action = pi.get_action(0)
        assert isinstance(action, int)
        assert 0 <= action < env.n_actions

    def test_state_value_access(self):
        """Test state value access methods."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        pi.solve()

        # Test get_state_value for all states
        all_values = pi.get_state_value()
        assert all_values.shape == (env.n_states,)

        # Test get_state_value for specific state
        state_value = pi.get_state_value(state=0)
        assert isinstance(state_value, float)
        assert state_value == all_values[0].item()

    def test_action_values_computation(self):
        """Test action values (Q-values) computation."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        pi.solve()

        action_values = pi.get_action_values(state=0)
        assert action_values.shape == (env.n_actions,)

        # Best action should have highest Q-value
        best_action = pi.get_action(0)
        assert action_values[best_action] == torch.max(action_values)

    def test_policy_performance_evaluation(self):
        """Test policy performance evaluation."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        pi.solve()

        # Evaluate from start state
        start_state = env.state_to_index(env.config.start_state)
        performance = pi.evaluate_policy_performance(start_state, max_steps=50)

        assert isinstance(performance, float)
        # Should be able to reach goal, so return should be reasonable
        assert performance > -10.0  # Not too negative

    def test_deterministic_environment(self):
        """Test on deterministic environment."""
        config = GridWorldConfig(height=3, width=3, step_reward=-1.0)
        env = GridWorldEnv(config, stochastic=False)

        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = pi.solve()

        assert result['converged'] is True

        # Check that transition probabilities are deterministic
        assert torch.allclose(torch.max(env.P, dim=2)[0], torch.ones(env.n_states, env.n_actions))

    def test_stochastic_environment(self):
        """Test on stochastic environment."""
        config = GridWorldConfig(height=3, width=3)
        env = GridWorldEnv(config, stochastic=True, noise_prob=0.2)

        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = pi.solve()

        assert result['converged'] is True

        # Check that some transitions are stochastic
        max_probs = torch.max(env.P, dim=2)[0]
        assert not torch.allclose(max_probs, torch.ones(env.n_states, env.n_actions))

    def test_terminal_states_handling(self):
        """Test handling of terminal states."""
        config = GridWorldConfig(
            height=3, width=3,
            goal_states=[(2, 2)],
            obstacles=[(1, 1)]
        )
        env = GridWorldEnv(config)

        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        pi.solve()

        # Terminal states should have self-loops
        goal_state = env.state_to_index((2, 2))
        obstacle_state = env.state_to_index((1, 1))

        for a in range(env.n_actions):
            assert env.P[goal_state, a, goal_state] == 1.0
            assert env.P[obstacle_state, a, obstacle_state] == 1.0

    def test_different_discount_factors(self):
        """Test with different discount factors."""
        env = create_simple_gridworld()

        for gamma in [0.5, 0.9, 0.99]:
            pi = PolicyIteration(
                env.n_states, env.n_actions,
                env.get_transition_probabilities(),
                env.get_reward_tensor(),
                gamma=gamma
            )

            result = pi.solve()
            assert result['converged'] is True
            assert pi.gamma == gamma

    def test_reset_functionality(self):
        """Test algorithm reset."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        # Solve algorithm
        pi.solve()
        assert pi.converged is True
        assert pi.n_policy_evaluations > 0

        # Reset
        pi.reset()

        assert pi.converged is False
        assert pi.n_policy_evaluations == 0
        assert pi.n_policy_improvements == 0
        assert len(pi.evaluation_history) == 0
        assert len(pi.policy_stable_history) == 0
        assert torch.allclose(pi.policy, torch.ones(env.n_states, env.n_actions) / env.n_actions)
        assert torch.allclose(pi.state_values, torch.zeros(env.n_states))

    def test_device_handling(self):
        """Test device handling for tensors."""
        env = create_simple_gridworld()
        device = torch.device('cpu')

        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            device=device
        )

        assert pi.device == device
        assert pi.P.device == device
        assert pi.R.device == device
        assert pi.policy.device == device
        assert pi.state_values.device == device

    def test_policy_ties_handling(self):
        """Test handling of ties in policy improvement."""
        # Create environment where multiple actions are equally good
        n_states, n_actions = 4, 4
        P = torch.zeros((n_states, n_actions, n_states))
        R = torch.zeros((n_states, n_actions, n_states))

        # All actions lead to same reward
        for s in range(n_states):
            for a in range(n_actions):
                next_s = (s + 1) % n_states
                P[s, a, next_s] = 1.0
                R[s, a, next_s] = 1.0

        pi = PolicyIteration(n_states, n_actions, P, R)
        pi.solve()

        # Policy should handle ties by uniform distribution
        for s in range(n_states):
            policy_sum = torch.sum(pi.policy[s])
            assert torch.isclose(policy_sum, torch.tensor(1.0))

    @pytest.mark.parametrize("height,width", [(2, 2), (3, 4), (5, 3)])
    def test_different_grid_sizes(self, height, width):
        """Test algorithm on different grid sizes."""
        config = GridWorldConfig(height=height, width=width)
        env = GridWorldEnv(config)

        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = pi.solve()

        assert result['converged'] is True
        assert pi.n_states == height * width

    @pytest.mark.parametrize("theta", [1e-4, 1e-6, 1e-8])
    def test_different_convergence_thresholds(self, theta):
        """Test different convergence thresholds."""
        env = create_simple_gridworld()

        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=theta
        )

        result = pi.solve()

        assert pi.theta == theta
        # Stricter thresholds may require more iterations
        if theta == 1e-8:
            assert result['iterations'] >= 1

    def test_repr_string(self):
        """Test string representation."""
        env = create_simple_gridworld()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            gamma=0.95
        )

        repr_str = repr(pi)

        assert "PolicyIteration" in repr_str
        assert f"n_states={env.n_states}" in repr_str
        assert f"n_actions={env.n_actions}" in repr_str
        assert "gamma=0.95" in repr_str
        assert "converged=False" in repr_str

    def test_cliff_world_example(self):
        """Test on cliff world environment."""
        from learnrl.utils.gridworld_env import create_cliff_world

        env = create_cliff_world()
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = pi.solve(max_iterations=100)

        assert result['converged'] is True

        # Check that optimal policy avoids cliff
        start_state = env.state_to_index(env.config.start_state)
        action = pi.get_action(start_state)

        # Should not immediately go into cliff
        next_state_coord = env.get_next_state(env.config.start_state, action)
        assert next_state_coord not in env.config.obstacles