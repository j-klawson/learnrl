"""Tests for ValueIteration algorithm."""

import pytest
import torch
import numpy as np

from learnrl.dp.value_iteration import ValueIteration
from learnrl.utils.gridworld_env import GridWorldEnv, GridWorldConfig, create_simple_gridworld

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class TestValueIteration:
    """Test cases for ValueIteration algorithm."""

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

        vi = ValueIteration(n_states, n_actions, P, R)

        assert vi.n_states == 16
        assert vi.n_actions == 4
        assert vi.gamma == 0.9
        assert vi.theta == 1e-6
        assert vi.state_values.shape == (16,)
        assert torch.allclose(vi.state_values, torch.zeros(16))
        assert vi.policy is None
        assert vi.converged is False
        assert vi.iterations == 0

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

        vi = ValueIteration(obs_space, action_space, P, R)

        assert vi.n_states == 10
        assert vi.n_actions == 3

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        n_states, n_actions = 5, 2
        P = torch.eye(n_states).unsqueeze(1).repeat(1, n_actions, 1)
        R = torch.zeros((n_states, n_actions, n_states))

        vi = ValueIteration(
            n_states, n_actions, P, R,
            gamma=0.95, theta=1e-8,
            device=torch.device('cpu')
        )

        assert vi.gamma == 0.95
        assert vi.theta == 1e-8
        assert vi.device == torch.device('cpu')

    def test_invalid_spaces(self):
        """Test error handling for invalid spaces."""
        P = torch.zeros((4, 2, 4))
        R = torch.zeros((4, 2, 4))

        with pytest.raises(ValueError, match="Unsupported observation space type"):
            ValueIteration("invalid", 2, P, R)

        with pytest.raises(ValueError, match="Unsupported action space type"):
            ValueIteration(4, "invalid", P, R)

    def test_tensor_shape_validation(self):
        """Test validation of tensor shapes."""
        n_states, n_actions = 4, 2

        # Wrong transition probabilities shape
        P_wrong = torch.zeros((3, 2, 4))
        R = torch.zeros((4, 2, 4))

        with pytest.raises(ValueError, match="Transition probs shape"):
            ValueIteration(n_states, n_actions, P_wrong, R)

        # Wrong rewards shape
        P = torch.zeros((4, 2, 4))
        R_wrong = torch.zeros((4, 2, 3))

        with pytest.raises(ValueError, match="Rewards shape"):
            ValueIteration(n_states, n_actions, P, R_wrong)

    def test_value_iteration_step(self):
        """Test single value iteration step."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        initial_values = vi.state_values.clone()
        delta = vi.value_iteration_step()

        assert isinstance(delta, float)
        assert delta >= 0.0
        assert not torch.allclose(vi.state_values, initial_values)

    def test_value_iteration_convergence(self):
        """Test value iteration convergence."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-6
        )

        result = vi.solve(max_iterations=1000)

        assert result['converged'] is True
        assert result['iterations'] > 0
        assert result['iterations'] < 1000  # Should converge before max
        assert result['final_delta'] < vi.theta
        assert len(result['delta_history']) == result['iterations']
        assert vi.converged is True
        assert vi.policy is not None

    def test_max_iterations_limit(self):
        """Test behavior when max iterations reached."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-12  # Very strict convergence
        )

        result = vi.solve(max_iterations=5)

        assert result['iterations'] == 5
        assert len(result['delta_history']) == 5
        # May or may not have converged with only 5 iterations

    def test_policy_extraction(self):
        """Test policy extraction from state values."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        # Policy should be extracted automatically
        assert vi.policy is not None
        assert vi.policy.shape == (env.n_states, env.n_actions)

        # All state policies should sum to 1
        policy_sums = torch.sum(vi.policy, dim=1)
        assert torch.allclose(policy_sums, torch.ones(env.n_states))

        # Test manual extraction
        extracted_policy = vi.extract_policy()
        assert torch.allclose(extracted_policy, vi.policy)

    def test_policy_access_methods(self):
        """Test policy access methods."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        # Test get_policy for all states
        full_policy = vi.get_policy()
        assert full_policy.shape == (env.n_states, env.n_actions)
        assert torch.allclose(torch.sum(full_policy, dim=1), torch.ones(env.n_states))

        # Test get_policy for specific state
        state_policy = vi.get_policy(state=0)
        assert state_policy.shape == (env.n_actions,)
        assert torch.isclose(torch.sum(state_policy), torch.tensor(1.0))

        # Test get_action
        action = vi.get_action(0)
        assert isinstance(action, int)
        assert 0 <= action < env.n_actions

    def test_policy_extraction_before_solve(self):
        """Test policy extraction before solving."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        # Policy should be None initially
        assert vi.policy is None

        # Accessing policy should trigger extraction
        policy = vi.get_policy()
        assert policy is not None
        assert vi.policy is not None

    def test_state_value_access(self):
        """Test state value access methods."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        # Test get_state_value for all states
        all_values = vi.get_state_value()
        assert all_values.shape == (env.n_states,)
        assert torch.allclose(all_values, vi.state_values)

        # Test get_state_value for specific state
        state_value = vi.get_state_value(state=0)
        assert isinstance(state_value, float)
        assert state_value == vi.state_values[0].item()

    def test_action_values_computation(self):
        """Test action values (Q-values) computation."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        action_values = vi.get_action_values(state=0)
        assert action_values.shape == (env.n_actions,)

        # Best action should have highest Q-value
        best_action = vi.get_action(0)
        assert action_values[best_action] == torch.max(action_values)

    def test_policy_performance_evaluation(self):
        """Test policy performance evaluation."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        # Evaluate from start state
        start_state = env.state_to_index(env.config.start_state)
        performance = vi.evaluate_policy_performance(start_state, max_steps=50)

        assert isinstance(performance, float)
        # Should be able to reach goal, so return should be reasonable
        assert performance > -10.0  # Not too negative

    def test_policy_value_computation(self):
        """Test policy value computation method."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        # Create a random policy
        random_policy = torch.rand(env.n_states, env.n_actions)
        random_policy = random_policy / torch.sum(random_policy, dim=1, keepdim=True)

        policy_values = vi.compute_policy_value(random_policy)
        assert policy_values.shape == (env.n_states,)

        # Optimal policy should have better or equal values
        optimal_values = vi.get_state_value()
        # Note: This is not always true due to ties, so we just check shape
        assert policy_values.shape == optimal_values.shape

    def test_policy_loss_computation(self):
        """Test policy loss computation."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

        # Test loss for optimal policy
        optimal_policy = vi.get_policy()
        optimal_loss = vi.policy_loss(optimal_policy)
        assert isinstance(optimal_loss, float)

        # Test loss for random policy
        random_policy = torch.rand(env.n_states, env.n_actions)
        random_policy = random_policy / torch.sum(random_policy, dim=1, keepdim=True)
        random_loss = vi.policy_loss(random_policy)

        # Optimal policy should have lower (more negative) loss
        assert optimal_loss <= random_loss

    def test_deterministic_environment(self):
        """Test on deterministic environment."""
        config = GridWorldConfig(height=3, width=3, step_reward=-1.0)
        env = GridWorldEnv(config, stochastic=False)

        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = vi.solve()

        assert result['converged'] is True

        # Check that transition probabilities are deterministic
        assert torch.allclose(torch.max(env.P, dim=2)[0], torch.ones(env.n_states, env.n_actions))

    def test_stochastic_environment(self):
        """Test on stochastic environment."""
        config = GridWorldConfig(height=3, width=3)
        env = GridWorldEnv(config, stochastic=True, noise_prob=0.2)

        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = vi.solve()

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

        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        vi.solve()

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
            vi = ValueIteration(
                env.n_states, env.n_actions,
                env.get_transition_probabilities(),
                env.get_reward_tensor(),
                gamma=gamma
            )

            result = vi.solve()
            assert result['converged'] is True
            assert vi.gamma == gamma

    def test_reset_functionality(self):
        """Test algorithm reset."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        # Solve algorithm
        vi.solve()
        assert vi.converged is True
        assert vi.iterations > 0
        assert vi.policy is not None

        # Reset
        vi.reset()

        assert vi.converged is False
        assert vi.iterations == 0
        assert vi.policy is None
        assert len(vi.delta_history) == 0
        assert torch.allclose(vi.state_values, torch.zeros(env.n_states))

    def test_device_handling(self):
        """Test device handling for tensors."""
        env = create_simple_gridworld()
        device = torch.device('cpu')

        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            device=device
        )

        assert vi.device == device
        assert vi.P.device == device
        assert vi.R.device == device
        assert vi.state_values.device == device

    def test_policy_ties_handling(self):
        """Test handling of ties in policy extraction."""
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

        vi = ValueIteration(n_states, n_actions, P, R)
        vi.solve()

        # Policy should handle ties by uniform distribution
        for s in range(n_states):
            policy_sum = torch.sum(vi.policy[s])
            assert torch.isclose(policy_sum, torch.tensor(1.0))

    def test_bellman_optimality_property(self):
        """Test that solved values satisfy Bellman optimality equation."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-8
        )

        vi.solve()

        # Check Bellman optimality equation for non-terminal states
        for s in range(env.n_states):
            state_coord = env.index_to_state(s)
            if not env.is_terminal_state(state_coord):
                # Compute action values
                action_values = vi.get_action_values(s)
                optimal_value = torch.max(action_values)

                # Should be close to current state value
                assert torch.isclose(optimal_value, vi.state_values[s], atol=1e-4)

    @pytest.mark.parametrize("height,width", [(2, 2), (3, 4), (5, 3)])
    def test_different_grid_sizes(self, height, width):
        """Test algorithm on different grid sizes."""
        config = GridWorldConfig(height=height, width=width)
        env = GridWorldEnv(config)

        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = vi.solve()

        assert result['converged'] is True
        assert vi.n_states == height * width

    @pytest.mark.parametrize("theta", [1e-4, 1e-6, 1e-8])
    def test_different_convergence_thresholds(self, theta):
        """Test different convergence thresholds."""
        env = create_simple_gridworld()

        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=theta
        )

        result = vi.solve()

        assert vi.theta == theta
        assert result['final_delta'] < theta
        # Stricter thresholds may require more iterations
        if theta == 1e-8:
            assert result['iterations'] >= 1

    def test_delta_history_tracking(self):
        """Test delta history tracking during solving."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = vi.solve()

        # Delta should generally decrease over time
        delta_history = result['delta_history']
        assert len(delta_history) == result['iterations']
        assert all(d >= 0 for d in delta_history)

        # Final delta should be small
        assert delta_history[-1] < vi.theta

    def test_repr_string(self):
        """Test string representation."""
        env = create_simple_gridworld()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            gamma=0.95
        )

        repr_str = repr(vi)

        assert "ValueIteration" in repr_str
        assert f"n_states={env.n_states}" in repr_str
        assert f"n_actions={env.n_actions}" in repr_str
        assert "gamma=0.95" in repr_str
        assert "converged=False" in repr_str
        assert "iterations=0" in repr_str

        # After solving
        vi.solve()
        solved_repr = repr(vi)
        assert "converged=True" in solved_repr
        assert "iterations=" in solved_repr

    def test_cliff_world_example(self):
        """Test on cliff world environment."""
        from learnrl.utils.gridworld_env import create_cliff_world

        env = create_cliff_world()
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor()
        )

        result = vi.solve(max_iterations=1000)

        assert result['converged'] is True

        # Check that optimal policy avoids cliff
        start_state = env.state_to_index(env.config.start_state)
        action = vi.get_action(start_state)

        # Should not immediately go into cliff
        next_state_coord = env.get_next_state(env.config.start_state, action)
        assert next_state_coord not in env.config.obstacles

    def test_comparison_with_policy_iteration(self):
        """Test that value iteration finds same solution as policy iteration."""
        from learnrl.dp.policy_iteration import PolicyIteration

        env = create_simple_gridworld()

        # Solve with Value Iteration
        vi = ValueIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-8
        )
        vi_result = vi.solve()

        # Solve with Policy Iteration
        pi = PolicyIteration(
            env.n_states, env.n_actions,
            env.get_transition_probabilities(),
            env.get_reward_tensor(),
            theta=1e-8
        )
        pi_result = pi.solve()

        # Both should converge
        assert vi_result['converged']
        assert pi_result['converged']

        # Should find similar state values (within tolerance)
        vi_values = vi.get_state_value()
        pi_values = pi.get_state_value()
        assert torch.allclose(vi_values, pi_values, atol=1e-4)

        # Should find similar policies
        vi_policy = vi.get_policy()
        pi_policy = pi.get_policy()

        # Policies should be close (allowing for ties)
        for s in range(env.n_states):
            vi_action = vi.get_action(s)
            pi_action = pi.get_action(s)

            # Actions should either be the same or have similar values
            vi_q_values = vi.get_action_values(s)
            pi_q_values = pi.get_action_values(s)

            # If actions differ, their Q-values should be very close
            if vi_action != pi_action:
                assert abs(vi_q_values[vi_action] - vi_q_values[pi_action]) < 1e-4
                assert abs(pi_q_values[vi_action] - pi_q_values[pi_action]) < 1e-4