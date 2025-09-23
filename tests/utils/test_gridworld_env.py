"""Tests for GridWorldEnv."""

import pytest
import torch
import numpy as np

from learnrl.utils.gridworld_env import (
    GridWorldEnv, GridWorldConfig, create_simple_gridworld, create_cliff_world
)


class TestGridWorldConfig:
    """Test cases for GridWorldConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GridWorldConfig()

        assert config.height == 4
        assert config.width == 4
        assert config.start_state == (0, 0)
        assert config.goal_states == [(3, 3)]
        assert config.obstacles == []
        assert config.step_reward == -0.1
        assert config.goal_reward == 1.0
        assert config.obstacle_reward == -1.0
        assert config.discount_factor == 0.9

    def test_custom_config(self):
        """Test custom configuration."""
        config = GridWorldConfig(
            height=6,
            width=8,
            start_state=(1, 1),
            goal_states=[(5, 7), (4, 6)],
            obstacles=[(2, 2), (3, 3)],
            step_reward=-0.2,
            goal_reward=10.0,
            obstacle_reward=-5.0,
            discount_factor=0.95
        )

        assert config.height == 6
        assert config.width == 8
        assert config.start_state == (1, 1)
        assert config.goal_states == [(5, 7), (4, 6)]
        assert config.obstacles == [(2, 2), (3, 3)]
        assert config.step_reward == -0.2
        assert config.goal_reward == 10.0
        assert config.obstacle_reward == -5.0
        assert config.discount_factor == 0.95


class TestGridWorldEnv:
    """Test cases for GridWorldEnv."""

    def test_basic_initialization(self):
        """Test basic environment initialization."""
        env = GridWorldEnv()

        assert env.height == 4
        assert env.width == 4
        assert env.n_states == 16
        assert env.n_actions == 4
        assert env.current_state == (0, 0)
        assert not env.stochastic

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = GridWorldConfig(height=3, width=5)
        env = GridWorldEnv(config)

        assert env.height == 3
        assert env.width == 5
        assert env.n_states == 15
        assert env.n_actions == 4

    def test_stochastic_initialization(self):
        """Test stochastic environment initialization."""
        env = GridWorldEnv(stochastic=True, noise_prob=0.2)

        assert env.stochastic is True
        assert env.noise_prob == 0.2

    def test_state_index_conversion(self):
        """Test state to index conversion."""
        env = GridWorldEnv()

        # Test state to index
        assert env.state_to_index((0, 0)) == 0
        assert env.state_to_index((0, 3)) == 3
        assert env.state_to_index((1, 0)) == 4
        assert env.state_to_index((3, 3)) == 15

        # Test index to state
        assert env.index_to_state(0) == (0, 0)
        assert env.index_to_state(3) == (0, 3)
        assert env.index_to_state(4) == (1, 0)
        assert env.index_to_state(15) == (3, 3)

    def test_state_validation(self):
        """Test state validation."""
        env = GridWorldEnv()

        # Valid states
        assert env.is_valid_state((0, 0)) is True
        assert env.is_valid_state((3, 3)) is True
        assert env.is_valid_state((2, 1)) is True

        # Invalid states
        assert env.is_valid_state((-1, 0)) is False
        assert env.is_valid_state((0, -1)) is False
        assert env.is_valid_state((4, 0)) is False
        assert env.is_valid_state((0, 4)) is False

    def test_terminal_states(self):
        """Test terminal state identification."""
        config = GridWorldConfig(
            goal_states=[(3, 3)],
            obstacles=[(1, 1), (2, 2)]
        )
        env = GridWorldEnv(config)

        # Terminal states
        assert env.is_terminal_state((3, 3)) is True  # Goal
        assert env.is_terminal_state((1, 1)) is True  # Obstacle
        assert env.is_terminal_state((2, 2)) is True  # Obstacle

        # Non-terminal states
        assert env.is_terminal_state((0, 0)) is False
        assert env.is_terminal_state((2, 1)) is False

    def test_next_state_computation(self):
        """Test next state computation."""
        env = GridWorldEnv()

        # Normal moves
        assert env.get_next_state((1, 1), 0) == (0, 1)  # Up
        assert env.get_next_state((1, 1), 1) == (1, 2)  # Right
        assert env.get_next_state((1, 1), 2) == (2, 1)  # Down
        assert env.get_next_state((1, 1), 3) == (1, 0)  # Left

        # Boundary conditions (stay in place)
        assert env.get_next_state((0, 0), 0) == (0, 0)  # Up from top
        assert env.get_next_state((0, 0), 3) == (0, 0)  # Left from left edge
        assert env.get_next_state((3, 3), 1) == (3, 3)  # Right from right edge
        assert env.get_next_state((3, 3), 2) == (3, 3)  # Down from bottom

    def test_reward_computation(self):
        """Test reward computation."""
        config = GridWorldConfig(
            goal_states=[(3, 3)],
            obstacles=[(1, 1)],
            step_reward=-0.1,
            goal_reward=1.0,
            obstacle_reward=-1.0
        )
        env = GridWorldEnv(config)

        # Step reward
        assert env.get_reward((0, 0), 1, (0, 1)) == -0.1

        # Goal reward
        assert env.get_reward((3, 2), 1, (3, 3)) == 1.0

        # Obstacle reward
        assert env.get_reward((1, 0), 1, (1, 1)) == -1.0

    def test_transition_model_shape(self):
        """Test transition model tensor shapes."""
        env = GridWorldEnv()

        assert env.P.shape == (16, 4, 16)  # [n_states, n_actions, n_states]
        assert env.R.shape == (16, 4, 16)  # [n_states, n_actions, n_states]

    def test_transition_probabilities_sum(self):
        """Test that transition probabilities sum to 1."""
        env = GridWorldEnv()

        # For each state-action pair, probabilities should sum to 1
        for s in range(env.n_states):
            for a in range(env.n_actions):
                prob_sum = torch.sum(env.P[s, a, :])
                assert torch.isclose(prob_sum, torch.tensor(1.0)), f"State {s}, Action {a}: {prob_sum}"

    def test_deterministic_transitions(self):
        """Test deterministic environment transitions."""
        env = GridWorldEnv(stochastic=False)

        # Test a specific transition
        s = env.state_to_index((1, 1))  # Middle state
        a = 1  # Right action
        next_s = env.state_to_index((1, 2))  # Expected next state

        # Should have probability 1 for correct transition, 0 for others
        assert env.P[s, a, next_s] == 1.0
        assert torch.sum(env.P[s, a, :]) == 1.0

    def test_stochastic_transitions(self):
        """Test stochastic environment transitions."""
        env = GridWorldEnv(stochastic=True, noise_prob=0.2)

        s = env.state_to_index((1, 1))  # Middle state
        a = 1  # Right action

        # Check that probabilities sum to 1
        assert torch.isclose(torch.sum(env.P[s, a, :]), torch.tensor(1.0))

        # Intended action should have higher probability
        next_s_intended = env.state_to_index((1, 2))
        intended_prob = env.P[s, a, next_s_intended]
        assert intended_prob > 0.2  # Should be higher than noise probability

    def test_reset_functionality(self):
        """Test environment reset."""
        config = GridWorldConfig(start_state=(2, 1))
        env = GridWorldEnv(config)

        # Move to different state
        env.current_state = (3, 3)

        # Reset should return to start state
        obs, info = env.reset()
        assert env.current_state == (2, 1)
        assert obs == env.state_to_index((2, 1))
        assert info["state"] == (2, 1)

    def test_step_functionality(self):
        """Test environment step function."""
        env = GridWorldEnv(stochastic=False)
        env.current_state = (1, 1)

        # Take action
        obs, reward, terminated, truncated, info = env.step(1)  # Right

        assert env.current_state == (1, 2)
        assert obs == env.state_to_index((1, 2))
        assert reward == env.config.step_reward
        assert terminated is False
        assert truncated is False
        assert info["state"] == (1, 2)

    def test_step_to_goal(self):
        """Test stepping to goal state."""
        config = GridWorldConfig(goal_states=[(1, 2)])
        env = GridWorldEnv(config, stochastic=False)
        env.current_state = (1, 1)

        # Step to goal
        obs, reward, terminated, truncated, info = env.step(1)  # Right

        assert env.current_state == (1, 2)
        assert reward == env.config.goal_reward
        assert terminated is True

    def test_step_from_terminal(self):
        """Test stepping from terminal state."""
        config = GridWorldConfig(goal_states=[(1, 1)])
        env = GridWorldEnv(config, stochastic=False)
        env.current_state = (1, 1)  # Start in terminal state

        # Step should stay in terminal state
        obs, reward, terminated, truncated, info = env.step(1)

        assert env.current_state == (1, 1)
        assert terminated is True
        assert reward == 0.0

    def test_render_ascii(self):
        """Test ASCII rendering."""
        config = GridWorldConfig(
            height=3,
            width=3,
            start_state=(0, 0),
            goal_states=[(2, 2)],
            obstacles=[(1, 1)]
        )
        env = GridWorldEnv(config)

        # Test basic rendering
        rendered = env.render(mode="ascii")
        assert isinstance(rendered, str)
        assert "A" in rendered  # Agent
        assert "G" in rendered  # Goal
        assert "X" in rendered  # Obstacle

    def test_render_with_policy(self):
        """Test rendering with policy."""
        env = GridWorldEnv()

        # Create dummy policy (all actions point right)
        policy = torch.zeros((env.n_states, env.n_actions))
        policy[:, 1] = 1.0  # All actions are "right"

        rendered = env.render(mode="ascii", policy=policy)
        assert isinstance(rendered, str)
        assert ">" in rendered  # Right arrows

    def test_tensor_methods(self):
        """Test tensor utility methods."""
        env = GridWorldEnv()

        # Test transition probability access
        P = env.get_transition_probabilities()
        assert P.shape == (16, 4, 16)
        assert torch.allclose(P, env.P)

        # Test reward tensor access
        R = env.get_reward_tensor()
        assert R.shape == (16, 4, 16)
        assert torch.allclose(R, env.R)

        # Test state values to grid conversion
        state_values = torch.randn(16)
        grid = env.get_state_values_tensor(state_values)
        assert grid.shape == (4, 4)

    def test_create_simple_gridworld(self):
        """Test simple gridworld creation."""
        env = create_simple_gridworld()

        assert env.height == 4
        assert env.width == 4
        assert env.config.start_state == (0, 0)
        assert env.config.goal_states == [(3, 3)]
        assert (1, 1) in env.config.obstacles
        assert (2, 2) in env.config.obstacles

    def test_create_cliff_world(self):
        """Test cliff world creation."""
        env = create_cliff_world()

        assert env.height == 4
        assert env.width == 12
        assert env.config.start_state == (3, 0)
        assert env.config.goal_states == [(3, 11)]
        assert len(env.config.obstacles) == 10  # Cliff states

    @pytest.mark.parametrize("height,width", [(3, 3), (5, 4), (2, 6)])
    def test_different_grid_sizes(self, height, width):
        """Test different grid sizes."""
        config = GridWorldConfig(height=height, width=width)
        env = GridWorldEnv(config)

        assert env.height == height
        assert env.width == width
        assert env.n_states == height * width
        assert env.P.shape == (height * width, 4, height * width)

    @pytest.mark.parametrize("noise_prob", [0.0, 0.1, 0.3, 0.5])
    def test_different_noise_levels(self, noise_prob):
        """Test different noise probability levels."""
        env = GridWorldEnv(stochastic=True, noise_prob=noise_prob)

        assert env.noise_prob == noise_prob

        # Test transition probabilities
        s = env.state_to_index((1, 1))
        for a in range(env.n_actions):
            prob_sum = torch.sum(env.P[s, a, :])
            assert torch.isclose(prob_sum, torch.tensor(1.0))

    def test_repr_string(self):
        """Test string representation."""
        config = GridWorldConfig(
            height=5,
            width=6,
            goal_states=[(4, 5)],
            obstacles=[(2, 2)]
        )
        env = GridWorldEnv(config, stochastic=True)

        repr_str = repr(env)
        assert "GridWorldEnv" in repr_str
        assert "5x6" in repr_str
        assert "stochastic=True" in repr_str