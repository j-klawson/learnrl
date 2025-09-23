"""GridWorld environment for testing Dynamic Programming algorithms."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


@dataclass
class GridWorldConfig:
    """Configuration for GridWorld environment."""

    height: int = 4
    width: int = 4
    start_state: Tuple[int, int] = (0, 0)
    goal_states: Optional[List[Tuple[int, int]]] = None
    obstacles: Optional[List[Tuple[int, int]]] = None
    step_reward: float = -0.1
    goal_reward: float = 1.0
    obstacle_reward: float = -1.0
    discount_factor: float = 0.9

    def __post_init__(self) -> None:
        if self.goal_states is None:
            self.goal_states = [(self.height - 1, self.width - 1)]
        if self.obstacles is None:
            self.obstacles = []


class GridWorldEnv:
    """
    Simple GridWorld environment for testing Dynamic Programming algorithms.

    This environment follows the GridWorld setup from Sutton & Barto Chapter 4.
    The agent navigates a grid to reach goal states while avoiding obstacles.
    """

    def __init__(
        self,
        config: Optional[GridWorldConfig] = None,
        stochastic: bool = False,
        noise_prob: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize GridWorld environment.

        Args:
            config: GridWorld configuration
            stochastic: Whether transitions are stochastic
            noise_prob: Probability of random action (if stochastic)
            device: PyTorch device for tensors
        """
        self.config = config or GridWorldConfig()
        self.stochastic = stochastic
        self.noise_prob = noise_prob
        self.device = device

        # Set up state and action spaces
        self.height = self.config.height
        self.width = self.config.width
        self.n_states = self.height * self.width
        self.n_actions = 4  # Up, Right, Down, Left

        # Define action directions
        self.actions = {
            0: (-1, 0),  # Up
            1: (0, 1),  # Right
            2: (1, 0),  # Down
            3: (0, -1),  # Left
        }

        # Current state
        self.current_state = self.config.start_state

        # Create Gymnasium spaces if available
        if HAS_GYMNASIUM:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(self.n_actions)

        # Pre-compute transition probabilities and rewards
        self._build_transition_model()

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to linear index."""
        row, col = state
        return row * self.width + col

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert linear index to (row, col) state."""
        row = index // self.width
        col = index % self.width
        return (row, col)

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is within grid bounds."""
        row, col = state
        return 0 <= row < self.height and 0 <= col < self.width

    def is_terminal_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is a terminal state (goal or obstacle)."""
        return (
            state in (self.config.goal_states or []) or
            state in (self.config.obstacles or [])
        )

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state given current state and action."""
        if self.is_terminal_state(state):
            return state

        row, col = state
        d_row, d_col = self.actions[action]
        next_state = (row + d_row, col + d_col)

        # If next state is invalid, stay in current state
        if not self.is_valid_state(next_state):
            return state

        return next_state

    def get_reward(
        self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]
    ) -> float:
        """Get reward for transition."""
        if next_state in (self.config.goal_states or []):
            return self.config.goal_reward
        elif next_state in (self.config.obstacles or []):
            return self.config.obstacle_reward
        else:
            return self.config.step_reward

    def _build_transition_model(self) -> None:
        """Build transition probability and reward tensors."""
        # P[s, a, s'] = probability of transitioning from s to s' with action a
        self.P = torch.zeros(
            (self.n_states, self.n_actions, self.n_states),
            device=self.device,
            dtype=torch.float32,
        )

        # R[s, a, s'] = reward for transitioning from s to s' with action a
        self.R = torch.zeros(
            (self.n_states, self.n_actions, self.n_states),
            device=self.device,
            dtype=torch.float32,
        )

        for s in range(self.n_states):
            state = self.index_to_state(s)

            if self.is_terminal_state(state):
                # Terminal states transition to themselves with 0 reward
                for a in range(self.n_actions):
                    self.P[s, a, s] = 1.0
                    self.R[s, a, s] = 0.0
            else:
                for a in range(self.n_actions):
                    if self.stochastic:
                        # Stochastic transitions
                        for actual_action in range(self.n_actions):
                            next_state = self.get_next_state(state, actual_action)
                            next_s = self.state_to_index(next_state)
                            reward = self.get_reward(state, actual_action, next_state)

                            if actual_action == a:
                                # Intended action
                                prob = (
                                    1.0
                                    - self.noise_prob
                                    + self.noise_prob / self.n_actions
                                )
                            else:
                                # Unintended action
                                prob = self.noise_prob / self.n_actions

                            self.P[s, a, next_s] += prob
                            self.R[s, a, next_s] = reward
                    else:
                        # Deterministic transitions
                        next_state = self.get_next_state(state, a)
                        next_s = self.state_to_index(next_state)
                        reward = self.get_reward(state, a, next_state)

                        self.P[s, a, next_s] = 1.0
                        self.R[s, a, next_s] = reward

    def reset(self) -> Tuple[int, Dict[str, Any]]:
        """Reset environment to start state."""
        self.current_state = self.config.start_state
        obs = self.state_to_index(self.current_state)
        info = {"state": self.current_state}
        return obs, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.is_terminal_state(self.current_state):
            # Already in terminal state
            obs = self.state_to_index(self.current_state)
            return obs, 0.0, True, False, {"state": self.current_state}

        # Determine actual action (for stochastic environments)
        if self.stochastic and np.random.random() < self.noise_prob:
            actual_action = np.random.randint(self.n_actions)
        else:
            actual_action = action

        next_state = self.get_next_state(self.current_state, actual_action)
        reward = self.get_reward(self.current_state, actual_action, next_state)

        self.current_state = next_state
        terminated = self.is_terminal_state(next_state)

        obs = self.state_to_index(self.current_state)
        info = {"state": self.current_state}

        return obs, reward, terminated, False, info

    def render(
        self, mode: str = "ascii", policy: Optional[torch.Tensor] = None
    ) -> Optional[str]:
        """Render the environment."""
        if mode == "ascii":
            return self._render_ascii(policy)
        elif mode == "matplotlib":
            return self._render_matplotlib(policy)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _render_ascii(self, policy: Optional[torch.Tensor] = None) -> str:
        """Render environment as ASCII text."""
        grid = []
        action_chars = ["^", ">", "v", "<"]  # Up, Right, Down, Left

        for row in range(self.height):
            row_str = ""
            for col in range(self.width):
                state = (row, col)

                if state == self.current_state:
                    char = "A"  # Agent
                elif state in (self.config.goal_states or []):
                    char = "G"  # Goal
                elif state in (self.config.obstacles or []):
                    char = "X"  # Obstacle
                elif policy is not None:
                    # Show policy
                    s = self.state_to_index(state)
                    action = int(torch.argmax(policy[s]).item())
                    char = action_chars[action]
                else:
                    char = "."  # Empty

                row_str += char + " "
            grid.append(row_str)

        return "\n".join(grid)

    def _render_matplotlib(self, policy: Optional[torch.Tensor] = None) -> Any:
        """Render environment using matplotlib."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Create grid
        grid = np.zeros((self.height, self.width))

        # Mark special states
        for goal in (self.config.goal_states or []):
            grid[goal[0], goal[1]] = 1  # Goal states

        for obstacle in (self.config.obstacles or []):
            grid[obstacle[0], obstacle[1]] = -1  # Obstacles

        # Plot grid
        im = ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)

        # Add agent position
        agent_row, agent_col = self.current_state
        ax.plot(agent_col, agent_row, "bo", markersize=15, label="Agent")

        # Add policy arrows if provided
        if policy is not None:
            action_dirs = [
                (-0.3, 0),
                (0, 0.3),
                (0.3, 0),
                (0, -0.3),
            ]  # Up, Right, Down, Left

            for row in range(self.height):
                for col in range(self.width):
                    state = (row, col)
                    if not self.is_terminal_state(state):
                        s = self.state_to_index(state)
                        action = int(torch.argmax(policy[s]).item())
                        dx, dy = action_dirs[action]
                        ax.arrow(
                            col,
                            row,
                            dy,
                            dx,
                            head_width=0.1,
                            head_length=0.1,
                            fc="black",
                            ec="black",
                        )

        # Formatting
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True)
        ax.legend()
        ax.set_title("GridWorld Environment")

        return fig

    def get_transition_probabilities(self) -> torch.Tensor:
        """Get transition probability tensor P[s, a, s']."""
        return self.P.clone()

    def get_reward_tensor(self) -> torch.Tensor:
        """Get reward tensor R[s, a, s']."""
        return self.R.clone()

    def get_state_values_tensor(self, state_values: torch.Tensor) -> torch.Tensor:
        """Convert state values to grid format for visualization."""
        grid = torch.zeros((self.height, self.width), device=self.device)
        for s in range(self.n_states):
            row, col = self.index_to_state(s)
            grid[row, col] = state_values[s]
        return grid

    def __repr__(self) -> str:
        return (
            f"GridWorldEnv({self.height}x{self.width}, "
            f"stochastic={self.stochastic}, "
            f"goals={self.config.goal_states}, "
            f"obstacles={self.config.obstacles})"
        )


def create_simple_gridworld() -> GridWorldEnv:
    """Create a simple 4x4 GridWorld for testing."""
    config = GridWorldConfig(
        height=4,
        width=4,
        start_state=(0, 0),
        goal_states=[(3, 3)],
        obstacles=[(1, 1), (2, 2)],
        step_reward=-0.1,
        goal_reward=1.0,
        obstacle_reward=-1.0,
    )
    return GridWorldEnv(config)


def create_cliff_world() -> GridWorldEnv:
    """Create the cliff walking problem from Sutton & Barto."""
    config = GridWorldConfig(
        height=4,
        width=12,
        start_state=(3, 0),
        goal_states=[(3, 11)],
        obstacles=[(3, i) for i in range(1, 11)],  # Cliff
        step_reward=-1.0,
        goal_reward=0.0,
        obstacle_reward=-100.0,
    )
    return GridWorldEnv(config)
