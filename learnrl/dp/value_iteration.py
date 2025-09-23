"""Value Iteration algorithm for Dynamic Programming."""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class ValueIteration:
    """
    Value Iteration algorithm implementation.

    This implementation follows Algorithm 4.4 from Sutton & Barto (2020).
    Value iteration combines policy evaluation and policy improvement in a single step,
    updating state values using the Bellman optimality equation.
    """

    def __init__(
        self,
        observation_space: Union[int, "spaces.Space"],
        action_space: Union[int, "spaces.Space"],
        transition_probs: torch.Tensor,
        rewards: torch.Tensor,
        gamma: float = 0.9,
        theta: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize Value Iteration algorithm.

        Args:
            observation_space: Environment observation space or number of states
            action_space: Environment action space or number of actions
            transition_probs: Transition probabilities P[s, a, s']
            rewards: Reward tensor R[s, a, s']
            gamma: Discount factor (0 <= gamma <= 1)
            theta: Convergence threshold for value function
            device: PyTorch device for tensors
        """
        # Handle both Gymnasium spaces and integers
        if isinstance(observation_space, int):
            self.n_states = observation_space
        elif HAS_GYMNASIUM and hasattr(observation_space, "n"):
            self.n_states = observation_space.n
        else:
            raise ValueError("Unsupported observation space type")

        if isinstance(action_space, int):
            self.n_actions = action_space
        elif HAS_GYMNASIUM and hasattr(action_space, "n"):
            self.n_actions = action_space.n
        else:
            raise ValueError("Unsupported action space type")

        self.gamma = gamma
        self.theta = theta
        self.device = device

        # Store transition model
        self.P = transition_probs.to(device)  # P[s, a, s']
        self.R = rewards.to(device)  # R[s, a, s']

        # Validate tensor shapes
        expected_shape = (self.n_states, self.n_actions, self.n_states)
        if self.P.shape != expected_shape:
            raise ValueError(
                f"Transition probs shape {self.P.shape} != expected {expected_shape}"
            )
        if self.R.shape != expected_shape:
            raise ValueError(
                f"Rewards shape {self.R.shape} != expected {expected_shape}"
            )

        # Initialize state values
        self.state_values = torch.zeros(
            self.n_states, device=device, dtype=torch.float32
        )

        # Policy will be extracted from state values
        self.policy: Optional[torch.Tensor] = None

        # Track algorithm progress
        self.iterations = 0
        self.converged = False
        self.delta_history: List[float] = []

    def value_iteration_step(self) -> float:
        """
        Perform one step of value iteration.

        Returns:
            Maximum change in state values (delta)
        """
        delta = 0.0
        new_state_values = torch.zeros_like(self.state_values)

        for s in range(self.n_states):
            # Compute action values for current state
            action_values = torch.zeros(self.n_actions, device=self.device)

            for a in range(self.n_actions):
                action_values[a] = torch.sum(
                    self.P[s, a, :] * (self.R[s, a, :] + self.gamma * self.state_values)
                )

            # Take maximum over actions (Bellman optimality equation)
            max_value = torch.max(action_values)
            new_state_values[s] = max_value

            # Track maximum change
            delta = max(delta, abs(max_value.item() - self.state_values[s].item()))

        self.state_values = new_state_values
        return delta

    def solve(self, max_iterations: int = 1000) -> dict:
        """
        Run complete value iteration algorithm.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary with convergence information
        """
        self.converged = False
        self.delta_history = []

        for iteration in range(max_iterations):
            # Perform value iteration step
            delta = self.value_iteration_step()
            self.iterations = iteration + 1
            self.delta_history.append(delta)

            # Check convergence
            if delta < self.theta:
                self.converged = True
                break

        # Extract optimal policy
        self.extract_policy()

        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_delta": delta,
            "delta_history": self.delta_history,
        }

    def extract_policy(self) -> torch.Tensor:
        """
        Extract optimal policy from state values.

        Returns:
            Optimal policy tensor
        """
        self.policy = torch.zeros(
            (self.n_states, self.n_actions), device=self.device, dtype=torch.float32
        )

        for s in range(self.n_states):
            # Compute action values
            action_values = torch.zeros(self.n_actions, device=self.device)

            for a in range(self.n_actions):
                action_values[a] = torch.sum(
                    self.P[s, a, :] * (self.R[s, a, :] + self.gamma * self.state_values)
                )

            # Find best action(s) - handle ties by uniform distribution
            max_value = torch.max(action_values)
            best_actions = (action_values == max_value).float()
            self.policy[s] = best_actions / torch.sum(best_actions)

        return self.policy.clone()

    def get_policy(self, state: Optional[int] = None) -> torch.Tensor:
        """
        Get policy for a specific state or all states.

        Args:
            state: State index (if None, returns policy for all states)

        Returns:
            Policy tensor
        """
        if self.policy is None:
            self.extract_policy()

        if state is None:
            return self.policy.clone()  # type: ignore
        else:
            return self.policy[state].clone()  # type: ignore

    def get_action(self, state: int) -> int:
        """
        Get best action for a given state.

        Args:
            state: State index

        Returns:
            Best action index
        """
        if self.policy is None:
            self.extract_policy()

        return int(torch.argmax(self.policy[state]).item())  # type: ignore

    def get_state_value(
        self, state: Optional[int] = None
    ) -> Union[float, torch.Tensor]:
        """
        Get state value for a specific state or all states.

        Args:
            state: State index (if None, returns values for all states)

        Returns:
            State value(s)
        """
        if state is None:
            return self.state_values.clone()
        else:
            return self.state_values[state].item()

    def get_action_values(self, state: int) -> torch.Tensor:
        """
        Get action values (Q-values) for a given state.

        Args:
            state: State index

        Returns:
            Action values tensor
        """
        action_values = torch.zeros(self.n_actions, device=self.device)

        for a in range(self.n_actions):
            action_values[a] = torch.sum(
                self.P[state, a, :]
                * (self.R[state, a, :] + self.gamma * self.state_values)
            )

        return action_values

    def evaluate_policy_performance(
        self, start_state: int, max_steps: int = 100
    ) -> float:
        """
        Evaluate policy performance by simulating an episode.

        Args:
            start_state: Starting state
            max_steps: Maximum episode length

        Returns:
            Total discounted return
        """
        if self.policy is None:
            self.extract_policy()

        total_return = 0.0
        discount = 1.0
        state = start_state

        for step in range(max_steps):
            # Sample action from policy
            action_probs = self.policy[state]  # type: ignore
            action = int(torch.multinomial(action_probs, 1).item())

            # Sample next state
            transition_probs = self.P[state, action, :]
            next_state = int(torch.multinomial(transition_probs, 1).item())

            # Get reward
            reward = self.R[state, action, next_state].item()

            # Update return
            total_return += discount * reward
            discount *= self.gamma

            state = int(next_state)

            # Check for terminal state (self-loop with 0 reward)
            if torch.allclose(self.P[state, :, state], torch.ones(self.n_actions)):
                break

        return total_return

    def compute_policy_value(
        self, policy: torch.Tensor, max_iterations: int = 1000
    ) -> torch.Tensor:
        """
        Compute state values for a given policy using policy evaluation.

        Args:
            policy: Policy tensor [n_states, n_actions]
            max_iterations: Maximum evaluation iterations

        Returns:
            State values under the given policy
        """
        values = torch.zeros(self.n_states, device=self.device, dtype=torch.float32)

        for iteration in range(max_iterations):
            delta = 0.0
            new_values = torch.zeros_like(values)

            for s in range(self.n_states):
                v = 0.0
                for a in range(self.n_actions):
                    action_prob = policy[s, a]
                    action_value = torch.sum(
                        self.P[s, a, :] * (self.R[s, a, :] + self.gamma * values)
                    )
                    v += action_prob * action_value

                new_values[s] = v
                delta = max(delta, float(abs(v - values[s].item())))

            values = new_values

            if delta < self.theta:
                break

        return values

    def policy_loss(self, policy: torch.Tensor) -> float:
        """
        Compute the loss (negative value) of a given policy.

        Args:
            policy: Policy tensor

        Returns:
            Negative expected return under the policy
        """
        policy_values = self.compute_policy_value(policy)
        # Assume uniform start state distribution
        return -torch.mean(policy_values).item()

    def reset(self) -> None:
        """Reset algorithm to initial state."""
        self.state_values = torch.zeros(
            self.n_states, device=self.device, dtype=torch.float32
        )
        self.policy = None
        self.iterations = 0
        self.converged = False
        self.delta_history = []

    def __repr__(self) -> str:
        return (
            f"ValueIteration(n_states={self.n_states}, "
            f"n_actions={self.n_actions}, gamma={self.gamma}, "
            f"theta={self.theta}, converged={self.converged}, "
            f"iterations={self.iterations})"
        )
