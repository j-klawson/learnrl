"""Policy Iteration algorithm for Dynamic Programming."""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class PolicyIteration:
    """
    Policy Iteration algorithm implementation.

    This implementation follows Algorithm 4.3 from Sutton & Barto (2020).
    Policy iteration alternates between policy evaluation and policy improvement
    until convergence to the optimal policy.
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
        Initialize Policy Iteration algorithm.

        Args:
            observation_space: Environment observation space or number of states
            action_space: Environment action space or number of actions
            transition_probs: Transition probabilities P[s, a, s']
            rewards: Reward tensor R[s, a, s']
            gamma: Discount factor (0 <= gamma <= 1)
            theta: Convergence threshold for policy evaluation
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

        # Initialize policy (uniform random)
        self.policy = (
            torch.ones(
                (self.n_states, self.n_actions), device=device, dtype=torch.float32
            )
            / self.n_actions
        )

        # Initialize state values
        self.state_values = torch.zeros(
            self.n_states, device=device, dtype=torch.float32
        )

        # Track algorithm progress
        self.n_policy_evaluations = 0
        self.n_policy_improvements = 0
        self.converged = False
        self.evaluation_history: List[int] = []
        self.policy_stable_history: List[bool] = []

    def policy_evaluation(self, max_iterations: int = 1000) -> int:
        """
        Evaluate current policy using iterative policy evaluation.

        Args:
            max_iterations: Maximum number of evaluation iterations

        Returns:
            Number of iterations until convergence
        """
        iterations = 0

        for iteration in range(max_iterations):
            delta = 0.0
            new_state_values = torch.zeros_like(self.state_values)

            for s in range(self.n_states):
                # Compute expected value under current policy
                v = 0.0
                for a in range(self.n_actions):
                    action_prob = self.policy[s, a]
                    action_value = torch.sum(
                        self.P[s, a, :]
                        * (self.R[s, a, :] + self.gamma * self.state_values)
                    )
                    v += action_prob * action_value

                new_state_values[s] = v
                delta = max(delta, float(abs(v - self.state_values[s].item())))

            self.state_values = new_state_values
            iterations += 1

            # Check convergence
            if delta < self.theta:
                break

        self.n_policy_evaluations += 1
        self.evaluation_history.append(iterations)
        return iterations

    def policy_improvement(self) -> bool:
        """
        Improve policy using current state values.

        Returns:
            True if policy changed, False if policy is stable
        """
        policy_stable = True
        new_policy = torch.zeros_like(self.policy)

        for s in range(self.n_states):
            # Find best action under current state values
            action_values = torch.zeros(self.n_actions, device=self.device)

            for a in range(self.n_actions):
                action_values[a] = torch.sum(
                    self.P[s, a, :] * (self.R[s, a, :] + self.gamma * self.state_values)
                )

            # Get best action(s) - handle ties by uniform distribution
            max_value = torch.max(action_values)
            best_actions = (action_values == max_value).float()
            new_policy[s] = best_actions / torch.sum(best_actions)

            # Check if policy changed for this state
            if not torch.allclose(self.policy[s], new_policy[s], atol=1e-8):
                policy_stable = False

        self.policy = new_policy
        self.n_policy_improvements += 1
        self.policy_stable_history.append(policy_stable)

        return policy_stable

    def solve(self, max_iterations: int = 100, max_eval_iterations: int = 1000) -> dict:
        """
        Run complete policy iteration algorithm.

        Args:
            max_iterations: Maximum number of policy iteration steps
            max_eval_iterations: Maximum iterations per policy evaluation

        Returns:
            Dictionary with convergence information
        """
        self.converged = False

        for iteration in range(max_iterations):
            # Policy Evaluation
            eval_iterations = self.policy_evaluation(max_eval_iterations)

            # Policy Improvement
            policy_stable = self.policy_improvement()

            # Check convergence
            if policy_stable:
                self.converged = True
                break

        return {
            "converged": self.converged,
            "iterations": iteration + 1,
            "policy_evaluations": self.n_policy_evaluations,
            "policy_improvements": self.n_policy_improvements,
            "evaluation_history": self.evaluation_history,
            "policy_stable_history": self.policy_stable_history,
        }

    def get_policy(self, state: Optional[int] = None) -> torch.Tensor:
        """
        Get policy for a specific state or all states.

        Args:
            state: State index (if None, returns policy for all states)

        Returns:
            Policy tensor
        """
        if state is None:
            return self.policy.clone()
        else:
            return self.policy[state].clone()

    def get_action(self, state: int) -> int:
        """
        Get best action for a given state.

        Args:
            state: State index

        Returns:
            Best action index
        """
        return int(torch.argmax(self.policy[state]).item())

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
        total_return = 0.0
        discount = 1.0
        state = start_state

        for step in range(max_steps):
            # Sample action from policy
            action_probs = self.policy[state]
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

    def reset(self) -> None:
        """Reset algorithm to initial state."""
        # Reset to uniform random policy
        self.policy = (
            torch.ones(
                (self.n_states, self.n_actions), device=self.device, dtype=torch.float32
            )
            / self.n_actions
        )

        # Reset state values
        self.state_values = torch.zeros(
            self.n_states, device=self.device, dtype=torch.float32
        )

        # Reset tracking
        self.n_policy_evaluations = 0
        self.n_policy_improvements = 0
        self.converged = False
        self.evaluation_history = []
        self.policy_stable_history = []

    def __repr__(self) -> str:
        return (
            f"PolicyIteration(n_states={self.n_states}, "
            f"n_actions={self.n_actions}, gamma={self.gamma}, "
            f"theta={self.theta}, converged={self.converged})"
        )
