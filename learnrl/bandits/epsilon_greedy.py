"""Epsilon-greedy multi-armed bandit algorithm."""

import torch
from typing import Optional


class EpsilonGreedyBandit:
    """
    Epsilon-greedy k-armed bandit agent.

    This implementation follows Section 2.3 of Sutton & Barto (2020).
    The agent maintains action-value estimates and selects actions using
    an epsilon-greedy policy.
    """

    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        initial_values: float = 0.0,
        step_size: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize epsilon-greedy bandit agent.

        Args:
            k: Number of arms/actions
            epsilon: Exploration probability (0 <= epsilon <= 1)
            initial_values: Initial action-value estimates
            step_size: Fixed step size (alpha). If None, uses sample averaging
            device: PyTorch device for tensors
        """
        self.k = k
        self.epsilon = epsilon
        self.step_size = step_size
        self.device = device

        # Action-value estimates Q(a)
        self.q_values = torch.full(
            (k,), initial_values, device=device, dtype=torch.float32
        )

        # Action counts N(a)
        self.action_counts = torch.zeros(k, device=device, dtype=torch.long)

        # Total time steps
        self.t = 0

    def select_action(self) -> int:
        """
        Select action using epsilon-greedy policy.

        Returns:
            Selected action index (0 to k-1)
        """
        if torch.rand(1).item() < self.epsilon:
            # Explore: choose random action
            action = int(torch.randint(0, self.k, (1,)).item())
        else:
            # Exploit: choose greedy action (break ties randomly)
            max_value = torch.max(self.q_values)
            best_actions = torch.where(self.q_values == max_value)[0]
            idx = torch.randint(0, len(best_actions), (1,))
            action = int(best_actions[idx].item())

        return action

    def update(self, action: int, reward: float) -> None:
        """
        Update action-value estimates based on received reward.

        Args:
            action: Action that was taken (0 to k-1)
            reward: Reward received for the action
        """
        self.action_counts[action] += 1
        self.t += 1

        # Update Q(a) using incremental update rule
        if self.step_size is None:
            # Sample averaging: alpha = 1/N(a)
            alpha = 1.0 / self.action_counts[action].item()
        else:
            # Fixed step size
            alpha = self.step_size

        # Q(a) ← Q(a) + α[R - Q(a)]
        self.q_values[action] += alpha * (reward - self.q_values[action])

    def reset(self, initial_values: float = 0.0) -> None:
        """
        Reset the agent to initial state.

        Args:
            initial_values: Initial action-value estimates
        """
        self.q_values.fill_(initial_values)
        self.action_counts.zero_()
        self.t = 0

    def get_action_probabilities(self) -> torch.Tensor:
        """
        Get current action selection probabilities.

        Returns:
            Tensor of shape (k,) with action probabilities
        """
        eps_prob = self.epsilon / self.k
        probs = torch.full((self.k,), eps_prob, device=self.device)

        # Find greedy actions
        max_value = torch.max(self.q_values)
        best_actions = torch.where(self.q_values == max_value)[0]

        # Add greedy probability mass
        greedy_prob = (1.0 - self.epsilon) / len(best_actions)
        probs[best_actions] += greedy_prob

        return probs

    def get_greedy_action(self) -> int:
        """
        Get the current greedy action (without exploration).

        Returns:
            Greedy action index (ties broken randomly)
        """
        max_value = torch.max(self.q_values)
        best_actions = torch.where(self.q_values == max_value)[0]
        idx = torch.randint(0, len(best_actions), (1,))
        return int(best_actions[idx].item())

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyBandit(k={self.k}, epsilon={self.epsilon}, "
            f"step_size={self.step_size}, t={self.t})"
        )
