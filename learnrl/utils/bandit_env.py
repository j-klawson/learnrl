"""Bandit test environment for generating random k-armed bandit problems."""

import torch
from typing import Optional


class BanditTestEnvironment:
    """
    Test environment for k-armed bandit problems.

    Generates random bandit problems where true action values q*(a) are drawn
    from a normal distribution, and rewards are generated with Gaussian noise.
    This follows the experimental setup from Sutton & Barto Section 2.3.
    """

    def __init__(
        self,
        k: int = 10,
        true_value_mean: float = 0.0,
        true_value_std: float = 1.0,
        reward_std: float = 1.0,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize bandit test environment.

        Args:
            k: Number of arms/actions
            true_value_mean: Mean for true action values q*(a)
            true_value_std: Standard deviation for true action values q*(a)
            reward_std: Standard deviation for reward noise
            seed: Random seed for reproducibility
            device: PyTorch device for tensors
        """
        self.k = k
        self.true_value_mean = true_value_mean
        self.true_value_std = true_value_std
        self.reward_std = reward_std
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Generate true action values q*(a) from normal distribution
        self.true_values = torch.normal(
            mean=true_value_mean, std=true_value_std, size=(k,), device=device
        )

        # Track optimal action for performance measurement
        self.optimal_action = int(torch.argmax(self.true_values).item())

    def get_reward(self, action: int) -> float:
        """
        Get reward for taking an action.

        The reward is drawn from a normal distribution with mean equal to
        the true action value q*(a) and standard deviation reward_std.

        Args:
            action: Action index (0 to k-1)

        Returns:
            Reward value
        """
        if not (0 <= action < self.k):
            msg = f"Action {action} not in valid range [0, {self.k-1}]"
            raise ValueError(msg)

        reward = torch.normal(mean=self.true_values[action], std=self.reward_std).item()

        return reward

    def is_optimal_action(self, action: int) -> bool:
        """
        Check if the given action is optimal.

        Args:
            action: Action index

        Returns:
            True if action is optimal, False otherwise
        """
        return action == self.optimal_action

    def get_optimal_value(self) -> float:
        """Get the true value of the optimal action."""
        return float(self.true_values[self.optimal_action].item())

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the environment with new random true values.

        Args:
            seed: Optional seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.true_values = torch.normal(
            mean=self.true_value_mean,
            std=self.true_value_std,
            size=(self.k,),
            device=self.device,
        )

        self.optimal_action = int(torch.argmax(self.true_values).item())

    def __repr__(self) -> str:
        return (
            f"BanditTestEnvironment(k={self.k}, "
            f"optimal_action={self.optimal_action}, "
            f"optimal_value={self.get_optimal_value():.3f})"
        )
