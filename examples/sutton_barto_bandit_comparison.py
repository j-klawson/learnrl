"""
Sutton & Barto Bandit Comparison Example

Reproduces the classic experiment from Section 2.3 of "Reinforcement Learning:
An Introduction" comparing greedy and ε-greedy action-value methods on 2000
randomly generated 10-armed bandit problems.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass

from learnrl.bandits import EpsilonGreedyBandit
from learnrl.utils import BanditTestEnvironment


@dataclass
class ExperimentResult:
    """Results from a bandit experiment."""

    algorithm_name: str
    average_rewards: np.ndarray  # Shape: (steps,)
    optimal_action_rates: np.ndarray  # Shape: (steps,)
    final_q_values: List[torch.Tensor]  # Q-values from each run


def run_single_algorithm(
    agent_class: Type,
    agent_kwargs: Dict[str, Any],
    algorithm_name: str,
    k: int = 10,
    num_problems: int = 2000,
    num_steps: int = 1000,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> ExperimentResult:
    """
    Run experiment for a single algorithm.

    Args:
        agent_class: Bandit algorithm class
        agent_kwargs: Keyword arguments for agent initialization
        algorithm_name: Name for the algorithm (for results)
        k: Number of arms for each bandit problem
        num_problems: Number of random bandit problems to test
        num_steps: Number of steps per problem
        seed: Random seed for reproducibility
        device: PyTorch device for tensors

    Returns:
        ExperimentResult containing averaged results
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Track results across all problems
    all_rewards = np.zeros((num_problems, num_steps))
    all_optimal_actions = np.zeros((num_problems, num_steps))
    final_q_values = []

    print(f"Running {algorithm_name}...")

    for problem_idx in range(num_problems):
        if problem_idx % 500 == 0:
            print(f"  Problem {problem_idx + 1}/{num_problems}")

        # Create new bandit problem
        env = BanditTestEnvironment(k=k, device=device)

        # Create new agent
        agent = agent_class(k=k, device=device, **agent_kwargs)

        # Run single problem
        for step in range(num_steps):
            action = agent.select_action()
            reward = env.get_reward(action)
            agent.update(action, reward)

            # Record results
            all_rewards[problem_idx, step] = reward
            all_optimal_actions[problem_idx, step] = env.is_optimal_action(action)

        # Store final Q-values
        final_q_values.append(agent.q_values.clone())

    # Average across all problems
    average_rewards = np.mean(all_rewards, axis=0)
    optimal_action_rates = np.mean(all_optimal_actions, axis=0)

    return ExperimentResult(
        algorithm_name=algorithm_name,
        average_rewards=average_rewards,
        optimal_action_rates=optimal_action_rates,
        final_q_values=final_q_values,
    )


def compare_algorithms(
    algorithms: List[Dict[str, Any]],
    k: int = 10,
    num_problems: int = 2000,
    num_steps: int = 1000,
    seed: Optional[int] = 42,
) -> List[ExperimentResult]:
    """
    Compare multiple algorithms on the same set of problems.

    Args:
        algorithms: List of algorithm specifications, each containing:
            - 'class': Algorithm class
            - 'kwargs': Keyword arguments for initialization
            - 'name': Display name for the algorithm
        k: Number of arms
        num_problems: Number of random bandit problems
        num_steps: Number of steps per problem
        seed: Random seed for reproducibility

    Returns:
        List of ExperimentResult, one for each algorithm
    """
    results = []

    print(f"Bandit Comparison Experiment:")
    print(f"- {k}-armed bandits")
    print(f"- {num_problems} random problems")
    print(f"- {num_steps} steps per problem")
    print(f"- Algorithms: {[algo['name'] for algo in algorithms]}")
    print()

    for i, algo_spec in enumerate(algorithms):
        # Use different seed for each algorithm to get independent runs
        algo_seed = seed + i * 1000 if seed is not None else None

        result = run_single_algorithm(
            agent_class=algo_spec["class"],
            agent_kwargs=algo_spec["kwargs"],
            algorithm_name=algo_spec["name"],
            k=k,
            num_problems=num_problems,
            num_steps=num_steps,
            seed=algo_seed,
        )
        results.append(result)
        print()

    return results


def plot_results(results: List[ExperimentResult], save_path: Optional[str] = None):
    """
    Plot the experimental results.

    Creates two subplots:
    1. Average reward over time steps
    2. Percentage of optimal actions over time steps

    Args:
        results: List of ExperimentResult to plot
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    colors = ["red", "green", "blue", "orange", "purple"]

    # Plot average reward
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        ax1.plot(result.average_rewards, label=result.algorithm_name, color=color)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Average Reward vs Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot percentage of optimal actions
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        ax2.plot(
            result.optimal_action_rates * 100, label=result.algorithm_name, color=color
        )

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal Action")
    ax2.set_title("Percentage of Optimal Actions vs Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def run_sutton_barto_experiment(
    epsilon_values: List[float] = [0.0, 0.01, 0.1],
    k: int = 10,
    num_problems: int = 2000,
    num_steps: int = 1000,
    seed: Optional[int] = 42,
    plot: bool = True,
    save_plot: Optional[str] = None,
) -> List[ExperimentResult]:
    """
    Run the classic Sutton & Barto experiment comparing greedy and ε-greedy.

    This reproduces the experiment from Section 2.3 that compares greedy
    (ε=0) and ε-greedy methods on 2000 randomly generated 10-armed bandit problems.

    Args:
        epsilon_values: List of epsilon values to test
        k: Number of arms
        num_problems: Number of random bandit problems
        num_steps: Number of steps per problem
        seed: Random seed for reproducibility
        plot: Whether to plot the results
        save_plot: Optional path to save the plot

    Returns:
        List of ExperimentResult for each epsilon value
    """
    # Define algorithms to compare
    algorithms = []
    for eps in epsilon_values:
        name = f"ε-greedy (ε={eps})" if eps > 0 else "greedy"
        algorithms.append(
            {
                "class": EpsilonGreedyBandit,
                "kwargs": {"epsilon": eps, "initial_values": 0.0},
                "name": name,
            }
        )

    # Run comparison
    results = compare_algorithms(
        algorithms=algorithms,
        k=k,
        num_problems=num_problems,
        num_steps=num_steps,
        seed=seed,
    )

    # Print summary
    print("Experiment Summary:")
    print("-" * 50)
    for result in results:
        final_avg_reward = result.average_rewards[-1]
        final_optimal_rate = result.optimal_action_rates[-1] * 100
        print(
            f"{result.algorithm_name:15} | "
            f"Final Avg Reward: {final_avg_reward:.3f} | "
            f"Final % Optimal: {final_optimal_rate:.1f}%"
        )
    print()

    # Plot results
    if plot:
        plot_results(results, save_path=save_plot)

    return results


if __name__ == "__main__":
    # Run the classic experiment
    print("Reproducing Sutton & Barto Section 2.3 Experiment")
    print("=" * 60)

    results = run_sutton_barto_experiment(
        epsilon_values=[0.0, 0.01, 0.1],
        k=10,
        num_problems=2000,
        num_steps=1000,
        seed=42,
        plot=True,
        save_plot="plots/sutton_barto_bandit_comparison.png",
    )

    print("Experiment completed!")
    print("\nKey findings:")
    print("- Greedy (ε=0) gets stuck in suboptimal actions")
    print("- ε-greedy with small ε (0.01) balances exploration and exploitation")
    print("- Higher ε (0.1) explores more but may sacrifice long-term performance")
