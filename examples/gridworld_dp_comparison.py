"""
GridWorld Dynamic Programming Algorithm Comparison.

This script demonstrates and compares Policy Iteration and Value Iteration
algorithms on various GridWorld environments, reproducing examples from
Sutton & Barto Chapter 4.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List

from learnrl.dp.policy_iteration import PolicyIteration
from learnrl.dp.value_iteration import ValueIteration
from learnrl.utils.gridworld_env import (
    GridWorldEnv, GridWorldConfig, create_simple_gridworld, create_cliff_world
)

# Default directory for saving plots
plotdir = "plots/gridworld/"


def compare_algorithms_on_environment(env: GridWorldEnv, title: str = "GridWorld") -> Dict:
    """
    Compare Policy Iteration and Value Iteration on a given environment.

    Args:
        env: GridWorld environment
        title: Title for the comparison

    Returns:
        Dictionary with comparison results
    """
    print(f"\n=== {title} ===")
    print(f"Environment: {env}")

    # Get transition model
    P = env.get_transition_probabilities()
    R = env.get_reward_tensor()

    results = {}

    # Policy Iteration
    print("\nRunning Policy Iteration...")
    pi = PolicyIteration(env.n_states, env.n_actions, P, R, theta=1e-6)
    start_time = time.time()
    pi_result = pi.solve(max_iterations=100)
    pi_time = time.time() - start_time

    results['policy_iteration'] = {
        'algorithm': pi,
        'result': pi_result,
        'time': pi_time,
        'state_values': pi.get_state_value(),
        'policy': pi.get_policy()
    }

    print(f"  Converged: {pi_result['converged']}")
    print(f"  Iterations: {pi_result['iterations']}")
    print(f"  Policy Evaluations: {pi_result['policy_evaluations']}")
    print(f"  Time: {pi_time:.4f}s")

    # Value Iteration
    print("\nRunning Value Iteration...")
    vi = ValueIteration(env.n_states, env.n_actions, P, R, theta=1e-6)
    start_time = time.time()
    vi_result = vi.solve(max_iterations=1000)
    vi_time = time.time() - start_time

    results['value_iteration'] = {
        'algorithm': vi,
        'result': vi_result,
        'time': vi_time,
        'state_values': vi.get_state_value(),
        'policy': vi.get_policy()
    }

    print(f"  Converged: {vi_result['converged']}")
    print(f"  Iterations: {vi_result['iterations']}")
    print(f"  Final Delta: {vi_result['final_delta']:.2e}")
    print(f"  Time: {vi_time:.4f}s")

    # Compare solutions
    print("\n--- Comparison ---")
    pi_values = results['policy_iteration']['state_values']
    vi_values = results['value_iteration']['state_values']

    value_diff = torch.max(torch.abs(pi_values - vi_values)).item()
    print(f"Max state value difference: {value_diff:.2e}")

    # Check if policies are the same
    pi_actions = []
    vi_actions = []
    for s in range(env.n_states):
        pi_actions.append(pi.get_action(s))
        vi_actions.append(vi.get_action(s))

    policy_agreement = sum(1 for a, b in zip(pi_actions, vi_actions) if a == b) / len(pi_actions)
    print(f"Policy agreement: {policy_agreement:.1%}")

    results['comparison'] = {
        'value_difference': value_diff,
        'policy_agreement': policy_agreement,
        'pi_actions': pi_actions,
        'vi_actions': vi_actions
    }

    return results


def visualize_results(env: GridWorldEnv, results: Dict, title: str = "GridWorld"):
    """Visualize algorithm results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title} - Dynamic Programming Comparison', fontsize=16)

    # Policy Iteration Results
    pi_values = results['policy_iteration']['state_values']
    pi_policy = results['policy_iteration']['policy']

    # Value Iteration Results
    vi_values = results['value_iteration']['state_values']
    vi_policy = results['value_iteration']['policy']

    # Convert state values to grid format
    pi_grid = env.get_state_values_tensor(pi_values).numpy()
    vi_grid = env.get_state_values_tensor(vi_values).numpy()
    value_diff_grid = env.get_state_values_tensor(torch.abs(pi_values - vi_values)).numpy()

    # Plot state values
    im1 = axes[0, 0].imshow(pi_grid, cmap='viridis')
    axes[0, 0].set_title('Policy Iteration\nState Values')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(vi_grid, cmap='viridis')
    axes[0, 1].set_title('Value Iteration\nState Values')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(value_diff_grid, cmap='Reds')
    axes[0, 2].set_title('Absolute Difference\nin State Values')
    plt.colorbar(im3, ax=axes[0, 2])

    # Plot policies
    for i, (policy, alg_name) in enumerate([(pi_policy, 'Policy Iteration'),
                                           (vi_policy, 'Value Iteration')]):
        ax = axes[1, i]

        # Create policy visualization
        policy_grid = np.zeros((env.height, env.width), dtype=int)
        for s in range(env.n_states):
            row, col = env.index_to_state(s)
            action = torch.argmax(policy[s]).item()
            policy_grid[row, col] = action

        # Plot with different colors for each action
        im = ax.imshow(policy_grid, cmap='Set3', vmin=0, vmax=3)
        ax.set_title(f'{alg_name}\nOptimal Policy')

        # Add arrows for actions
        action_chars = ['↑', '→', '↓', '←']
        for row in range(env.height):
            for col in range(env.width):
                state = (row, col)
                if not env.is_terminal_state(state):
                    s = env.state_to_index(state)
                    action = torch.argmax(policy[s]).item()
                    ax.text(col, row, action_chars[action],
                           ha='center', va='center', fontsize=12, fontweight='bold')
                elif state in env.config.goal_states:
                    ax.text(col, row, 'G', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='green')
                elif state in env.config.obstacles:
                    ax.text(col, row, 'X', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='red')

    # Convergence comparison
    ax = axes[1, 2]

    # Plot PI evaluation history (cumulative iterations)
    pi_result = results['policy_iteration']['result']
    pi_eval_history = pi_result['evaluation_history']
    pi_cumulative = np.cumsum([0] + pi_eval_history)

    ax.plot(range(len(pi_cumulative)), pi_cumulative, 'b-o', label='Policy Iteration', linewidth=2)

    # Plot VI delta history
    vi_result = results['value_iteration']['result']
    vi_delta_history = vi_result['delta_history']

    ax2 = ax.twinx()
    ax2.semilogy(range(1, len(vi_delta_history) + 1), vi_delta_history, 'r-s',
                label='Value Iteration', linewidth=2)

    ax.set_xlabel('Policy Iteration Steps')
    ax.set_ylabel('Cumulative Evaluation Iterations', color='b')
    ax2.set_ylabel('Value Change (log scale)', color='r')
    ax.set_title('Convergence Comparison')

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig


def run_experiments():
    """Run comprehensive experiments comparing algorithms."""
    print("Dynamic Programming Algorithm Comparison")
    print("=" * 50)

    experiments = []

    # Experiment 1: Simple 4x4 GridWorld
    print("\n" + "="*20 + " EXPERIMENT 1 " + "="*20)
    env1 = create_simple_gridworld()
    results1 = compare_algorithms_on_environment(env1, "Simple 4x4 GridWorld")
    experiments.append(("Simple GridWorld", env1, results1))

    # Experiment 2: Larger GridWorld
    print("\n" + "="*20 + " EXPERIMENT 2 " + "="*20)
    config2 = GridWorldConfig(
        height=6, width=6,
        start_state=(0, 0),
        goal_states=[(5, 5)],
        obstacles=[(2, 2), (2, 3), (3, 2), (3, 3)],
        step_reward=-0.04
    )
    env2 = GridWorldEnv(config2)
    results2 = compare_algorithms_on_environment(env2, "6x6 GridWorld with Obstacles")
    experiments.append(("Large GridWorld", env2, results2))

    # Experiment 3: Cliff World
    print("\n" + "="*20 + " EXPERIMENT 3 " + "="*20)
    env3 = create_cliff_world()
    results3 = compare_algorithms_on_environment(env3, "Cliff World")
    experiments.append(("Cliff World", env3, results3))

    # Experiment 4: Stochastic Environment
    print("\n" + "="*20 + " EXPERIMENT 4 " + "="*20)
    config4 = GridWorldConfig(height=4, width=4, step_reward=-0.1)
    env4 = GridWorldEnv(config4, stochastic=True, noise_prob=0.2)
    results4 = compare_algorithms_on_environment(env4, "Stochastic 4x4 GridWorld")
    experiments.append(("Stochastic GridWorld", env4, results4))

    # Create visualizations
    print("\n" + "="*20 + " GENERATING PLOTS " + "="*20)
    os.makedirs(plotdir, exist_ok=True)
    for title, env, results in experiments:
        fig = visualize_results(env, results, title)
        filename = f"dp_comparison_{title.lower().replace(' ', '_')}.png"
        filepath = f"{plotdir}{filename}"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        plt.close(fig)

    # Summary comparison
    print("\n" + "="*20 + " SUMMARY " + "="*20)
    print(f"{'Environment':<25} {'PI Time':<10} {'VI Time':<10} {'PI Iter':<8} {'VI Iter':<8} {'Agreement':<10}")
    print("-" * 75)

    for title, env, results in experiments:
        pi_time = results['policy_iteration']['time']
        vi_time = results['value_iteration']['time']
        pi_iter = results['policy_iteration']['result']['iterations']
        vi_iter = results['value_iteration']['result']['iterations']
        agreement = results['comparison']['policy_agreement']

        print(f"{title:<25} {pi_time:<10.4f} {vi_time:<10.4f} {pi_iter:<8} {vi_iter:<8} {agreement:<10.1%}")


def demonstrate_policy_evaluation():
    """Demonstrate policy evaluation convergence."""
    print("\n" + "="*20 + " POLICY EVALUATION DEMO " + "="*20)

    env = create_simple_gridworld()
    P = env.get_transition_probabilities()
    R = env.get_reward_tensor()

    # Create Policy Iteration instance
    pi = PolicyIteration(env.n_states, env.n_actions, P, R, theta=1e-8)

    print("Initial uniform random policy:")
    print(env.render(mode="ascii", policy=pi.get_policy()))

    # Perform several policy evaluations and improvements
    for iteration in range(3):
        print(f"\n--- Policy Iteration {iteration + 1} ---")

        # Policy Evaluation
        eval_iterations = pi.policy_evaluation(max_iterations=1000)
        print(f"Policy evaluation converged in {eval_iterations} iterations")

        # Show state values
        state_values = pi.get_state_value()
        value_grid = env.get_state_values_tensor(state_values)
        print("State Values:")
        for row in range(env.height):
            for col in range(env.width):
                print(f"{value_grid[row, col].item():6.2f}", end=" ")
            print()

        # Policy Improvement
        policy_stable = pi.policy_improvement()
        print(f"Policy stable: {policy_stable}")

        print("Updated policy:")
        print(env.render(mode="ascii", policy=pi.get_policy()))

        if policy_stable:
            print("Policy iteration converged!")
            break


def analyze_convergence_properties():
    """Analyze convergence properties of both algorithms."""
    print("\n" + "="*20 + " CONVERGENCE ANALYSIS " + "="*20)

    env = create_simple_gridworld()
    P = env.get_transition_probabilities()
    R = env.get_reward_tensor()

    # Test different convergence thresholds
    thresholds = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

    print(f"{'Threshold':<12} {'PI Iter':<8} {'VI Iter':<8} {'PI Time':<10} {'VI Time':<10}")
    print("-" * 50)

    for theta in thresholds:
        # Policy Iteration
        pi = PolicyIteration(env.n_states, env.n_actions, P, R, theta=theta)
        start_time = time.time()
        pi_result = pi.solve()
        pi_time = time.time() - start_time

        # Value Iteration
        vi = ValueIteration(env.n_states, env.n_actions, P, R, theta=theta)
        start_time = time.time()
        vi_result = vi.solve()
        vi_time = time.time() - start_time

        print(f"{theta:<12.0e} {pi_result['iterations']:<8} {vi_result['iterations']:<8} "
              f"{pi_time:<10.6f} {vi_time:<10.6f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all experiments
    run_experiments()
    demonstrate_policy_evaluation()
    analyze_convergence_properties()

    print("\n" + "="*50)
    print("All experiments completed!")
    print("Check the generated PNG files for visualizations.")
