"""
Dynamic Programming Visualization Demo.

This script provides interactive visualizations of Dynamic Programming
algorithms, showing step-by-step convergence and policy evolution.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Dict
import time

from learnrl.dp.policy_iteration import PolicyIteration
from learnrl.dp.value_iteration import ValueIteration
from learnrl.utils.gridworld_env import (
    GridWorldEnv, GridWorldConfig, create_simple_gridworld
)

# Default directory for saving plots
plotdir = "plots/dp/"


class DPVisualizer:
    """Visualizer for Dynamic Programming algorithms."""

    def __init__(self, env: GridWorldEnv):
        """
        Initialize visualizer with environment.

        Args:
            env: GridWorld environment
        """
        self.env = env
        self.fig = None
        self.axes = None

    def visualize_state_values(self, state_values: torch.Tensor, title: str = "State Values"):
        """Visualize state values as a heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Convert to grid format
        value_grid = self.env.get_state_values_tensor(state_values).numpy()

        # Create heatmap
        im = ax.imshow(value_grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax)

        # Add value annotations
        for row in range(self.env.height):
            for col in range(self.env.width):
                state = (row, col)
                value = value_grid[row, col]

                # Different formatting for terminal states
                if state in self.env.config.goal_states:
                    text = f"G\n{value:.2f}"
                    color = 'white'
                elif state in self.env.config.obstacles:
                    text = f"X\n{value:.2f}"
                    color = 'white'
                else:
                    text = f"{value:.2f}"
                    color = 'white' if value < 0 else 'black'

                ax.text(col, row, text, ha='center', va='center',
                       color=color, fontweight='bold')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(self.env.width))
        ax.set_yticks(range(self.env.height))
        ax.grid(True, alpha=0.3)

        return fig

    def visualize_policy(self, policy: torch.Tensor, title: str = "Policy"):
        """Visualize policy as arrows."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create background grid
        background = np.zeros((self.env.height, self.env.width))
        ax.imshow(background, cmap='gray', alpha=0.1)

        # Action symbols
        action_symbols = ['↑', '→', '↓', '←']
        action_colors = ['red', 'blue', 'green', 'orange']

        # Plot policy
        for s in range(self.env.n_states):
            row, col = self.env.index_to_state(s)
            state = (row, col)

            if state in self.env.config.goal_states:
                ax.text(col, row, 'G', ha='center', va='center',
                       fontsize=20, fontweight='bold', color='green',
                       bbox=dict(boxstyle="circle", facecolor='lightgreen'))
            elif state in self.env.config.obstacles:
                ax.text(col, row, 'X', ha='center', va='center',
                       fontsize=20, fontweight='bold', color='red',
                       bbox=dict(boxstyle="circle", facecolor='lightcoral'))
            else:
                # Get action probabilities
                action_probs = policy[s]
                best_action = torch.argmax(action_probs).item()

                # Show primary action
                symbol = action_symbols[best_action]
                color = action_colors[best_action]
                ax.text(col, row, symbol, ha='center', va='center',
                       fontsize=16, fontweight='bold', color=color)

                # Show action probabilities as pie chart for stochastic policies
                if not torch.allclose(action_probs, torch.zeros_like(action_probs)):
                    max_prob = torch.max(action_probs)
                    if max_prob < 0.99:  # Stochastic policy
                        # Add small indicators for other actions
                        for a, prob in enumerate(action_probs):
                            if a != best_action and prob > 0.01:
                                offset_x = 0.3 * np.cos(a * np.pi / 2)
                                offset_y = 0.3 * np.sin(a * np.pi / 2)
                                ax.text(col + offset_x, row + offset_y,
                                       action_symbols[a],
                                       ha='center', va='center',
                                       fontsize=8, alpha=0.7,
                                       color=action_colors[a])

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.set_xticks(range(self.env.width))
        ax.set_yticks(range(self.env.height))
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        return fig

    def visualize_algorithm_step(self, algorithm, step_num: int = 0):
        """Visualize current state of algorithm."""
        if hasattr(algorithm, 'policy') and algorithm.policy is not None:
            # Show both values and policy
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # State values
            state_values = algorithm.get_state_value()
            value_grid = self.env.get_state_values_tensor(state_values).numpy()

            im1 = ax1.imshow(value_grid, cmap='viridis')
            plt.colorbar(im1, ax=ax1)

            for row in range(self.env.height):
                for col in range(self.env.width):
                    value = value_grid[row, col]
                    ax1.text(col, row, f"{value:.2f}", ha='center', va='center',
                           color='white' if value < 0 else 'black', fontweight='bold')

            ax1.set_title(f'State Values (Step {step_num})')
            ax1.set_xticks(range(self.env.width))
            ax1.set_yticks(range(self.env.height))
            ax1.grid(True, alpha=0.3)

            # Policy
            policy = algorithm.get_policy()
            action_symbols = ['↑', '→', '↓', '←']

            background = np.zeros((self.env.height, self.env.width))
            ax2.imshow(background, cmap='gray', alpha=0.1)

            for s in range(self.env.n_states):
                row, col = self.env.index_to_state(s)
                state = (row, col)

                if state in self.env.config.goal_states:
                    ax2.text(col, row, 'G', ha='center', va='center',
                           fontsize=16, fontweight='bold', color='green')
                elif state in self.env.config.obstacles:
                    ax2.text(col, row, 'X', ha='center', va='center',
                           fontsize=16, fontweight='bold', color='red')
                else:
                    best_action = torch.argmax(policy[s]).item()
                    ax2.text(col, row, action_symbols[best_action],
                           ha='center', va='center', fontsize=14, fontweight='bold')

            ax2.set_title(f'Policy (Step {step_num})')
            ax2.set_xlim(-0.5, self.env.width - 0.5)
            ax2.set_ylim(-0.5, self.env.height - 0.5)
            ax2.set_xticks(range(self.env.width))
            ax2.set_yticks(range(self.env.height))
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()

        else:
            # Only show state values
            fig, ax1 = plt.subplots(figsize=(8, 6))

            state_values = algorithm.get_state_value()
            value_grid = self.env.get_state_values_tensor(state_values).numpy()

            im1 = ax1.imshow(value_grid, cmap='viridis')
            plt.colorbar(im1, ax=ax1)

            for row in range(self.env.height):
                for col in range(self.env.width):
                    value = value_grid[row, col]
                    ax1.text(col, row, f"{value:.2f}", ha='center', va='center',
                           color='white' if value < 0 else 'black', fontweight='bold')

            ax1.set_title(f'State Values (Step {step_num})')
            ax1.set_xticks(range(self.env.width))
            ax1.set_yticks(range(self.env.height))
            ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_convergence_animation(self, algorithm_class, max_iterations: int = 50,
                                   save_filename: str = None):
        """Create animation showing algorithm convergence."""
        # Initialize algorithm
        P = self.env.get_transition_probabilities()
        R = self.env.get_reward_tensor()
        algorithm = algorithm_class(self.env.n_states, self.env.n_actions, P, R, theta=1e-6)

        # Store snapshots
        snapshots = []
        iterations = []

        if algorithm_class == PolicyIteration:
            # Policy Iteration: show after each policy improvement
            for iteration in range(max_iterations):
                # Policy evaluation
                algorithm.policy_evaluation()

                # Store snapshot
                snapshots.append({
                    'state_values': algorithm.get_state_value().clone(),
                    'policy': algorithm.get_policy().clone()
                })
                iterations.append(iteration)

                # Policy improvement
                policy_stable = algorithm.policy_improvement()

                if policy_stable:
                    # Final snapshot
                    snapshots.append({
                        'state_values': algorithm.get_state_value().clone(),
                        'policy': algorithm.get_policy().clone()
                    })
                    iterations.append(iteration + 1)
                    break

        else:  # Value Iteration
            # Value Iteration: show after each value update
            for iteration in range(max_iterations):
                algorithm.value_iteration_step()

                # Store snapshot every few iterations
                if iteration % 5 == 0 or iteration < 10:
                    if algorithm.policy is None:
                        algorithm.extract_policy()

                    snapshots.append({
                        'state_values': algorithm.get_state_value().clone(),
                        'policy': algorithm.get_policy().clone()
                    })
                    iterations.append(iteration + 1)

                # Check convergence
                if len(algorithm.delta_history) > 0 and algorithm.delta_history[-1] < algorithm.theta:
                    if algorithm.policy is None:
                        algorithm.extract_policy()

                    snapshots.append({
                        'state_values': algorithm.get_state_value().clone(),
                        'policy': algorithm.get_policy().clone()
                    })
                    iterations.append(iteration + 1)
                    break

        # Create animation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{algorithm_class.__name__} Convergence', fontsize=16)

        def animate(frame):
            for ax in axes:
                ax.clear()

            snapshot = snapshots[frame]
            iteration = iterations[frame]

            # State values
            value_grid = self.env.get_state_values_tensor(snapshot['state_values']).numpy()
            im1 = axes[0].imshow(value_grid, cmap='viridis', vmin=value_grid.min(), vmax=value_grid.max())

            for row in range(self.env.height):
                for col in range(self.env.width):
                    value = value_grid[row, col]
                    axes[0].text(col, row, f"{value:.2f}", ha='center', va='center',
                               color='white' if value < 0 else 'black', fontweight='bold')

            axes[0].set_title(f'State Values (Iteration {iteration})')
            axes[0].set_xticks(range(self.env.width))
            axes[0].set_yticks(range(self.env.height))
            axes[0].grid(True, alpha=0.3)

            # Policy
            policy = snapshot['policy']
            action_symbols = ['↑', '→', '↓', '←']

            background = np.zeros((self.env.height, self.env.width))
            axes[1].imshow(background, cmap='gray', alpha=0.1)

            for s in range(self.env.n_states):
                row, col = self.env.index_to_state(s)
                state = (row, col)

                if state in self.env.config.goal_states:
                    axes[1].text(col, row, 'G', ha='center', va='center',
                               fontsize=16, fontweight='bold', color='green')
                elif state in self.env.config.obstacles:
                    axes[1].text(col, row, 'X', ha='center', va='center',
                               fontsize=16, fontweight='bold', color='red')
                else:
                    best_action = torch.argmax(policy[s]).item()
                    axes[1].text(col, row, action_symbols[best_action],
                               ha='center', va='center', fontsize=14, fontweight='bold')

            axes[1].set_title(f'Policy (Iteration {iteration})')
            axes[1].set_xlim(-0.5, self.env.width - 0.5)
            axes[1].set_ylim(-0.5, self.env.height - 0.5)
            axes[1].set_xticks(range(self.env.width))
            axes[1].set_yticks(range(self.env.height))
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()

            plt.tight_layout()

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(snapshots),
                                     interval=1000, repeat=True, blit=False)

        if save_filename:
            anim.save(save_filename, writer='pillow', fps=1)
            print(f"Animation saved as {save_filename}")

        return anim


def demonstrate_value_iteration_steps():
    """Demonstrate Value Iteration step by step."""
    print("=== Value Iteration Step-by-Step Demo ===")

    env = create_simple_gridworld()
    visualizer = DPVisualizer(env)

    P = env.get_transition_probabilities()
    R = env.get_reward_tensor()
    vi = ValueIteration(env.n_states, env.n_actions, P, R, theta=1e-6)

    # Ensure plot directory exists
    os.makedirs(plotdir, exist_ok=True)

    print("Initial state values (all zeros):")
    fig = visualizer.visualize_state_values(vi.get_state_value(), "Initial State Values")
    fig.savefig(f"{plotdir}vi_step_0.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Run several steps
    for step in range(1, 6):
        delta = vi.value_iteration_step()
        print(f"Step {step}: delta = {delta:.6f}")

        fig = visualizer.visualize_state_values(vi.get_state_value(), f"Value Iteration Step {step}")
        fig.savefig(f"{plotdir}vi_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Final solution
    vi.solve()
    print(f"Converged after {vi.iterations} iterations")

    fig = visualizer.visualize_state_values(vi.get_state_value(), "Final State Values")
    fig.savefig(f"{plotdir}vi_final_values.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = visualizer.visualize_policy(vi.get_policy(), "Optimal Policy")
    fig.savefig(f"{plotdir}vi_final_policy.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def demonstrate_policy_iteration_steps():
    """Demonstrate Policy Iteration step by step."""
    print("\n=== Policy Iteration Step-by-Step Demo ===")

    env = create_simple_gridworld()
    visualizer = DPVisualizer(env)

    P = env.get_transition_probabilities()
    R = env.get_reward_tensor()
    pi = PolicyIteration(env.n_states, env.n_actions, P, R, theta=1e-6)

    # Ensure plot directory exists
    os.makedirs(plotdir, exist_ok=True)

    print("Initial uniform random policy:")
    fig = visualizer.visualize_policy(pi.get_policy(), "Initial Random Policy")
    fig.savefig(f"{plotdir}pi_initial_policy.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Run several policy iteration steps
    for step in range(1, 4):
        print(f"\n--- Policy Iteration Step {step} ---")

        # Policy evaluation
        eval_iterations = pi.policy_evaluation()
        print(f"Policy evaluation converged in {eval_iterations} iterations")

        fig = visualizer.visualize_state_values(pi.get_state_value(), f"State Values After Evaluation {step}")
        fig.savefig(f"{plotdir}pi_values_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Policy improvement
        policy_stable = pi.policy_improvement()
        print(f"Policy stable: {policy_stable}")

        fig = visualizer.visualize_policy(pi.get_policy(), f"Policy After Improvement {step}")
        fig.savefig(f"{plotdir}pi_policy_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        if policy_stable:
            print("Policy iteration converged!")
            break


def compare_convergence_patterns():
    """Compare convergence patterns of both algorithms."""
    print("\n=== Convergence Pattern Comparison ===")

    env = create_simple_gridworld()
    P = env.get_transition_probabilities()
    R = env.get_reward_tensor()

    # Value Iteration convergence
    vi = ValueIteration(env.n_states, env.n_actions, P, R, theta=1e-8)
    vi.solve()

    # Policy Iteration convergence
    pi = PolicyIteration(env.n_states, env.n_actions, P, R, theta=1e-8)
    pi.solve()

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Value Iteration delta history
    ax1.semilogy(range(1, len(vi.delta_history) + 1), vi.delta_history, 'b-o', linewidth=2)
    ax1.axhline(y=vi.theta, color='r', linestyle='--', alpha=0.7, label='Convergence Threshold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Maximum Value Change (log scale)')
    ax1.set_title('Value Iteration Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Policy Iteration evaluation history
    pi_result = pi.evaluation_history
    cumulative_evals = np.cumsum([0] + pi_result)

    ax2.plot(range(len(cumulative_evals)), cumulative_evals, 'g-s', linewidth=2, label='Cumulative Evaluations')
    ax2.bar(range(1, len(pi_result) + 1), pi_result, alpha=0.6, color='orange', label='Evaluations per Step')
    ax2.set_xlabel('Policy Iteration Step')
    ax2.set_ylabel('Policy Evaluation Iterations')
    ax2.set_title('Policy Iteration Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(plotdir, exist_ok=True)
    fig.savefig(f"{plotdir}convergence_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Value Iteration: {len(vi.delta_history)} iterations")
    print(f"Policy Iteration: {len(pi_result)} steps, {sum(pi_result)} total evaluations")


def create_interactive_demo():
    """Create interactive demonstration."""
    print("\n=== Creating Interactive Animations ===")

    env = create_simple_gridworld()
    visualizer = DPVisualizer(env)

    # Ensure plot directory exists
    os.makedirs(plotdir, exist_ok=True)

    # Create animations
    print("Creating Value Iteration animation...")
    try:
        anim_vi = visualizer.create_convergence_animation(
            ValueIteration, max_iterations=100, save_filename=f"{plotdir}value_iteration_convergence.gif"
        )
        plt.close()
    except Exception as e:
        print(f"Could not create Value Iteration animation: {e}")

    print("Creating Policy Iteration animation...")
    try:
        anim_pi = visualizer.create_convergence_animation(
            PolicyIteration, max_iterations=20, save_filename=f"{plotdir}policy_iteration_convergence.gif"
        )
        plt.close()
    except Exception as e:
        print(f"Could not create Policy Iteration animation: {e}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Dynamic Programming Visualization Demo")
    print("=" * 50)

    # Run demonstrations
    demonstrate_value_iteration_steps()
    demonstrate_policy_iteration_steps()
    compare_convergence_patterns()
    create_interactive_demo()

    print("\n" + "=" * 50)
    print("Visualization demo completed!")
    print("Check the generated PNG and GIF files for step-by-step visualizations.")