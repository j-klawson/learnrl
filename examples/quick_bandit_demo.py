"""
Quick demo of the Sutton & Barto bandit comparison with reduced parameters.
"""

from sutton_barto_bandit_comparison import run_sutton_barto_experiment

# Default directory for saving plots
plotdir = "plots/bandits"

if __name__ == "__main__":
    print("Quick Bandit Comparison Demo")
    print("=" * 40)

    # Run smaller experiment for demonstration
    results = run_sutton_barto_experiment(
        epsilon_values=[0.0, 0.1],
        k=10,
        num_problems=100,  # Reduced from 2000
        num_steps=100,  # Reduced from 1000
        seed=42,
        plot=True,
        save_plot=f"{plotdir}quick_bandit_demo.png",
    )

    print("Quick demo completed!")
