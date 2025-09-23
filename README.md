# LearnRL

Simple PyTorch implementations of reinforcement learning algorithms from "Reinforcement Learning: An Introduction" by Sutton & Barto (2020).

## Overview

This package provides clean, educational implementations of classic RL algorithms designed to accompany the textbook. Each implementation focuses on clarity and understanding rather than performance optimization.

## Installation

```bash
# Clone the repository
git clone https://github.com/keith/learnrl.git
cd learnrl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install with Gymnasium support (required for TD/DP algorithms)
pip install -e ".[gym]"

# Install all dependencies (recommended for full development)
pip install -e ".[dev,gym]"
```

## Quick Start

### Multi-Armed Bandits

```python
import torch
from learnrl.bandits import EpsilonGreedyBandit
from learnrl.utils import BanditTestEnvironment

# Create a 10-armed bandit with ε=0.1
agent = EpsilonGreedyBandit(k=10, epsilon=0.1)

# Create test environment with true action values
env = BanditTestEnvironment(k=10, seed=42)

# Simulate interaction
for step in range(1000):
    action = agent.select_action()
    reward = env.get_reward(action)
    agent.update(action, reward)

print(f"Final Q-values: {agent.q_values}")
print(f"Optimal action: {env.optimal_action}")
print(f"Agent's best action: {agent.get_greedy_action()}")
```

### Sutton & Barto Experiments

```python
from examples.sutton_barto_bandit_comparison import run_sutton_barto_experiment

# Reproduce the classic ε-greedy vs greedy comparison
results = run_sutton_barto_experiment(
    epsilon_values=[0.0, 0.01, 0.1],
    k=10,
    num_problems=2000,
    num_steps=1000,
    plot=True
)
```

### RL Algorithms with Gymnasium

```python
import gymnasium as gym
from learnrl.td import QLearning  # Example (not yet implemented)

# Create environment and agent
env = gym.make('CartPole-v1')
agent = QLearning(env.observation_space, env.action_space, lr=0.1)

# Training loop
for episode in range(1000):
    observation, info = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.select_action(observation)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, action, reward, next_obs, terminated)
        observation = next_obs

print(f"Training completed over {episode + 1} episodes")
```

## Module Structure

```
learnrl/
├── bandits/           # Multi-armed bandit algorithms
│   ├── __init__.py
│   └── epsilon_greedy.py
├── dp/                # Dynamic programming (coming soon)
├── td/                # Temporal difference learning (coming soon)
├── utils/             # Utilities and test environments
│   ├── __init__.py
│   └── bandit_env.py  # BanditTestEnvironment for experiments
└── __init__.py
examples/
├── sutton_barto_bandit_comparison.py  # Full S&B experiment
└── quick_bandit_demo.py               # Quick demo
tests/
├── bandits/           # Tests for bandit algorithms
├── utils/             # Tests for utility functions
├── conftest.py        # Test fixtures and configuration
└── test_integration.py # Integration tests
```

## Algorithms Implemented

### Multi-Armed Bandits
- [x] **Epsilon-Greedy** (`EpsilonGreedyBandit`) - Section 2.3
  - ✅ Complete implementation with sample averaging and fixed step-size
  - ✅ Comprehensive test suite (23 tests)
  - ✅ Sutton & Barto experimental comparison
- [x] **Bandit Test Environment** (`BanditTestEnvironment`) - Experimental utility
  - ✅ Generates random k-armed bandit problems
  - ✅ Supports Sutton & Barto experimental setup
  - ✅ Full test coverage (20+ tests)
- [ ] **Upper Confidence Bound (UCB)** - Section 2.7
- [ ] **Thompson Sampling** - Section 2.8

### Dynamic Programming
- [ ] **Value Iteration** - Section 4.4
- [ ] **Policy Iteration** - Section 4.3

### Temporal Difference Learning
- [ ] **Monte Carlo** - Chapter 5
- [ ] **TD(0)** - Section 6.1
- [ ] **SARSA** - Section 6.4
- [ ] **Q-Learning** - Section 6.5

## Development

### Test Coverage
Current test statistics:
- **63 total tests** across all modules
- **100% coverage** for implemented algorithms
- **Integration tests** for algorithm comparisons
- **Parametrized tests** for edge cases and different configurations

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=learnrl

# Run specific test modules
pytest tests/bandits/
pytest tests/utils/
```

### Code Quality
All code must pass these quality checks:

```bash
# Format code
black learnrl/ tests/ examples/

# Type checking
mypy learnrl/

# Lint code
flake8 learnrl/

# Run all quality checks
black learnrl/ tests/ examples/ && mypy learnrl/ && flake8 learnrl/ && pytest
```

### Gymnasium Integration

All RL algorithms (TD/DP modules) are designed to work seamlessly with Gymnasium environments for testing and evaluation. This enables:

- **Standard Interface**: Algorithms accept `observation_space` and `action_space` parameters
- **Environment Compatibility**: Works with any Gymnasium environment (CartPole, MountainCar, etc.)
- **Proper Episode Handling**: Supports `terminated` and `truncated` flags
- **Learning Validation**: Test algorithms on standard benchmarks

**Supported Environments**: CartPole-v1, MountainCar-v0, LunarLander-v3, and more.

**Testing**: All algorithms include Gymnasium integration tests to ensure proper environment compatibility.

## License

MIT License - see LICENSE file for details.

Copyright (c) 2025 Keith Lawson

## Reference

- Sutton, R. S., & Barto, A. G. (2020). [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.). MIT Press.
- [Richard Sutton's personal page for the book](http://incompleteideas.net/book/the-book-2nd.html)

## Contact

- **Author**: Keith Lawson
- **Website**: [keithlawson.me](https://keithlawson.me)
- **LinkedIn**: [j-keith-lawson](https://linkedin.com/in/j-keith-lawson)