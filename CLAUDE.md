# Claude Code Configuration

This file contains configuration for Claude Code to help with development of the LearnRL package.

## Development Commands

### Testing
```bash
pytest
```

### Code Formatting
```bash
black learnrl/ tests/ examples/
```

### Type Checking
```bash
mypy learnrl/
```

### Linting
```bash
flake8 learnrl/
```

### Install Package in Development Mode
```bash
pip install -e .
```

### Install with Development Dependencies
```bash
pip install -e ".[dev]"
```

### Install with Gymnasium Support
```bash
pip install -e ".[gym]"
```

### Install with All Dependencies
```bash
pip install -e ".[dev,gym]"
```

## Project Structure

- `learnrl/` - Main package directory
  - `learnrl/bandits/` - Multi-armed bandit algorithms
    - `epsilon_greedy.py` - EpsilonGreedyBandit implementation
  - `learnrl/dp/` - Dynamic programming algorithms (coming soon)
  - `learnrl/td/` - Temporal difference learning algorithms (coming soon)
  - `learnrl/utils/` - Utility functions and test environments
    - `bandit_env.py` - BanditTestEnvironment for experiments
- `examples/` - Example scripts and demos
  - `sutton_barto_bandit_comparison.py` - Full S&B experimental comparison
  - `quick_bandit_demo.py` - Quick demonstration script
- `tests/` - Comprehensive test suite (63 tests)
  - `tests/bandits/` - Tests for bandit algorithms
  - `tests/utils/` - Tests for utility functions
  - `tests/conftest.py` - Test fixtures and configuration
  - `tests/test_integration.py` - Integration and experiment tests

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Include comprehensive docstrings following NumPy style
- Prefer PyTorch tensors over NumPy arrays for numerical computations
- Keep implementations simple and educational rather than optimized
- Follow the algorithm descriptions from Sutton & Barto (2020)

### Code Quality Checks

Before committing any code, ensure it passes all quality checks:

1. **Format with Black**: `black learnrl/ tests/ examples/`
2. **Type check with Mypy**: `mypy learnrl/`
3. **Lint with Flake8**: `flake8 learnrl/`
4. **Run tests**: `pytest`

All code must pass these checks before being considered complete.

**Quick command to run all checks:**
```bash
black learnrl/ tests/ examples/ && mypy learnrl/ && flake8 learnrl/ && pytest
```

### Current Implementation Status

**Completed:**
- ✅ EpsilonGreedyBandit with comprehensive test suite (23 tests)
- ✅ BanditTestEnvironment for experimental setup (20+ tests)
- ✅ Sutton & Barto Section 2.3 experiment reproduction
- ✅ Complete test infrastructure with fixtures and integration tests
- ✅ Full code quality pipeline (Black, MyPy, Flake8, Pytest)

**Test Coverage:**
- 63 total tests across all modules
- 100% coverage for implemented algorithms
- Integration tests for algorithm comparisons
- Parametrized tests for edge cases

## Dependencies

Core dependencies:
- `torch` - For tensor computations
- `numpy` - For numerical operations
- `matplotlib` - For plotting and visualization

Development dependencies:
- `pytest` - Testing framework
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checker

Optional dependencies:
- `gymnasium` - For environment-based testing and evaluation

## Gymnasium Integration

**CRITICAL**: All RL algorithm implementations MUST support Gymnasium environments for testing and evaluation.

### Algorithm Design Requirements

When implementing RL algorithms (TD/DP modules), ensure:

1. **Standard Gymnasium Interface**:
   - Accept `env.observation_space` and `env.action_space` as constructor parameters
   - Handle the standard `step()` return: `(observation, reward, terminated, truncated, info)`
   - Support both discrete and continuous action/observation spaces where applicable

2. **Environment Compatibility**:
   - Design agents to work with any Gymnasium environment
   - Keep core algorithm logic separate from environment interaction
   - Provide clear interfaces for environment integration

3. **Episode Management**:
   - Handle `terminated` and `truncated` flags properly
   - Support episodic and continuing tasks
   - Reset handling between episodes

4. **Testing Integration**:
   - Include Gymnasium environment tests in test suites
   - Test with common environments like CartPole, MountainCar, etc.
   - Validate learning curves and performance metrics

### Implementation Pattern

```python
class AlgorithmAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        # Initialize based on environment spaces
        pass

    def select_action(self, observation):
        # Return action compatible with environment
        pass

    def update(self, observation, action, reward, next_observation, terminated, truncated):
        # Update algorithm with environment feedback
        pass
```

### Testing with Gymnasium

All implementations should include tests demonstrating:
- Successful training on at least one discrete control task
- Proper handling of episode termination
- Learning progress over episodes

**Note**: Bandits use synthetic reward distributions rather than environments, so Gymnasium integration is not required for bandit algorithms.
- Be sure all algorithm implementations will work with gymnasium
- It's 2025. the copyright should be 2025, not 2024.