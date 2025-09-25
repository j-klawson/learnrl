# GitHub Actions CI/CD

This directory contains the GitHub Actions workflows for the LearnRL project.

## Workflows

### CI Workflow (`ci.yml`)
- **Runs on**: Debian 12 containers
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **Features**:
  - Automated testing with pytest
  - Code coverage reporting with codecov
  - Code quality checks (Black, Flake8, MyPy)
  - Security scanning (Bandit, Safety)
  - Gymnasium integration testing

### Jobs

1. **test**: Main testing job that runs on all Python versions
   - Installs dependencies
   - Runs linting (Flake8)
   - Type checking (MyPy)
   - Test execution with coverage
   - Uploads coverage to Codecov

2. **test-with-gymnasium**: Tests Gymnasium integration
   - Tests with optional Gymnasium dependencies
   - Runs CartPole example to verify functionality

3. **code-quality**: Comprehensive code quality checks
   - Black formatting verification
   - Extended Flake8 linting
   - Strict MyPy type checking

4. **security**: Security scanning
   - Bandit for security vulnerabilities
   - Safety for dependency security issues

## Coverage Reporting

Coverage reports are generated in multiple formats:
- Terminal output with missing lines
- HTML report (available as artifact)
- XML report (uploaded to Codecov)

## Running Locally

To run the same checks locally:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=learnrl --cov-report=term-missing

# Check formatting
black --check learnrl/

# Run linting
flake8 learnrl/ --max-line-length=88

# Type checking
mypy learnrl/

# Security checks
bandit -r learnrl/
safety check
```