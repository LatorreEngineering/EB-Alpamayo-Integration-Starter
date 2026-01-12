# Contributing to EB-Alpamayo-Integration-Starter

Thank you for your interest in contributing to this project! We welcome contributions from the autonomous driving and AI research community.

## 🚗 Automotive Software Engineering Principles

This project follows automotive software engineering best practices inspired by standards like ISO 26262, AUTOSAR, and ROS2. When contributing, please keep these principles in mind:

### 1. **Safety Awareness**
- Even in research code, consider potential safety implications
- Document assumptions about operating domains (ODDs)
- Include plausibility checks and validation where appropriate
- Never remove safety warnings or disclaimers

### 2. **Modularity & Interfaces**
- Design clear interfaces between components
- Use abstract base classes for swappable implementations
- Follow dependency inversion principles
- Keep coupling low, cohesion high

### 3. **Traceability**
- Link code to requirements (even informal ones in issues)
- Document design decisions in code comments
- Use structured logging with severity levels
- Maintain changelog discipline

### 4. **Testability**
- Write unit tests for all new functionality
- Include integration tests for end-to-end workflows
- Use mocking for external dependencies (model APIs, simulators)
- Aim for >80% code coverage

---

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+ 
- NVIDIA GPU with CUDA support (for testing)
- Git
- Docker (optional but recommended)

### Local Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/EB-Alpamayo-Integration-Starter.git
cd EB-Alpamayo-Integration-Starter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/ -v
```

### Docker Development

```bash
# Build development image
docker build -t eb-alpamayo:dev --target development .

# Run interactive container
docker run --gpus all -it -v $(pwd):/workspace eb-alpamayo:dev bash
```

---

## 📝 Code Style & Standards

### Python Style Guide

We follow **PEP 8** with these tools:

- **Black** (code formatting): `black src/ tests/`
- **Ruff** (linting): `ruff check src/ tests/`
- **MyPy** (type checking): `mypy src/`

Configuration is in `pyproject.toml`.

### Key Style Requirements

1. **Type Hints**: All functions must have type hints
   ```python
   def process_trajectory(
       waypoints: List[Tuple[float, float]], 
       dt: float = 0.1
   ) -> np.ndarray:
       """Process trajectory waypoints into time series.
       
       Args:
           waypoints: List of (x, y) coordinates in meters
           dt: Time step between waypoints in seconds
           
       Returns:
           Numpy array of shape (N, 3) with (x, y, t)
       """
       ...
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def validate_acceleration(
       trajectory: Trajectory,
       max_accel: float = 3.0
   ) -> ValidationResult:
       """Validate trajectory acceleration constraints.
       
       Checks if the trajectory respects maximum acceleration limits
       based on automotive safety standards (ISO 26262 inspired).
       
       Args:
           trajectory: Trajectory object to validate
           max_accel: Maximum allowed acceleration in m/s²
           
       Returns:
           ValidationResult with pass/fail and violation details
           
       Raises:
           ValueError: If trajectory is empty or malformed
           
       Example:
           >>> result = validate_acceleration(traj, max_accel=2.5)
           >>> if not result.is_valid:
           ...     print(f"Violations: {result.violations}")
       """
       ...
   ```

3. **Imports**: Organized and sorted (handled by Ruff)
   ```python
   # Standard library
   import logging
   from typing import List, Optional, Tuple
   
   # Third-party
   import numpy as np
   import torch
   from transformers import AutoModel
   
   # Local
   from src.safety_checks import TrajectoryValidator
   ```

4. **Line Length**: Max 100 characters (enforced by Black)

5. **Naming Conventions**:
   - Classes: `PascalCase` (e.g., `TrajectoryValidator`)
   - Functions/methods: `snake_case` (e.g., `validate_trajectory`)
   - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ACCELERATION`)
   - Private members: `_leading_underscore` (e.g., `_internal_cache`)

---

## 🧪 Testing Guidelines

### Writing Tests

- Use **pytest** framework
- Place tests in `tests/` directory mirroring `src/` structure
- Name test files `test_*.py`
- Name test functions `test_*`

### Test Categories

Mark tests appropriately:

```python
import pytest

@pytest.mark.gpu
def test_model_inference_gpu():
    """Tests that require GPU access."""
    ...

@pytest.mark.slow
def test_full_simulation_loop():
    """Tests that take >10 seconds."""
    ...

@pytest.mark.integration
def test_alpasim_integration():
    """Integration tests with external systems."""
    ...
```

Run specific categories:
```bash
# Run all tests except slow ones
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure (AAA Pattern)

```python
def test_trajectory_validation_exceeds_limits():
    """Test that trajectory validator detects excessive acceleration."""
    # Arrange
    validator = TrajectoryValidator(max_accel=2.0)
    trajectory = create_test_trajectory(accel=5.0)  # Exceeds limit
    
    # Act
    result = validator.validate(trajectory)
    
    # Assert
    assert not result.is_valid
    assert "acceleration" in result.violation_types
    assert result.max_violation > 2.0
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_model_inference_without_gpu():
    """Test model wrapper with mocked transformers."""
    with patch('transformers.AutoModel.from_pretrained') as mock_model:
        mock_model.return_value = Mock()
        
        wrapper = AlpamayoInference(model_name="nvidia/Alpamayo-R1-10B")
        result = wrapper.predict(images=[test_image])
        
        assert result is not None
        mock_model.assert_called_once()
```

---

## 🔀 Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

Examples:
- `feature/add-ros2-integration`
- `fix/trajectory-validation-bug`
- `docs/update-alpasim-instructions`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semi colons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(wrapper): add support for 4-bit quantization

Implements bitsandbytes 4-bit quantization for Alpamayo model
to enable inference on GPUs with <16GB VRAM.

Closes #42
```

```
fix(safety): correct lateral acceleration calculation

Previous implementation used incorrect coordinate transformation.
Now properly handles vehicle-frame to world-frame conversion.

Fixes #38
```

### Pull Request Process

1. **Create a fork** and work in a feature branch
2. **Write tests** for new functionality
3. **Update documentation** (README, docstrings, etc.)
4. **Run linters and tests** locally:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   pytest tests/ -v
   ```
5. **Create pull request** with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to related issues
   - Screenshots/videos if UI changes
   - Test results (paste pytest output)

6. **Respond to review feedback** promptly
7. **Squash commits** if requested before merge

---

## 📚 Documentation Standards

### Code Documentation

- **All public functions** must have docstrings
- **Complex algorithms** need inline comments explaining the approach
- **Automotive-specific logic** should reference standards/papers
  ```python
  # Curvature calculation based on ISO 26262 trajectory analysis
  # See: ISO 26262-6:2018, Section 7.4.3
  kappa = calculate_curvature(waypoints)
  ```

### README Updates

When adding new features, update:
- Feature list in main README
- Usage examples
- Requirements if new dependencies added

### Notebook Documentation

Jupyter notebooks should include:
- **Title and description** in first markdown cell
- **Clear section headers** organizing content
- **Explanatory markdown** before each code cell
- **Visualizations** with labeled axes and legends
- **Conclusion/summary** at the end

---

## 🎯 Contribution Ideas

We welcome contributions in these areas:

### High Priority
- [ ] Additional safety validators (TTC, RSS, etc.)
- [ ] ROS2 node wrapper with proper message types
- [ ] AUTOSAR interface adapters
- [ ] Performance benchmarking suite
- [ ] More AlpaSim scenario examples

### Medium Priority
- [ ] Integration with other VLA models (for comparison)
- [ ] Trajectory smoothing/post-processing
- [ ] Advanced visualization (3D, animations)
- [ ] Logging to automotive trace formats (MDF4, etc.)
- [ ] CI/CD pipeline setup

### Documentation
- [ ] Tutorial videos
- [ ] Architecture diagrams
- [ ] Performance optimization guide
- [ ] Deployment best practices

### Research Extensions
- [ ] Uncertainty quantification for VLA outputs
- [ ] Multi-agent scenario support
- [ ] Adversarial testing framework
- [ ] Sim-to-real transfer analysis

---

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be patient** with newcomers
- **Give constructive feedback** in reviews
- **Assume good intent** in discussions
- **Focus on the code**, not the person

### Getting Help

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Email**: For security issues or private concerns

### Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md` (to be created)
- Mentioned in release notes for significant contributions
- Credited in papers/publications using this work (if applicable)

---

## 📄 Legal

By contributing, you agree that:

1. Your contributions will be licensed under the Apache 2.0 License
2. You have the right to contribute the code
3. You understand this is research software, not for production use

For significant contributions (>100 lines), we may request a
[Contributor License Agreement (CLA)](https://en.wikipedia.org/wiki/Contributor_License_Agreement).

---

## 🚀 Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. Create GitHub release with notes
6. (Optional) Publish to PyPI

---

## 📧 Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Open a new discussion in GitHub Discussions


Thank you for helping advance automotive AI research! 🚗🤖
