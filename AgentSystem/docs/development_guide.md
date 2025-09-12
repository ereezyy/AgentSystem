# Development Guide

This guide provides instructions for developing, testing, and contributing to the AgentSystem project.

## Setup Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AgentSystem.git
   cd AgentSystem
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Running Tests

The project uses pytest for testing. Tests are located in the `tests/` directory.

### Running All Tests

```bash
pytest
```

This will run all tests and generate coverage reports.

### Running Specific Test Types

- Run unit tests only:
  ```bash
  pytest -m unit
  ```

- Run integration tests only:
  ```bash
  pytest -m integration
  ```

- Skip slow tests:
  ```bash
  pytest -m "not slow"
  ```

### Test Coverage

Coverage reports are generated automatically when running tests:

- Terminal report: Shows coverage in the console
- HTML report: Generated in `coverage_html/` directory
- XML report: Generated as `coverage.xml`

View the HTML coverage report:
```bash
open coverage_html/index.html  # On Windows: start coverage_html/index.html
```

## Code Quality

### Code Formatting

The project uses Black for code formatting:

```bash
# Check formatting
black --check .

# Apply formatting
black .
```

### Linting

Flake8 is used for linting:

```bash
flake8 .
```

### Type Checking

MyPy is used for static type checking:

```bash
mypy AgentSystem
```

## Project Structure

```
AgentSystem/
├── AgentSystem/
│   ├── modules/
│   │   ├── knowledge_manager.py    # Knowledge storage and retrieval
│   │   ├── web_researcher.py       # Web research capabilities
│   │   ├── code_modifier.py        # Code analysis and modification
│   │   └── learning_agent.py       # Main integration module
│   ├── utils/
│   │   └── logger.py              # Logging utilities
│   └── services/
│       └── ai.py                  # AI service integration
├── tests/
│   └── test_learning_system.py    # Test suite
├── docs/
│   └── development_guide.md       # This guide
└── examples/
    └── self_learning_example.py   # Usage examples
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

1. Tests are run on Python 3.8, 3.9, and 3.10
2. Code quality checks (Black, Flake8, MyPy)
3. Coverage reports are generated and uploaded to Codecov
4. Automatic deployment to PyPI on main branch pushes

### Local CI Checks

Run all CI checks locally before pushing:

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
./scripts/run_ci_checks.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and ensure:
   - All tests pass
   - Code is formatted with Black
   - No linting errors
   - Type hints are correct
   - New tests are added for new functionality

4. Update documentation if needed

5. Create a pull request:
   - Clear description of changes
   - Reference any related issues
   - Include test results and coverage report

## Best Practices

1. **Testing**
   - Write tests before implementing features (TDD)
   - Maintain high test coverage
   - Use appropriate test markers (unit/integration/slow)

2. **Code Quality**
   - Follow PEP 8 style guide
   - Use type hints consistently
   - Keep functions and methods focused and small
   - Document public APIs

3. **Git Workflow**
   - Make frequent, small commits
   - Write clear commit messages
   - Keep branches up to date with main
   - Squash commits before merging

4. **Documentation**
   - Update docstrings for all public APIs
   - Keep README and guides up to date
   - Document complex algorithms and decisions

## Troubleshooting

Common issues and solutions:

1. **Test Database Issues**
   - Ensure temp directories are cleaned up
   - Check file permissions
   - Verify SQLite installation

2. **Import Errors**
   - Verify virtual environment is activated
   - Check all dependencies are installed
   - Ensure Python path is correct

3. **CI Pipeline Failures**
   - Run checks locally first
   - Check Python version compatibility
   - Verify dependencies are up to date

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create and push tag:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```
4. CI will automatically deploy to PyPI

## Getting Help

- Create an issue for bugs or feature requests
- Join our Discord community for discussions
- Check the FAQ in the wiki
- Review existing issues and pull requests
