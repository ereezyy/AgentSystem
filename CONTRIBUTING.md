# Contributing to AgentSystem 🤝

First off, thank you for considering contributing to AgentSystem! It's people like you that make AgentSystem such a great tool for distributed AI orchestration.

This document outlines the process for contributing to the project.

## 🐛 Bug Reports

If you notice a bug, please create an issue and include:
- A clear and descriptive title.
- Steps to reproduce the behavior.
- Expected vs. actual behavior.
- Any relevant logs or error messages.
- Your environment details (OS, Python version, hardware).

## ✨ Feature Requests

Have an idea to improve AgentSystem? We'd love to hear it! Please create an issue and include:
- A clear description of the proposed feature.
- Use cases or reasons why this feature would be valuable.
- Any potential implementation ideas.

## 🛠️ Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Install development dependencies**. Make sure you have `pre-commit` installed.
   ```bash
   pip install pre-commit pytest pytest-cov black flake8
   pre-commit install
   ```
3. **Make your changes**. Ensure your code adheres to the project's style guidelines.
4. **Write tests** for your changes, if applicable.
5. **Run the test suite** to ensure everything passes:
   ```bash
   pytest AgentSystem/tests/
   ```
6. **Ensure pre-commit checks pass**.
7. **Commit your changes**. Write clear, concise commit messages.
8. **Push to your fork** and submit a Pull Request.

## 📝 Coding Standards

- **Formatting**: We use `black` for code formatting.
- **Linting**: We use `flake8` to catch linting errors.
- **Typing**: Please include type hints where possible to improve code clarity.
- **Documentation**: Update README.md or API docs if your changes affect usage.

## 💬 Community

Be kind and respectful to others. We welcome contributors of all skill levels.
If you need help, feel free to ask questions in the issues or pull requests.

Happy hacking! 🚀
