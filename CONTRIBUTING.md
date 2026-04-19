# Contributing to AgentSystem: Join the Orchestration

We welcome contributions from developers, researchers, and AI enthusiasts who wish to enhance AgentSystem. By contributing, you help us build a more robust, intelligent, and efficient system for distributed AI orchestration.

## 🤝 Code of Conduct

To ensure a welcoming and open environment, we adhere to a [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating in the project.

## 🛠️ How Can I Contribute?

There are several ways you can contribute to AgentSystem:

### 1. Reporting Bugs

*   **Check existing issues**: Before opening a new issue, please check if the bug has already been reported.
*   **Open a new issue**: If not, open a new issue with a clear and descriptive title. Include detailed steps to reproduce the bug, your environment details, and expected vs. actual behavior.

### 2. Suggesting Enhancements

*   Have an idea for a new feature or optimization? Open an issue with the label `enhancement`.
*   Describe the feature clearly and explain why it would be beneficial to the system.

### 3. Pull Requests

We encourage you to open pull requests (PRs) for any changes you wish to contribute. To ensure a smooth review process, please follow these guidelines:

## ✅ Development Setup

To set up your local development environment, please follow the [Installation](#installation) steps in the `README.md` file.

## 🌳 Branching Strategy

We use a `main` branch for stable releases and feature branches for ongoing development. All contributions should be made via feature branches.

*   `main`: The stable branch, always reflecting the latest release.
*   `feature/<feature-name>`: For new features.
*   `bugfix/<bug-description>`: For bug fixes.
*   `docs/<doc-update>`: For documentation improvements.

## 📥 Pull Request Guidelines

1.  **Fork the Repository**: Start by forking the `ereezyy/AgentSystem` repository to your own GitHub account.
2.  **Create a Feature Branch**: Create a new branch from `main` for your changes. Use a descriptive name (e.g., `feature/add-new-provider`, `bugfix/fix-celery-config`).

    ```bash
    git checkout main
    git pull origin main
    git checkout -b feature/your-feature-name
    ```

3.  **Implement Your Changes**: Make your code changes, add new features, or fix bugs.
4.  **Write Tests**: If you've added code that should be tested, please add appropriate unit or integration tests.
5.  **Update Documentation**: If your changes affect any APIs, features, or installation steps, update the relevant documentation (e.g., `README.md`).
6.  **Ensure Code Quality**: Run linting and tests to ensure your code adheres to our standards.
7.  **Commit Your Changes**: Write clear, concise, and descriptive commit messages. We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

    Example:
    ```bash
    git commit -m "feat: Add support for new AI provider"
    ```

8.  **Push to Your Fork**: Push your new branch to your forked repository on GitHub:

    ```bash
    git push origin feature/your-feature-name
    ```

9.  **Create a Pull Request**: Finally, open a pull request from your forked repository to the `main` branch of the original `ereezyy/AgentSystem` repository. Provide a detailed description of your changes and why they are necessary.

## 🔐 Security

If you discover a security vulnerability, please refer to our `SECURITY.md` for responsible disclosure guidelines. Do not open public issues for security vulnerabilities.

Let's build the orchestration together! 🚀
