from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agentsystem",
    version="0.1.0",
    description="A powerful agent framework for building AI-powered automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AgentSystem Team",
    author_email="info@agentsystem.ai",
    url="https://github.com/agentsystem/agentsystem",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "anthropic": ["anthropic>=0.5.0"],
        "local": ["llama-cpp-python>=0.2.0"],
        "all": ["anthropic>=0.5.0", "llama-cpp-python>=0.2.0"],
    },
    entry_points={
        "console_scripts": [
            "agentsystem=AgentSystem.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.8",
)
