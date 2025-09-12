from setuptools import setup, find_packages

setup(
    name="agentsystem-ultimate-sdk",
    version="3.0.0",
    description="The most powerful AI SDK ever created - AgentSystem Ultimate",
    author="AgentSystem",
    author_email="support@agentsystem.ai",
    url="https://github.com/agentsystem/ultimate-python-sdk",
    py_modules=["agentsystem"],
    install_requires=["aiohttp>=3.8.0"],
    python_requires=">=3.8",
)