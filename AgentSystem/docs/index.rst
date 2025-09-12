Welcome to AgentSystem's documentation!
=====================================

AgentSystem is a modular framework for building self-learning AI agents that can research, learn, and modify their own code.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   development_guide

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   modules/knowledge_manager
   modules/web_researcher
   modules/code_modifier
   modules/learning_agent

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/basic_usage
   examples/advanced_features
   examples/customization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/knowledge_manager
   api/web_researcher
   api/code_modifier
   api/learning_agent

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Features
--------

- Knowledge Base Management
    - Persistent storage of facts and documents
    - Full-text search capabilities
    - Automatic deduplication
    - Knowledge pruning

- Web Research
    - Autonomous web searching
    - Content extraction and sanitization
    - Rate limiting and caching
    - Robust error handling

- Code Modification
    - Safe code analysis and modification
    - Backup and rollback capabilities
    - Syntax validation
    - AI-powered improvements

- Continuous Learning
    - Background learning tasks
    - Autonomous research
    - Self-improvement capabilities
    - Progress monitoring

Installation
-----------

Install AgentSystem using pip:

.. code-block:: bash

   pip install agentsystem

For development installation:

.. code-block:: bash

   git clone https://github.com/yourusername/AgentSystem.git
   cd AgentSystem
   pip install -e ".[dev]"

Quick Start
----------

Here's a simple example of using AgentSystem:

.. code-block:: python

   from AgentSystem.modules.learning_agent import LearningAgent

   # Initialize the agent
   agent = LearningAgent()

   # Start background learning
   agent.start_learning()

   # Queue research topics
   agent.queue_research("artificial intelligence safety")
   agent.queue_research("machine learning best practices")

   # Queue code improvements
   agent.queue_code_improvement("my_module.py")

   # Monitor progress
   stats = agent.get_knowledge_stats()
   print(f"Learned {stats['facts']} facts across {stats['categories']} categories")

Contributing
-----------

We welcome contributions! Please see our :doc:`development_guide` for details on how to:

- Set up your development environment
- Run tests
- Submit pull requests
- Follow our coding standards

License
-------

AgentSystem is released under the MIT License. See LICENSE file for details.

Support
-------

- Documentation: https://agentsystem.readthedocs.io/
- GitHub Issues: https://github.com/yourusername/AgentSystem/issues
- Discord Community: https://discord.gg/agentsystem
