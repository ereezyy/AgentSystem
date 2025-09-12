Installation
============

AgentSystem can be installed using pip or from source.

Requirements
-----------

- Python 3.8 or higher
- pip package manager

Quick Installation
-----------------

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install agentsystem

Development Installation
-----------------------

For development or to get the latest features, install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/AgentSystem.git
   cd AgentSystem
   pip install -e ".[dev]"

This will install AgentSystem in development mode with all development dependencies.

Optional Dependencies
-------------------

AgentSystem has several optional dependencies for enhanced functionality:

Machine Learning Features
~~~~~~~~~~~~~~~~~~~~~~~~~

For similarity search and embeddings:

.. code-block:: bash

   pip install sentence-transformers scikit-learn

Natural Language Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

For query expansion and text processing:

.. code-block:: bash

   pip install nltk spacy

Web Research
~~~~~~~~~~~

For enhanced web scraping:

.. code-block:: bash

   pip install selenium beautifulsoup4 requests

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Install all optional dependencies at once:

.. code-block:: bash

   pip install "agentsystem[full]"

Verification
-----------

Verify your installation by running:

.. code-block:: python

   import AgentSystem
   print(f"AgentSystem version: {AgentSystem.__version__}")

   # Test basic functionality
   from AgentSystem.modules.knowledge_manager import KnowledgeManager
   km = KnowledgeManager()
   print("Knowledge Manager initialized successfully!")

Docker Installation
------------------

Run AgentSystem in a Docker container:

.. code-block:: bash

   docker pull agentsystem/agentsystem:latest
   docker run -it agentsystem/agentsystem:latest

Build from Dockerfile:

.. code-block:: bash

   git clone https://github.com/yourusername/AgentSystem.git
   cd AgentSystem
   docker build -t agentsystem .
   docker run -it agentsystem

Configuration
------------

Create a configuration file at `~/.agentsystem/config.yaml`:

.. code-block:: yaml

   # AgentSystem Configuration
   database:
     path: "~/.agentsystem/knowledge.db"
     
   ai_providers:
     openai:
       api_key: "your-api-key-here"
       model: "gpt-4"
     
   embeddings:
     model: "sentence-transformers/all-MiniLM-L6-v2"
     batch_size: 32
     
   web_research:
     max_pages: 10
     timeout: 30
     rate_limit: 1.0

Environment Variables
-------------------

Set the following environment variables:

.. code-block:: bash

   export AGENTSYSTEM_CONFIG_PATH="~/.agentsystem/config.yaml"
   export AGENTSYSTEM_DB_PATH="~/.agentsystem/knowledge.db"
   export AGENTSYSTEM_LOG_LEVEL="INFO"

Troubleshooting
--------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Permission Errors**

On Unix systems, you may need to use sudo:

.. code-block:: bash

   sudo pip install agentsystem

Or install for the current user only:

.. code-block:: bash

   pip install --user agentsystem

**Missing System Dependencies**

Some optional dependencies require system packages:

Ubuntu/Debian:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install build-essential python3-dev

CentOS/RHEL:

.. code-block:: bash

   sudo yum install gcc python3-devel

macOS:

.. code-block:: bash

   xcode-select --install

**SSL Certificate Issues**

If you encounter SSL errors:

.. code-block:: bash

   pip install --trusted-host pypi.org --trusted-host pypi.python.org agentsystem

Testing Installation
-------------------

Run the test suite to verify everything works:

.. code-block:: bash

   cd AgentSystem
   python -m pytest tests/

Or run specific tests:

.. code-block:: bash

   python -m pytest tests/test_knowledge_manager.py -v

Getting Help
-----------

If you encounter issues:

1. Check the `troubleshooting guide <https://agentsystem.readthedocs.io/en/latest/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/yourusername/AgentSystem/issues>`_
3. Create a new issue with detailed information about your problem
4. Join our `Discord community <https://discord.gg/agentsystem>`_
