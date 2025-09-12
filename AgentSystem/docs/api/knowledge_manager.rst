Knowledge Manager API
====================

The Knowledge Manager module provides advanced knowledge base management capabilities including fact deduplication, similarity detection, knowledge pruning, and context-aware query expansion.

.. automodule:: AgentSystem.modules.knowledge_manager
   :members:
   :undoc-members:
   :show-inheritance:

KnowledgeManager Class
---------------------

.. autoclass:: AgentSystem.modules.knowledge_manager.KnowledgeManager
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Core Methods
~~~~~~~~~~~~

.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.add_fact
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.search_facts
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.search_similar_facts
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.add_document
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.search_documents

Advanced Features
~~~~~~~~~~~~~~~~

.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.prune_knowledge
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.expand_query
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.get_knowledge_stats
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.get_fact_categories
.. automethod:: AgentSystem.modules.knowledge_manager.KnowledgeManager.update_fact_access

Database Schema
--------------

The Knowledge Manager uses the following database tables:

Facts Table
~~~~~~~~~~

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS facts (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       content TEXT NOT NULL,
       confidence REAL DEFAULT 1.0,
       source TEXT,
       category TEXT,
       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
       access_count INTEGER DEFAULT 0,
       last_accessed DATETIME,
       embedding_id INTEGER,
       FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
   );

Documents Table
~~~~~~~~~~~~~~

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS documents (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       title TEXT NOT NULL,
       content TEXT NOT NULL,
       source TEXT,
       category TEXT,
       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
       access_count INTEGER DEFAULT 0,
       last_accessed DATETIME,
       embedding_id INTEGER,
       FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
   );

Embeddings Table
~~~~~~~~~~~~~~~

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS embeddings (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       vector BLOB NOT NULL,
       model_name TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
       created_at DATETIME DEFAULT CURRENT_TIMESTAMP
   );

Usage Examples
-------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from AgentSystem.modules.knowledge_manager import KnowledgeManager

   # Initialize the knowledge manager
   km = KnowledgeManager()

   # Add facts
   km.add_fact("The Earth is approximately 4.5 billion years old", 
               confidence=0.95, source="geology textbook", category="science")
   
   # Search for facts
   results = km.search_facts("Earth age")
   print(f"Found {len(results)} facts about Earth's age")

   # Get knowledge statistics
   stats = km.get_knowledge_stats()
   print(f"Total facts: {stats['total_facts']}")
   print(f"Total documents: {stats['total_documents']}")

Advanced Features
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Search for similar facts (requires embeddings)
   similar = km.search_similar_facts("How old is our planet?", threshold=0.8)
   for fact in similar:
       print(f"Similarity: {fact['similarity']:.2f} - {fact['content']}")

   # Prune old or low-confidence knowledge
   pruned = km.prune_knowledge(max_age_days=30, min_confidence=0.6)
   print(f"Pruned {pruned} old or low-confidence facts")

   # Expand queries with synonyms
   expanded = km.expand_query("car")
   print(f"Expanded query: {expanded}")

Configuration
------------

The Knowledge Manager can be configured with the following parameters:

.. code-block:: python

   km = KnowledgeManager(
       db_path="custom_knowledge.db",
       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
       similarity_threshold=0.85,
       max_query_expansion=10
   )

Error Handling
-------------

The Knowledge Manager includes comprehensive error handling:

.. code-block:: python

   try:
       km.add_fact("Sample fact")
   except ValueError as e:
       print(f"Invalid fact: {e}")
   except Exception as e:
       print(f"Database error: {e}")

Performance Considerations
-------------------------

- **Embeddings**: Computing embeddings is computationally expensive. Enable only when similarity search is needed.
- **Database Size**: Large knowledge bases may require periodic pruning for optimal performance.
- **Query Expansion**: Limit the number of expanded terms to avoid overly broad searches.
- **Indexing**: The FTS5 index provides fast text search but requires additional storage space.

Dependencies
-----------

Required packages:

- ``sqlite3`` (built-in)
- ``sentence-transformers`` (optional, for similarity search)
- ``scikit-learn`` (optional, for cosine similarity)
- ``nltk`` (optional, for query expansion)

Optional packages for enhanced functionality:

.. code-block:: bash

   pip install sentence-transformers scikit-learn nltk
