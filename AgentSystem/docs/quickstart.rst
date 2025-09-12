Quick Start Guide
================

This guide will help you get started with AgentSystem in just a few minutes.

Installation
-----------

Install AgentSystem using pip:

.. code-block:: bash

   pip install agentsystem

Or install with all optional dependencies:

.. code-block:: bash

   pip install "agentsystem[full]"

Basic Usage
----------

Here's a simple example to get you started:

.. code-block:: python

   from AgentSystem.modules.learning_agent import LearningAgent

   # Create a learning agent
   agent = LearningAgent()

   # Add some initial knowledge
   agent.knowledge_manager.add_fact(
       "Python is a high-level programming language",
       confidence=0.95,
       source="programming guide",
       category="programming"
   )

   # Search for knowledge
   results = agent.knowledge_manager.search_facts("Python programming")
   print(f"Found {len(results)} facts about Python")

   # Get knowledge statistics
   stats = agent.knowledge_manager.get_knowledge_stats()
   print(f"Total facts: {stats['total_facts']}")

Knowledge Management
------------------

The Knowledge Manager is the core of AgentSystem:

.. code-block:: python

   from AgentSystem.modules.knowledge_manager import KnowledgeManager

   # Initialize knowledge manager
   km = KnowledgeManager()

   # Add facts with metadata
   km.add_fact(
       content="Machine learning is a subset of artificial intelligence",
       confidence=0.9,
       source="AI textbook",
       category="artificial intelligence"
   )

   # Search for similar facts (requires embeddings)
   similar_facts = km.search_similar_facts(
       "What is machine learning?", 
       threshold=0.8
   )

   # Add documents
   km.add_document(
       title="Introduction to AI",
       content="Artificial intelligence is the simulation of human intelligence...",
       source="AI course materials",
       category="education"
   )

Web Research
-----------

AgentSystem can autonomously research topics on the web:

.. code-block:: python

   from AgentSystem.modules.web_researcher import WebResearcher

   # Initialize web researcher
   researcher = WebResearcher()

   # Research a topic
   results = researcher.research_topic("artificial intelligence trends 2024")
   
   print(f"Found {len(results)} relevant articles")
   for result in results[:3]:  # Show first 3 results
       print(f"- {result['title']}")
       print(f"  Source: {result['source']}")
       print(f"  Relevance: {result['relevance_score']:.2f}")

Background Learning
-----------------

Set up continuous learning in the background:

.. code-block:: python

   from AgentSystem.modules.learning_agent import LearningAgent

   # Create agent with background learning
   agent = LearningAgent()

   # Start background learning
   agent.start_learning()

   # Queue research topics
   agent.queue_research("machine learning best practices")
   agent.queue_research("python development trends")

   # Queue code improvements
   agent.queue_code_improvement("my_script.py")

   # Monitor progress
   while agent.learning_active:
       stats = agent.get_learning_stats()
       print(f"Processed {stats['completed_tasks']} tasks")
       print(f"Queue size: {stats['queue_size']}")
       
       # Check every 30 seconds
       time.sleep(30)

   # Stop learning
   agent.stop_learning()

Code Modification
---------------

AgentSystem can analyze and improve your code:

.. code-block:: python

   from AgentSystem.modules.code_modifier import CodeModifier

   # Initialize code modifier
   modifier = CodeModifier()

   # Analyze code
   analysis = modifier.analyze_code("my_script.py")
   print(f"Found {len(analysis['issues'])} potential issues")

   # Apply improvements
   if analysis['issues']:
       success = modifier.apply_improvements("my_script.py")
       if success:
           print("Code improvements applied successfully")

Advanced Features
---------------

Similarity Search
~~~~~~~~~~~~~~~~

Find related knowledge using semantic similarity:

.. code-block:: python

   # Search for similar facts
   query = "How does neural network learning work?"
   similar = km.search_similar_facts(query, threshold=0.7)

   for fact in similar:
       print(f"Similarity: {fact['similarity']:.2f}")
       print(f"Content: {fact['content']}")

Knowledge Pruning
~~~~~~~~~~~~~~~~

Remove outdated or low-confidence knowledge:

.. code-block:: python

   # Prune knowledge older than 30 days with low confidence
   pruned = km.prune_knowledge(
       max_age_days=30,
       min_confidence=0.6,
       min_access_count=1
   )
   print(f"Pruned {pruned} outdated facts")

Query Expansion
~~~~~~~~~~~~~~

Expand queries with synonyms for better search results:

.. code-block:: python

   # Expand a query
   expanded = km.expand_query("car")
   print(f"Expanded query: {expanded}")
   # Output: "car automobile vehicle motor"

   # Use expanded query for search
   results = km.search_facts(expanded)

Configuration
-----------

Create a configuration file `~/.agentsystem/config.yaml`:

.. code-block:: yaml

   # Basic configuration
   database:
     path: "~/.agentsystem/knowledge.db"
     
   embeddings:
     model: "sentence-transformers/all-MiniLM-L6-v2"
     similarity_threshold: 0.85
     
   web_research:
     max_pages: 10
     timeout: 30
     rate_limit: 1.0
     
   learning:
     background_interval: 300  # 5 minutes
     max_queue_size: 100

Environment Variables
-------------------

Set environment variables for configuration:

.. code-block:: bash

   export AGENTSYSTEM_CONFIG_PATH="~/.agentsystem/config.yaml"
   export AGENTSYSTEM_DB_PATH="~/.agentsystem/knowledge.db"
   export AGENTSYSTEM_LOG_LEVEL="INFO"

Complete Example
--------------

Here's a complete example that demonstrates all features:

.. code-block:: python

   import time
   from AgentSystem.modules.learning_agent import LearningAgent

   def main():
       # Initialize the learning agent
       agent = LearningAgent()
       
       print("🚀 Starting AgentSystem Demo")
       
       # Add initial knowledge
       agent.knowledge_manager.add_fact(
           "AgentSystem is a self-learning AI framework",
           confidence=1.0,
           source="documentation",
           category="agentsystem"
       )
       
       # Start background learning
       agent.start_learning()
       print("📚 Background learning started")
       
       # Queue some research topics
       topics = [
           "artificial intelligence safety",
           "machine learning best practices",
           "python development trends"
       ]
       
       for topic in topics:
           agent.queue_research(topic)
           print(f"📋 Queued research: {topic}")
       
       # Monitor progress for 2 minutes
       for i in range(12):  # 12 * 10 seconds = 2 minutes
           stats = agent.get_learning_stats()
           knowledge_stats = agent.get_knowledge_stats()
           
           print(f"\n📊 Progress Update #{i+1}:")
           print(f"   Tasks completed: {stats['completed_tasks']}")
           print(f"   Queue size: {stats['queue_size']}")
           print(f"   Total facts: {knowledge_stats['facts']}")
           print(f"   Total documents: {knowledge_stats['documents']}")
           
           time.sleep(10)
       
       # Stop learning
       agent.stop_learning()
       print("\n✅ Demo completed!")
       
       # Show final statistics
       final_stats = agent.get_knowledge_stats()
       print(f"\n📈 Final Results:")
       print(f"   Facts learned: {final_stats['facts']}")
       print(f"   Documents stored: {final_stats['documents']}")
       print(f"   Categories: {final_stats['categories']}")

   if __name__ == "__main__":
       main()

Next Steps
---------

Now that you have AgentSystem running, explore these areas:

1. **Advanced Configuration**: Customize the system for your specific needs
2. **API Integration**: Connect to external APIs for enhanced capabilities
3. **Custom Modules**: Create your own modules for specialized functionality
4. **Performance Tuning**: Optimize for your hardware and use case

Resources
--------

- **Documentation**: https://agentsystem.readthedocs.io/
- **Examples**: Check the `examples/` directory for more complex use cases
- **API Reference**: Detailed API documentation for all modules
- **Community**: Join our Discord server for support and discussions

Troubleshooting
--------------

If you encounter issues:

1. Check the logs in `~/.agentsystem/logs/`
2. Verify your configuration file
3. Ensure all dependencies are installed
4. Check network connectivity for web research features

Happy learning! 🎉
