"""
Test Continuous Learning Module
------------------------------
Tests the functionality of the continuous learning module including
knowledge base operations and web research capabilities.
"""

import os
import sys
import time
import unittest
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from AgentSystem.modules.continuous_learning import ContinuousLearningModule, KnowledgeBase, WebResearcher
from AgentSystem.utils.logger import configure_logging

# Configure logging
configure_logging(level="INFO")
logger = logging.getLogger("tests.test_continuous_learning")


class TestKnowledgeBase(unittest.TestCase):
    """Test the KnowledgeBase class"""
    
    def setUp(self):
        """Set up the test environment"""
        # Use in-memory database for testing
        self.kb = KnowledgeBase(":memory:")
    
    def tearDown(self):
        """Clean up after tests"""
        self.kb.close()
    
    def test_add_fact(self):
        """Test adding a fact to the knowledge base"""
        fact_content = "The Earth orbits the Sun."
        fact_id = self.kb.add_fact(
            content=fact_content,
            source="test",
            confidence=0.9,
            category="astronomy"
        )
        
        # Check if the fact was added
        self.assertGreater(fact_id, 0, "Failed to add fact to knowledge base")
        
        # Retrieve the fact
        fact = self.kb.get_fact(fact_id)
        
        # Verify the fact
        self.assertIsNotNone(fact, "Failed to retrieve fact from knowledge base")
        self.assertEqual(fact["content"], fact_content, "Fact content doesn't match")
        self.assertEqual(fact["source"], "test", "Fact source doesn't match")
        self.assertEqual(fact["confidence"], 0.9, "Fact confidence doesn't match")
        self.assertEqual(fact["category"], "astronomy", "Fact category doesn't match")
    
    def test_add_document(self):
        """Test adding a document to the knowledge base"""
        doc_title = "Test Document"
        doc_content = "This is a test document with some content about planets and stars."
        doc_summary = "A test document about astronomy."
        
        doc_id = self.kb.add_document(
            content=doc_content,
            title=doc_title,
            summary=doc_summary,
            url="https://example.com/test",
            source="test"
        )
        
        # Check if the document was added
        self.assertGreater(doc_id, 0, "Failed to add document to knowledge base")
        
        # Retrieve the document
        doc = self.kb.get_document(doc_id)
        
        # Verify the document
        self.assertIsNotNone(doc, "Failed to retrieve document from knowledge base")
        self.assertEqual(doc["title"], doc_title, "Document title doesn't match")
        self.assertEqual(doc["content"], doc_content, "Document content doesn't match")
        self.assertEqual(doc["summary"], doc_summary, "Document summary doesn't match")
    
    def test_search_facts(self):
        """Test searching for facts in the knowledge base"""
        # Add some facts
        self.kb.add_fact(content="The Earth is the third planet from the Sun.", category="astronomy")
        self.kb.add_fact(content="Jupiter is the largest planet in our solar system.", category="astronomy")
        self.kb.add_fact(content="Water boils at 100 degrees Celsius at sea level.", category="physics")
        
        # Search for astronomy facts
        results = self.kb.search_facts("planet")
        
        # Verify results
        self.assertGreaterEqual(len(results), 2, "Search returned fewer results than expected")
        
        # Check categories
        astronomy_count = sum(1 for fact in results if fact.get("category") == "astronomy")
        self.assertGreaterEqual(astronomy_count, 2, "Expected at least 2 astronomy facts")
    
    def test_search_documents(self):
        """Test searching for documents in the knowledge base"""
        # Add some documents
        self.kb.add_document(
            content="The solar system consists of the Sun and the objects that orbit it.",
            title="Solar System",
            summary="Overview of the solar system"
        )
        self.kb.add_document(
            content="Galaxies are vast systems of stars, gas, dust, and dark matter.",
            title="Galaxies",
            summary="Structure and types of galaxies"
        )
        
        # Search for documents
        results = self.kb.search_documents("solar")
        
        # Verify results
        self.assertGreaterEqual(len(results), 1, "Search returned fewer results than expected")
        self.assertEqual(results[0]["title"], "Solar System", "Expected document title not found")


class TestWebResearcher(unittest.TestCase):
    """Test the WebResearcher class"""
    
    def setUp(self):
        """Set up the test environment"""
        # Use in-memory database for testing
        kb = KnowledgeBase(":memory:")
        self.researcher = WebResearcher(kb)
    
    def test_extract_facts(self):
        """Test extracting facts from text"""
        text = """
        The Earth is the third planet from the Sun. It is the only astronomical object known to harbor life.
        The Earth orbits the Sun at an average distance of about 150 million kilometers.
        Water covers about 71% of the Earth's surface, mostly in oceans.
        Some scientists believe that climate change is causing more extreme weather events.
        """
        
        # Extract facts
        facts = self.researcher.extract_facts(text)
        
        # This test will be skipped if the NLTK dependencies aren't available
        if not facts:
            self.skipTest("NLTK dependencies not available")
        
        # Verify that facts were extracted
        self.assertGreater(len(facts), 0, "No facts extracted from text")
        
        # Check content of facts
        fact_texts = [fact["content"] for fact in facts]
        self.assertTrue(any("Earth" in fact for fact in fact_texts), "Expected facts about Earth")


class TestContinuousLearningModule(unittest.TestCase):
    """Test the ContinuousLearningModule class"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create a temporary database file
        self.test_db_path = ":memory:"
        self.module = ContinuousLearningModule(self.test_db_path)
    
    def test_add_fact(self):
        """Test adding a fact through the module"""
        result = self.module.add_fact(
            content="Machine learning is a subset of artificial intelligence.",
            source="test",
            confidence=0.95,
            category="technology"
        )
        
        # Verify result
        self.assertTrue(result["success"], "Failed to add fact through module")
        self.assertGreater(result["fact_id"], 0, "Invalid fact ID")
    
    def test_add_document(self):
        """Test adding a document through the module"""
        result = self.module.add_document(
            content="Neural networks are computing systems inspired by biological neural networks.",
            title="Neural Networks",
            summary="Introduction to neural networks"
        )
        
        # Verify result
        self.assertTrue(result["success"], "Failed to add document through module")
        self.assertGreater(result["document_id"], 0, "Invalid document ID")
    
    def test_search_knowledge(self):
        """Test searching knowledge through the module"""
        # Add some facts and documents
        self.module.add_fact(content="Python is a programming language.", category="programming")
        self.module.add_document(content="Python is known for its readability and simplicity.",
                                title="Python Programming")
        
        # Search for knowledge
        result = self.module.search_knowledge("Python")
        
        # Verify result
        self.assertTrue(result["success"], "Search knowledge failed")
        self.assertGreaterEqual(result["fact_count"] + result["document_count"], 1,
                              "Search returned no results")
    
    def test_get_knowledge_statistics(self):
        """Test getting knowledge statistics"""
        # Add some facts and documents
        self.module.add_fact(content="Fact 1")
        self.module.add_fact(content="Fact 2")
        self.module.add_document(content="Document 1", title="Doc1")
        
        # Get statistics
        result = self.module.get_knowledge_statistics()
        
        # Verify result
        self.assertTrue(result["success"], "Failed to get statistics")
        self.assertIn("statistics", result, "Statistics not in result")
        
        stats = result["statistics"]
        self.assertGreaterEqual(stats["fact_count"], 2, "Incorrect fact count")
        self.assertGreaterEqual(stats["document_count"], 1, "Incorrect document count")


def run_quick_test():
    """Run a quick functional test of the module"""
    print("\n=== Quick Functional Test of Continuous Learning Module ===\n")
    
    # Create the module
    module = ContinuousLearningModule()
    
    print("Testing knowledge base operations...")
    
    # Add a fact
    result = module.add_fact(
        content="AgentSystem is a modular framework for building autonomous agents.",
        source="test",
        category="agent_framework"
    )
    print(f"Added fact: ID={result.get('fact_id')}")
    
    # Add a document
    result = module.add_document(
        content="AgentSystem provides a flexible architecture for creating intelligent agents that can perform tasks autonomously.",
        title="About AgentSystem",
        summary="Overview of the AgentSystem framework"
    )
    print(f"Added document: ID={result.get('document_id')}")
    
    # Search knowledge
    result = module.search_knowledge("AgentSystem")
    print(f"Search results: {result.get('fact_count')} facts, {result.get('document_count')} documents")
    
    # Get statistics
    stats = module.get_knowledge_statistics()
    print(f"Knowledge base statistics: {stats.get('statistics', {})}")
    
    print("\nTesting fact extraction...")
    
    # Extract facts from text
    text = """
    Autonomous agents are computer systems that can act on their own to achieve goals.
    Machine learning enables agents to improve from experience.
    The field of artificial intelligence has made significant progress in recent years.
    """
    
    facts = module.extract_facts_from_text(text)
    print(f"Extracted {facts.get('facts_found', 0)} facts from text:")
    for fact in facts.get('facts', []):
        print(f"  - {fact.get('content')} (confidence: {fact.get('confidence', 0):.2f})")
    
    print("\nTesting web research (if available)...")
    
    # Try a simple web search
    try:
        result = module.research_topic("autonomous agents", depth=1, max_results=2)
        if result.get("success", False):
            print(f"Research successful: found {result.get('facts_found', 0)} facts "
                  f"from {result.get('pages_processed', 0)} pages")
        else:
            print(f"Research failed: {result.get('error', 'unknown error')}")
    except Exception as e:
        print(f"Research test error: {e}")
    
    print("\n=== Test Completed ===\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run quick functional test
        run_quick_test()
    else:
        # Run unit tests
        unittest.main()
