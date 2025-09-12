"""
Learning System Tests
-------------------
Unit tests for the learning system components.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from AgentSystem.modules.knowledge_manager import KnowledgeManager
from AgentSystem.modules.web_researcher import WebResearcher
from AgentSystem.modules.code_modifier import CodeModifier
from AgentSystem.modules.learning_agent import LearningAgent

class TestKnowledgeManager(unittest.TestCase):
    def setUp(self):
        """Set up test knowledge base"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_knowledge.db"
        self.knowledge_manager = KnowledgeManager(str(self.db_path))
        
    def tearDown(self):
        """Clean up temporary files"""
        self.knowledge_manager.close()
        shutil.rmtree(self.temp_dir)
        
    def test_add_fact(self):
        """Test adding and retrieving facts"""
        # Add a fact
        fact_id = self.knowledge_manager.add_fact(
            content="Test fact",
            source="test",
            confidence=0.9,
            category="testing"
        )
        
        self.assertGreater(fact_id, 0)
        
        # Retrieve the fact
        fact = self.knowledge_manager.get_fact(fact_id)
        
        self.assertIsNotNone(fact)
        self.assertEqual(fact["content"], "Test fact")
        self.assertEqual(fact["source"], "test")
        self.assertEqual(fact["confidence"], 0.9)
        self.assertEqual(fact["category"], "testing")
        
    def test_search_facts(self):
        """Test searching facts"""
        # Add some facts
        self.knowledge_manager.add_fact(
            content="Python is a programming language",
            category="programming"
        )
        self.knowledge_manager.add_fact(
            content="JavaScript runs in browsers",
            category="programming"
        )
        
        # Search
        results = self.knowledge_manager.search_facts("Python")
        
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0]["content"])

class TestWebResearcher(unittest.TestCase):
    def setUp(self):
        """Set up test web researcher"""
        self.knowledge_manager = Mock()
        self.web_researcher = WebResearcher(self.knowledge_manager)
        
    def test_search(self):
        """Test web searching"""
        with patch.object(self.web_researcher, 'session') as mock_session:
            # Mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <body>
                    <table class="result-link">
                        <tr><td><a href="http://test.com">Test Result</a></td></tr>
                    </table>
                    <table class="result-snippet">
                        <tr><td>Test snippet</td></tr>
                    </table>
                </body>
            </html>
            """
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value = mock_response
            
            # Perform search
            results = self.web_researcher.search("test query")
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["title"], "Test Result")
            self.assertEqual(results[0]["url"], "http://test.com")
            self.assertEqual(results[0]["snippet"], "Test snippet")

class TestCodeModifier(unittest.TestCase):
    def setUp(self):
        """Set up test code modifier"""
        self.temp_dir = tempfile.mkdtemp()
        self.code_modifier = CodeModifier(self.temp_dir)
        
        # Create test file
        self.test_file = Path(self.temp_dir) / "test.py"
        with open(self.test_file, 'w') as f:
            f.write("""
class TestClass:
    def test_method(self):
        pass
""")
        
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
        
    def test_analyze_code(self):
        """Test code analysis"""
        analysis = self.code_modifier.analyze_code(str(self.test_file))
        
        self.assertEqual(len(analysis["classes"]), 1)
        self.assertEqual(analysis["classes"][0]["name"], "TestClass")
        self.assertEqual(analysis["classes"][0]["methods"], ["test_method"])
        
    def test_create_backup(self):
        """Test backup creation"""
        backup_path = self.code_modifier.create_backup(str(self.test_file))
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(Path(backup_path).exists())

class TestLearningAgent(unittest.TestCase):
    def setUp(self):
        """Set up test learning agent"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent = LearningAgent(
            knowledge_base_path=str(Path(self.temp_dir) / "test_kb.db"),
            backup_dir=self.temp_dir
        )
        
    def tearDown(self):
        """Clean up temporary files"""
        self.agent.shutdown()
        shutil.rmtree(self.temp_dir)
        
    def test_research_topic(self):
        """Test topic research"""
        with patch.object(self.agent.web_researcher, 'research_topic') as mock_research:
            mock_research.return_value = [{
                "content": "Test content",
                "url": "http://test.com",
                "summary": "Test summary"
            }]
            
            findings = self.agent.research_topic("test topic")
            
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["content"], "Test summary")
            
    def test_background_learning(self):
        """Test background learning queue"""
        self.agent.start_learning()
        self.assertTrue(self.agent.learning_thread.is_alive())
        
        self.agent.queue_research("test topic")
        self.assertEqual(self.agent.learning_queue.qsize(), 1)
        
        self.agent.stop_learning()
        self.assertFalse(self.agent.learning_active)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
