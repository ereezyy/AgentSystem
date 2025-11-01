"""
Learning System Tests
-------------------
Unit tests for the learning system components.
"""

import importlib.util
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"


def _load_module(module_name: str):
    module_path = MODULE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


knowledge_manager_module = _load_module("knowledge_manager")
learning_agent_module = _load_module("learning_agent")

KnowledgeManager = knowledge_manager_module.KnowledgeManager
LearningAgent = learning_agent_module.LearningAgent
ReflexRule = learning_agent_module.ReflexRule

try:
    web_researcher_module = _load_module("web_researcher")
    WebResearcher = web_researcher_module.WebResearcher
except Exception:  # pragma: no cover - optional dependency path
    WebResearcher = None

WEB_IMPORTS_AVAILABLE = bool(
    getattr(web_researcher_module, "WEB_IMPORTS_AVAILABLE", False)
) if 'web_researcher_module' in locals() else False

try:
    code_modifier_module = _load_module("code_modifier")
    CodeModifier = code_modifier_module.CodeModifier
except Exception:  # pragma: no cover - optional dependency path
    CodeModifier = None

CODE_MODIFIER_AVAILABLE = CodeModifier is not None

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

    def test_episodic_memory_and_fusion(self):
        episode_id = self.knowledge_manager.add_episode(
            event="Investigated water behaviour",
            outcome="understood flow",
            emotion="curious",
            salience=0.9,
            context={"topic": "water"},
        )
        self.assertGreater(episode_id, 0)

        fused = self.knowledge_manager.fusion_search("water")
        self.assertTrue(fused["episodes"])

        promoted = self.knowledge_manager.consolidate_memories(limit=1)
        self.assertIsInstance(promoted, list)

    def test_knowledge_synthesis_and_verification(self):
        self.knowledge_manager.add_fact("Water flows downhill", category="science")
        self.knowledge_manager.add_fact("Water flows through rivers", category="science")

        graph = self.knowledge_manager.synthesize_knowledge("water")
        self.assertIn("nodes", graph)
        self.assertTrue(graph["nodes"])

        hypotheses = self.knowledge_manager.generate_hypotheses("water")
        self.assertTrue(hypotheses)

        verdict = self.knowledge_manager.verify_claim("Water flows")
        self.assertIn(verdict["verdict"], {"supported", "partial", "unknown"})

    def test_source_trust_calibration(self):
        source = "http://example.com"
        baseline = self.knowledge_manager.get_source_trust(source)
        self.assertEqual(baseline["score"], 0.5)

        self.knowledge_manager.update_source_trust(source, success=True)
        increased = self.knowledge_manager.get_source_trust(source)
        self.assertGreater(increased["score"], baseline["score"])
        self.assertEqual(increased["success_count"], 1)

        self.knowledge_manager.update_source_trust(source, success=False, weight=2.0)
        reduced = self.knowledge_manager.get_source_trust(source)
        self.assertLess(reduced["score"], increased["score"])
        self.assertEqual(reduced["failure_count"], 1)

@unittest.skipIf(not WEB_IMPORTS_AVAILABLE, "Web researcher dependencies unavailable")
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

@unittest.skipIf(not CODE_MODIFIER_AVAILABLE, "Code modifier dependencies unavailable")
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

    def test_reward_tracking_metrics(self):
        """Ensure reward metrics update when feedback is recorded."""
        baseline = self.agent.get_learning_feedback()
        self.assertEqual(baseline["cumulative_reward"], 0.0)
        self.assertEqual(baseline["total_tasks"], 0)

        self.agent.submit_feedback(0.5, note="good progress")
        updated = self.agent.get_learning_feedback()
        self.assertAlmostEqual(updated["cumulative_reward"], 0.5)
        self.assertEqual(updated["total_tasks"], 1)
        self.assertGreaterEqual(updated["success_rate"], 0.0)

    def test_cognition_layers_and_meta_review(self):
        triggered: List[Dict[str, Any]] = []
        self.agent.cognition.reflex.register_rule(
            ReflexRule(trigger="alert", action=lambda event: triggered.append(event))
        )
        response = self.agent.process_event({"type": "alert", "reward": 0.2})
        self.assertTrue(response["handled"])
        self.assertTrue(triggered)

        planning = self.agent.process_event({"type": "observation", "goal": "research"})
        self.assertIn("plan", planning)

        review = self.agent.meta_review()
        self.assertIn("status", review)

    def test_react_reasoning_and_self_play(self):
        self.agent.knowledge_manager.add_fact("Water analysis complete", category="research")
        reason = self.agent.react_reason("water")
        self.assertIn("trace", reason)
        self.assertIn("memory", reason)

        simulation = self.agent.simulate_self_play("maze-run")
        self.assertIn(simulation["outcome"], {"win", "loss"})

    def test_distributed_mesh_and_routing(self):
        messages: List[Dict[str, Any]] = []
        self.agent.join_mesh("Planner", lambda payload: messages.append(payload))
        self.agent.share_mesh_state("goal", "expand-knowledge")
        self.agent.mesh.broadcast({"event": "test"})
        self.assertTrue(messages)
        self.assertEqual(self.agent.mesh.get_shared_state("goal"), "expand-knowledge")

        self.agent.inference_router.register_local(True)
        self.assertEqual(self.agent.choose_inference_path("analysis"), "local")

    def test_social_and_memory_enrichment(self):
        adapted = self.agent.adapt_response_tone("This is great progress")
        self.assertEqual(adapted["analysis"]["sentiment"], "positive")

        baseline = self.agent.knowledge_manager.contextual_recall("mission")
        self.agent.register_observation("Mission accomplished", success=True, emotion="proud")
        recall = self.agent.knowledge_manager.contextual_recall("Mission")
        self.assertGreaterEqual(len(recall), len(baseline))

    def test_reward_updates_source_trust(self):
        source = "http://trust.example"
        baseline = self.agent.knowledge_manager.get_source_trust(source)["score"]
        self.agent._record_reward(
            "research",
            reward=1.0,
            success=True,
            processing_time=0.5,
            details={"sources": [source]},
        )
        increased = self.agent.knowledge_manager.get_source_trust(source)["score"]
        self.assertGreaterEqual(increased, baseline)

        self.agent._record_reward(
            "research",
            reward=-1.0,
            success=False,
            processing_time=0.5,
            details={"sources": [source]},
        )
        reduced = self.agent.knowledge_manager.get_source_trust(source)["score"]
        self.assertLess(reduced, increased)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
