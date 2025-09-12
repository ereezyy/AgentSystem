#!/usr/bin/env python3
"""
Complete LearningAgent Functionality Test
========================================
Tests all LearningAgent functionality with the new AI provider system
without requiring Pi5 dependency.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add AgentSystem to path
sys.path.insert(0, str(Path(__file__).parent))

from AgentSystem.modules.learning_agent import LearningAgent
from AgentSystem.modules.knowledge_manager import KnowledgeManager
from AgentSystem.modules.web_researcher import WebResearcher
from AgentSystem.modules.code_modifier import CodeModifier
from AgentSystem.services.ai_providers import get_provider_manager
from AgentSystem.utils.logger import get_logger

logger = get_logger("test_learning_agent_complete")

def test_ai_providers():
    """Test AI provider system is working"""
    print("\n=== Testing AI Provider System ===")
    
    try:
        provider_manager = get_provider_manager()
        
        # Test text generation
        result = provider_manager.generate_text(
            "What is Python programming?",
            max_tokens=100,
            temperature=0.3
        )
        
        if result.get('text'):
            print(f"✓ AI Provider working: {result.get('provider_used', 'unknown')}")
            print(f"  Response: {result.get('text', '')[:100]}...")
            return True
        else:
            print(f"✗ AI Provider failed: {result.get('error', 'No text returned')}")
            return False
            
    except Exception as e:
        print(f"✗ AI Provider test failed: {e}")
        return False

def test_knowledge_manager():
    """Test KnowledgeManager functionality"""
    print("\n=== Testing KnowledgeManager ===")
    
    try:
        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_knowledge.db")
            
            km = KnowledgeManager(db_path)
            
            # Test adding facts
            fact_id = km.add_fact(
                "Python is a programming language",
                source="test",
                confidence=0.9,
                category="programming"
            )
            
            if fact_id:
                print("✓ KnowledgeManager can add facts")
                
                # Test querying facts
                facts = km.query_facts("Python programming")
                if facts:
                    print(f"✓ KnowledgeManager can query facts: {len(facts)} found")
                    return True
                else:
                    print("✗ KnowledgeManager query returned no results")
                    return False
            else:
                print("✗ KnowledgeManager failed to add fact")
                return False
                
    except Exception as e:
        print(f"✗ KnowledgeManager test failed: {e}")
        return False

def test_web_researcher():
    """Test WebResearcher functionality"""
    print("\n=== Testing WebResearcher ===")
    
    try:
        # Create temporary knowledge manager for WebResearcher
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_web_knowledge.db")
            km = KnowledgeManager(db_path)
            wr = WebResearcher(km)
        
            # Test search functionality
            results = wr.search("Python programming tutorial", max_results=3)
            
            if results and len(results) > 0:
                print(f"✓ WebResearcher can search: {len(results)} results found")
                
                # Test content extraction
                if results[0].get('url'):
                    content = wr.extract_content(results[0]['url'])
                    if content and len(content) > 100:
                        print("✓ WebResearcher can extract content")
                        return True
                    else:
                        print("✗ WebResearcher content extraction failed")
                        return False
                else:
                    print("✗ WebResearcher results missing URLs")
                    return False
            else:
                print("✗ WebResearcher search returned no results")
                return False
            
    except Exception as e:
        print(f"✗ WebResearcher test failed: {e}")
        return False

def test_code_modifier():
    """Test CodeModifier functionality with AI providers"""
    print("\n=== Testing CodeModifier ===")
    
    try:
        # Create temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_code = '''
def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return "Hello"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
            f.write(test_code)
            temp_file = f.name
        
        try:
            cm = CodeModifier()
            
            # Test code analysis
            analysis = cm.analyze_code(temp_file)
            if analysis and 'functions' in analysis:
                print(f"✓ CodeModifier can analyze code: {len(analysis['functions'])} functions found")
                
                # Test AI-enhanced improvement (without Pi5)
                improvements = cm.ai_enhanced_code_improvement(temp_file, use_pi5=False)
                if improvements.get('success'):
                    print(f"✓ CodeModifier AI improvements working with provider: {improvements.get('provider_used', 'unknown')}")
                    
                    # Test suggestion improvements
                    suggestions = cm.suggest_improvements(temp_file)
                    if suggestions:
                        print(f"✓ CodeModifier can suggest improvements: {len(suggestions)} suggestions")
                        return True
                    else:
                        print("✗ CodeModifier suggestion improvements failed")
                        return False
                else:
                    print(f"✗ CodeModifier AI improvements failed: {improvements.get('error', 'Unknown error')}")
                    return False
            else:
                print("✗ CodeModifier analysis failed")
                return False
                
        finally:
            # Clean up temp file
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ CodeModifier test failed: {e}")
        return False

def test_learning_agent_integration():
    """Test complete LearningAgent integration"""
    print("\n=== Testing LearningAgent Integration ===")
    
    try:
        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_learning.db")
            
            # Initialize LearningAgent with backup directory
            backup_dir = os.path.join(temp_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            agent = LearningAgent(
                knowledge_base_path=db_path,
                backup_dir=backup_dir
            )
            
            # Test initialization
            if agent.knowledge_manager and agent.web_researcher and agent.code_modifier:
                print("✓ LearningAgent initialized with all components")
                
                # Test research functionality
                results = agent.research_topic("Python best practices")
                if results is not None:
                    print(f"✓ LearningAgent can research topics: {len(results)} results")
                    
                    # Test knowledge stats
                    stats = agent.get_knowledge_stats()
                    if stats:
                        print(f"✓ LearningAgent stats: {stats}")
                        return True
                    else:
                        print("✗ LearningAgent stats retrieval failed")
                        return False
                else:
                    print("✗ LearningAgent topic research failed")
                    return False
            else:
                print("✗ LearningAgent initialization failed")
                return False
                
    except Exception as e:
        print(f"✗ LearningAgent integration test failed: {e}")
        return False

def test_learning_agent_background_learning():
    """Test LearningAgent background learning functionality"""
    print("\n=== Testing LearningAgent Background Learning ===")
    
    try:
        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_bg_learning.db")
            
            # Initialize LearningAgent with backup directory
            backup_dir = os.path.join(temp_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            agent = LearningAgent(
                knowledge_base_path=db_path,
                backup_dir=backup_dir
            )
            
            # Start background learning
            agent.start_learning()
            print("✓ LearningAgent background learning started")
            
            # Add some learning tasks
            agent.queue_research("machine learning basics")
            agent.queue_research("Python data structures")
            print("✓ LearningAgent tasks queued")
            
            # Wait a bit for background processing
            time.sleep(3)
            
            # Check status
            stats = agent.get_knowledge_stats()
            print(f"✓ Background learning stats: {stats}")
            
            # Stop background learning
            agent.stop_learning()
            print("✓ LearningAgent background learning stopped")
            
            return True
            
    except Exception as e:
        print(f"✗ LearningAgent background learning test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("LearningAgent Complete Functionality Test")
    print("=" * 50)
    
    tests = [
        ("AI Providers", test_ai_providers),
        ("KnowledgeManager", test_knowledge_manager),
        ("WebResearcher", test_web_researcher),
        ("CodeModifier", test_code_modifier),
        ("LearningAgent Integration", test_learning_agent_integration),
        ("Background Learning", test_learning_agent_background_learning)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LearningAgent is fully functional with AI providers.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)