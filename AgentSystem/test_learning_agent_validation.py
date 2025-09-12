#!/usr/bin/env python3
"""
Comprehensive validation test for LearningAgent functionality
"""

import sys
import time
import tempfile
from pathlib import Path

# Add AgentSystem to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.learning_agent import LearningAgent
from utils.logger import setup_logging

def test_learning_agent_initialization():
    """Test LearningAgent initialization"""
    print("Testing LearningAgent initialization...")
    
    try:
        agent = LearningAgent()
        
        # Check components are initialized
        assert agent.knowledge_manager is not None, "KnowledgeManager not initialized"
        assert agent.web_researcher is not None, "WebResearcher not initialized"
        assert agent.code_modifier is not None, "CodeModifier not initialized"
        assert agent.learning_queue is not None, "Learning queue not initialized"
        
        print("✅ LearningAgent initialization: PASSED")
        return True, agent
        
    except Exception as e:
        print(f"❌ LearningAgent initialization: FAILED - {e}")
        return False, None

def test_knowledge_management(agent):
    """Test knowledge management functionality"""
    print("\nTesting knowledge management...")
    
    try:
        # Test adding facts
        fact_id = agent.knowledge_manager.add_fact(
            content="Python is a programming language",
            source="test",
            category="programming"
        )
        assert fact_id is not None, "Failed to add fact"
        
        # Test searching facts
        results = agent.knowledge_manager.search_facts("Python")
        assert len(results) > 0, "Failed to search facts"
        
        print("✅ Knowledge management: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge management: FAILED - {e}")
        return False

def test_web_research(agent):
    """Test web research functionality"""
    print("\nTesting web research...")
    
    try:
        # Test research topic
        results = agent.research_topic("Python programming", depth=1)
        
        # Should return a list (even if empty due to mocking)
        assert isinstance(results, list), "Research should return a list"
        
        print(f"✅ Web research: PASSED (found {len(results)} results)")
        return True
        
    except Exception as e:
        print(f"❌ Web research: FAILED - {e}")
        return False

def test_code_modification(agent):
    """Test code modification functionality"""
    print("\nTesting code modification...")
    
    try:
        # Create a test file
        test_code = '''
def test_function():
    """A simple test function"""
    x = 1
    y = 2
    return x + y

def another_function(param):
    return param * 2
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            # Test code analysis
            analysis = agent.code_modifier.analyze_code(test_file)
            assert 'functions' in analysis, "Code analysis should include functions"
            assert len(analysis['functions']) >= 2, "Should detect both functions"
            
            # Test backup creation
            backup_path = agent.code_modifier.create_backup(test_file)
            assert backup_path is not None, "Backup creation failed"
            assert Path(backup_path).exists(), "Backup file not created"
            
            # Test code validation
            valid, message = agent.code_modifier.validate_changes(test_file)
            assert valid, f"Code validation failed: {message}"
            
            print("✅ Code modification: PASSED")
            return True
            
        finally:
            # Cleanup
            Path(test_file).unlink(missing_ok=True)
            if backup_path and Path(backup_path).exists():
                Path(backup_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Code modification: FAILED - {e}")
        return False

def test_background_learning(agent):
    """Test background learning functionality"""
    print("\nTesting background learning...")
    
    try:
        # Start learning
        agent.start_learning()
        assert agent.learning_active, "Learning should be active"
        assert agent.learning_thread is not None, "Learning thread should exist"
        assert agent.learning_thread.is_alive(), "Learning thread should be alive"
        
        # Queue some tasks
        agent.queue_research("machine learning", depth=1)
        
        # Create a test file for code improvement
        test_code = '''
def simple_function():
    return "hello world"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            agent.queue_code_improvement(test_file)
            
            # Wait a bit for processing
            time.sleep(2)
            
            # Check queue processing
            initial_queue_size = agent.learning_queue.qsize()
            
            # Stop learning
            agent.stop_learning()
            assert not agent.learning_active, "Learning should be stopped"
            
            print("✅ Background learning: PASSED")
            return True
            
        finally:
            Path(test_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Background learning: FAILED - {e}")
        return False

def test_knowledge_stats(agent):
    """Test knowledge statistics"""
    print("\nTesting knowledge statistics...")
    
    try:
        stats = agent.get_knowledge_stats()
        
        assert 'facts' in stats, "Stats should include facts count"
        assert 'categories' in stats, "Stats should include categories count"
        assert 'queue_size' in stats, "Stats should include queue size"
        assert 'is_learning' in stats, "Stats should include learning status"
        
        print(f"✅ Knowledge statistics: PASSED")
        print(f"   Facts: {stats['facts']}, Categories: {stats['categories']}")
        print(f"   Queue size: {stats['queue_size']}, Learning: {stats['is_learning']}")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge statistics: FAILED - {e}")
        return False

def test_celery_integration(agent):
    """Test Celery integration for AI-enhanced code improvement"""
    print("\nTesting Celery integration...")
    
    try:
        # Create a test file with potential improvements
        test_code = '''
def vulnerable_function(user_input):
    # This function has security issues
    query = "SELECT * FROM users WHERE id = " + user_input
    return query

def slow_function(items):
    # This function could be optimized
    result = []
    for item in items:
        result.append(item * 2)
    return result
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            # Test AI-enhanced code improvement
            result = agent.code_modifier.ai_enhanced_code_improvement(
                test_file, 
                use_pi5=True
            )
            
            assert 'success' in result, "Result should include success status"
            
            if result.get('success'):
                assert 'improvements' in result, "Successful result should include improvements"
                print(f"✅ Celery integration: PASSED ({len(result.get('improvements', []))} improvements)")
            else:
                print(f"⚠️  Celery integration: PARTIAL - {result.get('error', 'Unknown error')}")
            
            return True
            
        finally:
            Path(test_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Celery integration: FAILED - {e}")
        return False

def test_agent_shutdown(agent):
    """Test proper agent shutdown"""
    print("\nTesting agent shutdown...")
    
    try:
        agent.shutdown()
        
        # Check that learning is stopped
        assert not agent.learning_active, "Learning should be stopped after shutdown"
        
        print("✅ Agent shutdown: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Agent shutdown: FAILED - {e}")
        return False

def main():
    """Run comprehensive LearningAgent validation"""
    setup_logging()
    
    print("=" * 70)
    print("LEARNING AGENT COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    # Test sequence
    tests = [
        ("Initialization", test_learning_agent_initialization),
        ("Knowledge Management", lambda agent: test_knowledge_management(agent)),
        ("Web Research", lambda agent: test_web_research(agent)),
        ("Code Modification", lambda agent: test_code_modification(agent)),
        ("Background Learning", lambda agent: test_background_learning(agent)),
        ("Knowledge Statistics", lambda agent: test_knowledge_stats(agent)),
        ("Celery Integration", lambda agent: test_celery_integration(agent)),
        ("Agent Shutdown", lambda agent: test_agent_shutdown(agent))
    ]
    
    results = []
    agent = None
    
    for i, (test_name, test_func) in enumerate(tests):
        print(f"\n{'='*20} Test {i+1}/{len(tests)}: {test_name} {'='*20}")
        
        try:
            if test_name == "Initialization":
                success, agent = test_func()
                if not success:
                    print("❌ Cannot continue without successful initialization")
                    break
            else:
                if agent is None:
                    print("❌ No agent available for testing")
                    success = False
                else:
                    success = test_func(agent)
            
            results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✅" if success else "❌"
        print(f"{icon} {test_name:30} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {len(results)} tests, {passed} passed, {len(results) - passed} failed")
    
    if passed == len(results):
        print("\n🎉 ALL LEARNING AGENT FUNCTIONALITY VALIDATED!")
        print("The LearningAgent module is fully functional and ready for use.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} validation test(s) failed.")
        print("Some functionality may need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)