#!/usr/bin/env python3
"""
Test script to verify Celery integration between CodeModifier and Pi 5 worker
"""

import sys
import time
import json
from pathlib import Path

# Add AgentSystem to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.code_modifier import CodeModifier
from modules.learning_agent import LearningAgent
from utils.logger import setup_logging

def test_celery_connection():
    """Test basic Celery connection"""
    print("Testing Celery connection...")
    
    try:
        # Initialize code modifier
        code_modifier = CodeModifier()
        
        # Test health check task
        print("Sending health check task to Pi 5...")
        health_task = code_modifier.app.send_task('system.health_check')
        
        # Wait for result with timeout
        result = health_task.get(timeout=10)
        
        print(f"Health check result: {json.dumps(result, indent=2)}")
        return True
        
    except Exception as e:
        print(f"Celery connection test failed: {e}")
        return False

def test_ai_code_analysis():
    """Test AI code analysis task"""
    print("\nTesting AI code analysis...")
    
    try:
        # Initialize code modifier
        code_modifier = CodeModifier()
        
        # Create sample code data
        code_data = {
            'file_path': 'test.py',
            'code': '''
def vulnerable_function(user_input):
    # This function has security issues
    query = "SELECT * FROM users WHERE id = " + user_input
    return query
''',
            'analysis': {
                'functions': [{'name': 'vulnerable_function', 'args': ['user_input'], 'line': 1}],
                'classes': [],
                'imports': [],
                'loc': 4
            }
        }
        
        print("Sending code analysis task to Pi 5...")
        analysis_task = code_modifier.app.send_task(
            'codemodifier.ai_code_analysis',
            args=[code_data, '/models/code_analysis.hef']
        )
        
        # Wait for result
        result = analysis_task.get(timeout=30)
        
        print(f"AI analysis result: {json.dumps(result, indent=2)}")
        return result.get('status') == 'success'
        
    except Exception as e:
        print(f"AI code analysis test failed: {e}")
        return False

def test_learning_agent_integration():
    """Test LearningAgent with Celery integration"""
    print("\nTesting LearningAgent integration...")
    
    try:
        # Initialize learning agent
        learning_agent = LearningAgent()
        
        # Test code improvement functionality
        test_file = Path(__file__).parent / "test_sample.py"
        
        # Create a test file
        test_code = '''
def test_function():
    # Simple test function
    x = 1
    y = 2
    return x + y
'''
        
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        print(f"Testing code improvement on {test_file}")
        
        # Use learning agent to improve code
        result = learning_agent.code_modifier.ai_enhanced_code_improvement(
            str(test_file), 
            use_pi5=True
        )
        
        print(f"Code improvement result: {json.dumps(result, indent=2)}")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"LearningAgent integration test failed: {e}")
        return False

def main():
    """Run all Celery integration tests"""
    setup_logging()
    
    print("=" * 60)
    print("CELERY INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Celery Connection", test_celery_connection),
        ("AI Code Analysis", test_ai_code_analysis),
        ("LearningAgent Integration", test_learning_agent_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "PASSED" if success else "FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:30} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {len(results)} tests, {passed} passed, {len(results) - passed} failed")
    
    if passed == len(results):
        print("\n✅ All Celery integration tests PASSED!")
        return True
    else:
        print(f"\n❌ {len(results) - passed} test(s) FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)