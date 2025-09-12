"""
Self-Learning Agent Example
-------------------------
Demonstrates how to use the learning agent system for autonomous learning and self-improvement.
"""

import time
import logging
from pathlib import Path

from AgentSystem.utils.logger import get_logger
from AgentSystem.modules.learning_agent import LearningAgent

# Set up logging
logger = get_logger("examples.self_learning")
logger.setLevel(logging.INFO)

def main():
    # Initialize agent with custom paths
    agent = LearningAgent(
        knowledge_base_path="data/knowledge.db",
        backup_dir="data/backups"
    )
    
    try:
        # Start background learning
        agent.start_learning()
        
        # Queue some research tasks
        research_topics = [
            "artificial intelligence safety",
            "machine learning best practices",
            "software architecture patterns",
            "code optimization techniques"
        ]
        
        for topic in research_topics:
            logger.info(f"Queueing research on: {topic}")
            agent.queue_research(topic, depth=2)
        
        # Queue code improvement tasks
        code_files = [
            "AgentSystem/modules/knowledge_manager.py",
            "AgentSystem/modules/web_researcher.py",
            "AgentSystem/modules/code_modifier.py",
            "AgentSystem/modules/learning_agent.py"
        ]
        
        for file_path in code_files:
            if Path(file_path).exists():
                logger.info(f"Queueing improvement analysis for: {file_path}")
                agent.queue_code_improvement(file_path)
        
        # Monitor progress
        try:
            while True:
                stats = agent.get_knowledge_stats()
                logger.info(
                    f"Knowledge base stats: "
                    f"{stats['facts']} facts, "
                    f"{stats['categories']} categories, "
                    f"Queue size: {stats['queue_size']}"
                )
                
                if not stats['is_learning'] or stats['queue_size'] == 0:
                    logger.info("All tasks completed")
                    break
                    
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Stopping on user request...")
        
        # Example: Research a specific topic
        logger.info("\nResearching 'Python async programming'...")
        findings = agent.research_topic("Python async programming", depth=1)
        
        logger.info("\nKey findings:")
        for fact in findings:
            logger.info(f"- {fact['content']}")
            logger.info(f"  Source: {fact['source']}\n")
        
        # Example: Improve a specific file
        test_file = "AgentSystem/modules/learning_agent.py"
        if Path(test_file).exists():
            logger.info(f"\nAnalyzing {test_file} for improvements...")
            improvements = agent.improve_code(test_file)
            
            logger.info("\nSuggested improvements:")
            for improvement in improvements:
                logger.info(f"- {improvement['type']}: {improvement['description']}")
        
    finally:
        # Clean shutdown
        logger.info("\nShutting down agent...")
        agent.shutdown()

if __name__ == "__main__":
    main()
