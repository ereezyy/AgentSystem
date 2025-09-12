"""
Learning Agent Module
-------------------
Main integration module that combines knowledge management, web research,
and code modification capabilities into a cohesive learning agent.

Features:
- Autonomous learning and research
- Knowledge acquisition and management
- Self-modification and improvement
- Background learning tasks
"""

import threading
import queue
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from AgentSystem.utils.logger import get_logger
from AgentSystem.modules.knowledge_manager import KnowledgeManager
from AgentSystem.modules.web_researcher import WebResearcher
from AgentSystem.modules.code_modifier import CodeModifier

logger = get_logger("modules.learning_agent")

class LearningAgent:
    def __init__(self, 
                knowledge_base_path: Optional[str] = None,
                backup_dir: Optional[str] = None):
        """
        Initialize the learning agent
        
        Args:
            knowledge_base_path: Path to knowledge base file
            backup_dir: Directory for code backups
        """
        # Initialize components
        self.knowledge_manager = KnowledgeManager(knowledge_base_path)
        self.web_researcher = WebResearcher(self.knowledge_manager)
        self.code_modifier = CodeModifier(backup_dir)
        
        # Background learning queue
        self.learning_queue = queue.Queue()
        self.learning_thread = None
        self.learning_active = False
        self.learning_lock = threading.Lock()  # Protect learning_active flag
        
    def start_learning(self) -> None:
        """Start background learning thread"""
        with self.learning_lock:
            if self.learning_thread and self.learning_thread.is_alive():
                logger.warning("Learning thread already running")
                return
                
            logger.debug("Acquiring lock to start learning thread")
            self.learning_active = True
            self.learning_thread = threading.Thread(
                target=self._learning_loop,
                daemon=True
            )
            self.learning_thread.start()
            logger.info("Started background learning thread with thread-safe protection")
        
    def stop_learning(self) -> None:
        """Stop background learning thread"""
        with self.learning_lock:
            logger.debug("Acquiring lock to stop learning thread")
            self.learning_active = False
            
        # Join thread outside of lock to avoid deadlock
        if self.learning_thread:
            logger.debug("Waiting for learning thread to terminate")
            self.learning_thread.join(timeout=5.0)
            if self.learning_thread.is_alive():
                logger.warning("Learning thread did not terminate within timeout")
            else:
                logger.debug("Learning thread terminated successfully")
            self.learning_thread = None
        logger.info("Stopped background learning with thread-safe protection")
        
    def _learning_loop(self) -> None:
        """Background learning thread main loop"""
        while True:
            # Thread-safe check of learning_active flag
            with self.learning_lock:
                if not self.learning_active:
                    logger.debug("Learning loop terminating - flag set to False")
                    break
                    
            try:
                # Get next learning task
                try:
                    task = self.learning_queue.get(timeout=1.0)
                    start_time = time.time()  # Track processing start
                    logger.debug(f"Retrieved task from queue: {task}")
                except queue.Empty:
                    continue
                    
                # Process task with enhanced error recovery
                task_type = task.get("type")
                task_success = False
                
                try:
                    if task_type == "research":
                        topic = task["topic"]
                        depth = task.get("depth", 1)
                        logger.info(f"Starting research task: {topic} (depth={depth})")
                        results = self.research_topic(topic, depth)
                        logger.info(f"Completed research: {topic} - found {len(results)} results")
                        task_success = True
                        
                    elif task_type == "improve_code":
                        file_path = task["file"]
                        logger.info(f"Starting code improvement: {file_path}")
                        improvements = self.improve_code(file_path)
                        logger.info(f"Completed improvement: {file_path} - made {len(improvements)} improvements")
                        task_success = True
                        
                    else:
                        logger.warning(f"Unknown task type: {task_type}")
                        
                except Exception as task_error:
                    logger.error(f"Task processing failed for {task_type}: {task_error}")
                    logger.exception("Task processing error details:")
                    # Continue processing other tasks despite this failure
                    
                self.learning_queue.task_done()
                processing_time = time.time() - start_time
                status = "SUCCESS" if task_success else "FAILED"
                logger.info(f"Task {status}: {task_type} in {processing_time:.2f}s | Queue: {self.learning_queue.qsize()} pending")
                
            except Exception as e:
                logger.error(f"Critical error in learning loop: {e}")
                logger.exception("Learning loop critical error details:")
                # Add exponential backoff for critical errors
                time.sleep(min(5.0, 1.0 * 2))  # Start with 2s, could be extended
                
    def queue_research(self, topic: str, depth: int = 1) -> None:
        """
        Queue a research task
        
        Args:
            topic: Topic to research
            depth: How deep to follow links
        """
        self.learning_queue.put({
            "type": "research",
            "topic": topic,
            "depth": depth
        })
        
    def queue_code_improvement(self, file_path: str) -> None:
        """
        Queue a code improvement task
        
        Args:
            file_path: Path to file to improve
        """
        self.learning_queue.put({
            "type": "improve_code",
            "file": file_path
        })
        
    def research_topic(self, topic: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Research a topic and store findings with error recovery
        
        Args:
            topic: Topic to research
            depth: How deep to follow links
            
        Returns:
            List of research findings
        """
        facts = []
        
        try:
            logger.debug(f"Starting web research for topic: {topic}")
            # Search web with error handling
            results = self.web_researcher.research_topic(topic, depth)
            logger.debug(f"Web research returned {len(results)} results")
            
            if not results:
                logger.warning(f"No research results found for topic: {topic}")
                return facts
            
            # Extract and store facts with individual error handling
            for i, result in enumerate(results):
                try:
                    # Store document
                    doc_id = self.knowledge_manager.add_fact(
                        content=result["content"],
                        source=result["url"],
                        category=topic
                    )
                    logger.debug(f"Stored document {i+1}/{len(results)}: {result['url']}")
                    
                    # Extract key facts
                    summary = result.get("summary", "")
                    if summary:
                        fact_id = self.knowledge_manager.add_fact(
                            content=summary,
                            source=result["url"],
                            category=topic,
                            confidence=0.8
                        )
                        facts.append({
                            "id": fact_id,
                            "content": summary,
                            "source": result["url"]
                        })
                        logger.debug(f"Extracted fact from {result['url']}")
                        
                except Exception as fact_error:
                    logger.error(f"Failed to process research result {i+1}: {fact_error}")
                    # Continue with other results
                    continue
                    
        except Exception as research_error:
            logger.error(f"Research failed for topic '{topic}': {research_error}")
            logger.exception("Research error details:")
            # Return partial results if any were collected
            
        logger.info(f"Research completed for '{topic}': {len(facts)} facts extracted")
        return facts
        
    def improve_code(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze and improve code
        
        Args:
            file_path: Path to the file to improve
            
        Returns:
            List of improvements made
        """
        # Analyze current code
        analysis = self.code_modifier.analyze_code(file_path)
        
        # Get improvement suggestions
        suggestions = self.code_modifier.suggest_improvements(file_path)
        
        improvements = []
        for suggestion in suggestions:
            # Create changes dict
            changes = {
                "type": suggestion["type"],
                "description": suggestion["description"]
            }
            
            # Try to apply changes
            if self.code_modifier.modify_code(file_path, changes):
                improvements.append(changes)
                
        return improvements
        
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about acquired knowledge"""
        return {
            "facts": len(self.knowledge_manager.search_facts("")),
            "categories": len(set(f["category"] for f in self.knowledge_manager.search_facts(""))),
            "queue_size": self.learning_queue.qsize(),
            "is_learning": bool(self.learning_thread and self.learning_thread.is_alive())
        }
        
    def shutdown(self) -> None:
        """Cleanup and shutdown agent"""
        self.stop_learning()
        self.knowledge_manager.close()
