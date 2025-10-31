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
from collections import deque
from typing import Dict, List, Any, Optional
from pathlib import Path

from AgentSystem.utils.logger import get_logger

try:
    from AgentSystem.modules.knowledge_manager import KnowledgeManager
    from AgentSystem.modules.web_researcher import WebResearcher
    from AgentSystem.modules.code_modifier import CodeModifier
except ImportError:
    import importlib.util

    MODULE_DIR = Path(__file__).resolve().parent

    def _fallback_import(module_name: str):
        module_path = MODULE_DIR / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(f"learning_agent_{module_name}", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
        return module

    km_module = _fallback_import("knowledge_manager")
    if km_module and hasattr(km_module, "KnowledgeManager"):
        KnowledgeManager = km_module.KnowledgeManager  # type: ignore[attr-defined]
    else:  # pragma: no cover - knowledge manager is required for core operation
        raise

    wr_module = _fallback_import("web_researcher")
    if wr_module and hasattr(wr_module, "WebResearcher"):
        WebResearcher = wr_module.WebResearcher  # type: ignore[attr-defined]
    else:
        class WebResearcher:  # type: ignore[no-redef]
            """Minimal stub used when web research dependencies are unavailable."""

            def __init__(self, knowledge_manager: Any, *_, **__):
                self.knowledge_manager = knowledge_manager

            def research_topic(self, topic: str, depth: int = 1) -> List[Dict[str, Any]]:
                return []

    cm_module = _fallback_import("code_modifier")
    if cm_module and hasattr(cm_module, "CodeModifier"):
        CodeModifier = cm_module.CodeModifier  # type: ignore[attr-defined]
    else:
        class CodeModifier:  # type: ignore[no-redef]
            """Minimal stub used when code modification dependencies are unavailable."""

            def __init__(self, *_, **__):
                pass

            def analyze_code(self, file_path: str) -> Dict[str, Any]:
                return {}

            def suggest_improvements(self, file_path: str) -> List[Dict[str, Any]]:
                return []

            def modify_code(self, file_path: str, changes: Dict[str, Any]) -> bool:
                return False

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

        # Reward tracking
        self._reward_history: deque = deque(maxlen=100)
        self._cumulative_reward: float = 0.0
        self._task_outcomes = {"total": 0, "success": 0, "failure": 0}
        self._last_feedback: Optional[Dict[str, Any]] = None
        
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
                task_details: Dict[str, Any] = {}

                try:
                    if task_type == "research":
                        topic = task["topic"]
                        depth = task.get("depth", 1)
                        logger.info(f"Starting research task: {topic} (depth={depth})")
                        results = self.research_topic(topic, depth)
                        logger.info(f"Completed research: {topic} - found {len(results)} results")
                        task_details = {
                            "result_count": len(results),
                            "topic": topic
                        }
                        task_success = True

                    elif task_type == "improve_code":
                        file_path = task["file"]
                        logger.info(f"Starting code improvement: {file_path}")
                        improvements = self.improve_code(file_path)
                        logger.info(f"Completed improvement: {file_path} - made {len(improvements)} improvements")
                        task_details = {
                            "change_count": len(improvements),
                            "file_path": file_path
                        }
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

                reward = self._calculate_reward(task_type, task_success, processing_time, task_details)
                self._record_reward(task_type, reward, task_success, processing_time, task_details)

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

    def _calculate_reward(
        self,
        task_type: Optional[str],
        success: bool,
        processing_time: float,
        details: Dict[str, Any],
    ) -> float:
        """Compute a heuristic reward score for a completed task."""
        base_reward = 1.0 if success else -1.0

        if success:
            if task_type == "research":
                result_count = details.get("result_count", 0)
                base_reward += min(result_count, 5) * 0.1
            elif task_type == "improve_code":
                change_count = details.get("change_count", 0)
                base_reward += min(change_count, 5) * 0.2

        # Penalize long running tasks slightly to encourage efficiency
        base_reward -= min(processing_time / 60.0, 0.5)
        return base_reward

    def _record_reward(
        self,
        task_type: Optional[str],
        reward: float,
        success: bool,
        processing_time: float,
        details: Dict[str, Any],
    ) -> None:
        """Persist reward information for later introspection."""
        entry = {
            "task_type": task_type,
            "reward": reward,
            "success": success,
            "processing_time": processing_time,
            "details": details,
            "timestamp": time.time(),
        }
        self._reward_history.append(entry)
        self._cumulative_reward += reward
        self._task_outcomes["total"] += 1
        if success:
            self._task_outcomes["success"] += 1
        else:
            self._task_outcomes["failure"] += 1
        self._last_feedback = entry
        logger.info(
            "Recorded reward %.2f for task %s (success=%s, duration=%.2fs)",
            reward,
            task_type,
            success,
            processing_time,
        )

    def get_learning_feedback(self) -> Dict[str, Any]:
        """Return aggregate reward metrics for the learning loop."""
        recent = list(self._reward_history)
        recent_average = (
            sum(item["reward"] for item in recent) / len(recent)
            if recent
            else 0.0
        )
        success_rate = (
            self._task_outcomes["success"] / self._task_outcomes["total"]
            if self._task_outcomes["total"]
            else 0.0
        )
        return {
            "cumulative_reward": self._cumulative_reward,
            "recent_average_reward": recent_average,
            "total_tasks": self._task_outcomes["total"],
            "success_rate": success_rate,
            "last_feedback": self._last_feedback,
        }

    def submit_feedback(
        self,
        score: float,
        note: Optional[str] = None,
        task_type: str = "external_feedback",
    ) -> None:
        """Allow external systems to provide manual reward signals."""
        feedback_entry = {
            "task_type": task_type,
            "reward": score,
            "success": score >= 0,
            "processing_time": 0.0,
            "details": {"note": note} if note else {},
            "timestamp": time.time(),
        }
        self._reward_history.append(feedback_entry)
        self._cumulative_reward += score
        self._task_outcomes["total"] += 1
        if score >= 0:
            self._task_outcomes["success"] += 1
        else:
            self._task_outcomes["failure"] += 1
        self._last_feedback = feedback_entry
        logger.info("Manual feedback recorded with reward %.2f (%s)", score, note or "no note")

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about acquired knowledge"""
        stats = {
            "facts": len(self.knowledge_manager.search_facts("")),
            "categories": len(set(f["category"] for f in self.knowledge_manager.search_facts(""))),
            "queue_size": self.learning_queue.qsize(),
            "is_learning": bool(self.learning_thread and self.learning_thread.is_alive())
        }
        stats["performance"] = self.get_learning_feedback()
        return stats
        
    def shutdown(self) -> None:
        """Cleanup and shutdown agent"""
        self.stop_learning()
        self.knowledge_manager.close()
