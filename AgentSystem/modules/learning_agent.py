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
import random
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Iterable
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


@dataclass
class ReflexRule:
    """Map fast sensory events to immediate actions."""

    trigger: str
    action: Callable[[Dict[str, Any]], None]
    priority: int = 0


class ReflexLayer:
    """Fast response layer for immediate reactions."""

    def __init__(self) -> None:
        self._rules: List[ReflexRule] = []

    def register_rule(self, rule: ReflexRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def process(self, event: Dict[str, Any]) -> bool:
        signal = event.get("type") or event.get("signal")
        for rule in self._rules:
            if rule.trigger == signal:
                logger.debug("ReflexLayer matched rule %s for event %s", rule.trigger, event)
                rule.action(event)
                return True
        return False


class DeliberativeLayer:
    """Planning layer using lightweight tree search heuristics."""

    def __init__(self, evaluator: Optional[Callable[[Dict[str, Any]], float]] = None) -> None:
        self._evaluator = evaluator or (lambda plan: float(plan.get("expected_reward", 0)))

    def plan(self, goal: str, options: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pick a plan by scoring candidate actions."""
        best_score = float("-inf")
        best_plan: List[Dict[str, Any]] = []
        for idx, option in enumerate(options):
            option = dict(option)
            option.setdefault("steps", [option.get("action", goal)])
            option.setdefault("expected_reward", 0.0)
            score = self._evaluator(option) - idx * 0.01  # light tie-breaker
            if score > best_score:
                best_score = score
                best_plan = list(option["steps"])
        logger.debug("DeliberativeLayer chose plan %s for goal %s", best_plan, goal)
        return best_plan


class MetaCognitiveLayer:
    """Monitors performance and updates strategies."""

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=200)
        self._active_goals: deque = deque(maxlen=20)

    def record_outcome(self, feedback: Dict[str, Any]) -> None:
        self._history.append(feedback)

    def track_goal(self, goal: str) -> None:
        if goal not in self._active_goals:
            self._active_goals.append(goal)

    def review(self) -> Dict[str, Any]:
        if not self._history:
            return {"status": "insufficient-data"}
        recent_rewards = [item.get("reward", 0.0) for item in list(self._history)[-10:]]
        average = sum(recent_rewards) / max(len(recent_rewards), 1)
        recommendation = "maintain" if average >= 0 else "adjust-prompts"
        return {
            "status": "ok" if average >= 0 else "needs-adjustment",
            "recent_average_reward": average,
            "tracked_goals": list(self._active_goals),
            "recommendation": recommendation,
        }


class CausalInferencer:
    """Track simple cause-effect pairs to move beyond correlation."""

    def __init__(self) -> None:
        self._counts: Dict[tuple, Dict[str, int]] = {}

    def observe(self, cause: str, effect: str, success: bool) -> None:
        bucket = self._counts.setdefault((cause, effect), {"success": 0, "failure": 0})
        bucket["success" if success else "failure"] += 1

    def infer(self, cause: str) -> Optional[str]:
        best_effect = None
        best_ratio = 0.0
        for (observed_cause, effect), stats in self._counts.items():
            if observed_cause != cause:
                continue
            total = stats["success"] + stats["failure"]
            if not total:
                continue
            ratio = stats["success"] / total
            if ratio > best_ratio:
                best_ratio = ratio
                best_effect = effect
        return best_effect


class ReActReasoner:
    """Blend reasoning traces with acting hooks and memory introspection."""

    def __init__(self, knowledge_manager: "KnowledgeManager") -> None:
        self.knowledge_manager = knowledge_manager

    def reason(self, query: str) -> Dict[str, Any]:
        trace: List[str] = []
        trace.append(f"Thought: need information about {query}")
        memory = self.knowledge_manager.fusion_search(query, limit=5)
        trace.append(f"Retrieved {len(memory['facts'])} facts and {len(memory['episodes'])} episodes")
        action = "act:consult_knowledge_base" if memory["facts"] else "act:web_research"
        return {"trace": trace, "action": action, "memory": memory}


class DistributedAgentMesh:
    """Lightweight federated mesh for specialised agents."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._shared_state: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}

    def register(
        self,
        role: str,
        callback: Callable[[Dict[str, Any]], Any],
        *,
        weight: float = 1.0,
    ) -> None:
        self._subscribers[role] = callback
        try:
            self._weights[role] = max(float(weight), 0.0)
        except (TypeError, ValueError):
            self._weights[role] = 1.0

    def broadcast(self, message: Dict[str, Any]) -> None:
        for role, callback in self._subscribers.items():
            try:
                callback(dict(message, target_role=role))
            except Exception as exc:
                logger.warning("Mesh callback for %s failed: %s", role, exc)

    def update_shared_state(self, key: str, value: Any) -> None:
        self._shared_state[key] = value

    def get_shared_state(self, key: str, default: Any = None) -> Any:
        return self._shared_state.get(key, default)

    def request_consensus(
        self,
        question: Dict[str, Any],
        *,
        quorum: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Request proposals from subscribers and return an aggregated decision."""

        votes: List[Dict[str, Any]] = []
        total_weight = 0.0
        for role, callback in self._subscribers.items():
            payload = dict(question, target_role=role, kind="consensus_request")
            try:
                response = callback(payload)
            except Exception as exc:
                logger.warning("Consensus callback for %s failed: %s", role, exc)
                continue

            if response is None:
                continue

            if not isinstance(response, dict):
                response = {"vote": response}

            if "vote" not in response:
                continue

            response = dict(response)
            response.setdefault("role", role)
            weight = response.get("weight", self._weights.get(role, 1.0))
            try:
                weight_value = float(weight)
            except (TypeError, ValueError):
                weight_value = 1.0
            if weight_value < 0:
                weight_value = 0.0
            response["weight"] = weight_value
            votes.append(response)
            total_weight += weight_value

        if not votes:
            return {
                "decision": None,
                "votes": [],
                "passed": False,
                "total_weight": 0.0,
                "decision_weight": 0.0,
            }

        tallies: Dict[Any, float] = {}
        for entry in votes:
            vote_key = entry.get("vote")
            tallies[vote_key] = tallies.get(vote_key, 0.0) + entry["weight"]

        decision, decision_weight = max(tallies.items(), key=lambda item: item[1])
        threshold = quorum if quorum is not None else (total_weight / 2.0)
        passed = decision_weight >= threshold if total_weight else False

        return {
            "decision": decision,
            "votes": votes,
            "passed": passed,
            "total_weight": total_weight,
            "decision_weight": decision_weight,
            "threshold": threshold,
        }


class ResilienceManager:
    """Monitor agent health and auto-heal where possible."""

    def __init__(self) -> None:
        self._restart_hooks: List[Callable[[], None]] = []

    def register_restart(self, hook: Callable[[], None]) -> None:
        self._restart_hooks.append(hook)

    def ensure_thread(self, thread: Optional[threading.Thread], starter: Callable[[], None]) -> None:
        if thread and thread.is_alive():
            return
        logger.warning("Detected inactive thread; attempting automatic restart")
        for hook in self._restart_hooks:
            try:
                hook()
            except Exception as exc:
                logger.error("Restart hook failed: %s", exc)
        starter()


class InferenceRouter:
    """Select between local and cloud inference pathways."""

    def __init__(self) -> None:
        self.local_available = False
        self.cloud_available = True
        self.cached_prompts: Dict[str, Dict[str, Any]] = {}

    def register_local(self, available: bool) -> None:
        self.local_available = available

    def choose(self, task: str) -> str:
        if self.local_available:
            return "local"
        if not self.cloud_available:
            return "cached"
        return "cloud"

    def cache_prompt(self, key: str, payload: Dict[str, Any]) -> None:
        self.cached_prompts[key] = payload


class SocialIntelligenceLayer:
    """Minimal affective computing helper."""

    POSITIVE_WORDS = {"great", "good", "excellent", "awesome", "thanks"}
    NEGATIVE_WORDS = {"bad", "terrible", "angry", "upset", "frustrated"}

    def analyse(self, text: str) -> Dict[str, Any]:
        words = {w.strip(".,!?" ).lower() for w in text.split()}
        positivity = len(words & self.POSITIVE_WORDS)
        negativity = len(words & self.NEGATIVE_WORDS)
        sentiment = "neutral"
        if positivity > negativity:
            sentiment = "positive"
        elif negativity > positivity:
            sentiment = "negative"
        return {"sentiment": sentiment, "positivity": positivity, "negativity": negativity}

    def adapt_response(self, text: str, sentiment: str) -> str:
        if sentiment == "positive":
            return f"I'm glad to hear that! {text}"
        if sentiment == "negative":
            return f"I understand the concern. {text}"
        return text


class CognitionStack:
    """Aggregate reflex, deliberative, and meta-cognitive layers."""

    def __init__(self, meta_layer: MetaCognitiveLayer) -> None:
        self.reflex = ReflexLayer()
        self.deliberative = DeliberativeLayer()
        self.meta = meta_layer

    def handle_event(self, event: Dict[str, Any], planner: Callable[[str], List[str]]) -> Dict[str, Any]:
        handled = self.reflex.process(event)
        response: Dict[str, Any] = {"handled": handled}
        if not handled and event.get("goal"):
            plan = planner(event["goal"])
            response["plan"] = plan
        self.meta.record_outcome({"reward": event.get("reward", 0.0)})
        return response

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

        # Cognitive layers
        self.meta_layer = MetaCognitiveLayer()
        self.cognition = CognitionStack(self.meta_layer)
        self.causal_inferencer = CausalInferencer()
        self.reasoner = ReActReasoner(self.knowledge_manager)
        self.mesh = DistributedAgentMesh()
        self.resilience = ResilienceManager()
        self.inference_router = InferenceRouter()
        self.social_layer = SocialIntelligenceLayer()
        self._prompt_versions: Dict[str, Dict[str, Any]] = {}
        self._self_play_log: deque = deque(maxlen=50)
        self._distillation_buffer: deque = deque(maxlen=200)

        self.mesh.register("Observer", lambda msg: logger.debug("Observer received %s", msg))
        self.resilience.register_restart(self.start_learning)

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
        self.meta_layer.track_goal("background_learning")

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
                self.resilience.ensure_thread(self.learning_thread, self.start_learning)
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
                        sources = sorted(
                            {
                                entry.get("source")
                                for entry in results
                                if isinstance(entry, dict) and entry.get("source")
                            }
                        )
                        logger.info(f"Completed research: {topic} - found {len(results)} results")
                        task_details = {
                            "result_count": len(results),
                            "topic": topic,
                            "sources": sources,
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
                if task_type:
                    self.causal_inferencer.observe(task_type, status, task_success)
                self.meta_layer.record_outcome({"reward": reward, "task": task_type, "success": task_success})
                if task_type == "research" and task_success:
                    self.mesh.broadcast({"event": "research_complete", "details": task_details})

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
        self.meta_layer.track_goal(f"research:{topic}")

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
        self.meta_layer.track_goal(f"improve:{file_path}")
        
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

    # ------------------------------------------------------------------
    # Advanced cognition helpers
    # ------------------------------------------------------------------
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Route sensory or system events through the cognition stack."""

        def planner(goal: str) -> List[str]:
            options = (
                {
                    "action": "research",
                    "steps": [f"Research {goal}", "Summarise findings"],
                    "expected_reward": 0.6,
                },
                {
                    "action": "reflect",
                    "steps": [f"Consult memories about {goal}", "Draft reflection"],
                    "expected_reward": 0.5,
                },
            )
            return self.cognition.deliberative.plan(goal, options)

        response = self.cognition.handle_event(event, planner)
        if not response.get("handled") and event.get("goal"):
            self.learning_queue.put({"type": "research", "topic": event["goal"], "depth": 1})
        return response

    def deliberate(self, goal: str) -> List[str]:
        """Perform a deliberate planning pass for the provided goal."""
        options = [
            {"action": "research", "steps": [f"Investigate {goal}"], "expected_reward": 0.5},
            {"action": "consult", "steps": [f"Recall experiences about {goal}", "Synthesize learnings"], "expected_reward": 0.55},
        ]
        plan = self.cognition.deliberative.plan(goal, options)
        self.meta_layer.track_goal(goal)
        return plan

    def meta_review(self) -> Dict[str, Any]:
        """Expose meta-cognitive review data."""
        return self.meta_layer.review()

    def react_reason(self, query: str) -> Dict[str, Any]:
        """Run a ReAct-style reasoning cycle."""
        return self.reasoner.reason(query)

    # ------------------------------------------------------------------
    # Dynamic learning loop extensions
    # ------------------------------------------------------------------
    def simulate_self_play(self, scenario: str) -> Dict[str, Any]:
        """Run a lightweight self-play scenario to gather experience."""
        agent_score = random.uniform(-0.5, 1.0)
        outcome = "win" if agent_score > 0 else "loss"
        record = {
            "scenario": scenario,
            "score": agent_score,
            "outcome": outcome,
        }
        self._self_play_log.append(record)
        self.knowledge_manager.add_episode(
            event=f"Self-play {scenario}",
            outcome=outcome,
            emotion="positive" if agent_score > 0 else "frustrated",
            salience=min(1.0, max(0.1, 0.5 + agent_score / 2)),
            context={"score": agent_score},
        )
        return record

    def register_prompt(
        self,
        key: str,
        template: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a prompt template for evolution and tracking."""

        state = self._prompt_versions.setdefault(
            key,
            {
                "version": 1,
                "template": template,
                "history": [],
                "stats": {
                    "success": 0,
                    "failure": 0,
                    "success_streak": 0,
                    "failure_streak": 0,
                },
                "metadata": metadata.copy() if metadata else {},
                "adjustments": [],
            },
        )
        if not state.get("template"):
            state["template"] = template
        if metadata:
            state.setdefault("metadata", {}).update(metadata)
        return state

    def get_prompt_template(self, key: str) -> Optional[str]:
        """Return the currently active template for a prompt identifier."""

        state = self._prompt_versions.get(key)
        return state.get("template") if state else None

    def evolve_prompts(
        self,
        key: str,
        success: bool,
        notes: Optional[str] = None,
        reward: Optional[float] = None,
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adapt prompts based on performance history and feedback."""

        state = self._prompt_versions.setdefault(
            key,
            {
                "version": 1,
                "template": template or "",
                "history": [],
                "stats": {
                    "success": 0,
                    "failure": 0,
                    "success_streak": 0,
                    "failure_streak": 0,
                },
                "metadata": {},
                "adjustments": [],
            },
        )
        if template and not state.get("template"):
            state["template"] = template

        stats = state.setdefault(
            "stats",
            {
                "success": 0,
                "failure": 0,
                "success_streak": 0,
                "failure_streak": 0,
            },
        )

        entry = {
            "success": success,
            "notes": notes,
            "reward": reward,
            "version": state["version"],
            "timestamp": time.time(),
        }
        state.setdefault("history", []).append(entry)

        if success:
            stats["success"] = stats.get("success", 0) + 1
            stats["success_streak"] = stats.get("success_streak", 0) + 1
            stats["failure_streak"] = 0
        else:
            stats["failure"] = stats.get("failure", 0) + 1
            stats["failure_streak"] = stats.get("failure_streak", 0) + 1
            stats["success_streak"] = 0

        mutated = False
        if not success and stats["failure_streak"] >= 2:
            new_template = self._generate_prompt_variant(
                state.get("template", ""),
                notes,
                state["version"] + 1,
                state.setdefault("adjustments", []),
            )
            if new_template != state.get("template"):
                state["template"] = new_template
                state["version"] += 1
                stats["failure_streak"] = 0
                state.setdefault("adjustments", []).append(
                    {"note": notes or "", "version": state["version"], "timestamp": time.time()}
                )
                mutated = True
        elif success and stats.get("success_streak", 0) >= 3:
            state.setdefault("metadata", {})["stabilised"] = True

        cache_payload = {
            "template": state.get("template"),
            "version": state["version"],
            "mutated": mutated,
            "success": stats.get("success", 0),
            "failure": stats.get("failure", 0),
        }
        self.inference_router.cache_prompt(key, cache_payload)
        return state

    def auto_evolve_prompts(self) -> List[str]:
        """Automatically evolve prompts when the meta review indicates issues."""

        review = self.meta_layer.review()
        evolved: List[str] = []
        if review.get("status") == "needs-adjustment":
            for key, state in self._prompt_versions.items():
                stats = state.get("stats", {})
                if stats.get("failure", 0) > stats.get("success", 0):
                    self.evolve_prompts(key, success=False, notes="Meta-review requested refinement.")
                    evolved.append(key)
        return evolved

    def log_interaction(self, prompt_id: str, prompt: str, response: str, reward: float) -> None:
        """Record an interaction for later offline distillation."""

        record = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "timestamp": time.time(),
        }
        self._distillation_buffer.append(record)

    def distill_model(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Produce a distilled dataset from recent interactions for offline tuning."""

        samples = list(self._distillation_buffer)
        if not samples:
            return {"status": "no-data", "samples": 0}

        average_reward = sum(sample["reward"] for sample in samples) / len(samples)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample) + "\n")

        summary = {
            "status": "distilled",
            "samples": len(samples),
            "average_reward": average_reward,
            "prompt_ids": sorted({sample["prompt_id"] for sample in samples}),
        }
        return summary

    def _generate_prompt_variant(
        self,
        template: str,
        notes: Optional[str],
        version: int,
        adjustments: List[Dict[str, Any]],
    ) -> str:
        """Create a deterministic prompt variation based on prior adjustments."""

        base = template.strip() if template else "You are an adaptive research assistant."
        guidance_segments: List[str] = []
        if notes:
            guidance_segments.append(f"Feedback: {notes.strip()}")
        if not adjustments:
            guidance_segments.append("Incorporate verification and reflection before final answers.")
        else:
            guidance_segments.append("Provide deeper analysis and cite relevant memories when possible.")
        guidance = " ".join(segment for segment in guidance_segments if segment).strip()
        if guidance:
            base = f"{base}\n\n[Adjustment v{version}]: {guidance}"
        return base

    def reinforcement_feedback(self, module: str, reward: float) -> None:
        """Record reinforcement signal for the specified module."""
        self._record_reward(module, reward, reward >= 0, processing_time=0.0, details={"module": module})

    def register_observation(self, description: str, success: bool, emotion: Optional[str] = None) -> None:
        """Log an observation into episodic memory and cognition layers."""
        salience = 0.9 if success else 0.4
        self.knowledge_manager.add_episode(
            event=description,
            outcome="success" if success else "failure",
            emotion=emotion,
            salience=salience,
        )
        self.meta_layer.record_outcome({"reward": 1.0 if success else -0.5, "event": description})

    def contextual_recall(self, cue: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch related facts and episodes for the provided cue."""
        return self.knowledge_manager.fusion_search(cue)

    def choose_inference_path(self, task: str) -> str:
        """Select the inference strategy (local/cloud/cached)."""
        return self.inference_router.choose(task)

    def adapt_response_tone(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment and adapt the response tone."""
        analysis = self.social_layer.analyse(text)
        adapted = self.social_layer.adapt_response(text, analysis["sentiment"])
        return {"analysis": analysis, "response": adapted}

    def join_mesh(
        self,
        role: str,
        callback: Callable[[Dict[str, Any]], Any],
        *,
        weight: float = 1.0,
    ) -> None:
        """Register a specialised agent callback within the distributed mesh."""
        self.mesh.register(role, callback, weight=weight)

    def share_mesh_state(self, key: str, value: Any) -> None:
        """Publish shared state for other agents in the mesh."""
        self.mesh.update_shared_state(key, value)

    def get_mesh_state(self, key: str, default: Any = None) -> Any:
        """Retrieve shared mesh state."""
        return self.mesh.get_shared_state(key, default)

    def consensus_plan(
        self,
        goal: str,
        options: List[Dict[str, Any]],
        *,
        quorum: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Seek a consensus-backed plan with a deliberative fallback."""

        consensus = self.mesh.request_consensus(
            {"goal": goal, "options": options}, quorum=quorum
        )

        plan_steps = self.cognition.deliberative.plan(goal, options)

        decision_key = consensus.get("decision")
        if decision_key is not None:
            chosen_option = next(
                (
                    opt
                    for opt in options
                    if opt.get("id") == decision_key
                    or opt.get("action") == decision_key
                ),
                None,
            )
            if chosen_option and consensus.get("passed"):
                plan_steps = list(
                    chosen_option.get(
                        "steps", [chosen_option.get("action", goal)]
                    )
                )

        return {"plan": plan_steps, "consensus": consensus}

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
        sources_field = None
        if isinstance(details, dict):
            sources_field = details.get("sources")
        if sources_field:
            if isinstance(sources_field, (list, tuple, set)):
                candidates = [str(src) for src in sources_field if src]
            else:
                candidates = [str(sources_field)]
            for source in candidates:
                try:
                    self.knowledge_manager.update_source_trust(source, success)
                except Exception as exc:  # pragma: no cover - defensive safeguard
                    logger.debug("Skipping trust update for %s: %s", source, exc)
        self.meta_layer.record_outcome(entry)

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
            "meta_review": self.meta_layer.review(),
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
