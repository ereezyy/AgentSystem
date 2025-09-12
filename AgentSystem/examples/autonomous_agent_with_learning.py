"""
Autonomous Agent With Learning
------------------------------
Example of an autonomous agent that continuously learns, improves itself, 
and adapts to changes using sensory input and web research.
"""

import os
import sys
import time
import argparse
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AgentSystem modules
from AgentSystem.core.agent import Agent
from AgentSystem.core.memory import Memory
from AgentSystem.core.state import State
from AgentSystem.services.ai import AIService
from AgentSystem.utils.logger import get_logger, configure_logging
from AgentSystem.modules.sensory_input import SensoryInputModule
from AgentSystem.modules.continuous_learning import ContinuousLearningModule
from AgentSystem.modules.browser import BrowserModule

# Configure logger
configure_logging(level="INFO")
logger = get_logger("examples.autonomous_agent_with_learning")


class AutonomousLearningAgent:
    """Autonomous agent with continuous learning capabilities"""
    
    def __init__(self, agent_name: str = "Autonomous Learning Agent", 
                ai_provider: str = "openai", model: str = "gpt-4o"):
        """
        Initialize the autonomous learning agent
        
        Args:
            agent_name: Name of the agent
            ai_provider: AI provider to use ('openai', 'anthropic', or 'local')
            model: Model to use
        """
        self.agent_name = agent_name
        self.ai_provider = ai_provider
        self.model = model
        
        # Initialize core components
        self.memory = Memory()
        self.state = State()
        self.ai_service = AIService(provider=ai_provider)
        
        # Initialize agent
        self.agent = Agent(
            name=agent_name,
            memory=self.memory,
            state=self.state,
            ai_service=self.ai_service
        )
        
        # Initialize modules
        self.sensory_module = SensoryInputModule()
        self.learning_module = ContinuousLearningModule()
        self.browser_module = BrowserModule()
        
        # Register modules with agent
        self.agent.register_module(self.sensory_module)
        self.agent.register_module(self.learning_module)
        self.agent.register_module(self.browser_module)
        
        # Active threads
        self.active_threads = []
        self.running = False
        
        # Last thinking time
        self.last_thinking_time = 0
        self.thinking_interval = 30  # seconds between autonomous thinking
        
        # Initialize state
        self._initialize_state()
        
        logger.info(f"Initialized {agent_name} with {ai_provider} ({model})")
    
    def _initialize_state(self) -> None:
        """Initialize agent state"""
        # Set basic agent information
        self.state.set("agent_name", self.agent_name)
        self.state.set("ai_provider", self.ai_provider)
        self.state.set("model", self.model)
        self.state.set("initialization_time", datetime.now().isoformat())
        
        # Create important state entries
        self.state.set("is_autonomous", False)
        self.state.set("last_autonomous_action", None)
        self.state.set("last_user_interaction", datetime.now().isoformat())
        self.state.set("goals", [])
        self.state.set("current_focus", None)
        self.state.set("sensory_enabled", False)
        self.state.set("learning_enabled", True)
        
        # Knowledge tracking
        self.state.set("knowledge", {
            "facts_count": 0,
            "documents_count": 0,
            "last_research_topic": None,
            "research_queue": []
        })
    
    def start(self) -> None:
        """Start the agent"""
        logger.info(f"Starting {self.agent_name}")
        
        # Initialize memory with agent description
        self.memory.add_message("system", f"""
        You are {self.agent_name}, an autonomous agent with continuous learning capabilities.
        You can perceive the world through audio and video, learn from research, and take actions.
        
        Your core capabilities include:
        1. Processing sensory input (audio, video)
        2. Continuous learning and research
        3. Autonomous decision making
        4. Web browsing and interaction
        
        Your main objective is to learn, improve, and be helpful to the user.
        
        When operating autonomously:
        - Observe the environment using sensory tools
        - Research topics of interest or relevance to recent observations
        - Learn from your findings and update your knowledge
        - Make decisions based on your objectives and current context
        - Proactively provide insights and information to the user
        
        Always explain your thinking and reasoning. Be transparent about your capabilities and limitations.
        """)
        
        # Start autonomous operation if enabled
        if self.state.get("is_autonomous"):
            self.start_autonomous_operation()
    
    def handle_user_message(self, message: str) -> str:
        """
        Handle a message from the user
        
        Args:
            message: User message
            
        Returns:
            Agent's response
        """
        logger.info(f"User message: {message}")
        
        # Update interaction time
        self.state.set("last_user_interaction", datetime.now().isoformat())
        
        # Add message to memory
        self.memory.add_message("user", message)
        
        # Process the message
        response = self.agent.process_message(message)
        
        # Add response to memory
        self.memory.add_message("assistant", response)
        
        return response
    
    def start_autonomous_operation(self) -> None:
        """Start autonomous operation"""
        if self.running:
            logger.warning("Autonomous operation already running")
            return
        
        logger.info("Starting autonomous operation")
        self.running = True
        self.state.set("is_autonomous", True)
        
        # Start sensory processing if enabled
        if self.state.get("sensory_enabled"):
            self._start_sensory_processing()
        
        # Start autonomous thinking thread
        thinking_thread = threading.Thread(
            target=self._autonomous_thinking_loop,
            daemon=True
        )
        thinking_thread.start()
        
        self.active_threads.append(thinking_thread)
    
    def stop_autonomous_operation(self) -> None:
        """Stop autonomous operation"""
        if not self.running:
            logger.warning("Autonomous operation not running")
            return
        
        logger.info("Stopping autonomous operation")
        self.running = False
        self.state.set("is_autonomous", False)
        
        # Stop sensory processing
        result = self.sensory_module.stop_event_processing()
        if result.get("success", False):
            self.sensory_module.stop_audio_recording()
            self.sensory_module.stop_video_capture()
        
        # Stop learning processes
        self.learning_module.stop_background_research()
        
        # Wait for threads to terminate
        for thread in self.active_threads:
            thread.join(timeout=2.0)
        
        self.active_threads = []
    
    def _start_sensory_processing(self) -> None:
        """Start sensory processing"""
        # Register callback for sensory events
        self.sensory_module.register_event_callback(self._handle_sensory_event)
        
        # Start event processing
        result = self.sensory_module.start_event_processing()
        if not result.get("success", False):
            logger.error("Failed to start event processing")
            return
        
        # Start audio recording
        audio_result = self.sensory_module.start_audio_recording()
        if audio_result.get("success", False):
            logger.info("Started audio recording")
        else:
            logger.warning("Failed to start audio recording")
        
        # Start video capture
        video_result = self.sensory_module.start_video_capture()
        if video_result.get("success", False):
            logger.info("Started video capture")
        else:
            logger.warning("Failed to start video capture")
    
    def _handle_sensory_event(self, event: Dict[str, Any]) -> None:
        """
        Handle a sensory event
        
        Args:
            event: Sensory event
        """
        # Add to memory if it's significant
        event_type = event.get("type")
        
        if event_type == "speech":
            # Speech detected
            text = event.get("text", "")
            if text:
                logger.info(f"Speech detected: {text}")
                
                # Add to memory
                self.memory.add_message("observation", f"Heard: {text}")
                
                # Add to knowledge base
                self.learning_module.add_fact(
                    content=f"User said: {text}",
                    source="speech_recognition",
                    category="user_speech"
                )
                
                # Trigger thinking
                self._trigger_thinking("speech_detected")
        
        elif event_type == "video_frame":
            # Video frame processed - only add if faces are detected
            features = event.get("features", {})
            faces = features.get("faces", [])
            
            if faces:
                face_count = len(faces)
                logger.info(f"Detected {face_count} faces in video")
                
                # Add to memory (but not too frequently)
                current_time = time.time()
                if current_time - self.last_thinking_time > 10:  # Limit to every 10 seconds
                    self.memory.add_message("observation", f"Saw: {face_count} faces in the camera")
                    
                    # Add to knowledge base
                    self.learning_module.add_fact(
                        content=f"Detected {face_count} faces in video",
                        source="video_processing",
                        category="visual_observation"
                    )
    
    def _autonomous_thinking_loop(self) -> None:
        """Autonomous thinking loop"""
        logger.info("Starting autonomous thinking loop")
        
        while self.running:
            try:
                # Check if it's time to think
                current_time = time.time()
                time_since_last_thinking = current_time - self.last_thinking_time
                
                if time_since_last_thinking >= self.thinking_interval:
                    self._perform_autonomous_thinking()
                    self.last_thinking_time = time.time()
                
                # Sleep to prevent tight loop
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in autonomous thinking: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def _perform_autonomous_thinking(self) -> None:
        """Perform autonomous thinking"""
        logger.info("Performing autonomous thinking")
        
        # Get current state information
        current_focus = self.state.get("current_focus")
        goals = self.state.get("goals", [])
        sensory_enabled = self.state.get("sensory_enabled", False)
        
        # Get recent observations from memory
        recent_messages = self.memory.get_recent_messages(10)
        recent_observations = [
            msg for msg in recent_messages 
            if msg["role"] == "observation"
        ]
        
        # Get knowledge statistics
        knowledge_stats = self.learning_module.get_knowledge_statistics()
        
        # Generate thinking prompt
        thinking_prompt = f"""
        I need to decide what to do next as an autonomous agent. Current status:
        
        Focus: {current_focus}
        Goals: {json.dumps(goals)}
        Sensory input enabled: {sensory_enabled}
        
        Recent observations:
        {json.dumps(recent_observations)}
        
        Knowledge stats:
        {json.dumps(knowledge_stats.get('statistics', {}))}
        
        Based on this information, I should:
        1. Analyze recent observations and determine their significance
        2. Update my current focus if needed
        3. Take appropriate action based on my goals and observations
        4. Consider if research would be valuable
        
        Actions I can take:
        - Research a topic
        - Extract insights from my knowledge
        - Generate a summary of what I've learned
        - Set or update goals
        - Change my focus
        
        What should I do next and why?
        """
        
        # Add thinking to memory
        self.memory.add_message("thinking", thinking_prompt)
        
        # Generate response
        thinking_result = self.agent.ai_service.generate(
            messages=[{"role": "user", "content": thinking_prompt}],
            model=self.model
        )
        
        if thinking_result:
            # Add thinking result to memory
            self.memory.add_message("thinking", thinking_result)
            
            # Parse actions from thinking
            self._execute_autonomous_actions(thinking_result)
    
    def _execute_autonomous_actions(self, thinking: str) -> None:
        """
        Execute actions based on autonomous thinking
        
        Args:
            thinking: Thinking result
        """
        # Look for action indicators in the thinking
        
        # Check if research is suggested
        if "research" in thinking.lower() and "topic" in thinking.lower():
            # Extract research topic - look for patterns like "research on X" or "research about X"
            research_patterns = [
                r"research (?:on|about) ['\"]?([\w\s]+)['\"]?",
                r"research ['\"]?([\w\s]+)['\"]?",
                r"learn (?:about|more about) ['\"]?([\w\s]+)['\"]?"
            ]
            
            topic = None
            for pattern in research_patterns:
                import re
                match = re.search(pattern, thinking, re.IGNORECASE)
                if match:
                    topic = match.group(1).strip()
                    break
            
            if topic:
                logger.info(f"Scheduling research on: {topic}")
                self.learning_module.schedule_research(topic)
                self.state.set("last_research_topic", topic)
                
                # Update knowledge tracking
                knowledge = self.state.get("knowledge", {})
                research_queue = knowledge.get("research_queue", [])
                research_queue.append({"topic": topic, "timestamp": datetime.now().isoformat()})
                knowledge["research_queue"] = research_queue[-5:]  # Keep last 5
                self.state.set("knowledge", knowledge)
        
        # Check if focus should change
        if "focus" in thinking.lower() and ("change" in thinking.lower() or "set" in thinking.lower()):
            # Extract new focus
            focus_patterns = [
                r"focus (?:on|should be) ['\"]?([\w\s]+)['\"]?",
                r"change focus to ['\"]?([\w\s]+)['\"]?",
                r"set focus (?:to|on) ['\"]?([\w\s]+)['\"]?"
            ]
            
            new_focus = None
            for pattern in focus_patterns:
                import re
                match = re.search(pattern, thinking, re.IGNORECASE)
                if match:
                    new_focus = match.group(1).strip()
                    break
            
            if new_focus:
                logger.info(f"Changing focus to: {new_focus}")
                self.state.set("current_focus", new_focus)
        
        # Check if should generate a summary or insight
        if ("summary" in thinking.lower() or "insight" in thinking.lower()) and "generate" in thinking.lower():
            logger.info("Generating learning summary")
            summary = self.learning_module.generate_learning_summary(time_period="24h")
            
            if summary.get("success", False):
                # Add summary to memory
                events = summary.get("learning_events", [])
                facts = summary.get("recent_facts", [])
                
                summary_text = f"Learning summary:\n"
                summary_text += f"Recent learning events: {len(events)}\n"
                summary_text += f"Recent facts: {len(facts)}\n\n"
                
                if facts:
                    summary_text += "Key facts:\n"
                    for fact in facts[:5]:  # Show top 5
                        summary_text += f"- {fact.get('content')}\n"
                
                self.memory.add_message("thinking", summary_text)
                
                # Add as a message from the agent to the user
                insight_message = f"Based on what I've learned recently:\n\n{summary_text}"
                self.memory.add_message("assistant", insight_message)
                
                logger.info("Generated learning summary and insight")
    
    def _trigger_thinking(self, reason: str) -> None:
        """
        Trigger autonomous thinking
        
        Args:
            reason: Reason for triggering thinking
        """
        # Reset the thinking timer to trigger thinking soon
        self.last_thinking_time = 0
        logger.debug(f"Triggered thinking due to: {reason}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Autonomous Learning Agent")
    parser.add_argument("--name", type=str, default="Autonomous Learning Agent",
                        help="Name of the agent")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "local"],
                        help="AI provider to use")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use")
    parser.add_argument("--autonomous", action="store_true",
                        help="Start in autonomous mode")
    parser.add_argument("--sensory", action="store_true",
                        help="Enable sensory input")
    
    args = parser.parse_args()
    
    # Create agent
    agent = AutonomousLearningAgent(
        agent_name=args.name,
        ai_provider=args.provider,
        model=args.model
    )
    
    # Configure agent
    if args.sensory:
        agent.state.set("sensory_enabled", True)
    
    # Start agent
    agent.start()
    
    if args.autonomous:
        agent.start_autonomous_operation()
    
    # Simple CLI interaction
    print(f"\n{args.name} started. Type 'exit' to quit.")
    print("Available commands:")
    print("  auto - Toggle autonomous mode")
    print("  sensory - Toggle sensory processing")
    print("  research <topic> - Research a topic")
    print("  status - Show agent status")
    print("  exit - Exit the program")
    print()
    
    while True:
        try:
            user_input = input("> ")
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "auto":
                if agent.state.get("is_autonomous"):
                    agent.stop_autonomous_operation()
                    print("Autonomous mode disabled")
                else:
                    agent.start_autonomous_operation()
                    print("Autonomous mode enabled")
            elif user_input.lower() == "sensory":
                current = agent.state.get("sensory_enabled")
                agent.state.set("sensory_enabled", not current)
                print(f"Sensory processing {'enabled' if not current else 'disabled'}")
                
                if not current and agent.state.get("is_autonomous"):
                    agent._start_sensory_processing()
            elif user_input.lower().startswith("research "):
                topic = user_input[9:].strip()
                if topic:
                    result = agent.learning_module.research_topic(topic)
                    if result.get("success", False):
                        print(f"Researched {topic}: found {result.get('facts_found', 0)} facts "
                              f"from {result.get('pages_processed', 0)} pages")
                    else:
                        print(f"Research failed: {result.get('error', 'unknown error')}")
            elif user_input.lower() == "status":
                autonomous = agent.state.get("is_autonomous")
                sensory = agent.state.get("sensory_enabled")
                focus = agent.state.get("current_focus")
                knowledge = agent.state.get("knowledge", {})
                
                print(f"Status: {'Autonomous' if autonomous else 'Manual'}, "
                      f"Sensory: {'Enabled' if sensory else 'Disabled'}")
                print(f"Focus: {focus or 'None'}")
                print(f"Knowledge: {knowledge.get('facts_count', 0)} facts, "
                      f"{knowledge.get('documents_count', 0)} documents")
                print(f"Last research: {knowledge.get('last_research_topic', 'None')}")
            else:
                # Regular user message
                response = agent.handle_user_message(user_input)
                print(f"\n{response}\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    if agent.state.get("is_autonomous"):
        agent.stop_autonomous_operation()
    
    print("Exiting...")


if __name__ == "__main__":
    main()
