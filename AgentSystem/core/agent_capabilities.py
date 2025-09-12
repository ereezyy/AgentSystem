"""
Agent Capabilities
-----------------
Implements the reasoning and decision-making capabilities for agents
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import traceback

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.services.ai import ai_service, AIMessage, AIRequestOptions

# Get module logger
logger = get_logger("core.agent_capabilities")


class AgentThought:
    """Represents an agent's internal thought process"""
    
    def __init__(self, agent_id: str):
        """Initialize the thought process"""
        self.agent_id = agent_id
        self.task = None
        self.plan = []
        self.observations = []
        self.working_memory = []
        self.current_step_index = 0
        self.start_time = time.time()
        
    def set_task(self, task: str) -> None:
        """Set the current task"""
        self.task = task
        
    def create_plan(self, plan_steps: List[str]) -> None:
        """Create a plan from a list of steps"""
        self.plan = plan_steps
        
    def add_observation(self, observation: str) -> None:
        """Add an observation"""
        self.observations.append({
            "content": observation,
            "timestamp": time.time()
        })
        
    def remember(self, item: Any) -> None:
        """Store an item in working memory"""
        self.working_memory.append(item)
        
    def get_next_step(self) -> Optional[str]:
        """Get the next step in the plan"""
        if not self.plan or self.current_step_index >= len(self.plan):
            return None
        
        step = self.plan[self.current_step_index]
        self.current_step_index += 1
        return step
        
    def get_context(self) -> Dict[str, Any]:
        """Get the current context for decision making"""
        return {
            "agent_id": self.agent_id,
            "task": self.task,
            "plan": self.plan,
            "current_step_index": self.current_step_index,
            "observations": self.observations[-5:] if self.observations else [],  # Last 5 observations
            "working_memory": self.working_memory[-10:] if self.working_memory else [],  # Last 10 memory items
            "elapsed_time": time.time() - self.start_time
        }


class AgentReasoner:
    """Handles the reasoning and decision-making process for an agent"""
    
    def __init__(self, agent_id: str, model: str = None, provider: str = None):
        """
        Initialize the agent reasoner
        
        Args:
            agent_id: ID of the agent this reasoner belongs to
            model: AI model to use for reasoning (default: use system default)
            provider: AI provider to use (default: use system default)
        """
        self.agent_id = agent_id
        self.model = model
        self.provider = provider
        self.system_prompt = self._get_default_system_prompt()
        self.thought = AgentThought(agent_id)
        self.available_tools = {}
        
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent"""
        return """You are an autonomous agent tasked with accomplishing goals effectively.
Your process:
1. OBSERVE - Understand the current state based on context and observations
2. THINK - Consider options and plan your actions
3. DECIDE - Choose the best action to take
4. ACT - Execute the action using available tools

You have access to various tools that enable you to interact with the environment.
When using tools, carefully consider their parameters and expected outputs.
If a task is complex, break it down into smaller, manageable steps.
Always reflect on the results of your actions to improve future decisions.

When you need to use a tool:
1. Identify the most appropriate tool for the task
2. Provide the required parameters
3. Execute the tool and analyze the results

You are resourceful, thoughtful, and persistent in achieving your goals."""
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt"""
        self.system_prompt = prompt
        
    def register_tool(self, name: str, description: str, 
                      function: Callable, parameters: Dict[str, Any]) -> None:
        """
        Register a tool that the agent can use
        
        Args:
            name: Tool name
            description: Tool description
            function: Function to call when tool is used
            parameters: Parameter schema for the tool
        """
        self.available_tools[name] = {
            "name": name,
            "description": description,
            "function": function,
            "parameters": parameters
        }
        logger.debug(f"Registered tool '{name}'")
        
    def create_plan(self, task: str) -> List[str]:
        """
        Create a plan for accomplishing a task
        
        Args:
            task: Task description
            
        Returns:
            List of plan steps
        """
        logger.info(f"Creating plan for task: {task}")
        
        # Set the task
        self.thought.set_task(task)
        
        # Create the planning prompt
        messages = [
            AIMessage(role="system", content=self.system_prompt),
            AIMessage(role="user", content=f"""
Task: {task}

Create a step-by-step plan to accomplish this task. 
Break down the process into clear, logical steps.
Consider what tools or resources might be needed at each step.

Format your response as a numbered list of steps.
""")
        ]
        
        # Get the AI response
        options = AIRequestOptions(temperature=0.7)
        try:
            response = ai_service.complete(
                messages=messages,
                model=self.model,
                provider=self.provider,
                options=options
            )
            
            # Parse the response into plan steps
            content = response.content.strip()
            lines = content.split('\n')
            plan_steps = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering or bullet points
                    step = line.split('. ', 1)[-1] if '. ' in line else line[2:].strip()
                    plan_steps.append(step)
            
            if not plan_steps:
                # Fallback if no steps were parsed
                plan_steps = [line.strip() for line in lines if line.strip()]
            
            # Set the plan
            self.thought.create_plan(plan_steps)
            
            return plan_steps
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            # Create a simple fallback plan
            fallback = ["Analyze the task", "Gather necessary information", "Execute the task", "Verify results"]
            self.thought.create_plan(fallback)
            return fallback
        
    def decide_action(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decide on the next action to take
        
        Args:
            context: Additional context information
            
        Returns:
            Action decision
        """
        # Combine thought context with additional context
        thought_context = self.thought.get_context()
        if context:
            thought_context.update(context)
        
        # Format the tool descriptions for the prompt
        tools_description = ""
        if self.available_tools:
            tools_description = "Available tools:\n\n"
            for name, tool in self.available_tools.items():
                tools_description += f"- {name}: {tool['description']}\n"
                tools_description += f"  Parameters: {json.dumps(tool['parameters'], indent=2)}\n\n"
        
        # Create the decision prompt
        messages = [
            AIMessage(role="system", content=self.system_prompt),
            AIMessage(role="user", content=f"""
Current task: {thought_context['task']}

Current plan:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(thought_context['plan'])])}

Current step: {thought_context['current_step_index'] + 1}. {
    thought_context['plan'][thought_context['current_step_index']] 
    if thought_context['current_step_index'] < len(thought_context['plan']) else "Plan completed"
}

Recent observations:
{chr(10).join([f"- {obs['content']}" for obs in thought_context['observations']]) if thought_context['observations'] else "No recent observations."}

{tools_description}

Based on the current state, decide on the next action to take.
First, analyze the situation and think about what needs to be done.
Then, decide on the most appropriate action to take.

If you need to use a tool, specify:
1. The tool name
2. The parameters to use

If you need to create a sub-plan or modify the current plan, explain your reasoning.
""")
        ]
        
        # Create AI tools format if tools are available
        ai_tools = None
        if self.available_tools:
            ai_tools = []
            for name, tool in self.available_tools.items():
                ai_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
        
        # Get the AI response
        options = AIRequestOptions(temperature=0.7, tools=ai_tools)
        try:
            response = ai_service.complete(
                messages=messages,
                model=self.model,
                provider=self.provider,
                options=options
            )
            
            # Check if the response contains tool calls
            if response.tool_calls:
                # Process the first tool call
                tool_call = response.tool_calls[0]
                if tool_call["type"] == "function":
                    function = tool_call["function"]
                    tool_name = function["name"]
                    
                    if tool_name in self.available_tools:
                        # Parse the arguments
                        try:
                            arguments = json.loads(function["arguments"])
                            
                            return {
                                "action_type": "tool",
                                "tool": tool_name,
                                "parameters": arguments,
                                "reasoning": response.content
                            }
                        except json.JSONDecodeError:
                            logger.error(f"Invalid tool arguments: {function['arguments']}")
                    else:
                        logger.warning(f"Unknown tool: {tool_name}")
            
            # Default to a text response if no tool call was found
            return {
                "action_type": "text",
                "content": response.content
            }
            
        except Exception as e:
            logger.error(f"Error deciding action: {e}")
            return {
                "action_type": "error",
                "error": str(e)
            }
        
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action
        
        Args:
            action: Action to execute
            
        Returns:
            Result of the action
        """
        logger.debug(f"Executing action: {action['action_type']}")
        
        try:
            if action["action_type"] == "tool":
                tool_name = action["tool"]
                parameters = action["parameters"]
                
                if tool_name in self.available_tools:
                    tool = self.available_tools[tool_name]
                    result = tool["function"](**parameters)
                    
                    # Add observation
                    self.thought.add_observation(f"Used tool '{tool_name}' with result: {result}")
                    
                    return {
                        "success": True,
                        "result": result
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Unknown tool: {tool_name}"
                    }
            
            elif action["action_type"] == "text":
                # Add observation
                self.thought.add_observation(f"Thought: {action['content']}")
                
                return {
                    "success": True,
                    "result": action["content"]
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action['action_type']}"
                }
                
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error executing action: {e}\n{error_trace}")
            return {
                "success": False,
                "error": str(e),
                "traceback": error_trace
            }
        
    def reflect(self, task_result: Any) -> str:
        """
        Reflect on the task execution
        
        Args:
            task_result: Result of the task
            
        Returns:
            Reflection
        """
        logger.info("Reflecting on task execution")
        
        # Create the reflection prompt
        messages = [
            AIMessage(role="system", content=self.system_prompt),
            AIMessage(role="user", content=f"""
Task: {self.thought.task}

Plan:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(self.thought.plan)])}

Observations:
{chr(10).join([f"- {obs['content']}" for obs in self.thought.observations])}

Task result: {task_result}

Reflect on the task execution. Consider:
1. What went well?
2. What could have been improved?
3. Were there any unexpected challenges?
4. How could the plan be improved for similar tasks in the future?
5. What lessons can be learned from this experience?
""")
        ]
        
        # Get the AI response
        options = AIRequestOptions(temperature=0.7)
        try:
            response = ai_service.complete(
                messages=messages,
                model=self.model,
                provider=self.provider,
                options=options
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            return f"Unable to reflect due to error: {e}"


class ReasoningAgent:
    """
    A fully autonomous agent that can reason, plan, and act
    
    This class extends the base Agent with reasoning capabilities
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str = "Reasoning Agent",
                 description: str = "An autonomous agent with reasoning capabilities",
                 model: str = None,
                 provider: str = None):
        """
        Initialize the reasoning agent
        
        Args:
            agent_id: Agent ID
            name: Agent name
            description: Agent description
            model: AI model to use (default: use system default)
            provider: AI provider to use (default: use system default)
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.reasoner = AgentReasoner(agent_id, model, provider)
        self.modules = {}
        self.tools = {}
        self.is_running = False
        self.max_iterations = 50
        self.current_task = None
        
    def register_module(self, name: str, module: Any) -> None:
        """
        Register a module with the agent
        
        Args:
            name: Module name
            module: Module instance
        """
        self.modules[name] = module
        logger.debug(f"Registered module '{name}'")
        
        # Register module tools if available
        if hasattr(module, "get_tools") and callable(module.get_tools):
            tools = module.get_tools()
            for tool_name, tool_data in tools.items():
                self.register_tool(
                    f"{name}.{tool_name}",
                    tool_data["description"],
                    tool_data["function"],
                    tool_data["parameters"]
                )
        
    def register_tool(self, name: str, description: str, 
                      function: Callable, parameters: Dict[str, Any]) -> None:
        """
        Register a tool with the agent
        
        Args:
            name: Tool name
            description: Tool description
            function: Function to call when tool is used
            parameters: Parameter schema for the tool
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "function": function,
            "parameters": parameters
        }
        
        # Register with the reasoner
        self.reasoner.register_tool(name, description, function, parameters)
        
        logger.debug(f"Registered tool '{name}'")
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt"""
        self.reasoner.set_system_prompt(prompt)
        
    def run(self, task: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Run the agent on a task
        
        Args:
            task: Task to run
            max_iterations: Maximum number of iterations
            
        Returns:
            Task result
        """
        logger.info(f"Agent '{self.name}' starting task: {task}")
        
        # Set task and running state
        self.current_task = task
        self.is_running = True
        
        # Create a plan
        plan = self.reasoner.create_plan(task)
        logger.info(f"Created plan with {len(plan)} steps")
        
        # Initialize tracking variables
        start_time = time.time()
        iterations = 0
        max_iter = max_iterations or self.max_iterations
        
        # Results and observations
        results = []
        final_result = {"success": False, "message": "Task incomplete"}
        
        # Main agent loop
        try:
            while self.is_running and iterations < max_iter:
                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > 300:  # 5 minutes timeout
                    logger.warning(f"Agent timed out after {elapsed:.1f} seconds")
                    final_result = {"success": False, "message": "Task timed out"}
                    break
                
                # Decide on the next action
                logger.debug(f"Deciding action (iteration {iterations+1})")
                action = self.reasoner.decide_action()
                
                # Execute the action
                logger.debug(f"Executing action: {action.get('action_type', 'unknown')}")
                result = self.reasoner.execute_action(action)
                
                # Store the result
                results.append({
                    "iteration": iterations,
                    "action": action,
                    "result": result
                })
                
                # Check if plan is complete
                if (self.reasoner.thought.current_step_index >= len(self.reasoner.thought.plan) or
                    action.get("action_type") == "complete"):
                    logger.info("Plan completed")
                    final_result = {
                        "success": True,
                        "message": "Task completed successfully",
                        "iterations": iterations + 1,
                        "time_taken": time.time() - start_time
                    }
                    break
                
                # Increment iteration counter
                iterations += 1
                
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in agent execution: {e}\n{error_trace}")
            final_result = {
                "success": False,
                "message": f"Error: {str(e)}",
                "traceback": error_trace
            }
            
        finally:
            # Reset running state
            self.is_running = False
            
            # Reflect on the task execution
            reflection = self.reasoner.reflect(final_result)
            
            # Complete the final result
            final_result.update({
                "task": task,
                "plan": self.reasoner.thought.plan,
                "steps_completed": self.reasoner.thought.current_step_index,
                "observations": len(self.reasoner.thought.observations),
                "reflection": reflection,
                "iterations": iterations
            })
            
            logger.info(f"Agent '{self.name}' completed task after {iterations} iterations")
            
            return final_result
    
    def stop(self) -> None:
        """Stop the agent execution"""
        logger.info(f"Stopping agent '{self.name}'")
        self.is_running = False
