"""
Self-Improving Agent Example
---------------------------
Demonstrates an agent that can analyze, modify, and improve its own code
"""

import os
import sys
import uuid
import argparse
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path to import AgentSystem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AgentSystem components
from AgentSystem.core.agent_capabilities import ReasoningAgent
from AgentSystem.modules.code_editor import CodeEditorModule
from AgentSystem.services.ai import ai_service
from AgentSystem.utils.logger import get_logger, setup_logging

# Set up logging
setup_logging(level="INFO")
logger = get_logger("examples.self_improving_agent")


class SelfImprovingAgent:
    """An agent that can analyze and improve its own code"""
    
    def __init__(self, agent_id: str = None, model: str = None, provider: str = None):
        """
        Initialize the self-improving agent
        
        Args:
            agent_id: Unique identifier for the agent (default: generate UUID)
            model: AI model to use (default: use system default)
            provider: AI provider to use (default: use system default)
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Create workspace for code operations
        self.workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Create the modules
        self.code_editor = CodeEditorModule(workspace_dir=self.workspace_dir)
        
        # Create the reasoning agent
        self.agent = ReasoningAgent(
            agent_id=self.agent_id,
            name="Self-Improving Agent",
            description="An agent that can analyze and improve its own code",
            model=model,
            provider=provider
        )
        
        # Register modules
        self.agent.register_module("code_editor", self.code_editor)
        
        # Set a custom system prompt
        self.agent.set_system_prompt("""You are a self-improving agent capable of analyzing, modifying, and enhancing your own code.

Your capabilities:
1. Read and understand Python code
2. Analyze code structure and complexity
3. Generate and write new code
4. Modify existing code to improve it
5. Refactor code for better performance or readability
6. Test code to ensure it works correctly

When working with code:
- Think carefully about the implications of any changes
- Create backups before making significant modifications
- Test changes to ensure they work as expected
- Focus on maintainability and readability
- Document your changes and reasoning

You are thoughtful, methodical, and focused on making meaningful improvements while maintaining functionality.""")
    
    def improve_self(self, aspects_to_improve: List[str] = None, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Analyze and improve the agent's own code
        
        Args:
            aspects_to_improve: Specific aspects to focus on improving
                               (default: general improvements)
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Dictionary with results of the improvement process
        """
        # Get the current file path (this file)
        current_file = os.path.abspath(__file__)
        
        # Create a task for the agent
        aspects_str = ""
        if aspects_to_improve:
            aspects_str = f" Focus on these specific aspects: {', '.join(aspects_to_improve)}."
        
        task = f"""Analyze and improve your own code in the file: {current_file}.{aspects_str}

Follow these steps:
1. Read and analyze the current code to understand its structure and functionality
2. Identify areas for improvement (performance, readability, extensibility, etc.)
3. Make targeted improvements to the code
4. Test the changes to ensure they work correctly
5. Document the improvements you've made

Be especially careful since you are modifying your own code. Always create backups before making changes."""
        
        # Run the agent
        return self.agent.run(task, max_iterations=max_iterations)
    
    def extend_capabilities(self, new_capability: str, max_iterations: int = 15) -> Dict[str, Any]:
        """
        Extend the agent's capabilities by adding new functionality
        
        Args:
            new_capability: Description of the new capability to add
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Dictionary with results of the extension process
        """
        # Get the current file path (this file)
        current_file = os.path.abspath(__file__)
        
        # Create a task for the agent
        task = f"""Extend your capabilities by adding this new functionality: "{new_capability}"

Follow these steps:
1. Read and analyze your current code to understand your existing capabilities
2. Design and implement the new capability
3. Integrate it with your existing code
4. Test the new capability to ensure it works correctly
5. Document the changes and how to use the new capability

Be careful to maintain compatibility with your existing functionality."""
        
        # Run the agent
        return self.agent.run(task, max_iterations=max_iterations)
    
    def diagnose_and_fix(self, problem_description: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Diagnose and fix a problem in the agent's code
        
        Args:
            problem_description: Description of the problem to fix
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Dictionary with results of the diagnosis and fix
        """
        # Get the current file path (this file)
        current_file = os.path.abspath(__file__)
        
        # Create a task for the agent
        task = f"""Diagnose and fix this problem in your code: "{problem_description}"

Follow these steps:
1. Read and analyze your current code to understand the problem area
2. Identify the root cause of the problem
3. Develop a solution to fix the problem
4. Implement the fix and verify it resolves the issue
5. Document the problem, root cause, and solution

Be methodical in your approach and test your fix thoroughly."""
        
        # Run the agent
        return self.agent.run(task, max_iterations=max_iterations)
    
    def analyze_self(self) -> Dict[str, Any]:
        """
        Perform a detailed analysis of the agent's own code
        
        Returns:
            Dictionary with analysis results
        """
        # Get the current file path (this file)
        current_file = os.path.abspath(__file__)
        
        # Use the code editor module directly
        return self.code_editor.analyze_code(current_file)
    
    def create_improved_version(self, specification: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an improved version of the agent based on a specification
        
        Args:
            specification: Detailed specification for the improved agent
            output_path: Path to save the new agent code (default: generate path)
            
        Returns:
            Dictionary with the creation results
        """
        if not output_path:
            # Generate a path for the new version
            version_number = len(os.listdir(self.workspace_dir)) + 1
            output_path = os.path.join(
                self.workspace_dir, 
                f"self_improving_agent_v{version_number}.py"
            )
        
        # Create a task for the agent
        task = f"""Create an improved version of yourself based on this specification:

{specification}

Follow these steps:
1. Analyze your current code and capabilities
2. Design improvements that meet the specification
3. Implement a new version of your code with these improvements
4. Save the new version to: {output_path}
5. Verify the new version works correctly

Focus on creating a well-structured, maintainable, and efficient implementation."""
        
        # Run the agent
        return self.agent.run(task, max_iterations=20)


def main():
    """Run the self-improving agent example"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Self-Improving Agent Example")
    parser.add_argument("--action", type=str, choices=["improve", "extend", "diagnose", "analyze", "create"],
                       default="improve", help="Action to perform")
    parser.add_argument("--description", type=str,
                       help="Description for extend, diagnose, or create actions")
    parser.add_argument("--aspects", type=str, nargs="+",
                       help="Specific aspects to improve")
    parser.add_argument("--output", type=str,
                       help="Output path for create action")
    parser.add_argument("--model", type=str,
                       help="AI model to use")
    parser.add_argument("--provider", type=str,
                       help="AI provider to use")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Maximum number of iterations")
    args = parser.parse_args()
    
    # Create the self-improving agent
    agent = SelfImprovingAgent(model=args.model, provider=args.provider)
    
    # Print task information
    print(f"\n{'='*80}")
    print(f"Self-Improving Agent - {args.action.capitalize()} Action")
    print(f"{'='*80}\n")
    
    # Perform the requested action
    try:
        if args.action == "improve":
            print("Improving agent's own code...")
            result = agent.improve_self(
                aspects_to_improve=args.aspects,
                max_iterations=args.iterations
            )
            
        elif args.action == "extend":
            if not args.description:
                print("Error: --description is required for extend action")
                return
            
            print(f"Extending agent capabilities: {args.description}")
            result = agent.extend_capabilities(
                new_capability=args.description,
                max_iterations=args.iterations
            )
            
        elif args.action == "diagnose":
            if not args.description:
                print("Error: --description is required for diagnose action")
                return
            
            print(f"Diagnosing and fixing problem: {args.description}")
            result = agent.diagnose_and_fix(
                problem_description=args.description,
                max_iterations=args.iterations
            )
            
        elif args.action == "analyze":
            print("Analyzing agent's own code...")
            result = agent.analyze_self()
            
        elif args.action == "create":
            if not args.description:
                print("Error: --description is required for create action")
                return
            
            print(f"Creating improved version: {args.description}")
            result = agent.create_improved_version(
                specification=args.description,
                output_path=args.output
            )
            
        # Print the results
        print(f"\n{'='*80}")
        print(f"Action completed with result:")
        print(f"{'='*80}")
        
        if args.action == "analyze":
            # Format analysis results
            if result.get("success", False):
                print(f"File: {result['path']}")
                print(f"\nMetrics:")
                for key, value in result["metrics"].items():
                    print(f"- {key}: {value}")
                
                print(f"\nClasses ({len(result['classes'])}):")
                for cls in result["classes"]:
                    print(f"- {cls['name']} (line {cls['line']}, {len(cls['methods'])} methods)")
                
                print(f"\nFunctions ({len(result['functions'])}):")
                for func in result['functions']:
                    print(f"- {func['name']} (line {func['line']})")
            else:
                print(f"Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            # Format agent results
            print(f"Success: {result['success']}")
            print(f"Message: {result['message']}")
            print(f"Iterations: {result['iterations']}")
            print(f"Time taken: {result['time_taken']:.2f} seconds")
            print(f"\nPlan:")
            for i, step in enumerate(result['plan']):
                print(f"{i+1}. {step}")
            print(f"\nSteps completed: {result['steps_completed']}/{len(result['plan'])}")
        
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\nAgent execution interrupted by user")
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
