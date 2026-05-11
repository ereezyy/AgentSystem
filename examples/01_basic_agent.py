"""
Example 01: Basic Agent Setup

This tutorial demonstrates how to initialize a basic agent using the AgentSystem framework.
It shows how to configure the agent, provide a system prompt, and execute a simple task.

Prerequisites:
- Ensure you have an OpenAI API key set in your environment: `export OPENAI_API_KEY="your-key"`
"""

import asyncio
import os
import sys

# Add the project root to the python path if running directly from the examples folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies if they are not installed for the sake of this example script
# Note: In a real environment, you should have all required dependencies installed.
try:
    from AgentSystem.core.agent import Agent
    from AgentSystem.core.config import Config
except ImportError:
    print(
        "AgentSystem core modules not found. Ensure you are in the project root "
        "and dependencies are installed."
    )
    sys.exit(1)


async def main():
    print("🤖 Initializing Basic Agent...")

    # 1. Initialize Configuration
    # The config manages API keys, provider settings, and system parameters.
    config = Config()

    # 2. Define the Agent's identity and capabilities
    agent_id = "demo_assistant_01"
    system_prompt = (
        "You are a helpful, concise AI assistant demonstrating the AgentSystem framework. "
        "Keep your answers short and to the point."
    )
    capabilities = ["text_generation", "problem_solving"]

    # 3. Create the Agent instance
    # We specify the provider (e.g., "openai", "gemini", "local")
    agent = Agent(
        agent_id=agent_id,
        system_prompt=system_prompt,
        capabilities=capabilities,
        provider="openai",  # Change to your preferred configured provider
        config=config,
    )

    print(f"✅ Agent '{agent.agent_id}' initialized successfully.")

    # 4. Give the agent a task
    task_description = (
        "Explain the concept of a distributed task queue in two sentences."
    )
    print(f"\n📝 Task: {task_description}")

    try:
        # 5. Process the task asynchronously
        print("⏳ Processing...")
        result = await agent.process_task(task_description)

        print("\n🎯 Agent Response:")
        print("-" * 40)
        print(result)
        print("-" * 40)

    except Exception as e:
        print(f"\n❌ Error processing task: {e}")
        print(
            "Note: If you see an API error, ensure your OPENAI_API_KEY is set correctly."
        )


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
