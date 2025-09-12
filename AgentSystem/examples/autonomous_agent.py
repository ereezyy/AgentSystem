"""
Autonomous Agent Example
-----------------------
Demonstrates how to use the ReasoningAgent with various modules
"""

import os
import sys
import uuid
import argparse
from typing import Dict, Any

# Add the parent directory to sys.path to import AgentSystem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AgentSystem components
from AgentSystem.core.agent_capabilities import ReasoningAgent
from AgentSystem.modules.browser import BrowserModule
from AgentSystem.services.ai import ai_service
from AgentSystem.utils.logger import get_logger, setup_logging

# Set up logging
setup_logging(level="DEBUG")
logger = get_logger("examples.autonomous_agent")


def create_search_tool(browser_module):
    """Create a web search tool using the browser module"""
    
    def search_web(query: str, num_results: int = 3) -> Dict[str, Any]:
        """Search the web using a search engine"""
        # Open browser to search engine
        browser_module.navigate("https://www.google.com")
        
        # Find search box and enter query
        browser_module.type("input[name='q']", query)
        browser_module.press_key("Enter")
        
        # Wait for results
        browser_module.wait_for_selector(".g")
        
        # Extract results
        results = browser_module.evaluate_script(f"""
            Array.from(document.querySelectorAll('.g')).slice(0, {num_results}).map(el => {{
                const titleEl = el.querySelector('h3');
                const linkEl = el.querySelector('a');
                const snippetEl = el.querySelector('.VwiC3b');
                return {{
                    title: titleEl ? titleEl.innerText : 'No title',
                    url: linkEl ? linkEl.href : 'No URL',
                    snippet: snippetEl ? snippetEl.innerText : 'No snippet'
                }};
            }});
        """)
        
        return {
            "query": query,
            "results": results
        }
    
    # Define the tool parameters schema
    search_tool_params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 3
            }
        },
        "required": ["query"]
    }
    
    return {
        "description": "Search the web for information",
        "function": search_web,
        "parameters": search_tool_params
    }


def create_content_extraction_tool(browser_module):
    """Create a tool to extract content from a webpage"""
    
    def extract_webpage_content(url: str, selector: str = "body") -> Dict[str, Any]:
        """Extract content from a webpage"""
        # Navigate to the URL
        browser_module.navigate(url)
        
        # Wait for content to load
        browser_module.wait_for_selector(selector)
        
        # Extract the content
        content = browser_module.evaluate_script(f"""
            document.querySelector('{selector}').innerText;
        """)
        
        # Get the page title
        title = browser_module.evaluate_script("document.title;")
        
        return {
            "url": url,
            "title": title,
            "content": content[:5000] if content else "No content found"  # Limit content size
        }
    
    # Define the tool parameters schema
    extract_params = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to extract content from"
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the content to extract",
                "default": "body"
            }
        },
        "required": ["url"]
    }
    
    return {
        "description": "Extract content from a webpage",
        "function": extract_webpage_content,
        "parameters": extract_params
    }


def create_note_taking_tools():
    """Create tools for note-taking"""
    notes = {}
    
    def take_note(title: str, content: str) -> Dict[str, Any]:
        """Save a note with the given title and content"""
        note_id = str(uuid.uuid4())[:8]
        notes[note_id] = {
            "id": note_id,
            "title": title,
            "content": content,
            "created_at": "now"  # In a real implementation, use datetime
        }
        return {
            "success": True,
            "id": note_id,
            "message": f"Note '{title}' saved successfully"
        }
    
    def list_notes() -> Dict[str, Any]:
        """List all saved notes"""
        return {
            "count": len(notes),
            "notes": [{"id": id, "title": note["title"]} for id, note in notes.items()]
        }
    
    def get_note(note_id: str) -> Dict[str, Any]:
        """Get a specific note by ID"""
        if note_id in notes:
            return notes[note_id]
        else:
            return {
                "success": False,
                "error": f"Note with ID '{note_id}' not found"
            }
    
    take_note_params = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the note"
            },
            "content": {
                "type": "string",
                "description": "Content of the note"
            }
        },
        "required": ["title", "content"]
    }
    
    get_note_params = {
        "type": "object",
        "properties": {
            "note_id": {
                "type": "string",
                "description": "ID of the note to retrieve"
            }
        },
        "required": ["note_id"]
    }
    
    return {
        "take_note": {
            "description": "Save a note with a title and content",
            "function": take_note,
            "parameters": take_note_params
        },
        "list_notes": {
            "description": "List all saved notes",
            "function": list_notes,
            "parameters": {}
        },
        "get_note": {
            "description": "Get a specific note by ID",
            "function": get_note,
            "parameters": get_note_params
        }
    }


def main():
    """Run the autonomous agent example"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Autonomous Agent Example")
    parser.add_argument("--task", type=str, default="Research the latest AI advancements and take notes on the most interesting findings",
                        help="Task for the agent to perform")
    parser.add_argument("--model", type=str, default=None,
                        help="AI model to use")
    parser.add_argument("--provider", type=str, default=None,
                        help="AI provider to use")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode")
    parser.add_argument("--max-iterations", type=int, default=15,
                        help="Maximum number of agent iterations")
    args = parser.parse_args()
    
    # Create a unique agent ID
    agent_id = str(uuid.uuid4())
    
    # Create a browser module
    browser_module = BrowserModule(headless=args.headless)
    
    # Create the reasoning agent
    agent = ReasoningAgent(
        agent_id=agent_id,
        name="Research Assistant",
        description="An agent that can research topics and take notes",
        model=args.model,
        provider=args.provider
    )
    
    # Register the browser module
    agent.register_module("browser", browser_module)
    
    # Register web search tool
    search_tool = create_search_tool(browser_module)
    agent.register_tool(
        name="search_web",
        description=search_tool["description"],
        function=search_tool["function"],
        parameters=search_tool["parameters"]
    )
    
    # Register content extraction tool
    extract_tool = create_content_extraction_tool(browser_module)
    agent.register_tool(
        name="extract_webpage_content",
        description=extract_tool["description"],
        function=extract_tool["function"],
        parameters=extract_tool["parameters"]
    )
    
    # Register note-taking tools
    note_tools = create_note_taking_tools()
    for name, tool in note_tools.items():
        agent.register_tool(
            name=name,
            description=tool["description"],
            function=tool["function"],
            parameters=tool["parameters"]
        )
    
    # Set a custom system prompt
    agent.set_system_prompt("""You are a research assistant agent tasked with researching topics and taking detailed notes.

Your capabilities:
1. Search the web for information
2. Extract content from webpages
3. Take and organize notes on your findings

When researching:
- Start with broad searches to understand the topic
- Follow up with more specific searches as you learn more
- Extract content from authoritative sources
- Take organized, detailed notes on important information
- Summarize your findings with references

You are thorough, organized, and focused on providing accurate, useful information.""")
    
    # Print task information
    print(f"\n{'='*80}")
    print(f"Starting agent with task: {args.task}")
    print(f"{'='*80}\n")
    
    # Run the agent
    try:
        result = agent.run(args.task, max_iterations=args.max_iterations)
        
        # Print the results
        print(f"\n{'='*80}")
        print(f"Agent completed task with result:")
        print(f"{'='*80}")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Time taken: {result['time_taken']:.2f} seconds")
        print(f"\nPlan:")
        for i, step in enumerate(result['plan']):
            print(f"{i+1}. {step}")
        print(f"\nSteps completed: {result['steps_completed']}/{len(result['plan'])}")
        print(f"\nReflection:")
        print(result['reflection'])
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\nAgent execution interrupted by user")
        agent.stop()
    finally:
        # Clean up browser resources
        if browser_module:
            browser_module.close()


if __name__ == "__main__":
    main()
