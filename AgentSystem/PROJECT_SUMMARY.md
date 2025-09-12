# AgentSystem: Autonomous Learning Agent

## Project Summary

The AgentSystem is a cutting-edge framework for creating autonomous, self-improving AI agents that can interact with the real world, learn continuously, and adapt to changing environments. This project provides a comprehensive solution for building agents that can perceive their surroundings through audio and video, acquire knowledge through web research, and make autonomous decisions based on that information.

## Core Features

### 1. Modular Architecture

The system is built around a flexible, modular architecture that allows for easy extension and customization:

- **Core Framework**: Provides the fundamental agent structure with memory, state management, and decision-making capabilities
- **Service Layer**: Handles communication with AI providers (OpenAI, Anthropic, local LLMs)
- **Module System**: Offers plug-and-play functionality for different capabilities

### 2. Sensory Input

The agent can perceive the world through:

- **Audio Processing**: Records and analyzes audio from a microphone
- **Speech Recognition**: Converts spoken language to text
- **Video Capture**: Records and processes video from a webcam
- **Visual Analysis**: Detects faces, objects, and other visual features

### 3. Continuous Learning

The agent continuously improves through:

- **Knowledge Base**: A structured database for storing facts, documents, and relationships
- **Web Research**: Autonomous research capabilities to find information online
- **Fact Extraction**: Identification of factual statements from text
- **Learning History**: Tracking of what the agent has learned over time

### 4. Autonomous Operation

The agent can operate independently:

- **Autonomous Thinking**: Regular evaluation of current state and decision-making
- **Proactive Research**: Identification of knowledge gaps and research to fill them
- **Insight Generation**: Creation of summaries and insights based on knowledge
- **Goal Management**: Setting and pursuing goals based on observations

### 5. Web Interaction

- **Browser Automation**: Navigate and interact with websites
- **Content Processing**: Extract and understand web content
- **Research Capabilities**: Search for and synthesize information

## System Components

### Core Components

1. **Agent**: Central controller that coordinates all activities
2. **Memory**: Stores conversation history and observations
3. **State**: Maintains the agent's internal state
4. **AI Service**: Interfaces with large language models

### Modules

1. **Sensory Input Module**: Handles audio and video processing
2. **Continuous Learning Module**: Manages knowledge acquisition and storage
3. **Browser Module**: Enables web browsing and interaction
4. **Email Module**: Handles email communication
5. **Code Editor Module**: Provides code editing capabilities

## Implementation Details

### Knowledge Base

The knowledge base uses SQLite to store:

- **Facts**: Discrete pieces of information with confidence scores
- **Documents**: Larger texts with summaries and metadata
- **Relationships**: Connections between facts and documents
- **Embeddings**: Vector representations for semantic search

### Web Research

The web research component:

- Uses a respectful web scraper with rate limiting
- Extracts main content from web pages
- Identifies factual statements
- Stores information in the knowledge base
- Follows links to explore topics in depth

### Autonomous Agent Loop

The autonomous operation follows this process:

1. Collect sensory input (audio, video)
2. Process and analyze observations
3. Update internal state and memory
4. Decide on next actions (research, insight generation)
5. Execute actions and learn from results
6. Repeat

## Getting Started

### Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment: Copy `config/env.template` to `config/.env` and add API keys
3. Run the agent: `python run_learning_agent.py --full`

### Configuration Options

The agent can be configured through:

- Command-line arguments
- Environment variables
- Interactive commands during operation

### Command Line Interface

```
python run_learning_agent.py [options]

Options:
  --name NAME          Set agent name
  --provider PROVIDER  AI provider (openai, anthropic, local)
  --model MODEL        Model to use
  --autonomous         Enable autonomous operation
  --sensory            Enable sensory input
  --full               Enable all capabilities
```

## Use Cases

### 1. Research Assistant

Use the agent to research topics, synthesize information, and generate insights. The continuous learning capabilities allow it to build expertise in specific domains over time.

### 2. Personal Assistant

Leverage the sensory capabilities to create an assistant that can see, hear, and respond to your environment. The agent can learn your preferences and adapt to your needs.

### 3. Learning Companion

Use the agent as a learning tool that researches topics as you discuss them, providing additional context and information to enhance understanding.

### 4. Development Assistant

The code editing capabilities, combined with continuous learning, make the agent an effective programming assistant that improves as it learns about your codebase.

## Future Directions

This project provides a foundation for more advanced autonomous systems:

1. **Multimodal Learning**: Integrating text, audio, and visual learning
2. **Collaborative Agents**: Multiple specialized agents working together
3. **Physical Interaction**: Connecting to IoT devices and robotics platforms
4. **Personalized Knowledge**: Building user-specific knowledge bases
5. **Counterfactual Reasoning**: Enabling "what if" scenario exploration

## Documentation

For more detailed information, refer to:

- [README.md](README.md): Project overview and basic setup
- [docs/user_guide.md](docs/user_guide.md): Detailed usage guide
- [docs/advanced_setup.md](docs/advanced_setup.md): Advanced configuration
- Code documentation within module files

## Testing

Run the tests to verify functionality:

```bash
# Run all tests
python -m unittest discover -s tests

# Test specific modules
python -m AgentSystem.tests.test_sensory_input
python -m AgentSystem.tests.test_continuous_learning

# Quick functional tests
python -m AgentSystem.tests.test_sensory_input --quick
python -m AgentSystem.tests.test_continuous_learning --quick
```

## License

This project is available under the MIT License. See [LICENSE](LICENSE) for details.
