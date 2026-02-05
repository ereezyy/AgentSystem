# Autonomous Learning Agent User Guide

This guide will help you get started with the Autonomous Learning Agent, understand its capabilities, and make the most of its features.

## Overview

The Autonomous Learning Agent is a self-improving AI system that can:
- Process real-world sensory input (audio and video)
- Continuously learn through web research and knowledge gathering
- Autonomously make decisions and take actions
- Interact with you through a natural language interface

## Quick Start

### Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment:
   ```bash
   cp config/env.template config/.env
   ```
   
3. Edit the `.env` file to add your API keys and configuration.

### Running the Agent

To run the agent with all capabilities enabled:

```bash
python run_learning_agent.py --full
```

Or with specific options:

```bash
python run_learning_agent.py --name "My Assistant" --autonomous --sensory
```

## Command Line Options

- `--name`: Set the agent's name (default: "Autonomous Learning Agent")
- `--provider`: Choose the AI provider: "openai", "anthropic", or "local" (default: "openai")
- `--model`: Select the AI model to use (default: "gpt-4o")
- `--autonomous`: Enable autonomous operation mode
- `--sensory`: Enable sensory input processing (audio/video)
- `--full`: Enable all capabilities (equivalent to --autonomous --sensory)

## Interactive Commands

Once the agent is running, you can use these commands:

- `auto` - Toggle autonomous mode on/off
- `sensory` - Toggle sensory processing on/off
- `research <topic>` - Actively research a specific topic
- `status` - Display the agent's current status
- `exit` - Exit the program

Any other input will be treated as a message to the agent.

## Core Capabilities

### Sensory Processing

The agent can process:

- **Audio**: Listen through your microphone and recognize speech
- **Video**: Watch through your webcam and detect objects, faces, etc.

These inputs are processed and fed into the agent's memory, allowing it to respond to what it sees and hears.

### Continuous Learning

The agent builds a knowledge base by:

- Storing facts and documents it encounters
- Researching topics on the web
- Extracting factual information from text
- Learning from interactions with you

Over time, the agent builds a personalized knowledge base that improves its capabilities.

### Autonomous Operation

When in autonomous mode, the agent will:

1. Observe the environment through sensory input
2. Process and analyze those observations
3. Research topics related to observations or current focus
4. Generate insights based on its knowledge
5. Take appropriate actions based on its goals

You can set specific goals for the agent or let it operate based on its own assessment of what's important.

## Example Use Cases

### Research Assistant

```bash
python run_learning_agent.py --name "Research Assistant"
```

Then use the `research` command to explore topics:

```
> research quantum computing
> What are the practical applications of quantum computing?
```

### Environmental Monitor

```bash
python run_learning_agent.py --sensory --name "Environment Monitor"
```

The agent will process audio and video from your environment and alert you to significant events.

### Autonomous Learner

```bash
python run_learning_agent.py --full --name "Autonomous Learner"
```

Let the agent operate autonomously, learning from its environment and research, providing insights as it discovers new information.

## Troubleshooting

### Audio/Video Issues

If you encounter issues with sensory input:

1. Check that your microphone and webcam are connected and working
2. Verify you have the necessary permissions for the devices
3. Adjust the device indexes in the `.env` file if needed
4. Install system dependencies (e.g., `sudo apt-get install portaudio19-dev`) and then install the Python packages: `pip install pyaudio opencv-python SpeechRecognition`

### Research Issues

If web research isn't working:

1. Verify your internet connection
2. Check that you have the required packages: `pip install requests beautifulsoup4 nltk scikit-learn`
3. Try running with a lower research depth: `research <topic> --depth 1`

### Performance Considerations

The agent can use significant resources when all features are enabled:

- Audio and video processing can be CPU-intensive
- Large language models require significant memory
- Web research can use network bandwidth

Consider disabling features you're not using for better performance.

## Advanced Customization

### Knowledge Base Location

By default, the knowledge base is stored in `~/.agent_system/knowledge/knowledge.db`. You can change this location in your `.env` file:

```
KNOWLEDGE_DB_PATH=/path/to/your/knowledge.db
```

### Customizing Research Behavior

You can configure the research behavior in your `.env` file:

```
RESEARCH_DEPTH=2
RESEARCH_RATE_LIMIT=2
RESEARCH_MAX_RESULTS=5
```

### Adding Custom Modules

You can extend the agent with your own modules by:

1. Creating a new module class in the `modules` directory
2. Implementing the required interface (see existing modules for examples)
3. Registering your module with the agent in your application code

## Privacy Considerations

The agent processes and potentially stores:

- Audio from your microphone
- Video from your webcam
- Research conducted on your behalf
- Interactions you have with the agent

This data is stored locally on your machine, but be mindful of the information you allow the agent to process and store.

## Next Steps

- Try different configuration options to find what works best for your needs
- Explore the example scripts in the `examples` directory for more ideas
- Check out the developer documentation if you're interested in customizing or extending the agent
