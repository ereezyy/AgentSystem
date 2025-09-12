# Advanced Setup Guide: Sensory Input and Continuous Learning

This guide provides detailed setup instructions for the advanced capabilities of AgentSystem, specifically the Sensory Input and Continuous Learning modules.

## System Requirements

For full functionality, your system should have:

- Python 3.9 or higher
- A microphone (for audio input)
- A webcam (for video input)
- Internet connection (for web research)
- At least 4GB of available RAM
- Approximately 1GB of disk space for knowledge storage

## Installation Steps

### 1. Basic Installation

First, install the base system:

```bash
git clone https://github.com/yourusername/AgentSystem.git
cd AgentSystem
pip install -r requirements.txt
```

### 2. Additional Dependencies for Sensory Input

The audio components require additional system libraries:

#### On Windows:
- PyAudio should install automatically via pip
- For video, OpenCV may require Microsoft Visual C++ Redistributable

#### On macOS:
```bash
brew install portaudio
```

#### On Linux:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install libopencv-dev python3-opencv
```

### 3. Setting Up Continuous Learning

The knowledge base component requires initial setup:

```bash
# Create necessary directories
mkdir -p ~/.agent_system/knowledge
```

## Configuration

### Audio and Video Configuration

Edit your `.env` file to include these parameters:

```
# Sensory Input Configuration
AUDIO_DEVICE_INDEX=0  # Default microphone, change if needed
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024

VIDEO_DEVICE_INDEX=0  # Default webcam, change if needed
VIDEO_RESOLUTION=640x480
VIDEO_FPS=30

# Enable/disable components
ENABLE_AUDIO_PROCESSING=true
ENABLE_VIDEO_PROCESSING=true
ENABLE_FACE_DETECTION=true
```

### Continuous Learning Configuration

Add these to your `.env` file:

```
# Continuous Learning Configuration
KNOWLEDGE_DB_PATH=~/.agent_system/knowledge/knowledge.db
ENABLE_BACKGROUND_RESEARCH=true
RESEARCH_INTERVAL=3600  # 1 hour between auto-research cycles
RESEARCH_DEPTH=2  # How deep to follow links (1-3)
LEARNING_USER_AGENT="Mozilla/5.0 AgentSystem Research Bot"
```

## Testing Your Setup

### Test Sensory Input

Run the sensory input test to verify your microphone and webcam are working:

```bash
python -m AgentSystem.tests.test_sensory_input
```

You should see output indicating successful audio and video capture, along with any detected speech or faces.

### Test Continuous Learning

Run the learning test to verify the knowledge base and research capabilities:

```bash
python -m AgentSystem.tests.test_continuous_learning
```

This will create a test entry in the knowledge base and perform a simple web search.

## Troubleshooting

### Audio Issues

- If you see "No Default Input Device Available" error:
  - Check if your microphone is properly connected
  - Try setting a specific `AUDIO_DEVICE_INDEX` in your `.env` file
  
- If speech recognition fails:
  - Ensure you're in a quiet environment
  - Check your microphone volume levels
  - Try a different microphone

### Video Issues

- If the webcam doesn't activate:
  - Check if another application is using the webcam
  - Try setting a specific `VIDEO_DEVICE_INDEX` in your `.env` file
  
- If face detection doesn't work:
  - Ensure adequate lighting
  - Try adjusting the `VIDEO_RESOLUTION` to a lower value

### Learning Issues

- If web research fails:
  - Check your internet connection
  - Verify that your firewall isn't blocking requests
  - Try lowering the `RESEARCH_DEPTH` to 1

- If the knowledge base isn't persisting:
  - Check permissions on the `~/.agent_system/knowledge` directory
  - Verify the path in `KNOWLEDGE_DB_PATH` is correct for your OS

## Advanced Configuration

### Custom Face Detection Models

You can replace the default face detection with a custom model:

```
FACE_DETECTION_MODEL_PATH=/path/to/your/model
FACE_DETECTION_CONFIDENCE=0.6
```

### Web Research Customization

Fine-tune the web research behavior:

```
RESEARCH_RATE_LIMIT=2  # seconds between requests
RESEARCH_USER_AGENT="Your custom user agent"
RESEARCH_MAX_RESULTS=5
RESEARCH_CACHE_EXPIRY=3600  # seconds
```

## Running the Autonomous Learning Agent

Once everything is set up, run the autonomous learning agent:

```bash
python -m AgentSystem.examples.autonomous_agent_with_learning --autonomous --sensory
```

The agent will start with both autonomous operation and sensory processing enabled.
