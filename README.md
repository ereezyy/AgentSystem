![Apex Mind Avatar](assets/apex_mind_avatar.png)

# 🤖 AgentSystem

**Distributed AI orchestration system optimized for edge computing**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-red.svg)](https://www.raspberrypi.com/)

---

## What is AgentSystem?

AgentSystem is a distributed AI framework designed to run AI workloads across multiple devices, from high-performance servers to edge devices like Raspberry Pi 5. It uses Celery for task distribution, supports multiple AI providers (OpenAI, Gemini, OpenRouter, xAI), and features autonomous operations with memory persistence.

**Key Features:**
- 🚀 **Distributed Task Queue** — Celery-powered task distribution across multiple workers
- 🧠 **Multi-Provider AI** — Seamless switching between OpenAI, Gemini, OpenRouter, xAI, and local models
- 🔄 **Autonomous Operations** — Self-managing tasks with memory and context persistence
- 📡 **Edge Optimized** — Runs efficiently on Raspberry Pi 5 with AI HAT+
- 📊 **Real-Time Streaming** — WebSocket-based streaming for AI responses
- 💾 **Memory System** — Persistent conversation history and context management
- ⚡ **Multimodal Support** — Text, image, and vision model integration

---

## Architecture

```
┌─────────────────┐
│   Main Server   │  ← Central coordinator (Redis + Flask/FastAPI)
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Worker1│ │Worker2│ │ Pi5  │ │ Pi5  │  ← Distributed workers
│(Cloud)│ │(Local)│ │Worker│ │Worker│
└───────┘ └──────┘ └──────┘ └──────┘
```

---

## Quick Start

### Prerequisites
- **Python 3.9+**
- **Redis** (for task queue)
- **API Keys** (OpenAI, Gemini, or other supported providers)

### Installation

```bash
# Clone the repository
git clone https://github.com/ereezyy/AgentSystem.git
cd AgentSystem

# Install dependencies
pip install -r requirements-prod.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys and Redis URL
```

### Running a Worker

```bash
# Start a Celery worker
celery -A AgentSystem.core.celery_app worker --loglevel=info
```

### Raspberry Pi 5 Deployment

See [INSTALL.md](INSTALL.md) for detailed Pi5 setup instructions.

---

## Supported AI Providers

| Provider    | Text | Vision | Streaming | Local |
|-------------|------|--------|-----------|-------|
| OpenAI      | ✅   | ✅     | ✅        | ❌    |
| Gemini      | ✅   | ✅     | ✅        | ❌    |
| OpenRouter  | ✅   | ✅     | ✅        | ❌    |
| xAI (Grok)  | ✅   | ❌     | ✅        | ❌    |
| Local (Pi5) | ✅   | ❌     | ❌        | ✅    |

---

## Project Structure

```
AgentSystem/
├── core/                    # Core system modules
│   ├── celery_app.py       # Celery configuration
│   ├── memory.py           # Memory persistence
│   └── config.py           # System configuration
├── services/                # AI provider integrations
│   ├── ai_providers/       # OpenAI, Gemini, xAI, etc.
│   ├── ai.py               # Main AI service
│   └── streaming_service.py
├── autonomous/              # Autonomous operations engine
├── modules/                 # Feature modules
├── utils/                   # Utility functions
├── docs/                    # Documentation
└── tests/                   # Test suite
```

---

## Configuration

Key environment variables:

```bash
# Redis (required)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# AI Providers (at least one required)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
XAI_API_KEY=your_grok_key

# Worker Settings
CELERY_WORKER_CONCURRENCY=4  # Adjust based on hardware
```

---

## Use Cases

- **Multi-device AI workloads** — Distribute heavy AI tasks across multiple machines
- **Edge AI deployments** — Run AI on Raspberry Pi clusters for cost-effective inference
- **AI agent orchestration** — Build autonomous agents with memory and task queuing
- **Hybrid cloud/edge setups** — Combine cloud GPUs with local edge devices

---

## Performance

| Device          | Workers | Tasks/min | Avg Response Time |
|-----------------|---------|-----------|-------------------|
| Pi5 (4GB)       | 2       | ~30       | 2-5s              |
| Pi5 w/ AI HAT+  | 2       | ~50       | 1-3s              |
| Server (8-core) | 8       | ~200      | 0.5-1s            |

---

## Documentation

- [Installation Guide](INSTALL.md)
- [Backend Deployment](BACKEND_DEPLOYMENT_GUIDE.md)
- [Microcopy Analysis](MICROCOPY_ANALYSIS.md)

---

## Roadmap

- [ ] **Multi-modal RAG** — Knowledge base integration
- [ ] **Web UI** — Dashboard for monitoring workers
- [ ] **Auto-scaling** — Dynamic worker allocation
- [ ] **Function calling** — Tool use for autonomous agents
- [ ] **Voice support** — Speech-to-text/text-to-speech

---

## Contributing

Pull requests welcome. For major changes, open an issue first.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Eddy Woods** ([@ereezyy](https://github.com/ereezyy))
*AI Engineer & Game Developer*

---

**⭐ Star this repo if you find it useful!**
