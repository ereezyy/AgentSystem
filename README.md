# AgentSystem Pi5 Deployment Package

This package contains everything needed to deploy AgentSystem as a Celery worker on a Raspberry Pi 5 with AI HAT+. The Pi5 functions as a distributed worker alongside a primary CPU in the AgentSystem distributed AI architecture.

## Package Contents

```
pi5_deployment_package/
├── README.md                           # This file
├── .env.pi5                           # Pi5-specific environment template
├── pi5_requirements.txt               # Optimized dependencies for Pi5
├── agentsystem-pi5-worker.service     # Systemd service file
├── pi5_health_check.py               # Health monitoring script
├── scripts/                          # Deployment and setup scripts
│   ├── install.sh                    # Main installation script
│   └── setup_environment.sh          # Environment setup script
└── AgentSystem/                      # Complete AgentSystem codebase
    ├── pi5_worker.py                 # Main Pi5 worker entry point
    ├── core/                         # Core system modules
    ├── modules/                      # Feature modules
    ├── services/                     # AI and external services
    ├── utils/                        # Utility functions
    ├── config/                       # Configuration files
    ├── docs/                         # Documentation
    ├── examples/                     # Usage examples
    └── tests/                        # Test suite
```

## Prerequisites

### Hardware Requirements
- Raspberry Pi 5 (4GB+ RAM recommended)
- AI HAT+ (optional but recommended for local inference)
- MicroSD card (32GB+ Class 10)
- Stable network connection to main server

### Software Requirements
- Raspberry Pi OS (64-bit, Bullseye or newer)
- Python 3.9 or newer
- Git
- Redis server (running on main server)

## Quick Start

### 1. Transfer Package to Pi5
```bash
# Copy the entire package to your Pi5
scp -r pi5_deployment_package/ pi@your-pi5-ip:/home/pi/
```

### 2. Basic Installation
```bash
# SSH into your Pi5
ssh pi@your-pi5-ip

# Navigate to the package directory
cd /home/pi/pi5_deployment_package

# Make scripts executable
chmod +x scripts/*.sh

# Run the installation script
sudo ./scripts/install.sh
```

### 3. Configuration
```bash
# Copy and edit the environment file
sudo cp .env.pi5 /opt/agentsystem/.env
sudo nano /opt/agentsystem/.env

# Update the following required settings:
# - CELERY_BROKER_URL=redis://YOUR_MAIN_SERVER_IP:6379/0
# - CELERY_RESULT_BACKEND=redis://YOUR_MAIN_SERVER_IP:6379/0
# - MAIN_SERVER_HOST=YOUR_MAIN_SERVER_IP
# - OPENAI_API_KEY=your_api_key (or other AI provider)
```

### 4. Start the Service
```bash
# Enable and start the systemd service
sudo systemctl enable agentsystem-pi5-worker
sudo systemctl start agentsystem-pi5-worker

# Check status
sudo systemctl status agentsystem-pi5-worker
```

## Detailed Setup

### Environment Configuration

The `.env.pi5` file contains Pi5-optimized settings. Key configurations include:

**Celery Worker Settings:**
- `CELERY_WORKER_CONCURRENCY=2` - Optimized for Pi5 CPU cores
- `CELERY_WORKER_PREFETCH_MULTIPLIER=1` - Prevents memory overload
- `CELERY_WORKER_LOGLEVEL=INFO` - Appropriate logging level

**Resource Constraints:**
- `MAX_MEMORY_USAGE=6GB` - Prevents OOM on 8GB Pi5
- `MAX_CPU_USAGE=80` - Leaves headroom for system processes
- `THREAD_POOL_SIZE=4` - Optimized for Pi5 architecture

**Hardware Optimization:**
- `ENABLE_AI_HAT=true` - Enables AI HAT+ integration
- `VIDEO_RESOLUTION=320x240` - Reduced for Pi5 performance
- `AUDIO_CHUNK_SIZE=512` - Optimized buffer size

### Dependencies Installation

The `pi5_requirements.txt` file contains ARM64-optimized packages:

```bash
# Install with specific optimizations
pip install -r pi5_requirements.txt --no-cache-dir

# For better performance, consider system packages:
sudo apt install python3-opencv python3-numpy python3-scipy
```

### Service Management

The systemd service provides automatic startup and monitoring:

```bash
# Service management commands
sudo systemctl start agentsystem-pi5-worker    # Start
sudo systemctl stop agentsystem-pi5-worker     # Stop
sudo systemctl restart agentsystem-pi5-worker  # Restart
sudo systemctl status agentsystem-pi5-worker   # Check status

# View logs
sudo journalctl -u agentsystem-pi5-worker -f   # Follow logs
sudo journalctl -u agentsystem-pi5-worker -n 50 # Last 50 lines
```

### Health Monitoring

Use the health check script to monitor system status:

```bash
# Basic health check
python3 /opt/agentsystem/pi5_health_check.py

# Verbose output
python3 /opt/agentsystem/pi5_health_check.py --verbose

# JSON output for monitoring systems
python3 /opt/agentsystem/pi5_health_check.py --json

# Set up automated monitoring (crontab)
# Add to crontab (crontab -e):
*/5 * * * * /opt/agentsystem/venv/bin/python3 /opt/agentsystem/pi5_health_check.py --json >> /opt/agentsystem/logs/health_check.log 2>&1
```

## Architecture Overview

### Distributed System Role
The Pi5 worker operates as part of a distributed AgentSystem:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main Server   │    │   Pi5 Worker    │    │ Other Workers   │
│                 │    │                 │    │                 │
│ • Task Queue    │◄──►│ • Celery Worker │    │ • Additional    │
│ • Redis Broker  │    │ • AI HAT+       │    │   Pi5 Units     │
│ • Orchestrator  │    │ • Local Tasks   │    │ • Cloud Workers │
│ • Web Interface │    │ • Monitoring    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Task Distribution
The Pi5 worker handles specific task types:
- **Sensory Processing**: Audio/video analysis using AI HAT+
- **Local Inference**: Edge AI computations
- **Specialized Tasks**: Pi5-optimized workloads
- **Backup Processing**: Overflow from main server

## Performance Optimization

### Pi5-Specific Optimizations

1. **Memory Management:**
   ```bash
   # Add to /boot/config.txt for better memory allocation
   gpu_mem=128
   arm_boost=1
   ```

2. **CPU Scaling:**
   ```bash
   # Set performance governor
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

3. **Storage Optimization:**
   ```bash
   # Enable zram for better memory management
   sudo apt install zram-tools
   ```

### Monitoring Key Metrics

Monitor these Pi5-specific metrics:
- **CPU Temperature**: Keep below 75°C
- **Memory Usage**: Keep below 85%
- **CPU Usage**: Monitor for sustained 100% usage
- **Disk I/O**: SD card performance can be a bottleneck
- **Network Latency**: To main server (should be <50ms)

## Troubleshooting

### Common Issues

**1. Worker Not Connecting to Broker:**
```bash
# Test Redis connection
redis-cli -h YOUR_MAIN_SERVER_IP ping

# Check firewall on main server
sudo ufw status
```

**2. High CPU Temperature:**
```bash
# Check temperature
vcgencmd measure_temp

# Improve cooling or reduce concurrency in .env
CELERY_WORKER_CONCURRENCY=1
```

**3. Memory Issues:**
```bash
# Check memory usage
free -h

# Reduce memory limits in .env
MAX_MEMORY_USAGE=4GB
```

**4. Service Won't Start:**
```bash
# Check detailed logs
sudo journalctl -u agentsystem-pi5-worker -f

# Verify file permissions
sudo chown -R agentsystem:agentsystem /opt/agentsystem
```

### Log Files

Important log locations:
- **Service Logs**: `sudo journalctl -u agentsystem-pi5-worker`
- **Application Logs**: `/opt/agentsystem/logs/`
- **Health Check Logs**: `/opt/agentsystem/logs/health_check.log`
- **System Logs**: `/var/log/syslog`

## Development and Testing

### Running in Development Mode

```bash
# Activate development mode in .env
DEVELOPMENT_MODE=true
ENABLE_DEBUGGING=true

# Run worker manually for testing
cd /opt/agentsystem
source venv/bin/activate
celery -A AgentSystem.pi5_worker worker --loglevel=debug
```

### Testing the Installation

```bash
# Test basic functionality
python3 -c "
import sys
sys.path.append('/opt/agentsystem')
from AgentSystem.pi5_worker import app
print('Pi5 worker imported successfully')
"

# Test Celery connection
python3 -c "
import sys
sys.path.append('/opt/agentsystem')
from AgentSystem.pi5_worker import app
result = app.control.inspect().stats()
print('Celery connection test:', 'SUCCESS' if result else 'FAILED')
"
```

## Security Considerations

### Network Security
- Use VPN or private network for Pi5 ↔ Server communication
- Configure firewall to only allow necessary ports
- Use strong authentication tokens

### File Permissions
```bash
# Secure the installation
sudo chown -R agentsystem:agentsystem /opt/agentsystem
sudo chmod 750 /opt/agentsystem
sudo chmod 600 /opt/agentsystem/.env
```

### Updates and Maintenance
```bash
# Regular system updates
sudo apt update && sudo apt upgrade

# Update Python packages
pip install --upgrade -r pi5_requirements.txt

# Backup configuration before updates
sudo cp /opt/agentsystem/.env /opt/agentsystem/.env.backup
```

## Support and Documentation

- **Full Documentation**: See `AgentSystem/docs/` directory
- **Pi5 Setup Guide**: `AgentSystem/docs/pi5_setup_guide.md`
- **API Documentation**: `AgentSystem/docs/api/`
- **Examples**: `AgentSystem/examples/`

## License

This project is licensed under the same terms as the main AgentSystem project. See `AgentSystem/LICENSE` for details.

---

**Note**: This deployment package is specifically optimized for Raspberry Pi 5 with AI HAT+. For other platforms, use the standard AgentSystem installation process.