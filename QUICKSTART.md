# AgentSystem Pi5 Worker - Quick Start Guide

**For experienced users who want fast deployment without detailed explanations.**

## Prerequisites Checklist

- [ ] Pi5 with Raspberry Pi OS 64-bit
- [ ] Network access to main AgentSystem server
- [ ] Redis running on main server and accessible
- [ ] Pi5 deployment package transferred to Pi5

## Hardware Quick Reference

| Pi5 Model | Recommended Settings |
|-----------|---------------------|
| **4GB** | Concurrency: 1, Memory: 3GB, Conservative settings |
| **8GB** | Concurrency: 2, Memory: 6GB, Performance settings |

## 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Configure boot optimizations
echo -e "gpu_mem=128\narm_boost=1\nmax_usb_current=1" | sudo tee -a /boot/config.txt

# Set performance governor
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo apt install -y cpufrequtils
```

## 2. Automated Installation

```bash
cd /home/pi/pi5_deployment_package
chmod +x scripts/*.sh
sudo ./scripts/install.sh
```

**Installation does:** System deps → User creation → Python environment → Service setup → Permissions → Optimizations

## 3. Configuration

### Automated Configuration
```bash
sudo ./scripts/setup_environment.sh
```

### Manual Configuration
```bash
sudo cp /opt/agentsystem/.env.template /opt/agentsystem/.env
sudo nano /opt/agentsystem/.env
```

**Essential settings:**
```bash
CELERY_BROKER_URL=redis://YOUR_MAIN_SERVER_IP:6379/0
CELERY_RESULT_BACKEND=redis://YOUR_MAIN_SERVER_IP:6379/0
MAIN_SERVER_HOST=YOUR_MAIN_SERVER_IP
OPENAI_API_KEY=your_api_key_here
SECRET_KEY=$(openssl rand -hex 32)
WORKER_AUTH_TOKEN=$(openssl rand -hex 32)
```

### RAM-Specific Settings

**Pi5 4GB:**
```bash
CELERY_WORKER_CONCURRENCY=1
MAX_MEMORY_USAGE=3GB
VIDEO_RESOLUTION=240x180
CACHE_SIZE_LIMIT=256MB
```

**Pi5 8GB:**
```bash
CELERY_WORKER_CONCURRENCY=2
MAX_MEMORY_USAGE=6GB
VIDEO_RESOLUTION=640x480
CACHE_SIZE_LIMIT=512MB
```

## 4. Service Management

```bash
# Start and enable service
sudo systemctl enable --now agentsystem-pi5-worker

# Check status
sudo systemctl status agentsystem-pi5-worker

# View logs
sudo journalctl -u agentsystem-pi5-worker -f
```

## 5. Quick Verification

```bash
# Health check
python3 /opt/agentsystem/pi5_health_check.py --verbose

# Redis connectivity
redis-cli -h YOUR_MAIN_SERVER_IP -p 6379 ping

# Worker registration
sudo -u agentsystem bash -c "cd /opt/agentsystem && source venv/bin/activate && celery -A AgentSystem.pi5_worker inspect active"

# Resource check
vcgencmd measure_temp  # Should be < 75°C
free -h                # Check available memory
```

## 6. Security Hardening (Optional)

```bash
# Firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw allow ssh
sudo ufw allow 8001

# Secure permissions
sudo chmod 600 /opt/agentsystem/.env
sudo chmod 750 /opt/agentsystem
```

## Quick Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| **Service won't start** | `sudo journalctl -u agentsystem-pi5-worker -n 50` |
| **Redis connection failed** | `redis-cli -h MAIN_SERVER_IP -p 6379 ping` |
| **High CPU temp** | Reduce `CELERY_WORKER_CONCURRENCY=1` |
| **Memory issues** | Lower `MAX_MEMORY_USAGE` for your Pi5 model |
| **Import errors** | Check Python environment: `sudo -u agentsystem bash -c "cd /opt/agentsystem && source venv/bin/activate && python -c 'from AgentSystem.pi5_worker import app'"` |

## Manual Installation Commands (Advanced)

If you prefer manual control over automated installation:

```bash
# System packages
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential \
    git redis-tools sqlite3 libffi-dev libssl-dev libjpeg-dev \
    python3-opencv python3-numpy libraspberrypi-bin

# Create user and directories
sudo groupadd --system agentsystem
sudo useradd --system --gid agentsystem --home /opt/agentsystem --shell /bin/bash agentsystem
sudo mkdir -p /opt/agentsystem/{data,logs,models}

# Copy files
sudo cp -r AgentSystem/* /opt/agentsystem/
sudo cp .env.pi5 /opt/agentsystem/.env.template
sudo cp pi5_requirements.txt pi5_health_check.py /opt/agentsystem/

# Python environment
sudo -u agentsystem python3 -m venv /opt/agentsystem/venv
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r pi5_requirements.txt --no-cache-dir
    pip install -e .
"

# Service and permissions
sudo cp agentsystem-pi5-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo chown -R agentsystem:agentsystem /opt/agentsystem
sudo chmod +x /opt/agentsystem/pi5_health_check.py
```

## File Transfer Quick Reference

| Method | Command |
|--------|---------|
| **SCP** | `scp -r pi5_deployment_package/ pi@PI5_IP:/home/pi/` |
| **USB** | `cp -r /media/pi/USB/pi5_deployment_package/ /home/pi/` |
| **wget** | `wget http://server/pi5_deployment_package.tar.gz && tar -xzf pi5_deployment_package.tar.gz` |

## Performance Monitoring

```bash
# Continuous monitoring
htop
watch -n 1 'vcgencmd measure_temp && free -h'

# Automated health checks (add to crontab)
*/5 * * * * /opt/agentsystem/venv/bin/python3 /opt/agentsystem/pi5_health_check.py --json >> /opt/agentsystem/logs/health.log 2>&1
```

## Expected Results

✅ **Service Status:** `active (running)`  
✅ **Health Check:** `HEALTHY`  
✅ **Redis Connection:** `PONG`  
✅ **Worker Registration:** Visible in Celery inspect  
✅ **Temperature:** < 75°C  
✅ **Memory Usage:** < 85% of available  

---

**Total deployment time:** ~10-15 minutes with automated installation  
**Manual deployment time:** ~30-45 minutes

For detailed explanations and comprehensive troubleshooting, see [`INSTALL.md`](INSTALL.md).  
For file transfer methods, see [`TRANSFER.md`](TRANSFER.md).