# AgentSystem Pi5 Worker - Complete Installation Guide

This comprehensive guide covers the complete installation and deployment of AgentSystem as a Celery worker on Raspberry Pi 5. This documentation is designed for users of all skill levels, from beginners to advanced system administrators.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Pre-Installation Checklist](#pre-installation-checklist)
- [File Transfer to Pi5](#file-transfer-to-pi5)
- [Pi5 System Preparation](#pi5-system-preparation)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [RAM-Specific Optimizations](#ram-specific-optimizations)
- [Service Management](#service-management)
- [Verification and Testing](#verification-and-testing)
- [Security Hardening](#security-hardening)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Overview

The AgentSystem Pi5 Worker operates as a distributed Celery worker in the AgentSystem ecosystem. This deployment package is completely self-contained and designed to work offline for core functionality.

### Architecture Overview

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

## Hardware Requirements

### Minimum Requirements

| Component | Specification | Notes |
|-----------|---------------|-------|
| **Pi5 Board** | Raspberry Pi 5 | 4GB or 8GB RAM |
| **Storage** | 32GB+ microSD (Class 10) | 64GB+ recommended |
| **Power** | Official Pi5 PSU (5V/5A) | USB-C, 27W recommended |
| **Network** | Ethernet or Wi-Fi | Stable connection to main server |
| **AI HAT+** | Optional but recommended | For local inference |

### Recommended Configurations

#### 🟢 Pi5 4GB Configuration
- **Best for**: Light workloads, basic AI tasks, development
- **Concurrency**: 1-2 workers
- **Memory limit**: 3GB for AgentSystem
- **Ideal use cases**: 
  - Sensor data processing
  - Basic computer vision
  - Text processing tasks

#### 🔵 Pi5 8GB Configuration  
- **Best for**: Heavy AI workloads, production environments
- **Concurrency**: 2-4 workers
- **Memory limit**: 6GB for AgentSystem
- **Ideal use cases**:
  - Complex AI inference
  - Video processing
  - Multiple concurrent tasks
  - Local model hosting

### Storage Recommendations

| Storage Type | Performance | Durability | Cost | Recommendation |
|--------------|-------------|------------|------|----------------|
| **microSD Class 10** | Good | Fair | Low | Development/Testing |
| **microSD A2** | Better | Fair | Medium | Light Production |
| **USB 3.0 SSD** | Excellent | Excellent | Medium | Recommended |
| **NVMe HAT** | Excellent | Excellent | High | High Performance |

## Pre-Installation Checklist

### ✅ Before You Begin

- [ ] Raspberry Pi 5 with Raspberry Pi OS (64-bit) installed
- [ ] Network access from Pi5 to your main AgentSystem server
- [ ] Main server running with Redis accessible
- [ ] Pi5 deployment package ready for transfer
- [ ] SSH access configured (for remote installation)
- [ ] Basic system updates completed

### Main Server Requirements

Your main AgentSystem server must have:
- [ ] Redis server running and accessible
- [ ] Firewall configured to allow Pi5 connections
- [ ] AgentSystem orchestrator running
- [ ] Network connectivity to Pi5

### Network Configuration

Ensure network connectivity between Pi5 and main server:

```bash
# Test from Pi5 to main server
ping your-main-server-ip

# Test Redis connectivity (if redis-cli is installed)
redis-cli -h your-main-server-ip -p 6379 ping
```

## File Transfer to Pi5

Choose the method that best fits your setup. For detailed instructions on all transfer methods, see [`TRANSFER.md`](TRANSFER.md).

### Quick Transfer Options

#### Option 1: SCP/SSH (Recommended for Remote Setup)
```bash
# From your computer, copy the package to Pi5
scp -r pi5_deployment_package/ pi@your-pi5-ip:/home/pi/
```

#### Option 2: USB Drive (Physical Access)
1. Copy `pi5_deployment_package/` to USB drive
2. Insert USB drive into Pi5
3. Copy files: `cp -r /media/pi/USB_DRIVE/pi5_deployment_package/ /home/pi/`

#### Option 3: Direct Network Share
```bash
# If using SMB/CIFS share
sudo mount -t cifs //your-computer-ip/shared-folder /mnt/share
cp -r /mnt/share/pi5_deployment_package/ /home/pi/
```

## Pi5 System Preparation

### 1. Update Raspberry Pi OS

```bash
# Update package lists and system packages
sudo apt update && sudo apt upgrade -y

# Reboot to ensure all updates are applied
sudo reboot
```

**💡 Explanation**: This ensures your Pi5 has the latest security patches and system improvements.

### 2. Enable Required Features

```bash
# Enable SSH (if not already enabled)
sudo systemctl enable ssh
sudo systemctl start ssh

# Enable SPI and I2C for AI HAT+ (if using)
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
```

### 3. Configure Boot Options

Add these lines to `/boot/config.txt` for optimal Pi5 performance:

```bash
# Edit boot configuration
sudo nano /boot/config.txt

# Add these lines at the end:
# GPU memory allocation (AI HAT+ optimization)
gpu_mem=128

# Enable ARM performance boost
arm_boost=1

# Increase USB current (for external storage)
max_usb_current=1
```

**💡 Explanation**: These settings optimize memory allocation and performance for AI workloads.

### 4. Set CPU Performance Governor

```bash
# Set CPU to performance mode for consistent performance
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

# Install cpufrequtils if not present
sudo apt install -y cpufrequtils
```

## Installation Methods

Choose between automated or manual installation based on your preference and requirements.

### Method 0: Graphical Installer (Beginner Friendly)

For non-technical users we include a lightweight installer wizard that walks through
creating a virtual environment and installing Python dependencies.

```bash
cd /home/pi/pi5_deployment_package
python3 scripts/gui_installer.py
```

The wizard lets you choose the installation folder and whether to create a dedicated
virtual environment. Progress is displayed inside the window and detailed logs help
diagnose any issues without leaving the graphical interface.

### Method 1: Automated Installation (Recommended)

The automated installation handles everything for you:

```bash
# Navigate to the package directory
cd /home/pi/pi5_deployment_package

# Make installation script executable
chmod +x scripts/*.sh

# Run the automated installation
sudo ./scripts/install.sh
```

**What the automated installer does:**
- ✅ Installs system dependencies
- ✅ Creates system user and directories
- ✅ Sets up Python virtual environment
- ✅ Installs Python packages
- ✅ Configures systemd service
- ✅ Sets appropriate permissions
- ✅ Applies Pi5 optimizations

### Method 2: Manual Installation (Advanced Users)

For users who prefer step-by-step control or need to customize the installation:

#### Step 1: Install System Dependencies

```bash
# Essential packages
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential git curl wget htop \
    redis-tools sqlite3 libffi-dev libssl-dev \
    libjpeg-dev libopenblas-dev libatlas-base-dev \
    libhdf5-dev pkg-config cmake

# Pi5 specific packages for performance
sudo apt install -y \
    python3-opencv python3-numpy python3-scipy \
    libraspberrypi-bin raspi-config
```

#### Step 2: Create System User

```bash
# Create dedicated user for AgentSystem
sudo groupadd --system agentsystem
sudo useradd --system \
    --gid agentsystem \
    --home /opt/agentsystem \
    --shell /bin/bash \
    --comment "AgentSystem Pi5 Worker" \
    agentsystem
```

#### Step 3: Create Directory Structure

```bash
# Create application directories
sudo mkdir -p /opt/agentsystem/{data,logs,models}
sudo mkdir -p /opt/agentsystem/data/{frames,memory}
sudo mkdir -p /tmp/agentsystem

# Copy AgentSystem files
sudo cp -r AgentSystem/* /opt/agentsystem/
sudo cp .env.pi5 /opt/agentsystem/.env.template
sudo cp pi5_requirements.txt /opt/agentsystem/
sudo cp pi5_health_check.py /opt/agentsystem/
```

#### Step 4: Set Up Python Environment

```bash
# Create virtual environment
sudo -u agentsystem python3 -m venv /opt/agentsystem/venv

# Activate and install packages
sudo -u agentsystem bash -c "
    source /opt/agentsystem/venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r /opt/agentsystem/pi5_requirements.txt --no-cache-dir
    cd /opt/agentsystem
    pip install -e .
"
```

#### Step 5: Install Systemd Service

```bash
# Copy and enable service
sudo cp agentsystem-pi5-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agentsystem-pi5-worker
```

#### Step 6: Set Permissions

```bash
# Set ownership and permissions
sudo chown -R agentsystem:agentsystem /opt/agentsystem
sudo chown -R agentsystem:agentsystem /tmp/agentsystem
sudo chmod -R 755 /opt/agentsystem
sudo chmod +x /opt/agentsystem/pi5_health_check.py
```

## Configuration

### Environment Setup

Choose between automated or manual configuration:

#### Automated Configuration (Recommended)

```bash
# Run the interactive configuration script
sudo ./scripts/setup_environment.sh
```

This script will guide you through:
- Redis/Celery broker configuration
- AI provider setup (OpenAI, Gemini)
- Worker settings
- Security configuration
- Resource limits

#### Manual Configuration

```bash
# Copy template and edit manually
sudo cp /opt/agentsystem/.env.template /opt/agentsystem/.env
sudo nano /opt/agentsystem/.env
```

### Required Configuration Values

Edit `/opt/agentsystem/.env` and set these essential values:

```bash
# === REQUIRED SETTINGS ===

# Redis/Celery Configuration
CELERY_BROKER_URL=redis://YOUR_MAIN_SERVER_IP:6379/0
CELERY_RESULT_BACKEND=redis://YOUR_MAIN_SERVER_IP:6379/0
MAIN_SERVER_HOST=YOUR_MAIN_SERVER_IP

# AI Provider (choose at least one)
OPENAI_API_KEY=your_openai_api_key_here
# OR
GEMINI_API_KEY=your_gemini_api_key_here

# Security
SECRET_KEY=your_secure_random_string_here
WORKER_AUTH_TOKEN=your_worker_auth_token_here

# === WORKER CONFIGURATION ===
CELERY_WORKER_NAME=pi5_worker_$(hostname)
WORKER_ID=pi5_$(hostname)
```

### Test Configuration

```bash
# Test Redis connection
redis-cli -h YOUR_MAIN_SERVER_IP -p 6379 ping

# Test Python imports
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    python -c 'from AgentSystem.pi5_worker import app; print(\"Import successful\")'
"
```

## RAM-Specific Optimizations

### Pi5 4GB Configuration

For Pi5 with 4GB RAM, use these conservative settings:

```bash
# Edit /opt/agentsystem/.env
sudo nano /opt/agentsystem/.env

# Pi5 4GB Optimizations
CELERY_WORKER_CONCURRENCY=1
MAX_MEMORY_USAGE=3GB
THREAD_POOL_SIZE=2
CACHE_SIZE_LIMIT=256MB

# Reduce video processing
VIDEO_RESOLUTION=240x180
VIDEO_FPS=10
ENABLE_OBJECT_DETECTION=false

# Lightweight AI settings
AI_HAT_MAX_BATCH_SIZE=1
RESEARCH_MAX_RESULTS=1
MAX_KNOWLEDGE_ITEMS=2000
```

**Additional 4GB optimizations:**

```bash
# Enable zram for better memory management
sudo apt install -y zram-tools

# Configure swap on zram
echo 'ALGO=lz4' | sudo tee -a /etc/default/zramswap
echo 'PERCENT=25' | sudo tee -a /etc/default/zramswap
sudo systemctl enable zramswap
```

### Pi5 8GB Configuration

For Pi5 with 8GB RAM, use these performance settings:

```bash
# Edit /opt/agentsystem/.env
sudo nano /opt/agentsystem/.env

# Pi5 8GB Optimizations
CELERY_WORKER_CONCURRENCY=2
MAX_MEMORY_USAGE=6GB
THREAD_POOL_SIZE=4
CACHE_SIZE_LIMIT=512MB

# Enhanced video processing
VIDEO_RESOLUTION=640x480
VIDEO_FPS=15
ENABLE_OBJECT_DETECTION=true

# Enhanced AI settings
AI_HAT_MAX_BATCH_SIZE=2
RESEARCH_MAX_RESULTS=3
MAX_KNOWLEDGE_ITEMS=5000
ENABLE_BACKGROUND_RESEARCH=true
```

### Systemd Service Resource Limits

Update the service file for your RAM configuration:

```bash
# Edit service file
sudo nano /etc/systemd/system/agentsystem-pi5-worker.service

# For 4GB Pi5:
MemoryMax=3G
CPUQuota=200%

# For 8GB Pi5:
MemoryMax=6G
CPUQuota=320%

# Reload service
sudo systemctl daemon-reload
```

## Service Management

### Starting the Service

```bash
# Start the AgentSystem worker
sudo systemctl start agentsystem-pi5-worker

# Check status
sudo systemctl status agentsystem-pi5-worker

# Enable auto-start on boot
sudo systemctl enable agentsystem-pi5-worker
```

### Service Management Commands

```bash
# Essential service commands
sudo systemctl start agentsystem-pi5-worker    # Start service
sudo systemctl stop agentsystem-pi5-worker     # Stop service
sudo systemctl restart agentsystem-pi5-worker  # Restart service
sudo systemctl status agentsystem-pi5-worker   # Check status

# View logs
sudo journalctl -u agentsystem-pi5-worker -f   # Follow logs live
sudo journalctl -u agentsystem-pi5-worker -n 50 # Last 50 lines
sudo journalctl -u agentsystem-pi5-worker --since "1 hour ago"
```

### Manual Testing (Development Mode)

For debugging or development:

```bash
# Run worker manually (as agentsystem user)
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    celery -A AgentSystem.pi5_worker worker --loglevel=debug
"
```

## Verification and Testing

### 1. System Health Check

```bash
# Run comprehensive health check
python3 /opt/agentsystem/pi5_health_check.py --verbose

# Expected output should show:
# ✅ AgentSystem Pi5 Worker Health: HEALTHY
```

### 2. Service Status Verification

```bash
# Check service is running
sudo systemctl is-active agentsystem-pi5-worker
# Expected: active

# Check service is enabled
sudo systemctl is-enabled agentsystem-pi5-worker
# Expected: enabled
```

### 3. Network Connectivity Test

```bash
# Test Redis connection
redis-cli -h YOUR_MAIN_SERVER_IP -p 6379 ping
# Expected: PONG

# Test main server connectivity
curl -I http://YOUR_MAIN_SERVER_IP:8000
# Expected: HTTP response headers
```

### 4. Worker Registration Test

```bash
# Check if worker appears in Celery
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    celery -A AgentSystem.pi5_worker inspect active
"
# Expected: Worker information and active tasks
```

### 5. Resource Usage Monitoring

```bash
# Check CPU temperature
vcgencmd measure_temp
# Expected: temp=45.0'C (should be < 75°C)

# Check memory usage
free -h
# Expected: Available memory appropriate for your Pi5 model

# Check disk usage
df -h
# Expected: Root filesystem < 90% usage
```

### 6. Log File Verification

```bash
# Check application logs exist and are being written
ls -la /opt/agentsystem/logs/
sudo tail -f /opt/agentsystem/logs/*.log

# Check systemd logs
sudo journalctl -u agentsystem-pi5-worker --since "10 minutes ago"
```

## Security Hardening

### 1. File Permissions

```bash
# Secure the installation directory
sudo chmod 750 /opt/agentsystem
sudo chmod 600 /opt/agentsystem/.env

# Verify permissions
ls -la /opt/agentsystem/.env
# Expected: -rw------- 1 agentsystem agentsystem
```

### 2. Firewall Configuration

```bash
# Install and configure UFW firewall
sudo apt install -y ufw

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port if changed)
sudo ufw allow ssh

# Allow AgentSystem worker port (if needed for direct access)
sudo ufw allow 8001

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### 3. SSH Hardening

```bash
# Edit SSH configuration
sudo nano /etc/ssh/sshd_config

# Recommended settings:
# PermitRootLogin no
# PasswordAuthentication no
# PubkeyAuthentication yes
# Port 2222  # Change default port

# Restart SSH service
sudo systemctl restart ssh
```

### 4. Automatic Updates

```bash
# Install unattended upgrades
sudo apt install -y unattended-upgrades

# Configure automatic security updates
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 5. Network Security

```bash
# Configure worker to only accept connections from main server
sudo nano /opt/agentsystem/.env

# Add these security settings:
ALLOWED_HOSTS=YOUR_MAIN_SERVER_IP,localhost,127.0.0.1
REQUIRE_AUTH_TOKEN=true
ENABLE_FIREWALL_INTEGRATION=true
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Worker Not Connecting to Redis

**Symptoms:**
- Service fails to start
- Logs show "Connection refused" errors
- Health check shows Redis connection failed

**Solutions:**

```bash
# 1. Test Redis connectivity
redis-cli -h YOUR_MAIN_SERVER_IP -p 6379 ping

# 2. Check firewall on main server
# On main server:
sudo ufw allow from PI5_IP_ADDRESS to any port 6379

# 3. Verify Redis configuration
# Check Redis is bound to correct interface (not just 127.0.0.1)

# 4. Check network connectivity
ping YOUR_MAIN_SERVER_IP
telnet YOUR_MAIN_SERVER_IP 6379
```

#### Issue: High CPU Temperature

**Symptoms:**
- `vcgencmd measure_temp` shows > 75°C
- System throttling occurs
- Performance degradation

**Solutions:**

```bash
# 1. Check current temperature
vcgencmd measure_temp

# 2. Reduce worker concurrency
sudo nano /opt/agentsystem/.env
# Set: CELERY_WORKER_CONCURRENCY=1

# 3. Improve cooling
# - Add heatsink or fan
# - Ensure good airflow
# - Check ambient temperature

# 4. Monitor throttling
vcgencmd get_throttled
# 0x0 = no throttling, other values indicate throttling occurred
```

#### Issue: Memory Issues / OOM Kills

**Symptoms:**
- Worker process killed unexpectedly
- "Out of memory" in system logs
- Service restarts frequently

**Solutions:**

```bash
# 1. Check memory usage
free -h
dmesg | grep -i "killed process"

# 2. Reduce memory limits
sudo nano /opt/agentsystem/.env
# For 4GB Pi5:
MAX_MEMORY_USAGE=2GB
CACHE_SIZE_LIMIT=128MB

# 3. Enable swap if not present
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. Monitor memory usage
htop
# or
watch -n 1 'free -h'
```

#### Issue: Service Won't Start

**Symptoms:**
- `systemctl start` fails
- Service shows "failed" status
- No worker processes visible

**Solutions:**

```bash
# 1. Check detailed status
sudo systemctl status agentsystem-pi5-worker -l

# 2. Check logs for errors
sudo journalctl -u agentsystem-pi5-worker -n 50

# 3. Test manual startup
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    celery -A AgentSystem.pi5_worker worker --loglevel=debug
"

# 4. Check file permissions
ls -la /opt/agentsystem/
sudo chown -R agentsystem:agentsystem /opt/agentsystem

# 5. Verify Python environment
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    python -c 'import sys; print(sys.path)'
    python -c 'from AgentSystem.pi5_worker import app'
"
```

#### Issue: AI HAT+ Not Detected

**Symptoms:**
- AI HAT+ features not working
- Hardware acceleration disabled
- Local inference failing

**Solutions:**

```bash
# 1. Check HAT+ detection
lsusb | grep -i hailo
dmesg | grep -i hailo

# 2. Enable required interfaces
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
sudo reboot

# 3. Install HAT+ drivers (if needed)
# Follow manufacturer's installation instructions

# 4. Verify HAT+ in environment
sudo nano /opt/agentsystem/.env
# Ensure: ENABLE_AI_HAT=true
```

#### Issue: Disk Space Issues

**Symptoms:**
- Service fails with disk space errors
- Logs not writing
- Performance degradation

**Solutions:**

```bash
# 1. Check disk usage
df -h
du -sh /opt/agentsystem/*

# 2. Clean up logs
sudo find /opt/agentsystem/logs -name "*.log" -mtime +7 -delete
sudo journalctl --vacuum-time=7d

# 3. Clean package cache
sudo apt clean
pip cache purge

# 4. Move logs to USB storage (if available)
# Mount USB drive and symlink logs directory
```

### Advanced Debugging

#### Enable Debug Logging

```bash
# Edit environment for more verbose logging
sudo nano /opt/agentsystem/.env

# Add/modify these settings:
LOG_LEVEL=DEBUG
CELERY_WORKER_LOGLEVEL=DEBUG
ENABLE_DEBUGGING=true

# Restart service
sudo systemctl restart agentsystem-pi5-worker
```

#### Monitor System Resources

```bash
# Real-time monitoring
htop
iotop  # Disk I/O
nethogs  # Network usage

# Continuous monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "CPU Temp: $(vcgencmd measure_temp)"
    echo "Memory: $(free -h | grep Mem)"
    echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
    echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
    echo
    sleep 60
done
EOF

chmod +x monitor.sh
./monitor.sh
```

## Maintenance

### Regular Maintenance Tasks

#### Weekly Tasks

```bash
# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Check service health
python3 /opt/agentsystem/pi5_health_check.py --verbose

# 3. Check disk usage
df -h

# 4. Review logs for errors
sudo journalctl -u agentsystem-pi5-worker --since "1 week ago" | grep -i error
```

#### Monthly Tasks

```bash
# 1. Update Python packages
sudo -u agentsystem bash -c "
    cd /opt/agentsystem
    source venv/bin/activate
    pip list --outdated
    # Update specific packages as needed
"

# 2. Clean old logs
sudo find /opt/agentsystem/logs -name "*.log" -mtime +30 -delete
sudo journalctl --vacuum-time=30d

# 3. Check certificate expiration (if using SSL)
# 4. Review security updates
# 5. Backup configuration
sudo cp /opt/agentsystem/.env /opt/agentsystem/.env.backup.$(date +%Y%m%d)
```

### Backup and Recovery

#### Configuration Backup

```bash
# Create backup script
cat > /opt/agentsystem/backup_config.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/agentsystem/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration
cp /opt/agentsystem/.env $BACKUP_DIR/env_$DATE
cp /etc/systemd/system/agentsystem-pi5-worker.service $BACKUP_DIR/service_$DATE

# Backup data directory (excluding large files)
tar -czf $BACKUP_DIR/data_$DATE.tar.gz \
    --exclude='*.log' \
    --exclude='frames/*' \
    /opt/agentsystem/data/

echo "Backup completed: $BACKUP_DIR/*_$DATE"
EOF

chmod +x /opt/agentsystem/backup_config.sh
```

#### Recovery Procedure

```bash
# To restore from backup:
# 1. Stop service
sudo systemctl stop agentsystem-pi5-worker

# 2. Restore configuration
sudo cp /opt/agentsystem/backups/env_YYYYMMDD_HHMMSS /opt/agentsystem/.env

# 3. Restore data
sudo tar -xzf /opt/agentsystem/backups/data_YYYYMMDD_HHMMSS.tar.gz -C /

# 4. Set permissions
sudo chown -R agentsystem:agentsystem /opt/agentsystem

# 5. Start service
sudo systemctl start agentsystem-pi5-worker
```

### Performance Monitoring

Set up automated monitoring:

```bash
# Add to crontab for automated health checks
sudo crontab -e

# Add these lines:
# Health check every 5 minutes
*/5 * * * * /opt/agentsystem/venv/bin/python3 /opt/agentsystem/pi5_health_check.py --json >> /opt/agentsystem/logs/health_check.log 2>&1

# Temperature monitoring every minute
* * * * * echo "$(date): $(vcgencmd measure_temp)" >> /opt/agentsystem/logs/temperature.log

# Weekly cleanup
0 2 * * 0 find /opt/agentsystem/logs -name "*.log" -mtime +7 -delete
```

## Support and Additional Resources

- **Full Documentation**: See `AgentSystem/docs/` directory in the deployment package
- **Quick Start Guide**: [`QUICKSTART.md`](QUICKSTART.md) for experienced users
- **File Transfer Guide**: [`TRANSFER.md`](TRANSFER.md) for detailed transfer methods
- **Health Monitoring**: Use [`pi5_health_check.py`](pi5_health_check.py) for system monitoring
- **Configuration Reference**: See [`.env.pi5`](.env.pi5) for all available settings

---

**🎉 Congratulations!** Your AgentSystem Pi5 Worker should now be successfully deployed and running. The worker will automatically connect to your main server and begin processing tasks assigned to it.

For ongoing monitoring, regularly run the health check and review the service logs to ensure optimal performance.