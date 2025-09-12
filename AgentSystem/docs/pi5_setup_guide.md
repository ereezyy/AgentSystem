# Raspberry Pi 5 Setup Guide for AgentSystem

This guide provides step-by-step instructions for setting up a Raspberry Pi 5 with AI HAT+ to work as a distributed AI worker for the AgentSystem.

## Hardware Requirements

### Essential Components
- **Raspberry Pi 5** (8GB RAM recommended for AI workloads)
- **AI HAT+** (Hailo-8 chip providing 26 TOPS AI acceleration)
- **MicroSD Card** (64GB+ Class 10 or UHS-I for better performance)
- **Power Supply** (27W USB-C official Pi 5 power supply)
- **Cooling Solution** (Active cooling fan or heatsink for sustained AI workloads)

### Optional Components
- **Ethernet Cable** (for stable network connection)
- **HDMI Cable** (for initial setup if needed)
- **USB Keyboard/Mouse** (for initial setup if needed)

## Step 1: Prepare the Raspberry Pi OS

### 1.1 Flash the OS
```bash
# Download Raspberry Pi Imager from https://rpi.org/imager
# Flash Raspberry Pi OS (64-bit) to SD card
# Recommended: Use "Raspberry Pi OS (64-bit)" - full version
```

### 1.2 Enable SSH and Configure WiFi (Optional)
During imaging with Raspberry Pi Imager:
1. Click the gear icon for advanced options
2. Enable SSH with password authentication
3. Set username: `pi` and password: `your_secure_password`
4. Configure WiFi if needed (SSID and password)
5. Set locale settings

Alternatively, manually configure:
```bash
# Create SSH enable file
touch /boot/ssh

# Create WiFi configuration (if using WiFi)
cat > /boot/wpa_supplicant.conf << EOF
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YourWiFiName"
    psk="YourWiFiPassword"
}
EOF
```

## Step 2: Initial Pi 5 Setup

### 2.1 Connect and Update System
```bash
# SSH into your Pi 5 (find IP with router admin or nmap)
ssh pi@<your-pi-ip>

# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3.11 \
    python3.11-pip \
    python3.11-venv \
    python3.11-dev \
    git \
    redis-server \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    python3-pyqt5 \
    python3-h5py \
    libjasper-dev \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    python3-pyqt5.qtopengl
```

### 2.2 Configure Redis
```bash
# Configure Redis for Celery
sudo nano /etc/redis/redis.conf

# Make these changes:
# bind 127.0.0.1 ::1  # Allow local connections
# maxmemory 1gb       # Set memory limit
# maxmemory-policy allkeys-lru  # Memory eviction policy

# Enable and start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test Redis
redis-cli ping  # Should return PONG
```

## Step 3: Install AI HAT+ Drivers and Software

### 3.1 Install Hailo Software Stack
```bash
# Download and install Hailo software
curl -sSL https://install.hailo.ai/hailo-all | sudo bash

# Alternative manual installation:
# wget https://hailo.ai/downloads/hailo-rpi-image/hailo-all.deb
# sudo dpkg -i hailo-all.deb
# sudo apt-get install -f  # Fix any dependency issues

# Reboot to load kernel modules
sudo reboot
```

### 3.2 Verify AI HAT+ Installation
```bash
# After reboot, verify installation
hailo-info

# Expected output should show:
# - Hailo device detected
# - Driver version
# - Firmware version
# - Available neural networks

# Check device permissions
ls -la /dev/hailo*

# Should show hailo device files with proper permissions
```

### 3.3 Install Hailo Python Packages
```bash
# Install Hailo Python runtime
pip3 install hailort hailo-platform

# Install additional AI/ML packages
pip3 install \
    numpy \
    opencv-python \
    pillow \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
```

## Step 4: Setup AgentSystem on Pi 5

### 4.1 Clone Repository
```bash
# Clone AgentSystem repository
cd /home/pi
git clone <your-agentsystem-repo-url> AgentSystem
cd AgentSystem

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 4.2 Install Python Dependencies
```bash
# Install base requirements
pip install -r requirements.txt

# Install Pi 5 specific packages
pip install \
    psutil \
    RPi.GPIO \
    gpiozero \
    celery[redis] \
    redis \
    paramiko \
    APScheduler \
    hailort \
    hailo-platform

# Install AI/ML packages optimized for ARM
pip install \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    opencv-python-headless
```

## Step 5: Configure Environment Variables

### 5.1 Create Environment File
```bash
# Create .env file
nano /home/pi/AgentSystem/.env
```

### 5.2 Environment Configuration
```env
# AI Provider API Keys
GEMINI_API_KEY=AIzaSyClEDbUL2W4BawoYBvwbF9E8ejbPsXw3qc
OPENROUTER_API_KEY=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
OPENROUTER_FALLBACK_API_KEY=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
XAI_API_KEY=xai-hI30Sx8voDdgwC7dfRYlBwrkAGJF1JwPfr0hvbUtKMArGdaCbT9acyKFpl6qMgx4gh6YmN7IpZH5QkhE

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Pi 5 Worker Configuration
PI5_WORKER_NAME=pi5-worker-01
PI5_WORKER_QUEUES=ai_high,ai_medium,ai_low
PI5_WORKER_CONCURRENCY=4
PI5_MAX_TEMP=75.0
PI5_THERMAL_THROTTLE=70.0
PI5_MEMORY_LIMIT=6.0

# Hailo AI HAT+ Configuration
HAILO_DEVICE_ID=0
HAILO_MODEL_PATH=/home/pi/AgentSystem/models/hailo
HAILO_BATCH_SIZE=1
HAILO_TIMEOUT=30

# Database Configuration
DATABASE_URL=sqlite:///agentsystem.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/home/pi/AgentSystem/logs/pi5_worker.log

# Security Configuration
ALLOWED_HOSTS=localhost,127.0.0.1,<your-main-system-ip>
SECRET_KEY=your-secret-key-here
```

## Step 6: Configure Celery Worker Service

### 6.1 Create Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/celery-pi5-worker.service
```

### 6.2 Service Configuration
```ini
[Unit]
Description=Celery Pi 5 AI Worker
Documentation=https://docs.celeryproject.org/
After=network.target redis.service
Wants=redis.service

[Service]
Type=exec
User=pi
Group=pi
EnvironmentFile=/home/pi/AgentSystem/.env
WorkingDirectory=/home/pi/AgentSystem
ExecStart=/home/pi/AgentSystem/venv/bin/celery -A pi5_worker worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=ai_high,ai_medium,ai_low \
    --hostname=pi5-worker-01@%%h \
    --pidfile=/var/run/celery/pi5_worker.pid \
    --logfile=/home/pi/AgentSystem/logs/celery_worker.log
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=300

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/home/pi/AgentSystem/logs /home/pi/AgentSystem/data /var/run/celery

[Install]
WantedBy=multi-user.target
```

### 6.3 Create Required Directories
```bash
# Create directories for Celery
sudo mkdir -p /var/run/celery
sudo mkdir -p /home/pi/AgentSystem/logs
sudo chown pi:pi /var/run/celery
sudo chown pi:pi /home/pi/AgentSystem/logs

# Create log rotation configuration
sudo nano /etc/logrotate.d/celery-pi5
```

### 6.4 Log Rotation Configuration
```
/home/pi/AgentSystem/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 pi pi
    postrotate
        systemctl reload celery-pi5-worker
    endscript
}
```

## Step 7: Configure Monitoring and Health Checks

### 7.1 Create Health Check Script
```bash
# Create health check script
nano /home/pi/AgentSystem/scripts/health_check.sh
```

```bash
#!/bin/bash
# Pi 5 Health Check Script

LOG_FILE="/home/pi/AgentSystem/logs/health_check.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting health check..." >> $LOG_FILE

# Check CPU temperature
TEMP=$(vcgencmd measure_temp | cut -d= -f2 | cut -d\' -f1)
echo "[$DATE] CPU Temperature: ${TEMP}°C" >> $LOG_FILE

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "[$DATE] Memory Usage: ${MEM_USAGE}%" >> $LOG_FILE

# Check Redis status
if redis-cli ping > /dev/null 2>&1; then
    echo "[$DATE] Redis: OK" >> $LOG_FILE
else
    echo "[$DATE] Redis: FAILED" >> $LOG_FILE
fi

# Check Celery worker status
if systemctl is-active --quiet celery-pi5-worker; then
    echo "[$DATE] Celery Worker: OK" >> $LOG_FILE
else
    echo "[$DATE] Celery Worker: FAILED" >> $LOG_FILE
fi

# Check Hailo device
if hailo-info > /dev/null 2>&1; then
    echo "[$DATE] Hailo AI HAT+: OK" >> $LOG_FILE
else
    echo "[$DATE] Hailo AI HAT+: FAILED" >> $LOG_FILE
fi

echo "[$DATE] Health check completed." >> $LOG_FILE
```

### 7.2 Make Health Check Executable and Schedule
```bash
# Make script executable
chmod +x /home/pi/AgentSystem/scripts/health_check.sh

# Add to crontab for regular monitoring
crontab -e

# Add this line to run every 5 minutes:
*/5 * * * * /home/pi/AgentSystem/scripts/health_check.sh
```

## Step 8: Enable and Start Services

### 8.1 Enable Services
```bash
# Reload systemd daemon
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable redis-server
sudo systemctl enable celery-pi5-worker

# Start services
sudo systemctl start redis-server
sudo systemctl start celery-pi5-worker
```

### 8.2 Verify Service Status
```bash
# Check Redis status
sudo systemctl status redis-server

# Check Celery worker status
sudo systemctl status celery-pi5-worker

# View Celery worker logs
sudo journalctl -u celery-pi5-worker -f

# Check if worker is receiving tasks
celery -A pi5_worker inspect active
```

## Step 9: Test AI HAT+ Integration

### 9.1 Basic Hailo Test
```bash
# Test Hailo device detection
cd /home/pi/AgentSystem
source venv/bin/activate

python3 -c "
import hailort
print('Hailo Runtime Version:', hailort.get_version())
devices = hailort.scan_devices()
print('Available Hailo devices:', len(devices))
for i, device in enumerate(devices):
    print(f'Device {i}: {device}')
"
```

### 9.2 AI Model Test
```bash
# Test AI model loading and inference
python3 -c "
from pi5_worker import test_hailo_inference
result = test_hailo_inference()
print('Hailo inference test result:', result)
"
```

## Step 10: Network Configuration

### 10.1 Configure Firewall
```bash
# Install and configure UFW firewall
sudo apt install ufw

# Allow SSH
sudo ufw allow ssh

# Allow Redis port for Celery communication
sudo ufw allow 6379

# Allow custom ports if needed
# sudo ufw allow 5672  # RabbitMQ (if using instead of Redis)

# Enable firewall
sudo ufw enable

# Check firewall status
sudo ufw status
```

### 10.2 Get Network Information
```bash
# Get Pi 5 IP address
hostname -I

# Get network interface information
ip addr show

# Test network connectivity
ping -c 4 google.com
```

## Step 11: Update Main System Configuration

### 11.1 Update Celery Configuration on Main System
On your main Windows system, update the Celery broker URL to point to your Pi 5:

```python
# In AgentSystem/modules/code_modifier.py
# Update the Celery configuration:

CELERY_BROKER_URL = 'redis://<PI5_IP_ADDRESS>:6379/0'
CELERY_RESULT_BACKEND = 'redis://<PI5_IP_ADDRESS>:6379/0'

# Example:
# CELERY_BROKER_URL = 'redis://192.168.1.100:6379/0'
# CELERY_RESULT_BACKEND = 'redis://192.168.1.100:6379/0'
```

### 11.2 Update Environment Variables on Main System
```env
# Add to your main system's .env file:
PI5_WORKER_HOST=<PI5_IP_ADDRESS>
PI5_WORKER_PORT=6379
PI5_WORKER_ENABLED=true
```

## Step 12: Test the Complete Setup

### 12.1 Test Celery Connection
```bash
# On Pi 5, monitor worker logs
sudo journalctl -u celery-pi5-worker -f

# On main system, run integration test
python test_celery_integration.py
```

### 12.2 Test AI Code Analysis
```bash
# On main system, test AI code analysis task
python -c "
from AgentSystem.modules.code_modifier import CodeModifier
cm = CodeModifier()
result = cm.ai_enhanced_code_improvement('print(\"Hello World\")')
print('AI analysis result:', result)
"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Hailo Device Not Detected
```bash
# Check if AI HAT+ is properly connected
lsusb | grep -i hailo

# Check kernel modules
lsmod | grep hailo

# Reinstall Hailo drivers
sudo apt remove hailo-all
curl -sSL https://install.hailo.ai/hailo-all | sudo bash
sudo reboot
```

#### 2. Celery Worker Not Starting
```bash
# Check service logs
sudo journalctl -u celery-pi5-worker -n 50

# Check Redis connection
redis-cli ping

# Test Celery manually
cd /home/pi/AgentSystem
source venv/bin/activate
celery -A pi5_worker worker --loglevel=debug
```

#### 3. High Temperature Issues
```bash
# Check current temperature
vcgencmd measure_temp

# Monitor temperature continuously
watch -n 1 vcgencmd measure_temp

# Check cooling solution
# Ensure fan is working and heatsink is properly attached

# Reduce CPU frequency if needed
echo 'arm_freq=1500' | sudo tee -a /boot/config.txt
sudo reboot
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h

# Check swap usage
swapon --show

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 5. Network Connectivity Issues
```bash
# Check network interface
ip addr show

# Check routing
ip route show

# Test DNS resolution
nslookup google.com

# Check firewall rules
sudo ufw status verbose
```

### Performance Optimization

#### 1. GPU Memory Configuration
```bash
# Edit boot configuration
sudo nano /boot/config.txt

# Add these lines:
gpu_mem=128
gpu_freq=500
```

#### 2. CPU Governor Settings
```bash
# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make permanent
sudo nano /etc/rc.local
# Add before 'exit 0':
echo 'performance' > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### 3. I/O Optimization
```bash
# Optimize SD card performance
sudo nano /boot/cmdline.txt
# Add: elevator=deadline

# Mount tmpfs for temporary files
sudo nano /etc/fstab
# Add:
tmpfs /tmp tmpfs defaults,noatime,nosuid,size=100m 0 0
tmpfs /var/tmp tmpfs defaults,noatime,nosuid,size=30m 0 0
```

### Monitoring Commands

```bash
# System monitoring
htop                    # CPU and memory usage
iotop                   # I/O usage
vcgencmd measure_temp   # CPU temperature
vcgencmd get_throttled  # Throttling status

# Celery monitoring
celery -A pi5_worker inspect active     # Active tasks
celery -A pi5_worker inspect stats      # Worker statistics
celery -A pi5_worker inspect registered # Registered tasks

# Service monitoring
sudo systemctl status celery-pi5-worker
sudo systemctl status redis-server
sudo journalctl -u celery-pi5-worker -f
```

## Maintenance Tasks

### Daily
- Check system temperature
- Monitor Celery worker logs
- Verify AI HAT+ functionality

### Weekly
- Update system packages: `sudo apt update && sudo apt upgrade`
- Check disk space: `df -h`
- Review log files for errors

### Monthly
- Clean log files: `sudo logrotate -f /etc/logrotate.conf`
- Check for Hailo driver updates
- Backup configuration files

## Security Considerations

1. **Change default passwords** for pi user
2. **Enable SSH key authentication** and disable password auth
3. **Keep system updated** with security patches
4. **Configure firewall** to allow only necessary ports
5. **Use VPN** for remote access if needed
6. **Monitor access logs** regularly

## Support and Resources

- **Hailo Documentation**: https://hailo.ai/developer-zone/
- **Raspberry Pi Documentation**: https://www.raspberrypi.org/documentation/
- **Celery Documentation**: https://docs.celeryproject.org/
- **Redis Documentation**: https://redis.io/documentation

For issues specific to this setup, check the AgentSystem repository issues or create a new issue with detailed logs and system information.