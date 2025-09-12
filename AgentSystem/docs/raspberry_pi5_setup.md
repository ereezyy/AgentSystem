# Raspberry Pi 5 with AI HAT+ Setup Guide

This guide covers setting up the Raspberry Pi 5 with AI HAT+ (26 TOPS) for the Continuous Learning Module's ethical hacking and social engineering capabilities.

## Hardware Requirements

### Raspberry Pi 5 Specifications
- **Model**: Raspberry Pi 5 (8GB RAM recommended)
- **Processor**: Broadcom BCM2712, Quad-core ARM Cortex-A76 @ 2.4 GHz
- **Storage**: 128GB microSD (Class U3, A2) + 500GB USB SSD
- **Network**: Gigabit Ethernet (required for low-latency Redis communication)
- **Power**: Official 5V, 3A USB-C power supply
- **Cooling**: Raspberry Pi Active Cooler (mandatory)

### AI HAT+ (26 TOPS)
- **Model**: Hailo-8 Neural Network Accelerator
- **Performance**: 26 TOPS (INT8 precision)
- **Interface**: PCIe Gen 3
- **Cooling**: Included heatsink + Pi 5 Active Cooler

## Software Setup

### 1. Install Raspberry Pi OS

```bash
# Download Raspberry Pi OS (64-bit) Lite
# Flash to microSD using Raspberry Pi Imager
# Enable SSH during setup
```

### 2. Initial System Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3-pip \
    python3-venv \
    redis-tools \
    git \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev

# Install monitoring tools
sudo apt install -y htop iotop nmon
```

### 3. Install AI HAT+ Software

```bash
# Add Hailo repository
curl -s https://hailo.ai/raspberrypi/KEY.gpg | sudo apt-key add -
echo "deb https://hailo.ai/raspberrypi/ $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/hailo.list

# Install Hailo software (~900MB)
sudo apt update
sudo apt install -y hailo-all

# Verify installation
hailortcli fw-control identify
```

### 4. Configure PCIe for AI HAT+

```bash
# Edit config.txt
sudo nano /boot/firmware/config.txt

# Add these lines:
dtparam=pciex1
dtparam=pcie0_gen=3
dtoverlay=pcie-32bit-dma

# Reboot
sudo reboot
```

### 5. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv /home/pi/agentsystem_env
source /home/pi/agentsystem_env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install \
    celery[redis] \
    psutil \
    paramiko \
    psycopg2-binary \
    python-nmap \
    zlib \
    numpy

# Install Hailo Python SDK
pip install hailo_sdk
```

### 6. Create Directory Structure

```bash
# Create required directories
sudo mkdir -p /models
sudo chown pi:pi /models

mkdir -p ~/AgentSystem/logs
mkdir -p ~/AgentSystem/data
```

### 7. Deploy Pi 5 Worker

```bash
# Copy pi5_worker.py to Pi 5
scp AgentSystem/pi5_worker.py pi@<PI5_IP>:~/AgentSystem/

# The worker is now configured via an environment variable.
# See the Celery Worker Service configuration section below for details.
```

### 8. Configure Celery Worker Service

```bash
# Create systemd service
sudo nano /etc/systemd/system/celery-worker.service
```

Add the following content:

```ini
[Unit]
Description=Celery Worker for AI HAT+
After=network.target

[Service]
Type=forking
User=pi
Group=pi
WorkingDirectory=/home/pi/AgentSystem
# Set the path to the virtual environment
Environment="PATH=/home/pi/agentsystem_env/bin"
# Set the IP of the primary machine (your Windows PC) where Redis is running.
# IMPORTANT: Replace 192.168.1.100 with your Windows machine's actual local IP address.
Environment="PRIMARY_CPU_IP=192.168.1.100"
ExecStart=/home/pi/agentsystem_env/bin/celery -A pi5_worker worker \
    --concurrency=2 \
    --loglevel=info \
    --logfile=/home/pi/AgentSystem/logs/celery.log \
    --detach

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable celery-worker
sudo systemctl start celery-worker
```

## Network Configuration

### 1. Static IP Configuration

```bash
# Edit dhcpcd.conf
sudo nano /etc/dhcpcd.conf

# Add static IP configuration:
interface eth0
static ip_address=192.168.1.150/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

### 2. Firewall Configuration

```bash
# Install and configure UFW
sudo apt install -y ufw

# Allow SSH and Celery communication
sudo ufw allow 22/tcp
sudo ufw allow from 192.168.1.100 to any port 6379
sudo ufw enable
```

## Performance Optimization

### 1. CPU Overclocking (Optional)

```bash
# Edit config.txt
sudo nano /boot/firmware/config.txt

# Add overclocking settings:
over_voltage=4
arm_freq=3000
gpu_freq=800

# Monitor temperature after overclocking
vcgencmd measure_temp
```

### 2. Memory Split Configuration

```bash
# Allocate more memory to CPU (less to GPU)
sudo raspi-config
# Advanced Options > Memory Split > 16
```

### 3. Swap Configuration

```bash
# Increase swap size for large models
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Model Deployment

### 1. Compile Models on Primary CPU

```bash
# On primary CPU (x86)
# Install Hailo Dataflow Compiler
wget https://hailo.ai/download/hailo_dataflow_compiler.tar.gz
tar -xzf hailo_dataflow_compiler.tar.gz
cd hailo_dataflow_compiler
./install.sh

# Compile YOLO model for vulnerability detection
hailo_compiler \
    --input yolov5s.onnx \
    --output /models/compiled/yolo_vuln.hef \
    --arch hailo8 \
    --optimization-level 3

# Compile NLP model for phishing
hailo_compiler \
    --input minilm.onnx \
    --output /models/compiled/nlp_phishing.hef \
    --arch hailo8 \
    --quantization int8
```

### 2. Transfer Models to Pi 5

```bash
# From primary CPU
scp /models/compiled/*.hef pi@192.168.1.150:/models/
```

## Monitoring and Maintenance

### 1. Temperature Monitoring Script

```bash
# Create monitoring script
nano ~/monitor_temp.sh
```

```bash
#!/bin/bash
while true; do
    temp=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
    echo "$(date): CPU Temp: $temp°C"
    
    if (( $(echo "$temp > 80" | bc -l) )); then
        echo "WARNING: High temperature detected!"
        # Reduce Celery concurrency
        celery -A pi5_worker control pool_shrink 1
    fi
    
    sleep 60
done
```

### 2. Resource Monitoring

```bash
# Monitor system resources
htop  # CPU and RAM usage
iotop  # Disk I/O
vcgencmd get_throttled  # Check for throttling
```

### 3. Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/celery
```

```
/home/pi/AgentSystem/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 pi pi
}
```

## Security Hardening

### 1. SSH Key Authentication

```bash
# Generate SSH key on primary CPU
ssh-keygen -t ed25519 -C "agentsystem@primary"

# Copy to Pi 5
ssh-copy-id pi@192.168.1.150

# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh
```

### 2. Fail2ban Configuration

```bash
# Install fail2ban
sudo apt install -y fail2ban

# Configure for SSH protection
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
# Enable SSH jail
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Troubleshooting

### AI HAT+ Not Detected

```bash
# Check PCIe devices
lspci -v

# Check kernel messages
dmesg | grep hailo

# Verify firmware
sudo rpi-eeprom-update -a
```

### High Temperature Issues

1. Ensure Active Cooler is properly installed
2. Check thermal paste application
3. Reduce overclocking
4. Add additional cooling (case fan)

### Memory Issues

```bash
# Check memory usage
free -h

# Clear cache
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Reduce Celery concurrency
celery -A pi5_worker control pool_shrink 1
```

### Network Latency

1. Use wired Ethernet (not Wi-Fi)
2. Check network cable quality (Cat6 recommended)
3. Ensure direct connection or high-quality switch
4. Monitor with: `ping -c 100 <PRIMARY_CPU_IP>`

## Performance Benchmarks

Expected performance with proper setup:

- **AI Inference**: 
  - YOLO vulnerability detection: ~50ms per scan result
  - NLP phishing generation: ~150ms per email
- **Network Latency**: <1ms to primary CPU
- **Task Throughput**: 10-20 AI tasks/second
- **Temperature**: 45-65°C under load
- **Power Consumption**: 15-20W total

## Maintenance Schedule

- **Daily**: Check logs and temperature
- **Weekly**: Update system packages, check disk space
- **Monthly**: Clean dust filters, verify AI model accuracy
- **Quarterly**: Full system backup, security audit

## Integration Testing

After setup, run integration tests:

```bash
# On primary CPU
python AgentSystem/examples/ethical_hacking_example.py

# Monitor Pi 5 logs
tail -f ~/AgentSystem/logs/celery.log
```

Verify:
- Tasks are received and processed
- AI models load successfully
- Results return to primary CPU
- No thermal throttling occurs
