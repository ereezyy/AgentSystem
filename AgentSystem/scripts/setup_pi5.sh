#!/bin/bash
# Raspberry Pi 5 Setup Script for AgentSystem
# Run this script on your Pi 5 to automate the setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    error "This script must be run on a Raspberry Pi"
    exit 1
fi

# Check if running as pi user
if [ "$USER" != "pi" ]; then
    error "This script must be run as the 'pi' user"
    exit 1
fi

log "Starting Raspberry Pi 5 setup for AgentSystem..."

# Update system
log "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
log "Installing essential packages..."
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
    python3-pyqt5.qtopengl \
    ufw \
    htop \
    iotop

# Configure Redis
log "Configuring Redis..."
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup
sudo sed -i 's/^# maxmemory <bytes>/maxmemory 1gb/' /etc/redis/redis.conf
sudo sed -i 's/^# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf

# Enable and start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test Redis
if redis-cli ping | grep -q "PONG"; then
    log "Redis is working correctly"
else
    error "Redis is not responding"
    exit 1
fi

# Install Hailo software stack
log "Installing Hailo AI HAT+ software..."
if ! command -v hailo-info &> /dev/null; then
    curl -sSL https://install.hailo.ai/hailo-all | sudo bash
    log "Hailo software installed. A reboot will be required."
    REBOOT_REQUIRED=true
else
    log "Hailo software already installed"
fi

# Create AgentSystem directory structure
log "Setting up AgentSystem directory structure..."
cd /home/pi

# Create directories
mkdir -p AgentSystem/{logs,data,scripts,models/hailo}
cd AgentSystem

# Create Python virtual environment
log "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python packages
log "Installing Python packages..."
pip install \
    celery[redis] \
    redis \
    psutil \
    RPi.GPIO \
    gpiozero \
    paramiko \
    APScheduler \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    opencv-python-headless \
    requests \
    beautifulsoup4 \
    lxml \
    sqlalchemy \
    python-dotenv

# Install Hailo Python packages (if available)
if command -v hailo-info &> /dev/null; then
    log "Installing Hailo Python packages..."
    pip install hailort hailo-platform || warn "Hailo Python packages not available yet"
fi

# Create environment file template
log "Creating environment file template..."
cat > .env.template << 'EOF'
# AI Provider API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_FALLBACK_API_KEY=your_fallback_api_key_here
XAI_API_KEY=your_xai_api_key_here

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
ALLOWED_HOSTS=localhost,127.0.0.1
SECRET_KEY=change-this-secret-key
EOF

# Create health check script
log "Creating health check script..."
cat > scripts/health_check.sh << 'EOF'
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
if command -v hailo-info &> /dev/null && hailo-info > /dev/null 2>&1; then
    echo "[$DATE] Hailo AI HAT+: OK" >> $LOG_FILE
else
    echo "[$DATE] Hailo AI HAT+: FAILED" >> $LOG_FILE
fi

echo "[$DATE] Health check completed." >> $LOG_FILE
EOF

chmod +x scripts/health_check.sh

# Create systemd service file
log "Creating Celery systemd service..."
sudo tee /etc/systemd/system/celery-pi5-worker.service > /dev/null << 'EOF'
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
EOF

# Create required directories
sudo mkdir -p /var/run/celery
sudo chown pi:pi /var/run/celery

# Create log rotation configuration
log "Setting up log rotation..."
sudo tee /etc/logrotate.d/celery-pi5 > /dev/null << 'EOF'
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
EOF

# Configure firewall
log "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 6379

# Configure performance settings
log "Configuring performance settings..."
# Add GPU memory split
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
fi

# Add CPU frequency settings
if ! grep -q "arm_freq=" /boot/config.txt; then
    echo "arm_freq=2400" | sudo tee -a /boot/config.txt
fi

# Set up cron job for health checks
log "Setting up health check cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/pi/AgentSystem/scripts/health_check.sh") | crontab -

# Get network information
IP_ADDRESS=$(hostname -I | awk '{print $1}')

log "Setup completed successfully!"
log "Pi 5 IP Address: $IP_ADDRESS"

echo ""
echo -e "${BLUE}=== NEXT STEPS ===${NC}"
echo "1. Copy your AgentSystem code to /home/pi/AgentSystem/"
echo "2. Edit /home/pi/AgentSystem/.env with your API keys (use .env.template as reference)"
echo "3. Enable the Celery worker service:"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable celery-pi5-worker"
echo "4. Update your main system's Celery configuration to use: redis://$IP_ADDRESS:6379/0"

if [ "$REBOOT_REQUIRED" = true ]; then
    echo ""
    warn "A reboot is required to complete the Hailo driver installation."
    echo "Run 'sudo reboot' when ready."
fi

echo ""
echo -e "${GREEN}Setup script completed!${NC}"