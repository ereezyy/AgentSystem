#!/bin/bash
# AgentSystem Pi5 Worker Installation Script
# This script automates the installation of AgentSystem as a Celery worker on Raspberry Pi 5

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/agentsystem"
SERVICE_USER="agentsystem"
SERVICE_GROUP="agentsystem"
VENV_DIR="${INSTALL_DIR}/venv"
LOG_DIR="${INSTALL_DIR}/logs"
DATA_DIR="${INSTALL_DIR}/data"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check if running on Raspberry Pi 5
check_pi5() {
    log "Checking if running on Raspberry Pi 5..."
    
    if ! grep -q "Raspberry Pi 5" /proc/cpuinfo 2>/dev/null; then
        log_warning "This doesn't appear to be a Raspberry Pi 5"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "Raspberry Pi 5 detected"
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    apt-get update
    apt-get upgrade -y
    log_success "System packages updated"
}

# Install required system packages
install_system_deps() {
    log "Installing system dependencies..."
    
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        curl \
        wget \
        htop \
        redis-tools \
        sqlite3 \
        libffi-dev \
        libssl-dev \
        libjpeg-dev \
        libopenblas-dev \
        libatlas-base-dev \
        libhdf5-dev \
        pkg-config \
        cmake
    
    # Pi5 specific packages
    apt-get install -y \
        python3-opencv \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        libraspberrypi-bin \
        raspi-config
    
    log_success "System dependencies installed"
}

# Create system user and group
create_user() {
    log "Creating system user and group..."
    
    if ! getent group ${SERVICE_GROUP} > /dev/null 2>&1; then
        groupadd --system ${SERVICE_GROUP}
        log_success "Created group: ${SERVICE_GROUP}"
    else
        log "Group ${SERVICE_GROUP} already exists"
    fi
    
    if ! getent passwd ${SERVICE_USER} > /dev/null 2>&1; then
        useradd --system \
            --gid ${SERVICE_GROUP} \
            --home ${INSTALL_DIR} \
            --shell /bin/bash \
            --comment "AgentSystem Pi5 Worker" \
            ${SERVICE_USER}
        log_success "Created user: ${SERVICE_USER}"
    else
        log "User ${SERVICE_USER} already exists"
    fi
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    
    mkdir -p ${INSTALL_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ${DATA_DIR}
    mkdir -p ${DATA_DIR}/models
    mkdir -p ${DATA_DIR}/frames
    mkdir -p ${DATA_DIR}/memory
    mkdir -p /tmp/agentsystem
    
    log_success "Directory structure created"
}

# Copy AgentSystem files
copy_files() {
    log "Copying AgentSystem files..."
    
    # Get the script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PACKAGE_DIR="$(dirname "${SCRIPT_DIR}")"
    
    # Copy AgentSystem directory
    if [[ -d "${PACKAGE_DIR}/AgentSystem" ]]; then
        cp -r "${PACKAGE_DIR}/AgentSystem/"* ${INSTALL_DIR}/
        log_success "AgentSystem files copied"
    else
        log_error "AgentSystem directory not found in package"
        exit 1
    fi
    
    # Copy Pi5 specific files
    cp "${PACKAGE_DIR}/.env.pi5" ${INSTALL_DIR}/.env.template
    cp "${PACKAGE_DIR}/pi5_requirements.txt" ${INSTALL_DIR}/
    cp "${PACKAGE_DIR}/pi5_health_check.py" ${INSTALL_DIR}/
    cp "${PACKAGE_DIR}/agentsystem-pi5-worker.service" /etc/systemd/system/
    
    log_success "Pi5 specific files copied"
}

# Set up Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv ${VENV_DIR}
    
    # Activate virtual environment
    source ${VENV_DIR}/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install AgentSystem requirements
    pip install -r ${INSTALL_DIR}/pi5_requirements.txt --no-cache-dir
    
    # Install AgentSystem package in development mode
    cd ${INSTALL_DIR}
    pip install -e .
    
    log_success "Python virtual environment set up"
}

# Set file permissions
set_permissions() {
    log "Setting file permissions..."
    
    # Set ownership
    chown -R ${SERVICE_USER}:${SERVICE_GROUP} ${INSTALL_DIR}
    chown -R ${SERVICE_USER}:${SERVICE_GROUP} /tmp/agentsystem
    
    # Set directory permissions
    chmod -R 755 ${INSTALL_DIR}
    chmod -R 750 ${LOG_DIR}
    chmod -R 750 ${DATA_DIR}
    
    # Set script permissions
    chmod +x ${INSTALL_DIR}/pi5_health_check.py
    
    # Secure environment file (will be created later)
    if [[ -f "${INSTALL_DIR}/.env" ]]; then
        chmod 600 ${INSTALL_DIR}/.env
    fi
    
    log_success "File permissions set"
}

# Configure systemd service
configure_service() {
    log "Configuring systemd service..."
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service (but don't start yet)
    systemctl enable agentsystem-pi5-worker
    
    log_success "Systemd service configured"
}

# Optimize Pi5 settings
optimize_pi5() {
    log "Applying Pi5 optimizations..."
    
    # GPU memory split
    if ! grep -q "gpu_mem=128" /boot/config.txt; then
        echo "gpu_mem=128" >> /boot/config.txt
        log "Added gpu_mem=128 to /boot/config.txt"
    fi
    
    # ARM boost
    if ! grep -q "arm_boost=1" /boot/config.txt; then
        echo "arm_boost=1" >> /boot/config.txt
        log "Added arm_boost=1 to /boot/config.txt"
    fi
    
    # Set CPU governor to performance
    echo 'GOVERNOR="performance"' > /etc/default/cpufrequtils
    
    log_success "Pi5 optimizations applied"
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    # Install ufw if not present
    if ! command -v ufw &> /dev/null; then
        apt-get install -y ufw
    fi
    
    # Basic firewall rules
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 8001  # AgentSystem Pi5 worker port
    
    # Enable firewall
    ufw --force enable
    
    log_success "Firewall configured"
}

# Create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > ${INSTALL_DIR}/start_worker.sh << 'EOF'
#!/bin/bash
# AgentSystem Pi5 Worker Startup Script

INSTALL_DIR="/opt/agentsystem"
VENV_DIR="${INSTALL_DIR}/venv"

cd ${INSTALL_DIR}
source ${VENV_DIR}/bin/activate

# Load environment variables
if [[ -f "${INSTALL_DIR}/.env" ]]; then
    export $(grep -v '^#' ${INSTALL_DIR}/.env | xargs)
fi

# Start Celery worker
exec celery -A AgentSystem.pi5_worker worker \
    --loglevel=info \
    --hostname=pi5_worker@$(hostname) \
    --concurrency=${CELERY_WORKER_CONCURRENCY:-2} \
    --prefetch-multiplier=${CELERY_WORKER_PREFETCH_MULTIPLIER:-1} \
    --max-tasks-per-child=1000 \
    --time-limit=300 \
    --soft-time-limit=240
EOF
    
    chmod +x ${INSTALL_DIR}/start_worker.sh
    chown ${SERVICE_USER}:${SERVICE_GROUP} ${INSTALL_DIR}/start_worker.sh
    
    log_success "Startup script created"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/agentsystem-pi5 << EOF
${LOG_DIR}/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ${SERVICE_USER} ${SERVICE_GROUP}
    postrotate
        systemctl reload agentsystem-pi5-worker > /dev/null 2>&1 || true
    endscript
}
EOF
    
    log_success "Log rotation configured"
}

# Main installation function
main() {
    log "Starting AgentSystem Pi5 Worker installation..."
    
    check_root
    check_pi5
    update_system
    install_system_deps
    create_user
    create_directories
    copy_files
    setup_venv
    set_permissions
    configure_service
    optimize_pi5
    configure_firewall
    create_startup_script
    setup_log_rotation
    
    log_success "Installation completed successfully!"
    echo
    log "Next steps:"
    log "1. Copy .env.template to .env and configure your settings:"
    log "   sudo cp ${INSTALL_DIR}/.env.template ${INSTALL_DIR}/.env"
    log "   sudo nano ${INSTALL_DIR}/.env"
    echo
    log "2. Start the AgentSystem Pi5 worker service:"
    log "   sudo systemctl start agentsystem-pi5-worker"
    echo
    log "3. Check the service status:"
    log "   sudo systemctl status agentsystem-pi5-worker"
    echo
    log "4. Monitor the health:"
    log "   python3 ${INSTALL_DIR}/pi5_health_check.py --verbose"
    echo
    log_warning "A reboot is recommended to apply all Pi5 optimizations:"
    log "   sudo reboot"
}

# Run main function
main "$@"