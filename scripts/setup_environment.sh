#!/bin/bash
# AgentSystem Pi5 Environment Setup Script
# This script helps configure the environment variables for the Pi5 worker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INSTALL_DIR="/opt/agentsystem"
ENV_FILE="${INSTALL_DIR}/.env"
ENV_TEMPLATE="${INSTALL_DIR}/.env.template"

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

# Prompt for user input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local result
    
    echo -n "${prompt} [${default}]: "
    read result
    echo "${result:-$default}"
}

# Prompt for required input
prompt_required() {
    local prompt="$1"
    local result
    
    while [[ -z "$result" ]]; do
        echo -n "${prompt}: "
        read result
        if [[ -z "$result" ]]; then
            log_warning "This field is required. Please enter a value."
        fi
    done
    echo "$result"
}

# Prompt for sensitive input (hidden)
prompt_password() {
    local prompt="$1"
    local result
    
    while [[ -z "$result" ]]; do
        echo -n "${prompt}: "
        read -s result
        echo
        if [[ -z "$result" ]]; then
            log_warning "This field is required. Please enter a value."
        fi
    done
    echo "$result"
}

# Test Redis connection
test_redis_connection() {
    local redis_url="$1"
    log "Testing Redis connection..."
    
    # Extract host and port from Redis URL
    local host=$(echo "$redis_url" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    local port=$(echo "$redis_url" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h "$host" -p "$port" ping &> /dev/null; then
            log_success "Redis connection successful"
            return 0
        else
            log_error "Failed to connect to Redis at $host:$port"
            return 1
        fi
    else
        log_warning "redis-cli not available, skipping connection test"
        return 0
    fi
}

# Generate secure random string
generate_secure_key() {
    openssl rand -hex 32
}

# Interactive configuration
configure_environment() {
    log "Starting interactive environment configuration..."
    echo
    
    # Copy template if .env doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        if [[ -f "$ENV_TEMPLATE" ]]; then
            cp "$ENV_TEMPLATE" "$ENV_FILE"
            log "Created .env file from template"
        else
            log_error "Template file not found: $ENV_TEMPLATE"
            exit 1
        fi
    fi
    
    echo "========================================"
    echo "  AgentSystem Pi5 Environment Setup"
    echo "========================================"
    echo
    
    # Required settings
    log "📡 Celery/Redis Configuration (Required)"
    echo "Enter the details for your main AgentSystem server:"
    echo
    
    MAIN_SERVER_IP=$(prompt_required "Main server IP address")
    REDIS_PORT=$(prompt_with_default "Redis port" "6379")
    REDIS_DB=$(prompt_with_default "Redis database number" "0")
    
    CELERY_BROKER_URL="redis://${MAIN_SERVER_IP}:${REDIS_PORT}/${REDIS_DB}"
    CELERY_RESULT_BACKEND="$CELERY_BROKER_URL"
    
    # Test Redis connection
    if ! test_redis_connection "$CELERY_BROKER_URL"; then
        log_warning "Redis connection failed. Please verify the server is running and accessible."
        if ! prompt_with_default "Continue anyway?" "n" | grep -q "^[Yy]"; then
            exit 1
        fi
    fi
    
    echo
    log "🤖 AI Provider Configuration (Required)"
    echo "Choose your primary AI provider:"
    echo "1) OpenAI"
    echo "2) Google Gemini"
    echo "3) Both"
    echo
    
    AI_PROVIDER=$(prompt_with_default "Choice (1-3)" "1")
    
    case $AI_PROVIDER in
        1|3)
            echo
            OPENAI_API_KEY=$(prompt_password "OpenAI API Key")
            OPENAI_ORG_ID=$(prompt_with_default "OpenAI Organization ID (optional)" "")
            ;;
    esac
    
    case $AI_PROVIDER in
        2|3)
            echo
            GEMINI_API_KEY=$(prompt_password "Google Gemini API Key")
            ;;
    esac
    
    echo
    log "🔧 Pi5 Worker Configuration"
    WORKER_NAME=$(prompt_with_default "Worker name" "pi5_worker_$(hostname)")
    WORKER_CONCURRENCY=$(prompt_with_default "Worker concurrency (CPU cores to use)" "2")
    ENABLE_AI_HAT=$(prompt_with_default "Enable AI HAT+ integration" "true")
    
    echo
    log "🔐 Security Configuration"
    SECRET_KEY=$(generate_secure_key)
    WORKER_AUTH_TOKEN=$(generate_secure_key)
    
    echo
    log "📊 Resource Limits"
    MAX_MEMORY_USAGE=$(prompt_with_default "Maximum memory usage" "6GB")
    MAX_CPU_USAGE=$(prompt_with_default "Maximum CPU usage %" "80")
    
    echo
    log "📝 Logging Configuration"
    LOG_LEVEL=$(prompt_with_default "Log level (DEBUG/INFO/WARNING/ERROR)" "INFO")
    
    # Write configuration to .env file
    log "Writing configuration to $ENV_FILE..."
    
    # Update the .env file with collected values
    update_env_var "CELERY_BROKER_URL" "$CELERY_BROKER_URL"
    update_env_var "CELERY_RESULT_BACKEND" "$CELERY_RESULT_BACKEND"
    update_env_var "MAIN_SERVER_HOST" "$MAIN_SERVER_IP"
    update_env_var "CELERY_WORKER_NAME" "$WORKER_NAME"
    update_env_var "CELERY_WORKER_CONCURRENCY" "$WORKER_CONCURRENCY"
    update_env_var "SECRET_KEY" "$SECRET_KEY"
    update_env_var "WORKER_AUTH_TOKEN" "$WORKER_AUTH_TOKEN"
    update_env_var "MAX_MEMORY_USAGE" "$MAX_MEMORY_USAGE"
    update_env_var "MAX_CPU_USAGE" "$MAX_CPU_USAGE"
    update_env_var "LOG_LEVEL" "$LOG_LEVEL"
    update_env_var "ENABLE_AI_HAT" "$ENABLE_AI_HAT"
    
    if [[ -n "$OPENAI_API_KEY" ]]; then
        update_env_var "OPENAI_API_KEY" "$OPENAI_API_KEY"
        if [[ -n "$OPENAI_ORG_ID" ]]; then
            update_env_var "OPENAI_ORG_ID" "$OPENAI_ORG_ID"
        fi
    fi
    
    if [[ -n "$GEMINI_API_KEY" ]]; then
        update_env_var "GEMINI_API_KEY" "$GEMINI_API_KEY"
    fi
    
    # Set secure permissions
    chmod 600 "$ENV_FILE"
    chown agentsystem:agentsystem "$ENV_FILE"
    
    log_success "Environment configuration completed!"
}

# Update environment variable in .env file
update_env_var() {
    local key="$1"
    local value="$2"
    
    if grep -q "^${key}=" "$ENV_FILE"; then
        # Update existing variable
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        # Add new variable
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# Validate configuration
validate_configuration() {
    log "Validating configuration..."
    
    # Check required variables
    local required_vars=(
        "CELERY_BROKER_URL"
        "CELERY_RESULT_BACKEND" 
        "MAIN_SERVER_HOST"
        "SECRET_KEY"
        "WORKER_AUTH_TOKEN"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$ENV_FILE" || grep -q "^${var}=$" "$ENV_FILE"; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required configuration variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        return 1
    fi
    
    # Check for at least one AI provider
    if ! grep -q "^OPENAI_API_KEY=.\+" "$ENV_FILE" && ! grep -q "^GEMINI_API_KEY=.\+" "$ENV_FILE"; then
        log_error "At least one AI provider API key must be configured"
        return 1
    fi
    
    log_success "Configuration validation passed"
    return 0
}

# Show configuration summary
show_summary() {
    log "Configuration Summary:"
    echo "========================"
    
    # Extract key values from .env file
    local broker_url=$(grep "^CELERY_BROKER_URL=" "$ENV_FILE" | cut -d'=' -f2-)
    local worker_name=$(grep "^CELERY_WORKER_NAME=" "$ENV_FILE" | cut -d'=' -f2-)
    local concurrency=$(grep "^CELERY_WORKER_CONCURRENCY=" "$ENV_FILE" | cut -d'=' -f2-)
    local ai_hat=$(grep "^ENABLE_AI_HAT=" "$ENV_FILE" | cut -d'=' -f2-)
    
    echo "Broker URL: $broker_url"
    echo "Worker Name: $worker_name"
    echo "Concurrency: $concurrency"
    echo "AI HAT+ Enabled: $ai_hat"
    
    if grep -q "^OPENAI_API_KEY=.\+" "$ENV_FILE"; then
        echo "OpenAI: ✅ Configured"
    fi
    
    if grep -q "^GEMINI_API_KEY=.\+" "$ENV_FILE"; then
        echo "Gemini: ✅ Configured"
    fi
    
    echo "========================"
}

# Test configuration
test_configuration() {
    log "Testing configuration..."
    
    # Source the environment file
    set -a
    source "$ENV_FILE"
    set +a
    
    # Test Python import
    if ! sudo -u agentsystem bash -c "cd $INSTALL_DIR && source venv/bin/activate && python -c 'from AgentSystem.pi5_worker import app; print(\"Worker import successful\")'"; then
        log_error "Failed to import worker module"
        return 1
    fi
    
    log_success "Configuration test passed"
}

# Main function
main() {
    log "Starting AgentSystem Pi5 environment setup..."
    
    check_root
    
    if [[ ! -d "$INSTALL_DIR" ]]; then
        log_error "AgentSystem installation directory not found: $INSTALL_DIR"
        log "Please run the installation script first."
        exit 1
    fi
    
    configure_environment
    
    if validate_configuration; then
        show_summary
        echo
        
        if prompt_with_default "Test the configuration now?" "y" | grep -q "^[Yy]"; then
            test_configuration
        fi
        
        echo
        log_success "Environment setup completed successfully!"
        echo
        log "You can now start the AgentSystem Pi5 worker service:"
        log "  sudo systemctl start agentsystem-pi5-worker"
        echo
        log "To check the status:"
        log "  sudo systemctl status agentsystem-pi5-worker"
        echo
        log "To view logs:"
        log "  sudo journalctl -u agentsystem-pi5-worker -f"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo
        echo "Interactive environment configuration for AgentSystem Pi5 worker"
        echo
        echo "This script will guide you through configuring the required"
        echo "environment variables for the Pi5 worker to connect to your"
        echo "main AgentSystem server."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac