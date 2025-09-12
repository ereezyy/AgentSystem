# Multi-stage Docker build for AgentSystem
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libsqlite3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash agentsystem

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY AgentSystem/requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-prod.txt

# Copy application code
COPY AgentSystem/ ./AgentSystem/
COPY scripts/ ./scripts/

# Set ownership and permissions
RUN chown -R agentsystem:agentsystem /app
USER agentsystem

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/backups

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "AgentSystem.main", "--mode", "server", "--host", "0.0.0.0", "--port", "8000"]
