# 🚀 AgentSystem Backend Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying the full Python AgentSystem backend to various server environments with enterprise-grade scalability, security, and monitoring.

## 🛠 Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for local development)
- Git access to the repository
- Basic server administration knowledge

## 🎯 Deployment Options

### **Option 1: Docker Compose (Recommended for Production)**

### **Option 2: Railway Platform (Easiest)**

### **Option 3: AWS ECS/Fargate (Enterprise)**

### **Option 4: DigitalOcean App Platform**

### **Option 5: Google Cloud Run**

### **Option 6: Traditional VPS/Server**

---

## 🐳 Option 1: Docker Compose Deployment

### **Quick Start (5 minutes)**

```bash
# 1. Clone and navigate to deployment directory
cd pi5_deployment_package

# 2. Create environment file
cp .env.example .env.production

# 3. Configure essential variables
nano .env.production
# Set: DB_PASSWORD, REDIS_PASSWORD, FLOWER_PASSWORD, GRAFANA_PASSWORD

# 4. Deploy the stack
docker-compose up -d

# 5. Check deployment status
docker-compose ps
docker-compose logs agentsystem
```

### **Services Deployed**

- **AgentSystem App**: Main application server (Port 8000)
- **PostgreSQL**: Primary database (Port 5432)
- **Redis**: Cache and message broker (Port 6379)
- **Celery Worker**: Background task processing
- **Celery Beat**: Task scheduler
- **Flower**: Task monitoring UI (Port 5555)
- **Nginx**: Reverse proxy (Ports 80/443)
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Monitoring dashboard (Port 3001)

### **Access Points**

- **Main App**: <http://localhost:8000>
- **API Health**: <http://localhost:8000/health>
- **Task Monitor**: <http://localhost:5555>
- **Metrics**: <http://localhost:9090>
- **Dashboard**: <http://localhost:3001>

---

## 🚂 Option 2: Railway Platform Deployment

### **Step 1: Prepare Railway Configuration**

Create `railway.toml`:

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[environments.production]
variables = { NODE_ENV = "production" }
```

### **Step 2: Deploy to Railway**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and initialize
railway login
railway init

# 3. Add services
railway add --database postgresql
railway add --database redis

# 4. Set environment variables
railway variables set NODE_ENV=production
railway variables set LOG_LEVEL=info

# 5. Deploy
railway up
```

### **Railway Environment Variables**

```bash
DATABASE_URL=postgresql://... # Auto-generated
REDIS_URL=redis://... # Auto-generated
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

---

## ☁️ Option 3: AWS ECS/Fargate Deployment

### **Step 1: Create ECS Task Definition**

```bash
# Build and push to ECR
aws ecr create-repository --repository-name agentsystem
docker build -t agentsystem .
docker tag agentsystem:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/agentsystem:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/agentsystem:latest
```

### **Step 2: Deploy Infrastructure**

Use AWS CloudFormation or Terraform:

```yaml
# cloudformation-template.yml
Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: agentsystem-cluster

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: agentsystem-task
      Cpu: 1024
      Memory: 2048
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
```

---

## 🌊 Option 4: DigitalOcean App Platform

### **Deploy with App Spec**

Create `.do/app.yaml`:

```yaml
name: agentsystem
services:
- name: web
  source_dir: /
  github:
    repo: your-username/agentsystem
    branch: main
  dockerfile_path: pi5_deployment_package/Dockerfile
  http_port: 8000
  instance_count: 1
  instance_size_slug: basic-xxs

databases:
- name: postgres
  engine: PG
  size: basic-xs

- name: redis
  engine: REDIS
  size: basic-xs
```

Deploy:

```bash
doctl apps create --spec .do/app.yaml
```

---

## 🏗 Option 5: Google Cloud Run

### **Deploy to Cloud Run**

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/agentsystem

# 2. Deploy to Cloud Run
gcloud run deploy agentsystem \
  --image gcr.io/PROJECT_ID/agentsystem \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10
```

---

## 🖥 Option 6: Traditional VPS/Server

### **Ubuntu 22.04 Setup**

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Deploy application
git clone YOUR_REPO
cd agentsystem/pi5_deployment_package
cp .env.example .env.production
# Edit .env.production with your settings
docker-compose up -d
```

---

## 🔧 Configuration

### **Essential Environment Variables**

```bash
# Application
NODE_ENV=production
LOG_LEVEL=info
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_TIMEOUT=30000

# Redis
REDIS_URL=redis://host:6379/0
REDIS_PASSWORD=secure_redis_password

# Security
SESSION_SECRET=your-super-secure-session-secret-32-chars
ENCRYPTION_KEY=your-32-character-encryption-key-here
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# AI Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Monitoring
SENTRY_DSN=https://...
ENABLE_METRICS=true
PROMETHEUS_ENDPOINT=/metrics

# Celery
CELERY_BROKER_URL=redis://host:6379/1
CELERY_RESULT_BACKEND=redis://host:6379/2
CELERY_WORKER_CONCURRENCY=4
```

### **Production Security Settings**

```bash
# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/fullchain.pem
SSL_KEY_PATH=/etc/ssl/private/privkey.pem

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_MS=3600000

# CORS
CORS_ORIGINS=https://yourdomain.com
CORS_CREDENTIALS=true

# Authentication
JWT_SECRET=your-jwt-secret-key
JWT_EXPIRY=24h
BCRYPT_ROUNDS=12
```

---

## 📊 Monitoring and Health Checks

### **Health Check Endpoints**

- **Application Health**: `GET /health`
- **Database Health**: `GET /health/db`
- **Redis Health**: `GET /health/redis`
- **Detailed Status**: `GET /status`

### **Prometheus Metrics**

```bash
# View metrics
curl http://localhost:8000/metrics

# Key metrics to monitor:
# - http_requests_total
# - http_request_duration_seconds
# - agentsystem_active_users
# - agentsystem_task_queue_size
# - agentsystem_memory_usage
```

### **Grafana Dashboard Setup**

1. Access Grafana at <http://localhost:3001>
2. Login with admin/admin_password
3. Import dashboard from monitoring/agentsystem-dashboard.json
4. Configure alerts for critical metrics

---

## 🔍 Troubleshooting

### **Common Issues**

**Database Connection Failed**

```bash
# Check database status
docker-compose logs postgres

# Test connection
docker-compose exec agentsystem python -c "from AgentSystem.utils.db import test_connection; test_connection()"
```

**High Memory Usage**

```bash
# Check memory usage
docker stats

# Restart services
docker-compose restart agentsystem celery-worker
```

**Task Queue Backup**

```bash
# Check Celery status
docker-compose exec celery-worker celery -A AgentSystem.celery_app inspect active

# Purge queue if needed
docker-compose exec celery-worker celery -A AgentSystem.celery_app purge
```

---

## 🔄 Maintenance

### **Regular Tasks**

**Daily:**

```bash
# Check logs
docker-compose logs --tail=100 agentsystem

# Monitor disk usage
df -h
docker system df
```

**Weekly:**

```bash
# Update containers
docker-compose pull
docker-compose up -d

# Clean up old images
docker image prune -f
```

**Monthly:**

```bash
# Full system update
docker-compose down
docker-compose pull
docker-compose up -d

# Database backup
docker-compose exec postgres pg_dump -U agentsystem agentsystem > backup_$(date +%Y%m%d).sql
```

### **Scaling Considerations**

**Horizontal Scaling:**

```yaml
# Scale workers
docker-compose up -d --scale celery-worker=3

# Load balancer configuration
services:
  agentsystem:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

---

## 🚀 Performance Optimization

### **Database Optimization**

```sql
-- Create indexes for better performance
CREATE INDEX idx_facts_timestamp ON facts(timestamp);
CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_activity_log_type ON activity_log(activity_type);
```

### **Redis Configuration**

```conf
# redis.conf optimizations
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### **Application Tuning**

```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
```

---

## 📞 Support and Next Steps

### **Immediate Actions**

1. Choose your deployment platform
2. Configure environment variables
3. Deploy and test the health endpoints
4. Set up monitoring and alerts
5. Configure backups

### **Production Checklist**

- [ ] SSL certificates configured
- [ ] Environment variables secured
- [ ] Database backups scheduled
- [ ] Monitoring alerts set up
- [ ] Log aggregation configured
- [ ] Performance testing completed
- [ ] Security audit performed
- [ ] Documentation updated

### **Need Help?**

- Review logs: `docker-compose logs`
- Check health: `curl http://localhost:8000/health`
- Monitor metrics: <http://localhost:9090>
- Task monitoring: <http://localhost:5555>

The AgentSystem backend is now ready for enterprise-scale deployment with full monitoring, security, and scalability features!
