# TimesFM API Production Runbook

This runbook provides comprehensive procedures for deploying, monitoring, and maintaining the TimesFM API in production environments.

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Tuning](#performance-tuning)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Emergency Procedures](#emergency-procedures)
9. [Backup and Recovery](#backup-and-recovery)

## System Overview

### Architecture Components

- **TimesFM API**: FastAPI application serving time series forecasting
- **Redis**: Caching layer for forecast results
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting dashboard
- **Traefik**: Reverse proxy and load balancer
- **Node Exporter**: System metrics collection
- **cAdvisor**: Container metrics collection
- **AlertManager**: Alert routing and notification

### Resource Requirements

#### Minimum Resources
- **CPU**: 4 cores per TimesFM instance
- **Memory**: 8GB RAM per TimesFM instance
- **GPU**: 1 NVIDIA GPU (V100 or better) per instance
- **Storage**: 50GB SSD for models and logs
- **Network**: 1Gbps connectivity

#### Recommended Resources (Production)
- **CPU**: 8 cores per TimesFM instance
- **Memory**: 16GB RAM per TimesFM instance
- **GPU**: 1 NVIDIA A100 or V100 per instance
- **Storage**: 100GB NVMe SSD for models and logs
- **Network**: 10Gbps connectivity

## Prerequisites

### Infrastructure Requirements

1. **Kubernetes Cluster** (v1.25+) with:
   - NVIDIA GPU support
   - Persistent volume support
   - Ingress controller
   - RBAC enabled

2. **Docker Environment** with:
   - NVIDIA container runtime
   - Docker Compose v2.0+
   - Sufficient disk space for images and models

3. **Monitoring Stack**:
   - Prometheus server
   - Grafana instance
   - AlertManager configured

### Model Artifacts

1. **TimesFM Model Files**:
   ```bash
   # Download models to /models directory
   wget https://storage.googleapis.com/timesfm-models/timesfm-200m.ckpt -O /models/timesfm-200m.ckpt
   wget https://storage.googleapis.com/timesfm-models/timesfm-50m.ckpt -O /models/timesfm-50m.ckpt
   ```

2. **Model Metadata**:
   - Model configuration files
   - Tokenizer files (if applicable)
   - Version information

### Configuration Files

1. **Environment Variables**:
   ```bash
   # Create .env file
   echo "MODEL_SIZE=200M" > .env
   echo "MODEL_CHECKPOINT_PATH=/models/timesfm-200m.ckpt" >> .env
   echo "REDIS_URL=redis://redis:6379" >> .env
   echo "API_WORKERS=2" >> .env
   echo "LOG_LEVEL=INFO" >> .env
   ```

## Deployment Procedures

### Docker Compose Deployment

#### 1. Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd timesfm/timesfm-api

# Download models
mkdir -p models
cd models
wget https://storage.googleapis.com/timesfm-models/timesfm-200m.ckpt

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

#### 2. Deploy Stack

```bash
# Deploy all services
cd docker
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs timesfm-api
```

#### 3. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Grafana
curl http://localhost:3000/api/health

# Check Prometheus
curl http://localhost:9090/-/healthy
```

### Kubernetes Deployment

#### 1. Create Namespace

```bash
# Create namespace
kubectl create namespace timesfm

# Set default namespace
kubectl config set-context --current --namespace=timesfm
```

#### 2. Deploy Infrastructure

```bash
# Deploy monitoring stack
kubectl apply -f k8s/monitoring.yaml

# Wait for services to be ready
kubectl wait --for=condition=available --timeout=300s deployment/prometheus
kubectl wait --for=condition=available --timeout=300s deployment/grafana
kubectl wait --for=condition=available --timeout=300s deployment/redis
```

#### 3. Deploy TimesFM API

```bash
# Create secrets (if not already created)
kubectl create secret generic timesfm-secrets \
  --from-literal=redis-url=redis://redis:6379 \
  --from-literal=grafana-password=admin123

# Deploy API
kubectl apply -f k8s/timesfm-api-deployment.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=600s deployment/timesfm-api
```

#### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -l app=timesfm-api

# Check service endpoints
kubectl get endpoints

# Test API
kubectl port-forward service/timesfm-api-service 8000:80 &
curl http://localhost:8000/health
```

### Configuration Management

#### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_SIZE` | Model size (50M, 200M) | 200M | Yes |
| `MODEL_CHECKPOINT_PATH` | Path to model file | /models/timesfm-200m.ckpt | Yes |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 | Yes |
| `API_WORKERS` | Number of worker processes | 2 | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | 0 | No |
| `PRELOAD_MODEL` | Preload model on startup | true | No |

#### Secrets Management

```bash
# Kubernetes secrets
kubectl create secret generic timesfm-secrets \
  --from-literal=redis-url=redis://redis:6379 \
  --from-literal=grafana-password=secure-password \
  --dry-run=client -o yaml | kubectl apply -f -

# Docker secrets
echo "secure-password" | docker secret create grafana-password -
```

## Monitoring and Alerting

### Key Metrics

#### Application Metrics
- **Request Rate**: `timesfm_requests_total`
- **Response Time**: `timesfm_request_duration_seconds`
- **Error Rate**: `timesfm_requests_total{status="5.."}`
- **Inference Time**: `timesfm_inference_duration_seconds`
- **Active Connections**: `timesfm_active_connections`
- **Memory Usage**: `timesfm_memory_usage_bytes`
- **GPU Memory Usage**: `timesfm_gpu_memory_usage_bytes`

#### System Metrics
- **CPU Usage**: `node_cpu_seconds_total`
- **Memory Usage**: `node_memory_MemAvailable_bytes`
- **Disk Usage**: `node_filesystem_size_bytes`
- **Network I/O**: `node_network_receive_bytes_total`

### Alerting Rules

#### Critical Alerts
- **Service Down**: API service unavailable
- **High Error Rate**: >10% error rate over 5 minutes
- **High Latency**: 95th percentile > 30 seconds
- **GPU Memory High**: >15GB GPU memory usage
- **Model Loading Failure**: Model fails to load

#### Warning Alerts
- **High Memory Usage**: >7GB system memory
- **High Request Rate**: >50 requests/second
- **Cache Hit Rate Low**: <50% cache hit rate
- **Redis Down**: Redis service unavailable

### Grafana Dashboards

#### Main Dashboard: "TimesFM API Overview"
- Request rate and error rates
- Response time percentiles
- Model inference performance
- Resource utilization
- Cache performance

#### Additional Dashboards
- **System Metrics**: Node and container performance
- **Infrastructure**: Redis and monitoring stack health
- **Business Metrics**: Forecast accuracy and usage patterns

### Monitoring Commands

```bash
# Check API metrics
curl http://localhost:8000/metrics

# Prometheus queries
curl 'http://localhost:9090/api/v1/query?query=rate(timesfm_requests_total[5m])'

# Grafana API
curl -u admin:admin123 http://localhost:3000/api/dashboards/home

# Kubernetes logs
kubectl logs -l app=timesfm-api --tail=100

# Docker logs
docker-compose -f docker-compose.prod.yml logs -f timesfm-api
```

## Troubleshooting Guide

### Common Issues

#### 1. API Not Responding

**Symptoms**: Health check failing, timeouts

**Troubleshooting Steps**:
```bash
# Check container/pod status
docker-compose -f docker-compose.prod.yml ps
kubectl get pods -l app=timesfm-api

# Check logs
docker-compose -f docker-compose.prod.yml logs timesfm-api
kubectl logs -l app=timesfm-api --tail=50

# Check resource usage
docker stats
kubectl top pods

# Check health endpoint
curl -v http://localhost:8000/health
```

**Common Causes**:
- Out of memory errors
- GPU not available
- Model loading failure
- Port conflicts

#### 2. High Latency

**Symptoms**: Response times > 30 seconds

**Troubleshooting Steps**:
```bash
# Check resource utilization
kubectl top pods
nvidia-smi

# Check model inference time
curl 'http://localhost:8000/metrics' | grep inference_duration

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check concurrent requests
curl 'http://localhost:8000/metrics' | grep active_connections
```

**Solutions**:
- Scale horizontally (add more pods)
- Optimize batch sizes
- Use larger GPU
- Implement request queuing

#### 3. Memory Issues

**Symptoms**: OOMKilled, restarts

**Troubleshooting Steps**:
```bash
# Check memory usage
kubectl describe pod <pod-name>
kubectl top pods

# Check memory limits
kubectl get deployment timesfm-api -o yaml | grep -A5 resources

# Check memory leaks
docker stats --no-stream
```

**Solutions**:
- Increase memory limits
- Optimize model loading
- Implement memory cleanup
- Add memory profiling

#### 4. GPU Issues

**Symptoms**: CUDA errors, GPU not detected

**Troubleshooting Steps**:
```bash
# Check GPU availability
nvidia-smi
kubectl describe nodes | grep nvidia.com/gpu

# Check GPU in container
kubectl exec -it <pod-name> -- nvidia-smi

# Check CUDA drivers
nvidia-smi --query-gpu=driver_version --format=csv
```

**Solutions**:
- Install correct NVIDIA drivers
- Check GPU device mapping
- Verify NVIDIA container runtime
- Check GPU resource requests

### Performance Issues

#### 1. Slow Model Loading

**Optimization Steps**:
```bash
# Preload model
export PRELOAD_MODEL=true

# Use faster storage
# Move model to NVMe SSD

# Optimize model format
# Convert to optimized checkpoint
```

#### 2. Cache Inefficiency

**Optimization Steps**:
```bash
# Check Redis performance
redis-cli info stats
redis-cli latency doctor

# Optimize cache keys
# Use consistent hashing

# Increase cache size
# Update Redis maxmemory setting
```

## Performance Tuning

### API Tuning

#### Worker Configuration
```python
# Optimize based on CPU cores
workers = min(os.cpu_count(), 4)

# For GPU workloads, single process often better
if torch.cuda.is_available():
    workers = 1
```

#### Batch Processing
```python
# Optimize batch sizes based on GPU memory
batch_size = {
    "50M": 32,
    "200M": 16,
    "500M": 8
}
```

### Infrastructure Tuning

#### Kubernetes Resources
```yaml
# Optimize resource requests/limits
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "4"
    memory: "8Gi"
    nvidia.com/gpu: "1"
```

#### HPA Configuration
```yaml
# Configure autoscaling
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Pods
  pods:
    metric:
      name: timesfm_requests_per_second
    target:
      type: AverageValue
      averageValue: "100"
```

### Database/Caching Tuning

#### Redis Optimization
```bash
# Redis configuration
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### Model Caching Strategy
```python
# Implement multi-level caching
@lru_cache(maxsize=100)
def load_model_cached(model_path):
    # Cache loaded models in memory

# Use Redis for result caching
cache_key = f"{hash(data)}_{horizon}_{model_size}"
cached_result = redis.get(cache_key)
```

## Maintenance Procedures

### Regular Maintenance

#### Daily Checks
```bash
#!/bin/bash
# daily_checks.sh

# Check service health
curl -f http://localhost:8000/health || exit 1

# Check resource usage
kubectl top pods

# Check error rates
curl 'http://localhost:9090/api/v1/query?query=rate(timesfm_requests_total{status="5.."}[5m])'

# Check disk space
df -h

# Check GPU status
nvidia-smi
```

#### Weekly Maintenance
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update containers
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Clean up old logs
find /app/logs -name "*.log" -mtime +7 -delete

# Backup configurations
kubectl get configmaps,secrets -o yaml > backup-$(date +%Y%m%d).yaml

# Check model updates
# Verify latest model versions
```

#### Monthly Maintenance
```bash
#!/bin/bash
# monthly_maintenance.sh

# Update models
# Download new model versions
# Test model compatibility

# Performance testing
./scripts/load_test.py --users=20 --requests=200

# Security updates
# Update base images
# Scan for vulnerabilities

# Capacity planning
# Review resource usage trends
# Plan for scaling
```

### Model Updates

#### Model Deployment Procedure
```bash
# 1. Test new model
kubectl run model-test --image=timesfm-api:latest \
  --env="MODEL_CHECKPOINT_PATH=/models/new-model.ckpt" \
  --restart=Never

# 2. Validate performance
./scripts/load_test.py --url=http://model-test:8000

# 3. Gradual rollout
kubectl patch deployment timesfm-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"timesfm-api","env":[{"name":"MODEL_CHECKPOINT_PATH","value":"/models/new-model.ckpt"}]}]}}}}'

# 4. Monitor performance
# Watch metrics and error rates
# Rollback if issues detected
```

### Backup Procedures

#### Configuration Backup
```bash
# Backup Kubernetes configurations
kubectl get all,configmaps,secrets,pvc -o yaml > backup-$(date +%Y%m%d).yaml

# Backup Docker configurations
cp -r docker/ backup/docker-$(date +%Y%m%d)/
cp .env backup/.env-$(date +%Y%m%d)
```

#### Model Backup
```bash
# Backup model files
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage
gsutil cp models-backup-$(date +%Y%m%d).tar.gz gs://backups/
```

## Emergency Procedures

### Service Outage Response

#### 1. Immediate Response (0-5 minutes)
```bash
# Check service status
kubectl get pods -l app=timesfm-api

# Restart if needed
kubectl rollout restart deployment/timesfm-api

# Check logs for errors
kubectl logs -l app=timesfm-api --tail=100
```

#### 2. Investigation (5-15 minutes)
```bash
# Check resource usage
kubectl top pods
kubectl describe nodes

# Check external dependencies
kubectl logs -l app=redis
kubectl logs -l app=prometheus

# Check network connectivity
kubectl exec -it <pod-name> -- nslookup redis
```

#### 3. Recovery (15-60 minutes)
```bash
# Scale up if resource constrained
kubectl scale deployment timesfm-api --replicas=4

# Rollback if recent deployment
kubectl rollout undo deployment/timesfm-api

# Restore from backup if needed
kubectl apply -f backup-<timestamp>.yaml
```

### Security Incident Response

#### 1. Containment
```bash
# Isolate affected services
kubectl scale deployment timesfm-api --replicas=0

# Enable network policies
kubectl apply -f security/network-policy.yaml

# Change credentials
kubectl delete secret timesfm-secrets
kubectl create secret generic timesfm-secrets --from-literal=...
```

#### 2. Investigation
```bash
# Check logs for suspicious activity
kubectl logs -l app=timesfm-api --since=1h | grep -i error

# Check network traffic
kubectl get networkpolicy
kubectl logs -l app=traefik | grep -i suspicious

# Audit user access
kubectl auth can-i --list
```

#### 3. Recovery
```bash
# Rebuild clean environment
kubectl delete namespace timesfm
kubectl create namespace timesfm

# Restore from clean backup
kubectl apply -f backup-clean-<timestamp>.yaml

# Update security policies
# Apply latest security patches
```

## Backup and Recovery

### Backup Strategy

#### 1. Configuration Backup
- **Frequency**: Daily
- **Retention**: 30 days
- **Storage**: Cloud storage (S3/GS)
- **Content**: Kubernetes manifests, secrets, configmaps

#### 2. Model Backup
- **Frequency**: Weekly
- **Retention**: 90 days
- **Storage**: Cloud storage + local
- **Content**: Model checkpoints, metadata

#### 3. Data Backup
- **Frequency**: Daily
- **Retention**: 7 days
- **Storage**: Local SSD
- **Content**: Logs, metrics, cache data

### Recovery Procedures

#### Complete System Recovery
```bash
# 1. Restore namespace
kubectl create namespace timesfm

# 2. Restore configurations
kubectl apply -f backup-<timestamp>.yaml

# 3. Restore models
gsutil cp gs://backups/models-backup-<timestamp>.tar.gz .
tar -xzf models-backup-<timestamp>.tar.gz

# 4. Restart services
kubectl rollout restart deployment/timesfm-api

# 5. Verify functionality
./scripts/health_check.sh
```

#### Partial Recovery
```bash
# Restore specific component
kubectl apply -f backup-<timestamp>-timesfm-api.yaml

# Or scale down/up specific service
kubectl scale deployment timesfm-api --replicas=0
kubectl scale deployment timesfm-api --replicas=2
```

### Disaster Recovery Plan

#### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 24 hours

#### Disaster Scenarios
1. **Single Node Failure**: Automatic failover
2. **Region Failure**: Manual failover to backup region
3. **Data Corruption**: Restore from latest backup
4. **Security Breach**: Isolate and rebuild from clean backup

#### Recovery Testing
```bash
# Monthly disaster recovery test
# Schedule during maintenance window
# Document recovery times
# Update procedures based on results
```

---

## Contact Information

- **Primary Support**: devops@example.com
- **On-call Engineer**: oncall@example.com
- **Emergency Contact**: +1-555-EMERGENCY

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-10-02 | Initial production runbook | Claude |
| 1.1 | TBD | TBD | TBD |