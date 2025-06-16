# Ultimate Arbitrage System - Deployment Guide

## Overview

This guide covers the complete CI/CD, Infrastructure-as-Code, and rollout strategy for the Ultimate Arbitrage System with multi-region active-active deployment.

## Architecture Overview

### Pipeline Flow
```
GitHub → GitHub Actions → Argo CD (GitOps) → Kubernetes
```

### Infrastructure Stack
- **IaC**: Terraform + Terragrunt
- **Container Orchestration**: Kubernetes (EKS)
- **GitOps**: Argo CD
- **Database**: PostgreSQL (RDS)
- **Feature Flags**: OpenFeature with flagd
- **Monitoring**: Prometheus + Grafana
- **Error Budget**: Automated monitoring and rollback

### Multi-Region Setup
- **Active-Active Clusters**: us-west-2, eu-west-1, ap-southeast-1
- **Global Load Balancer**: AWS Global Accelerator with 50ms failover
- **Data Replication**: Cross-region PostgreSQL read replicas

## Deployment Strategies

### 1. Blue/Green Deployment

**Configuration**:
- Two identical production environments (Blue/Green)
- Instant traffic switch between environments
- Zero-downtime deployments
- Quick rollback capability

**Feature Flag Control**:
```json
{
  "blue-green-deployment": {
    "state": "ENABLED",
    "variants": {
      "blue": "blue",
      "green": "green"
    },
    "defaultVariant": "blue"
  }
}
```

### 2. Canary Deployment with Progressive Rollout

**Progressive Traffic Split**:
- **Step 1**: 1% of traffic → 10 minutes observation
- **Step 2**: 5% of traffic → 10 minutes observation
- **Step 3**: 25% of traffic → 10 minutes observation
- **Step 4**: 50% of traffic → 10 minutes observation
- **Step 5**: 75% of traffic → 5 minutes observation
- **Step 6**: 100% of traffic (full rollout)

**Automated Analysis**:
- Success rate ≥ 95%
- Average response time < 1000ms
- Error rate ≤ 5%

**Manual Trigger via GitHub Actions**:
```bash
gh workflow run ci-cd.yml \
  --field environment=prod \
  --field canary_percentage=5
```

### 3. Feature Flag Management

**OpenFeature Integration**:
- Dashboard available at: `http://feature-flag-dashboard.feature-flags.svc.cluster.local`
- Real-time flag toggling
- Percentage-based rollouts
- Environment-specific targeting

**Key Feature Flags**:
- `canary-deployment`: Enable/disable canary strategy
- `progressive-rollout`: Control rollout percentage
- `error-budget-enforcement`: Strict vs permissive policies
- `monitoring-enhanced`: Enhanced monitoring features

## Safety Nets

### 1. Automated Rollback on Error Budget Breach

**Error Budget Policy**:
- **Threshold**: 1% error rate (99% SLA)
- **Monitoring Interval**: 30 seconds
- **Action**: Automatic rollback if threshold exceeded during active deployment

**Implementation**:
```bash
# Monitor runs continuously
kubectl get deployment error-budget-monitor -n monitoring

# Manual rollback if needed
kubectl argo rollouts abort arbitrage-app -n arbitrage
kubectl argo rollouts undo arbitrage-app -n arbitrage
```

### 2. Database Migrations with Liquibase

**Migration Process**:
1. **Pre-migration backup**: Automatic RDS snapshot
2. **Migration execution**: Liquibase with verification
3. **Post-migration validation**: Schema and data integrity checks
4. **Rollback capability**: Automatic restore from snapshot if needed

**Manual Migration**:
```bash
# Run migration
liquibase --url="$DB_URL" \
  --username="$DB_USER" \
  --password="$DB_PASS" \
  --changeLogFile=database/changelog.xml \
  update

# Verify migration
liquibase --url="$DB_URL" \
  --username="$DB_USER" \
  --password="$DB_PASS" \
  --changeLogFile=database/changelog.xml \
  status
```

## Multi-Region Active-Active Setup

### 1. Global Load Balancer Configuration

**AWS Global Accelerator**:
```bash
# Create Global Accelerator
aws globalaccelerator create-accelerator \
  --name arbitrage-global \
  --ip-address-type IPV4 \
  --enabled

# Add listeners for HTTP/HTTPS
aws globalaccelerator create-listener \
  --accelerator-arn $ACCELERATOR_ARN \
  --protocol TCP \
  --port-ranges FromPort=80,ToPort=80 \
  --client-affinity SOURCE_IP
```

**Health Check Configuration**:
- **Health Check Path**: `/health`
- **Healthy Threshold**: 3 consecutive successes
- **Unhealthy Threshold**: 3 consecutive failures
- **Timeout**: 5 seconds
- **Interval**: 10 seconds
- **Failover Time**: 50ms

### 2. Region-Specific Deployments

**us-west-2 (Primary)**:
```bash
# Deploy to us-west-2
terragrunt apply --terragrunt-working-dir infrastructure/environments/prod/us-west-2
```

**eu-west-1 (Secondary)**:
```bash
# Deploy to eu-west-1
terragrunt apply --terragrunt-working-dir infrastructure/environments/prod/eu-west-1
```

**ap-southeast-1 (Tertiary)**:
```bash
# Deploy to ap-southeast-1
terragrunt apply --terragrunt-working-dir infrastructure/environments/prod/ap-southeast-1
```

### 3. Database Cross-Region Replication

**Master-Slave Setup**:
- **Master**: us-west-2 (read/write)
- **Read Replicas**: eu-west-1, ap-southeast-1 (read-only)
- **Replication Lag**: < 100ms
- **Automatic Failover**: RDS Multi-AZ in each region

## Monitoring and Observability

### 1. Key Metrics

**Application Metrics**:
- Request rate (requests/second)
- Response time (p95, p99)
- Error rate (%)
- Arbitrage opportunities detected
- Trade execution success rate

**Infrastructure Metrics**:
- CPU and Memory utilization
- Network latency between regions
- Database connection pool usage
- Kubernetes pod health

**Business Metrics**:
- Revenue per region
- Active users
- Feature flag adoption rates
- Deployment frequency and success rate

### 2. Alerting Rules

**Critical Alerts**:
- Error rate > 1% for 5 minutes
- Response time > 2 seconds (p95) for 3 minutes
- Database connection failures
- Region unavailability

**Warning Alerts**:
- Error rate > 0.5% for 10 minutes
- Memory usage > 80%
- Disk usage > 85%
- Feature flag percentage changes

## Deployment Commands

### 1. Infrastructure Deployment

```bash
# Initialize Terraform backend
terragrunt init --terragrunt-working-dir infrastructure/environments/prod

# Plan infrastructure changes
terragrunt plan --terragrunt-working-dir infrastructure/environments/prod

# Apply infrastructure changes
terragrunt apply --terragrunt-working-dir infrastructure/environments/prod
```

### 2. Application Deployment

```bash
# Trigger manual deployment
gh workflow run ci-cd.yml \
  --field environment=prod \
  --field canary_percentage=1

# Monitor rollout progress
kubectl argo rollouts get rollout arbitrage-app -n arbitrage --watch

# Promote canary to stable
kubectl argo rollouts promote arbitrage-app -n arbitrage
```

### 3. Feature Flag Management

```bash
# Access feature flag dashboard
kubectl port-forward svc/feature-flag-dashboard 8080:80 -n feature-flags
# Open http://localhost:8080

# Update feature flags via API
curl -X POST http://flagd-service.feature-flags.svc.cluster.local:8014/flagd.evaluation.v1.Service/ResolveBoolean \
  -H "Content-Type: application/json" \
  -d '{
    "flagKey": "canary-deployment",
    "context": {
      "environment": "prod"
    }
  }'
```

## Troubleshooting

### 1. Rollback Procedures

**Immediate Rollback**:
```bash
# Abort current rollout
kubectl argo rollouts abort arbitrage-app -n arbitrage

# Rollback to previous version
kubectl argo rollouts undo arbitrage-app -n arbitrage

# Verify rollback
kubectl argo rollouts status arbitrage-app -n arbitrage
```

**Database Rollback**:
```bash
# Restore from automated backup
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier arbitrage-prod-restored \
  --db-snapshot-identifier backup-20231201-120000-abcd1234
```

### 2. Region Failover

**Manual Region Failover**:
```bash
# Update Global Accelerator endpoint weights
aws globalaccelerator update-endpoint-group \
  --endpoint-group-arn $ENDPOINT_GROUP_ARN \
  --endpoint-configurations EndpointId=$NEW_ALB_ARN,Weight=100
```

**Health Check Validation**:
```bash
# Check regional health
curl -f https://arbitrage-us-west-2.example.com/health
curl -f https://arbitrage-eu-west-1.example.com/health
curl -f https://arbitrage-ap-southeast-1.example.com/health
```

## Security Considerations

### 1. Secrets Management
- AWS Secrets Manager for database credentials
- Kubernetes secrets for API keys
- Rotation policies for all secrets
- Encryption at rest and in transit

### 2. Network Security
- VPC isolation for each environment
- Security groups with least privilege
- WAF protection for public endpoints
- Private subnets for database and backend services

### 3. Compliance
- Audit logs for all deployment activities
- Immutable infrastructure
- Signed container images
- Regular security scanning

## Performance Optimization

### 1. Caching Strategy
- Redis cluster for application caching
- CDN for static assets
- Database query optimization
- Connection pooling

### 2. Scaling Policies
- Horizontal Pod Autoscaler (HPA)
- Cluster Autoscaler for nodes
- Predictive scaling based on historical data
- Regional traffic distribution

## Disaster Recovery

### 1. Backup Strategy
- Automated daily database backups
- Cross-region backup replication
- Infrastructure state backups
- Application configuration backups

### 2. Recovery Procedures
- RTO (Recovery Time Objective): < 15 minutes
- RPO (Recovery Point Objective): < 5 minutes
- Automated failover for database
- Manual failover for application traffic

This deployment guide ensures a robust, scalable, and highly available arbitrage system with comprehensive safety nets and monitoring capabilities.

