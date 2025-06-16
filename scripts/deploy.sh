#!/bin/bash

# Ultimate Arbitrage System - Deployment Script
# This script handles the complete deployment process

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-dev}
REGION=${2:-us-west-2}
CANARY_PERCENTAGE=${3:-1}
SKIP_TESTS=${4:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required tools
    local tools=("terraform" "terragrunt" "kubectl" "docker" "aws" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_warning "kubectl cannot connect to cluster. Will attempt to configure..."
        aws eks update-kubeconfig --region "$REGION" --name "arbitrage-$ENVIRONMENT-$REGION"
    fi
    
    log_success "Prerequisites validated"
}

# Run tests
run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running tests..."
    
    # Unit tests
    npm test
    
    # Integration tests
    npm run test:integration
    
    # Security audit
    npm audit --audit-level high
    
    log_success "All tests passed"
}

# Build and push container image
build_and_push_image() {
    log_info "Building and pushing container image..."
    
    local image_tag="$(git rev-parse --short HEAD)"
    local image_name="ghcr.io/ultimate-arbitrage/arbitrage-app:$image_tag"
    
    # Build image
    docker build -t "$image_name" .
    
    # Push to registry
    docker push "$image_name"
    
    # Update image tag in kustomization
    sed -i "s|newTag: .*|newTag: $image_tag|" k8s/base/kustomization.yaml
    
    log_success "Container image built and pushed: $image_name"
    echo "$image_tag" > .image-tag
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure for $ENVIRONMENT in $REGION..."
    
    local tf_dir="infrastructure/environments/$ENVIRONMENT"
    
    # Initialize Terragrunt
    terragrunt init --terragrunt-working-dir "$tf_dir"
    
    # Plan infrastructure changes
    terragrunt plan --terragrunt-working-dir "$tf_dir" -out=tfplan
    
    # Apply infrastructure changes
    terragrunt apply --terragrunt-working-dir "$tf_dir" tfplan
    
    log_success "Infrastructure deployed successfully"
}

# Database migration
run_database_migration() {
    log_info "Running database migrations..."
    
    # Get database credentials from AWS Secrets Manager
    local db_secret=$(aws secretsmanager get-secret-value \
        --secret-id "arbitrage-$ENVIRONMENT-rds-password" \
        --query 'SecretString' --output text)
    
    local db_host=$(aws ssm get-parameter \
        --name "/arbitrage/$ENVIRONMENT/database/host" \
        --query 'Parameter.Value' --output text)
    
    local db_user=$(aws ssm get-parameter \
        --name "/arbitrage/$ENVIRONMENT/database/user" \
        --query 'Parameter.Value' --output text)
    
    # Create backup before migration
    local backup_name="backup-$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)"
    aws rds create-db-snapshot \
        --db-instance-identifier "arbitrage-$ENVIRONMENT" \
        --db-snapshot-identifier "$backup_name"
    
    log_info "Created database backup: $backup_name"
    
    # Wait for backup to complete
    aws rds wait db-snapshot-completed --db-snapshot-identifier "$backup_name"
    
    # Run Liquibase migrations
    liquibase \
        --url="jdbc:postgresql://$db_host:5432/arbitrage" \
        --username="$db_user" \
        --password="$db_secret" \
        --changeLogFile=database/changelog.xml \
        update
    
    # Verify migration
    liquibase \
        --url="jdbc:postgresql://$db_host:5432/arbitrage" \
        --username="$db_user" \
        --password="$db_secret" \
        --changeLogFile=database/changelog.xml \
        status
    
    log_success "Database migrations completed successfully"
}

# Deploy application using ArgoCD
deploy_application() {
    log_info "Deploying application with canary percentage: $CANARY_PERCENTAGE%"
    
    local image_tag
    if [ -f .image-tag ]; then
        image_tag=$(cat .image-tag)
    else
        image_tag="$(git rev-parse --short HEAD)"
    fi
    
    # Create ArgoCD application manifest
    cat > "argocd-app-$ENVIRONMENT-$REGION.yaml" << EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: arbitrage-app-$ENVIRONMENT-$REGION
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/ultimate-arbitrage/UltimateArbitrageSystem.git
    targetRevision: HEAD
    path: k8s/overlays/$ENVIRONMENT
    helm:
      valueFiles:
        - values-$REGION.yaml
      parameters:
        - name: image.tag
          value: $image_tag
        - name: canary.percentage
          value: "$CANARY_PERCENTAGE"
        - name: region
          value: $REGION
  destination:
    server: https://kubernetes.default.svc
    namespace: arbitrage-$ENVIRONMENT
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
EOF
    
    # Apply ArgoCD application
    kubectl apply -f "argocd-app-$ENVIRONMENT-$REGION.yaml"
    
    # Wait for application to sync
    kubectl wait --for=condition=Synced application "arbitrage-app-$ENVIRONMENT-$REGION" -n argocd --timeout=300s
    
    log_success "Application deployed and synced"
}

# Monitor rollout
monitor_rollout() {
    log_info "Monitoring rollout progress..."
    
    # Wait for rollout to complete or fail
    local timeout=1800  # 30 minutes
    local elapsed=0
    local check_interval=30
    
    while [ $elapsed -lt $timeout ]; do
        local status=$(kubectl argo rollouts status arbitrage-app -n "arbitrage-$ENVIRONMENT" --timeout=30s 2>/dev/null || echo "UNKNOWN")
        
        case $status in
            *"successfully rolled out"*)
                log_success "Rollout completed successfully"
                return 0
                ;;
            *"rollout aborted"*)
                log_error "Rollout was aborted"
                return 1
                ;;
            *"rollout failed"*)
                log_error "Rollout failed"
                return 1
                ;;
        esac
        
        log_info "Rollout in progress... (${elapsed}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    log_error "Rollout monitoring timed out after ${timeout}s"
    return 1
}

# Health check
perform_health_check() {
    log_info "Performing health checks..."
    
    local health_url
    if [ "$ENVIRONMENT" = "prod" ]; then
        health_url="https://arbitrage-$REGION.example.com/health"
    else
        health_url="https://arbitrage-$ENVIRONMENT-$REGION.example.com/health"
    fi
    
    local retries=10
    local wait_time=30
    
    for i in $(seq 1 $retries); do
        if curl -f -s "$health_url" > /dev/null; then
            log_success "Health check passed for $health_url"
            return 0
        fi
        
        log_warning "Health check failed (attempt $i/$retries). Retrying in ${wait_time}s..."
        sleep $wait_time
    done
    
    log_error "Health check failed after $retries attempts"
    return 1
}

# Rollback on failure
rollback_deployment() {
    log_error "Deployment failed. Initiating rollback..."
    
    # Abort current rollout
    kubectl argo rollouts abort arbitrage-app -n "arbitrage-$ENVIRONMENT" || true
    
    # Rollback to previous version
    kubectl argo rollouts undo arbitrage-app -n "arbitrage-$ENVIRONMENT" || true
    
    # Wait for rollback to complete
    kubectl argo rollouts status arbitrage-app -n "arbitrage-$ENVIRONMENT" --timeout=300s || true
    
    log_warning "Rollback completed. Please investigate the failure."
}

# Send deployment notification
send_notification() {
    local status=$1
    local message=$2
    
    log_info "Sending deployment notification..."
    
    # Send to Slack (if webhook URL is configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"Deployment $status\",
                \"attachments\": [{
                    \"color\": \"$([ \"$status\" = \"SUCCESS\" ] && echo \"good\" || echo \"danger\")\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"$ENVIRONMENT\", \"short\": true},
                        {\"title\": \"Region\", \"value\": \"$REGION\", \"short\": true},
                        {\"title\": \"Canary %\", \"value\": \"$CANARY_PERCENTAGE%\", \"short\": true},
                        {\"title\": \"Message\", \"value\": \"$message\", \"short\": false}
                    ]
                }]
            }"
    fi
    
    log_success "Notification sent"
}

# Main deployment function
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT, region: $REGION"
    
    # Validate prerequisites
    validate_prerequisites
    
    # Run tests
    run_tests
    
    # Build and push image
    build_and_push_image
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Run database migrations
    run_database_migration
    
    # Deploy application
    deploy_application
    
    # Monitor rollout
    if monitor_rollout; then
        # Perform health check
        if perform_health_check; then
            send_notification "SUCCESS" "Deployment completed successfully"
            log_success "Deployment completed successfully!"
        else
            rollback_deployment
            send_notification "FAILED" "Health check failed, deployment rolled back"
            exit 1
        fi
    else
        rollback_deployment
        send_notification "FAILED" "Rollout failed, deployment rolled back"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f "argocd-app-$ENVIRONMENT-$REGION.yaml" .image-tag
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"

