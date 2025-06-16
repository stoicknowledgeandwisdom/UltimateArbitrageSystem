# Airflow Automation Pipeline Deployment Script for Windows
# PowerShell version

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting Airflow Automation Pipeline Deployment..." -ForegroundColor Green

# Check if kubectl is available
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "❌ kubectl is not installed. Please install kubectl first." -ForegroundColor Red
    exit 1
}

# Check if docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed. Please install Docker first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Prerequisites check passed" -ForegroundColor Green

# Build Docker image
Write-Host "🔨 Building Airflow Docker image..." -ForegroundColor Blue
docker build -t arbitrage-airflow:latest .

Write-Host "✅ Docker image built successfully" -ForegroundColor Green

# Apply Kubernetes configurations
Write-Host "☸️ Deploying to Kubernetes..." -ForegroundColor Blue

# Create namespace
kubectl apply -f k8s/airflow-namespace.yaml

# Create RBAC
kubectl apply -f k8s/airflow-rbac.yaml

# Create ConfigMaps and Secrets
kubectl apply -f k8s/airflow-configmap.yaml
kubectl apply -f k8s/airflow-secrets.yaml

# Deploy PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml

# Wait for PostgreSQL to be ready
Write-Host "⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n airflow

# Initialize Airflow database
Write-Host "🗄️ Initializing Airflow database..." -ForegroundColor Blue
kubectl run airflow-init --rm -i --tty --restart=Never `
    --image=arbitrage-airflow:latest `
    --namespace=airflow `
    --env="AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow" `
    -- airflow db init

# Create admin user
Write-Host "👤 Creating Airflow admin user..." -ForegroundColor Blue
kubectl run airflow-create-user --rm -i --tty --restart=Never `
    --image=arbitrage-airflow:latest `
    --namespace=airflow `
    --env="AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow" `
    -- airflow users create `
    --username admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com `
    --password admin

# Deploy Airflow components
kubectl apply -f k8s/airflow-deployment.yaml

# Wait for deployments to be ready
Write-Host "⏳ Waiting for Airflow components to be ready..." -ForegroundColor Yellow
kubectl wait --for=condition=available --timeout=300s deployment/airflow-scheduler -n airflow
kubectl wait --for=condition=available --timeout=300s deployment/airflow-webserver -n airflow

Write-Host "✅ Airflow Automation Pipeline deployed successfully!" -ForegroundColor Green

# Get service information
Write-Host "📊 Service Information:" -ForegroundColor Cyan
kubectl get services -n airflow

Write-Host ""
Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host "📝 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Access Airflow UI at http://localhost:8080 (if using port-forward)"
Write-Host "   2. Default credentials: admin/admin"
Write-Host "   3. Configure your API keys in the secrets"
Write-Host "   4. Set up Kafka for event triggers"
Write-Host ""
Write-Host "💡 Useful commands:" -ForegroundColor Cyan
Write-Host "   - Port forward: kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow"
Write-Host "   - View logs: kubectl logs -f deployment/airflow-scheduler -n airflow"
Write-Host "   - Scale workers: kubectl scale deployment airflow-scheduler --replicas=2 -n airflow"

