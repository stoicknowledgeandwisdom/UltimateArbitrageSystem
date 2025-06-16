#!/bin/bash
# Airflow Automation Pipeline Deployment Script

set -e

echo "🚀 Starting Airflow Automation Pipeline Deployment..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Build Docker image
echo "🔨 Building Airflow Docker image..."
docker build -t arbitrage-airflow:latest .

echo "✅ Docker image built successfully"

# Apply Kubernetes configurations
echo "☸️ Deploying to Kubernetes..."

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
echo "⏳ Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n airflow

# Initialize Airflow database
echo "🗄️ Initializing Airflow database..."
kubectl run airflow-init --rm -i --tty --restart=Never \
    --image=arbitrage-airflow:latest \
    --namespace=airflow \
    --env="AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow" \
    -- airflow db init

# Create admin user
echo "👤 Creating Airflow admin user..."
kubectl run airflow-create-user --rm -i --tty --restart=Never \
    --image=arbitrage-airflow:latest \
    --namespace=airflow \
    --env="AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow" \
    -- airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Deploy Airflow components
kubectl apply -f k8s/airflow-deployment.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for Airflow components to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/airflow-scheduler -n airflow
kubectl wait --for=condition=available --timeout=300s deployment/airflow-webserver -n airflow

echo "✅ Airflow Automation Pipeline deployed successfully!"

# Get service information
echo "📊 Service Information:"
kubectl get services -n airflow

echo ""
echo "🎉 Deployment completed successfully!"
echo "📝 Next steps:"
echo "   1. Access Airflow UI at http://localhost:8080 (if using port-forward)"
echo "   2. Default credentials: admin/admin"
echo "   3. Configure your API keys in the secrets"
echo "   4. Set up Kafka for event triggers"
echo ""
echo "💡 Useful commands:"
echo "   - Port forward: kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow"
echo "   - View logs: kubectl logs -f deployment/airflow-scheduler -n airflow"
echo "   - Scale workers: kubectl scale deployment airflow-scheduler --replicas=2 -n airflow"

