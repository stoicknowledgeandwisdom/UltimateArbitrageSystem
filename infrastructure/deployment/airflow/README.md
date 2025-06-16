# Airflow Automation & Orchestration Pipeline

A comprehensive automation and orchestration pipeline using Apache Airflow with KubernetesExecutor, implementing event-driven and cron-based workflows with SAGA pattern for resilience.

## 🏗️ Architecture Overview

### Workflow Engine
- **Apache Airflow 2.7.0** with KubernetesExecutor
- **DAGs as Code** approach for version control and CI/CD
- **Event-driven triggers** from Kafka topics (market shock, capital threshold)
- **Hybrid scheduling** - separation of cron vs event-driven for predictability

### Core Tasks
1. **Auto-rebalance funds** across exchanges using Transferwise API for cheapest routes
2. **Update funding-rate hedges** every 15 minutes or on >20 bps deviation
3. **Emergency stop-loss** that liquidates all positions on critical alerts

### Resilience Features
- **SAGA Pattern** implementation for distributed transactions
- **Idempotent tasks** with retry & compensation actions
- **Exponential backoff** retry strategies
- **Circuit breaker** patterns for external API calls

## 📁 Project Structure

```
airflow/
├── dags/                           # Airflow DAGs
│   ├── auto_rebalance_saga.py      # Event-driven fund rebalancing
│   ├── funding_rate_hedge_saga.py  # Hybrid hedge updates
│   └── emergency_stop_loss_saga.py # Critical position liquidation
├── sensors/                        # Custom Kafka sensors
│   └── kafka_event_sensor.py       # Event sensors for market data
├── operators/                      # SAGA pattern operators
│   └── saga_operators.py           # Resilient task operators
├── k8s/                           # Kubernetes configurations
│   ├── airflow-namespace.yaml     # Namespace setup
│   ├── airflow-rbac.yaml          # RBAC configuration
│   ├── airflow-configmap.yaml     # Environment configuration
│   ├── airflow-secrets.yaml       # API keys and secrets
│   ├── postgres-deployment.yaml   # Database deployment
│   └── airflow-deployment.yaml    # Airflow components
├── config/                        # Airflow configuration
│   └── airflow.cfg               # Main Airflow config
├── Dockerfile                     # Container image
├── requirements.txt              # Python dependencies
├── deploy.sh                     # Linux deployment script
├── deploy.ps1                    # Windows deployment script
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured and connected to cluster
- Docker installed and running

### 1. Clone and Navigate
```bash
cd UltimateArbitrageSystem/airflow
```

### 2. Deploy (Linux/macOS)
```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Deploy (Windows)
```powershell
.\deploy.ps1
```

### 4. Access Airflow UI
```bash
# Port forward to access locally
kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow
```

Open http://localhost:8080 in your browser.

**Default credentials:** admin/admin

## 📊 DAG Overview

### 1. Auto-Rebalance SAGA (`auto_rebalance_saga`)
- **Trigger:** Event-driven (market shocks, capital thresholds)
- **Function:** Rebalances funds across exchanges using Transferwise API
- **Resilience:** SAGA pattern with compensation for failed transfers
- **Idempotency:** Prevents duplicate rebalancing operations

### 2. Funding Rate Hedge Updates

#### Cron-based (`funding_rate_hedge_cron`)
- **Schedule:** Every 15 minutes
- **Function:** Regular hedge ratio updates
- **Predictability:** Scheduled maintenance of hedge positions

#### Event-driven (`funding_rate_hedge_event`)
- **Trigger:** >20 bps deviation in funding rates
- **Function:** Immediate hedge adjustments for large deviations
- **Responsiveness:** Real-time reaction to market conditions

### 3. Emergency Stop-Loss SAGA (`emergency_stop_loss_saga`)
- **Trigger:** Critical capital thresholds, extreme market shocks
- **Function:** Liquidates all positions immediately
- **Priority:** Highest priority with immediate execution
- **Notifications:** Email alerts to risk management team

## 🔧 Configuration

### API Keys Configuration
Update the secrets in `k8s/airflow-secrets.yaml`:

```bash
# Encode your API keys in base64
echo -n "your-api-key" | base64

# Update the secret file
kubectl edit secret airflow-secrets -n airflow
```

### Kafka Integration
Set up Kafka topics for event triggers:

```bash
# Market events topic
kafka-topics.sh --create --topic market-events --bootstrap-server localhost:9092

# Capital events topic
kafka-topics.sh --create --topic capital-events --bootstrap-server localhost:9092

# Funding rate events topic
kafka-topics.sh --create --topic funding-rate-events --bootstrap-server localhost:9092
```

### Environment Variables
Key configuration in `k8s/airflow-configmap.yaml`:
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka cluster address
- `TRANSFERWISE_API_URL`: Transferwise API endpoint
- Exchange API endpoints and credentials

## 🛡️ SAGA Pattern Implementation

### Core Features
1. **Transaction Execution:** Each task is wrapped in a transaction
2. **Compensation Logic:** Automatic rollback on failures
3. **Idempotency:** Tasks can be safely retried
4. **State Management:** XCom-based state tracking

### Example SAGA Operator
```python
class FundRebalanceOperator(SAGAOperator):
    def __init__(self, *args, **kwargs):
        def rebalance_transaction(context, saga_context):
            # Main rebalancing logic
            return execute_rebalance()
        
        def rebalance_compensation(context, saga_context):
            # Rollback logic
            return reverse_rebalance()
        
        super().__init__(
            saga_id='fund_rebalance',
            transaction_func=rebalance_transaction,
            compensation_func=rebalance_compensation,
            *args, **kwargs
        )
```

## 📈 Monitoring & Observability

### Airflow UI
- DAG status and execution history
- Task logs and error tracking
- Performance metrics

### Kubernetes Monitoring
```bash
# View pod status
kubectl get pods -n airflow

# Check logs
kubectl logs -f deployment/airflow-scheduler -n airflow

# Monitor resource usage
kubectl top pods -n airflow
```

### Health Checks
- Airflow webserver health endpoint: `/health`
- Scheduler job monitoring
- Database connectivity checks

## 🔄 Operational Commands

### Scaling
```bash
# Scale scheduler replicas
kubectl scale deployment airflow-scheduler --replicas=2 -n airflow

# Scale webserver replicas
kubectl scale deployment airflow-webserver --replicas=2 -n airflow
```

### Updates
```bash
# Update Docker image
docker build -t arbitrage-airflow:v2.0 .
kubectl set image deployment/airflow-scheduler airflow-scheduler=arbitrage-airflow:v2.0 -n airflow
```

### Backup
```bash
# Backup Airflow database
kubectl exec -it deployment/postgres -n airflow -- pg_dump -U airflow airflow > airflow_backup.sql
```

## 🚨 Troubleshooting

### Common Issues

1. **DAG Import Errors**
   ```bash
   kubectl logs deployment/airflow-scheduler -n airflow | grep "Failed to import"
   ```

2. **Kafka Connection Issues**
   - Check Kafka bootstrap servers configuration
   - Verify network connectivity from Airflow pods

3. **Database Connection**
   ```bash
   kubectl exec -it deployment/postgres -n airflow -- psql -U airflow
   ```

4. **Worker Pod Issues**
   ```bash
   kubectl get pods -n airflow | grep airflow-worker
   kubectl describe pod <worker-pod-name> -n airflow
   ```

### Log Analysis
```bash
# Scheduler logs
kubectl logs -f deployment/airflow-scheduler -n airflow

# Webserver logs
kubectl logs -f deployment/airflow-webserver -n airflow

# Worker logs (for specific tasks)
kubectl logs <worker-pod-name> -n airflow
```

## 🔐 Security Considerations

1. **API Key Management**
   - Use Kubernetes secrets for sensitive data
   - Rotate keys regularly
   - Implement least-privilege access

2. **Network Security**
   - Use NetworkPolicies for pod communication
   - TLS encryption for external communications
   - Secure Kafka connections

3. **RBAC**
   - Minimal permissions for Airflow service account
   - Regular audit of cluster permissions

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [SAGA Pattern Overview](https://microservices.io/patterns/data/saga.html)
- [Kafka Integration Guide](https://kafka.apache.org/documentation/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

