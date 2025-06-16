# Comprehensive Monitoring, Logging & Alerting System

This directory contains a world-class observability stack designed specifically for the Ultimate Arbitrage Trading System. The monitoring system provides real-time insights, anomaly detection, security monitoring, and time-travel debugging capabilities.

## 🏗️ Architecture Overview

### Core Components

**Metrics Collection & Monitoring**
- **Prometheus**: Time-series metrics collection with custom trading metrics
- **Grafana**: Rich dashboards for trading performance, PnL, and system health
- **Node Exporter**: System-level metrics (CPU, memory, disk, network)
- **cAdvisor**: Container-level metrics for Docker environments

**Logging & Log Analysis**
- **Loki**: Log aggregation with structured JSON parsing
- **Promtail**: Log shipping with custom parsing for trading logs
- **Elasticsearch**: Advanced log analytics and searching

**Distributed Tracing**
- **Jaeger**: End-to-end request tracing for strategy call-graphs
- **OpenTelemetry**: Auto-instrumentation for Python, PostgreSQL, Redis

**Time-Travel Debugging**
- **ClickHouse**: High-performance analytics database for historical analysis
- **Custom Time-Travel UI**: Retroactive debugging capabilities

**Alerting & Incident Management**
- **Alertmanager**: Intelligent alert routing and deduplication
- **PagerDuty Integration**: Critical trading alerts with escalation
- **Slack Integration**: Team notifications with rich formatting

**Security Analytics**
- **Zeek**: Network security monitoring with custom trading patterns
- **OSQuery**: Host-level security monitoring
- **Sigma Rules**: Threat detection rules

**Advanced Features**
- **Real-time Anomaly Detection**: ML-based anomaly detection using Isolation Forest
- **SLO Monitoring**: Error budget tracking with burn rate alerts
- **Business KPI Tracking**: Trading-specific metrics and thresholds

## 🚀 Quick Start

### Prerequisites

```bash
# Required tools
docker --version
docker-compose --version
python3 --version
```

### 1. Deploy the Monitoring Stack

```bash
# Navigate to monitoring directory
cd monitoring

# Deploy complete stack
python3 deploy_monitoring.py deploy

# Or deploy with custom options
python3 deploy_monitoring.py deploy --skip-build --no-wait
```

### 2. Access the Dashboards

Once deployed, access the monitoring interfaces:

- **Grafana Dashboards**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686
- **Alertmanager**: http://localhost:9093
- **ClickHouse Analytics**: http://localhost:8123

### 3. Start Application Monitoring

```python
from monitoring.monitoring_system import MonitoringSystem

# Initialize monitoring
monitoring = MonitoringSystem()

# Record trading events
monitoring.record_trading_event(
    'order_placed',
    exchange='binance',
    symbol='BTC/USDT',
    side='buy',
    amount=0.1,
    price=50000
)
```

## 📊 Key Metrics Tracked

### Trading Performance Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `trading_order_latency_seconds` | Order placement latency | >500ms |
| `trading_fill_rate` | Order fill rate percentage | <80% |
| `trading_pnl_current` | Real-time profit/loss | <-5% drawdown |
| `arbitrage_opportunities_total` | Detected opportunities | - |
| `arbitrage_executed_total` | Executed arbitrage trades | - |
| `trading_margin_used` | Margin utilization | >85% |
| `exchange_connectivity_status` | Exchange connectivity | 0 (disconnected) |
| `exchange_api_rate_limit_remaining` | API rate limits | <10% remaining |

### System Health Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `http_requests_total` | HTTP request count | - |
| `http_request_duration_seconds` | Request latency | >200ms (p95) |
| `database_connections_active` | DB connection pool | >80% utilization |
| `cache_hits_total` / `cache_misses_total` | Cache performance | <90% hit rate |
| `node_cpu_seconds_total` | CPU usage | >80% |
| `node_memory_MemAvailable_bytes` | Memory usage | >85% |
| `node_filesystem_avail_bytes` | Disk usage | <10% available |

### Business KPI Metrics

| Metric | Description | SLO Target |
|--------|-------------|-----------|
| **Availability SLO** | API uptime | 99.9% |
| **Latency SLO** | P95 response time | <200ms |
| **Error Rate SLO** | 5xx error rate | <0.1% |
| **Daily PnL Target** | Daily profit target | Custom threshold |
| **Volume Anomaly** | Trading volume deviation | >3σ from mean |

## 🔧 Configuration

### Environment Variables

Edit `monitoring/.env` to configure alerting:

```bash
# Slack Integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# PagerDuty Integration
PAGERDUTY_ROUTING_KEY=YOUR_PAGERDUTY_ROUTING_KEY
PAGERDUTY_RISK_ROUTING_KEY=YOUR_PAGERDUTY_RISK_ROUTING_KEY

# Email Alerts
SMTP_PASSWORD=your_smtp_password

# Database Credentials
CLICKHOUSE_PASSWORD=clickhouse123
ELASTIC_PASSWORD=elastic123
```

### Alert Configuration

Alert rules are defined in `prometheus/rules/trading-alerts.yml`:

```yaml
groups:
  - name: trading_system_alerts
    rules:
      - alert: HighLatencyAlert
        expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[2m])) by (le)) > 0.5
        for: 30s
        labels:
          severity: critical
          service: trading
        annotations:
          summary: "High latency detected in trading system"
          description: "95th percentile latency is {{ $value }}s"
```

### Custom Dashboards

Grafana dashboards are automatically provisioned:

- **Trading System Overview**: Real-time trading metrics
- **Security Monitoring**: Threat detection and analysis
- **Infrastructure Overview**: System health and performance
- **SLO Dashboard**: Error budget and burn rate tracking

## 🛡️ Security Monitoring

### Network Security (Zeek)

Zeek monitors network traffic for:
- Suspicious API call patterns
- SQL injection attempts
- Data exfiltration patterns
- Geographic anomalies
- Rate limiting violations

Configuration: `security/zeek/site.zeek`

### Host Security (OSQuery)

OSQuery provides host-level monitoring:
- Process monitoring for trading applications
- File integrity monitoring
- Network connection analysis
- User activity tracking
- Container security

Configuration: `security/osquery.conf`

### Threat Detection

Custom security patterns detect:
- Trading system-specific threats
- API abuse patterns
- Anomalous trading behavior
- Unauthorized access attempts

## 📈 Anomaly Detection

The system includes ML-based anomaly detection:

```python
# Real-time anomaly detection
anomaly = monitoring.anomaly_detector.add_data_point(
    metric_name='trading_volume',
    value=current_volume,
    context={'exchange': 'binance', 'symbol': 'BTC/USDT'}
)

if anomaly and anomaly.is_anomaly:
    print(f"Anomaly detected: {anomaly.description}")
    # Trigger alert
```

**Features:**
- Isolation Forest algorithm for outlier detection
- Dynamic model retraining
- Context-aware anomaly scoring
- Integration with alerting system

## 🕰️ Time-Travel Debugging

ClickHouse enables retroactive debugging:

```sql
-- Query historical metrics
SELECT 
    timestamp,
    metric_name,
    value,
    labels['exchange'] as exchange
FROM metrics_events 
WHERE 
    timestamp BETWEEN '2024-01-01 00:00:00' AND '2024-01-01 23:59:59'
    AND metric_name = 'trading_order_latency_seconds'
ORDER BY timestamp;

-- Analyze security events
SELECT 
    event_type,
    count() as event_count,
    avg(threat_score) as avg_threat_score
FROM security_events 
WHERE timestamp >= now() - INTERVAL 24 HOUR
GROUP BY event_type
ORDER BY event_count DESC;

-- Anomaly analysis
SELECT 
    metric_name,
    count() as anomaly_count,
    avg(deviation_score) as avg_deviation
FROM anomaly_detections 
WHERE 
    is_anomaly = 1
    AND timestamp >= now() - INTERVAL 7 DAY
GROUP BY metric_name;
```

## 📱 Alerting & Escalation

### Alert Severity Levels

1. **Critical**: Immediate PagerDuty escalation
   - Exchange connectivity loss
   - High error rates (>5%)
   - Significant PnL drawdown
   - Security breaches

2. **Warning**: Slack notifications
   - High latency (>500ms)
   - Low fill rates (<80%)
   - Resource constraints
   - Anomaly detections

3. **Info**: Dashboard notifications
   - Performance metrics
   - Business KPIs
   - System status

### Alert Routing

```yaml
# Risk management alerts - highest priority
- match:
    service: risk
  receiver: 'risk-team-urgent'
  group_wait: 0s
  repeat_interval: 2m

# Trading system alerts
- match:
    service: trading
  receiver: 'trading-team'
  repeat_interval: 30m

# Security alerts
- match_re:
    alertname: '.*Security.*|.*Intrusion.*'
  receiver: 'security-team'
  repeat_interval: 15m
```

## 🔄 Operations

### Daily Operations

```bash
# Check system status
python3 deploy_monitoring.py status

# View recent alerts
curl -s http://localhost:9093/api/v1/alerts | jq '.data[] | select(.status.state=="active")'

# Check metric ingestion
curl -s http://localhost:9090/api/v1/query?query=up | jq '.data.result[] | select(.value[1]=="0")'
```

### Troubleshooting

**Service Not Starting:**
```bash
# Check logs
docker-compose logs -f [service_name]

# Restart specific service
docker-compose restart [service_name]
```

**Missing Metrics:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metric endpoint
curl http://localhost:9999/metrics
```

**Alert Not Firing:**
```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Test alert expression
curl "http://localhost:9090/api/v1/query?query=YOUR_ALERT_EXPRESSION"
```

### Backup & Recovery

```bash
# Backup Grafana dashboards
docker exec grafana grafana-cli admin export-dashboard > dashboards_backup.json

# Backup Prometheus data
docker exec prometheus tar -czf /tmp/prometheus_backup.tar.gz /prometheus

# Backup ClickHouse data
docker exec clickhouse clickhouse-backup create
```

## 📚 API Reference

### Monitoring System Python API

```python
from monitoring.monitoring_system import MonitoringSystem

# Initialize monitoring
monitoring = MonitoringSystem(
    clickhouse_host='localhost',
    clickhouse_port=9000,
    prometheus_port=9999
)

# Record trading events
monitoring.record_trading_event(
    event_type='order_placed',
    exchange='binance',
    symbol='BTC/USDT',
    side='buy',
    amount=0.1,
    price=50000
)

# Record security events
security_event = SecurityEvent(
    timestamp=datetime.utcnow(),
    event_type='suspicious_api_call',
    severity='warning',
    source_ip='192.168.1.100',
    description='Unusual API pattern detected',
    threat_score=0.6,
    action_taken='logged'
)
monitoring.record_security_event(security_event)

# Use decorators for automatic instrumentation
@monitor_execution_time('trading_strategy_execution')
@monitor_errors('trading_strategy_errors')
def execute_arbitrage_strategy():
    # Your trading logic here
    pass
```

### REST API Endpoints

```bash
# Prometheus metrics
GET http://localhost:9999/metrics

# Grafana dashboards
GET http://localhost:3000/api/dashboards/home
POST http://localhost:3000/api/dashboards/db

# Alertmanager alerts
GET http://localhost:9093/api/v1/alerts
POST http://localhost:9093/api/v1/silences

# ClickHouse queries
POST http://localhost:8123/
Content-Type: text/plain
SELECT * FROM metrics_events LIMIT 10
```

## 🎯 Performance Tuning

### Prometheus Optimization

```yaml
# Increase retention for trading data
storage:
  tsdb:
    retention: 90d
    retention-size: 100GB

# Optimize for high cardinality metrics
global:
  scrape_interval: 5s
  evaluation_interval: 5s
```

### ClickHouse Optimization

```xml
<!-- Optimize for time-series data -->
<compression>
  <case>
    <method>lz4</method>
  </case>
</compression>

<!-- Partition by date for efficient queries -->
<partition_by>toYYYYMMDD(timestamp)</partition_by>
<order_by>(timestamp, metric_name, service)</order_by>
```

### Grafana Performance

```bash
# Enable query caching
GF_QUERY_CACHE_ENABLED=true
GF_QUERY_CACHE_TTL=1m

# Optimize dashboard refresh
refresh: 5s  # For real-time dashboards
refresh: 30s  # For overview dashboards
```

## 🔐 Security Considerations

### Network Security
- All services run in isolated Docker network
- TLS encryption for external connections
- API authentication and authorization
- Rate limiting on all endpoints

### Data Protection
- Sensitive metrics are encrypted at rest
- Log data sanitization for PII
- Secure credential management
- Regular security updates

### Access Control
- Role-based access control (RBAC)
- Multi-factor authentication (MFA)
- Audit logging for all access
- Regular access reviews

## 🚀 Advanced Features

### Custom Metrics

```python
# Define custom trading metrics
from prometheus_client import Counter, Histogram, Gauge

# Custom arbitrage metrics
arbitrage_profit = Gauge(
    'arbitrage_profit_usd',
    'Arbitrage profit in USD',
    ['strategy', 'exchange_pair']
)

risk_exposure = Gauge(
    'portfolio_risk_exposure',
    'Current risk exposure',
    ['asset_class', 'strategy']
)

# Update metrics in your trading code
arbitrage_profit.labels(
    strategy='cross_exchange',
    exchange_pair='binance_coinbase'
).set(125.50)
```

### Machine Learning Integration

```python
# Predictive anomaly detection
from sklearn.ensemble import IsolationForest
import numpy as np

# Train on historical data
historical_data = get_historical_metrics('trading_volume', days=30)
model = IsolationForest(contamination=0.1)
model.fit(historical_data)

# Real-time prediction
current_volume = get_current_volume()
anomaly_score = model.decision_function([[current_volume]])[0]

if anomaly_score < -0.5:
    trigger_anomaly_alert(current_volume, anomaly_score)
```

### Multi-Environment Support

```bash
# Deploy to different environments
python3 deploy_monitoring.py deploy --env production
python3 deploy_monitoring.py deploy --env staging
python3 deploy_monitoring.py deploy --env development
```

## 📖 Further Reading

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Design](https://grafana.com/docs/grafana/latest/best-practices/)
- [OpenTelemetry Python Guide](https://opentelemetry.io/docs/instrumentation/python/)
- [ClickHouse Performance Guide](https://clickhouse.com/docs/en/operations/performance/)
- [Jaeger Deployment Guide](https://www.jaegertracing.io/docs/deployment/)

## 🤝 Contributing

To contribute to the monitoring system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This monitoring system is part of the Ultimate Arbitrage Trading System and is proprietary software.

---

*For support, contact the DevOps team or create an issue in the project repository.*

