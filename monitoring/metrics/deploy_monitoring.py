#!/usr/bin/env python3
"""
Monitoring System Deployment Script

This script deploys and configures the comprehensive monitoring stack:
- Prometheus & Grafana
- Jaeger for tracing
- Loki for logs
- ClickHouse for time-travel debugging
- Security monitoring with Zeek & OSQuery
- Alerting with PagerDuty & Slack
"""

import os
import sys
import time
import json
import yaml
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional


class MonitoringDeployer:
    """Deploy and configure the monitoring stack"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.getcwd())
        self.monitoring_dir = self.base_dir / "monitoring"
        self.docker_compose_file = self.monitoring_dir / "docker-compose.yml"
        
        # Service health check endpoints
        self.health_checks = {
            'prometheus': 'http://localhost:9090/-/ready',
            'grafana': 'http://localhost:3000/api/health',
            'jaeger': 'http://localhost:16686/',
            'loki': 'http://localhost:3100/ready',
            'clickhouse': 'http://localhost:8123/ping',
            'elasticsearch': 'http://localhost:9200/_cluster/health',
            'alertmanager': 'http://localhost:9093/-/ready'
        }
    
    def deploy(self, 
               skip_build: bool = False,
               wait_for_health: bool = True,
               configure_dashboards: bool = True):
        """Deploy the complete monitoring stack"""
        
        print("🚀 Starting monitoring stack deployment...")
        
        # Validate prerequisites
        self._validate_prerequisites()
        
        # Create required directories
        self._create_directories()
        
        # Generate environment configuration
        self._generate_environment_config()
        
        # Deploy with Docker Compose
        self._deploy_docker_stack(skip_build)
        
        if wait_for_health:
            # Wait for services to be healthy
            self._wait_for_services()
        
        # Configure ClickHouse
        self._configure_clickhouse()
        
        if configure_dashboards:
            # Import Grafana dashboards
            self._configure_grafana_dashboards()
        
        # Configure Prometheus rules
        self._configure_prometheus_rules()
        
        # Test alerting
        self._test_alerting()
        
        print("✅ Monitoring stack deployment completed successfully!")
        self._print_access_urls()
    
    def _validate_prerequisites(self):
        """Validate that required tools are available"""
        required_tools = ['docker', 'docker-compose']
        
        for tool in required_tools:
            if not self._command_exists(tool):
                raise RuntimeError(f"Required tool '{tool}' not found. Please install it first.")
        
        # Check Docker daemon
        try:
            subprocess.run(['docker', 'info'], check=True, 
                         capture_output=True, text=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("Docker daemon is not running. Please start Docker first.")
        
        print("✅ Prerequisites validated")
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        try:
            subprocess.run(['which', command], check=True, 
                         capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _create_directories(self):
        """Create required directories for data persistence"""
        directories = [
            'prometheus/data',
            'grafana/data',
            'loki/data',
            'clickhouse/data',
            'elasticsearch/data',
            'alertmanager/data',
            'zeek/logs',
            'osquery/logs'
        ]
        
        for directory in directories:
            dir_path = self.monitoring_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Set appropriate permissions
            os.chmod(dir_path, 0o755)
        
        print("✅ Directories created")
    
    def _generate_environment_config(self):
        """Generate environment configuration file"""
        env_config = {
            # Grafana
            'GF_SECURITY_ADMIN_PASSWORD': 'admin123',
            'GF_USERS_ALLOW_SIGN_UP': 'false',
            
            # ClickHouse
            'CLICKHOUSE_DB': 'monitoring',
            'CLICKHOUSE_USER': 'admin',
            'CLICKHOUSE_PASSWORD': 'clickhouse123',
            
            # Elasticsearch
            'ELASTIC_PASSWORD': 'elastic123',
            'discovery.type': 'single-node',
            'xpack.security.enabled': 'false',
            
            # Alert configuration (replace with actual values)
            'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'PAGERDUTY_ROUTING_KEY': 'YOUR_PAGERDUTY_ROUTING_KEY',
            'PAGERDUTY_RISK_ROUTING_KEY': 'YOUR_PAGERDUTY_RISK_ROUTING_KEY',
            'SMTP_PASSWORD': 'your_smtp_password',
        }
        
        env_file = self.monitoring_dir / '.env'
        with open(env_file, 'w') as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")
        
        print("✅ Environment configuration generated")
    
    def _deploy_docker_stack(self, skip_build: bool = False):
        """Deploy the Docker Compose stack"""
        os.chdir(self.monitoring_dir)
        
        try:
            # Pull latest images
            if not skip_build:
                print("📥 Pulling Docker images...")
                subprocess.run(['docker-compose', 'pull'], check=True)
            
            # Start the stack
            print("🐳 Starting Docker Compose stack...")
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
            print("✅ Docker stack deployed")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to deploy Docker stack: {e}")
    
    def _wait_for_services(self, timeout: int = 300):
        """Wait for all services to be healthy"""
        print("⏳ Waiting for services to be healthy...")
        
        start_time = time.time()
        healthy_services = set()
        
        while time.time() - start_time < timeout:
            for service, url in self.health_checks.items():
                if service in healthy_services:
                    continue
                
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code in [200, 204]:
                        healthy_services.add(service)
                        print(f"✅ {service} is healthy")
                except requests.RequestException:
                    pass
            
            if len(healthy_services) == len(self.health_checks):
                print("✅ All services are healthy")
                return
            
            time.sleep(10)
        
        unhealthy = set(self.health_checks.keys()) - healthy_services
        print(f"⚠️ Some services are not healthy: {unhealthy}")
        print("Continuing with deployment...")
    
    def _configure_clickhouse(self):
        """Configure ClickHouse for time-travel debugging"""
        print("⚙️ Configuring ClickHouse...")
        
        # Wait a bit more for ClickHouse to be fully ready
        time.sleep(10)
        
        try:
            # Create database and tables using the monitoring system
            from monitoring_system import MonitoringSystem
            
            # This will create the necessary tables
            monitoring = MonitoringSystem(
                clickhouse_host='localhost',
                clickhouse_port=9000
            )
            
            print("✅ ClickHouse configured")
            
        except Exception as e:
            print(f"⚠️ ClickHouse configuration failed: {e}")
            print("You may need to configure it manually later")
    
    def _configure_grafana_dashboards(self):
        """Configure Grafana dashboards"""
        print("📊 Configuring Grafana dashboards...")
        
        # Wait for Grafana to be ready
        time.sleep(10)
        
        try:
            # Create trading dashboard
            self._create_trading_dashboard()
            
            # Create security dashboard
            self._create_security_dashboard()
            
            # Create infrastructure dashboard
            self._create_infrastructure_dashboard()
            
            print("✅ Grafana dashboards configured")
            
        except Exception as e:
            print(f"⚠️ Grafana dashboard configuration failed: {e}")
    
    def _create_trading_dashboard(self):
        """Create comprehensive trading dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Trading System Overview",
                "uid": "trading-dashboard",
                "tags": ["trading", "arbitrage"],
                "timezone": "utc",
                "refresh": "5s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "Order Latency",
                        "type": "stat",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, sum(rate(trading_order_latency_seconds_bucket[5m])) by (le))",
                            "legendFormat": "95th percentile"
                        }],
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "s",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 0.1},
                                        {"color": "red", "value": 0.5}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Fill Rate",
                        "type": "stat",
                        "targets": [{
                            "expr": "(sum(rate(orders_filled_total[5m])) / sum(rate(orders_placed_total[5m]))) * 100",
                            "legendFormat": "Fill Rate %"
                        }],
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 70},
                                        {"color": "green", "value": 90}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Current PnL",
                        "type": "stat",
                        "targets": [{
                            "expr": "sum(trading_pnl_current)",
                            "legendFormat": "Total PnL"
                        }],
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "currencyUSD",
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": -1000},
                                        {"color": "yellow", "value": 0},
                                        {"color": "green", "value": 1000}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 4,
                        "title": "Arbitrage Opportunities",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(arbitrage_opportunities_total[1m])",
                            "legendFormat": "Opportunities/min"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    }
                ]
            },
            "overwrite": True
        }
        
        self._post_to_grafana('/api/dashboards/db', dashboard_config)
    
    def _create_security_dashboard(self):
        """Create security monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Security Monitoring",
                "uid": "security-dashboard",
                "tags": ["security", "threats"],
                "timezone": "utc",
                "refresh": "30s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Threat Score Distribution",
                        "type": "histogram",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, threat_score_bucket)",
                            "legendFormat": "95th percentile threat score"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Security Events by Type",
                        "type": "piechart",
                        "targets": [{
                            "expr": "sum by (event_type) (increase(security_events_total[1h]))",
                            "legendFormat": "{{event_type}}"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    }
                ]
            },
            "overwrite": True
        }
        
        self._post_to_grafana('/api/dashboards/db', dashboard_config)
    
    def _create_infrastructure_dashboard(self):
        """Create infrastructure monitoring dashboard"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Infrastructure Overview",
                "uid": "infrastructure-dashboard",
                "tags": ["infrastructure", "system"],
                "timezone": "utc",
                "refresh": "30s",
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
                            "legendFormat": "{{instance}}"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Memory Usage",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                            "legendFormat": "Memory Usage %"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    }
                ]
            },
            "overwrite": True
        }
        
        self._post_to_grafana('/api/dashboards/db', dashboard_config)
    
    def _post_to_grafana(self, endpoint: str, data: dict):
        """Post data to Grafana API"""
        url = f"http://localhost:3000{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Basic YWRtaW46YWRtaW4xMjM='  # admin:admin123
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
    
    def _configure_prometheus_rules(self):
        """Configure Prometheus alerting rules"""
        print("⚙️ Configuring Prometheus rules...")
        
        # Reload Prometheus configuration
        try:
            response = requests.post('http://localhost:9090/-/reload')
            if response.status_code == 200:
                print("✅ Prometheus rules configured")
            else:
                print(f"⚠️ Failed to reload Prometheus: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ Failed to reload Prometheus: {e}")
    
    def _test_alerting(self):
        """Test alerting configuration"""
        print("🧪 Testing alerting configuration...")
        
        # Check Alertmanager status
        try:
            response = requests.get('http://localhost:9093/api/v1/status')
            if response.status_code == 200:
                print("✅ Alertmanager is responding")
            else:
                print(f"⚠️ Alertmanager status check failed: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ Alertmanager test failed: {e}")
    
    def _print_access_urls(self):
        """Print access URLs for all services"""
        print("\n🌐 Service Access URLs:")
        print("=" * 50)
        
        services = {
            'Grafana (Dashboards)': 'http://localhost:3000 (admin/admin123)',
            'Prometheus (Metrics)': 'http://localhost:9090',
            'Jaeger (Tracing)': 'http://localhost:16686',
            'Alertmanager (Alerts)': 'http://localhost:9093',
            'ClickHouse (Analytics)': 'http://localhost:8123',
            'Elasticsearch': 'http://localhost:9200',
        }
        
        for service, url in services.items():
            print(f"📊 {service:25} → {url}")
        
        print("\n📋 Metrics endpoints:")
        print(f"📈 Application metrics    → http://localhost:9999/metrics")
        print(f"🖥️  System metrics        → http://localhost:9100/metrics")
        print(f"🐳 Container metrics      → http://localhost:8080/metrics")
    
    def stop(self):
        """Stop the monitoring stack"""
        print("🛑 Stopping monitoring stack...")
        
        os.chdir(self.monitoring_dir)
        
        try:
            subprocess.run(['docker-compose', 'down'], check=True)
            print("✅ Monitoring stack stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop monitoring stack: {e}")
    
    def restart(self):
        """Restart the monitoring stack"""
        print("🔄 Restarting monitoring stack...")
        self.stop()
        time.sleep(5)
        self.deploy(skip_build=True, wait_for_health=True)
    
    def status(self):
        """Check status of monitoring services"""
        print("📊 Monitoring Stack Status:")
        print("=" * 40)
        
        for service, url in self.health_checks.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 204]:
                    status = "✅ Healthy"
                else:
                    status = f"⚠️ Status {response.status_code}"
            except requests.RequestException:
                status = "❌ Unhealthy"
            
            print(f"{service:15} → {status}")


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy monitoring stack')
    parser.add_argument('action', choices=['deploy', 'stop', 'restart', 'status'],
                       help='Action to perform')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip building/pulling images')
    parser.add_argument('--no-wait', action='store_true',
                       help='Don\'t wait for services to be healthy')
    parser.add_argument('--no-dashboards', action='store_true',
                       help='Skip dashboard configuration')
    parser.add_argument('--base-dir', type=str,
                       help='Base directory for monitoring stack')
    
    args = parser.parse_args()
    
    deployer = MonitoringDeployer(args.base_dir)
    
    try:
        if args.action == 'deploy':
            deployer.deploy(
                skip_build=args.skip_build,
                wait_for_health=not args.no_wait,
                configure_dashboards=not args.no_dashboards
            )
        elif args.action == 'stop':
            deployer.stop()
        elif args.action == 'restart':
            deployer.restart()
        elif args.action == 'status':
            deployer.status()
    
    except KeyboardInterrupt:
        print("\n⏹️ Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

