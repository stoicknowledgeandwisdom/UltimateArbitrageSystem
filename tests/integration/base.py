"""Integration test base infrastructure with Testcontainers and LocalStack"""

import pytest
import asyncio
import docker
import time
import requests
import json
from typing import Dict, List, Any, Optional
from testcontainers.localstack import LocalStackContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.kafka import KafkaContainer
from testcontainers.compose import DockerCompose
import boto3
from contextlib import contextmanager
import logging
from datetime import datetime


class IntegrationTestBase:
    """Base class for integration tests with container management"""
    
    @pytest.fixture(scope="session")
    def localstack_container(self):
        """LocalStack container for AWS services"""
        with LocalStackContainer(
            image="localstack/localstack:latest",
            services=["s3", "sqs", "sns", "dynamodb", "lambda", "cloudwatch"]
        ) as localstack:
            # Wait for LocalStack to be ready
            time.sleep(10)
            yield localstack
    
    @pytest.fixture(scope="session")
    def postgres_container(self):
        """PostgreSQL container for database testing"""
        with PostgresContainer(
            image="postgres:15",
            username="test",
            password="test",
            dbname="arbitrage_test"
        ) as postgres:
            yield postgres
    
    @pytest.fixture(scope="session")
    def redis_container(self):
        """Redis container for caching and pub/sub testing"""
        with RedisContainer(image="redis:7") as redis:
            yield redis
    
    @pytest.fixture(scope="session")
    def kafka_container(self):
        """Kafka container for message streaming testing"""
        with KafkaContainer(image="confluentinc/cp-kafka:latest") as kafka:
            yield kafka
    
    @pytest.fixture(scope="session")
    def exchange_simulator_container(self):
        """Custom exchange simulator container"""
        dockerfile_content = """
        FROM python:3.11-slim
        
        WORKDIR /app
        
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        
        COPY exchange_simulator.py .
        
        EXPOSE 8080
        
        CMD ["python", "exchange_simulator.py"]
        """
        
        requirements_content = """
        fastapi==0.104.1
        uvicorn==0.24.0
        websockets==12.0
        pydantic==2.5.0
        numpy==1.24.3
        """
        
        simulator_code = self._get_exchange_simulator_code()
        
        # Create temporary directory with Dockerfile and code
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write Dockerfile
            with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
                f.write(dockerfile_content)
            
            # Write requirements
            with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                f.write(requirements_content)
            
            # Write simulator code
            with open(os.path.join(temp_dir, "exchange_simulator.py"), "w") as f:
                f.write(simulator_code)
            
            # Build and run container
            client = docker.from_env()
            image, _ = client.images.build(path=temp_dir, tag="exchange-simulator:test")
            
            container = client.containers.run(
                image.id,
                ports={'8080/tcp': None},
                detach=True,
                remove=True
            )
            
            # Wait for container to be ready
            time.sleep(5)
            
            try:
                yield container
            finally:
                container.stop()
    
    def _get_exchange_simulator_code(self) -> str:
        """Exchange simulator implementation"""
        return '''
import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np

app = FastAPI(title="Exchange Simulator")

# Global state
orderbooks = {}
balances = {}
orders = {}
trades = []
active_websockets = []

# Initialize mock data
def initialize_data():
    global orderbooks, balances
    
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
    
    for symbol in symbols:
        base_price = random.uniform(100, 50000)
        spread = base_price * 0.001
        
        bids = []
        asks = []
        
        for i in range(20):
            bid_price = base_price - spread - (i * spread * 0.1)
            ask_price = base_price + spread + (i * spread * 0.1)
            
            bids.append([bid_price, random.uniform(0.1, 10.0)])
            asks.append([ask_price, random.uniform(0.1, 10.0)])
        
        orderbooks[symbol] = {
            "bids": bids,
            "asks": asks,
            "timestamp": time.time()
        }
    
    # Initialize balances
    balances["test_user"] = {
        "BTC": {"free": 10.0, "used": 0.0},
        "ETH": {"free": 100.0, "used": 0.0},
        "USDT": {"free": 100000.0, "used": 0.0}
    }

@app.on_event("startup")
async def startup_event():
    initialize_data()
    # Start background task to update orderbooks
    asyncio.create_task(update_orderbooks())

async def update_orderbooks():
    """Continuously update orderbook data"""
    while True:
        await asyncio.sleep(0.1)  # Update every 100ms
        
        for symbol in orderbooks:
            # Small random price movements
            price_change = random.uniform(-0.001, 0.001)
            
            for bid in orderbooks[symbol]["bids"]:
                bid[0] *= (1 + price_change)
                bid[1] = max(0.1, bid[1] + random.uniform(-0.1, 0.1))
            
            for ask in orderbooks[symbol]["asks"]:
                ask[0] *= (1 + price_change)
                ask[1] = max(0.1, ask[1] + random.uniform(-0.1, 0.1))
            
            orderbooks[symbol]["timestamp"] = time.time()
        
        # Broadcast to websocket clients
        if active_websockets:
            message = {
                "type": "orderbook_update",
                "data": orderbooks
            }
            for ws in active_websockets.copy():
                try:
                    await ws.send_text(json.dumps(message))
                except:
                    active_websockets.remove(ws)

@app.get("/api/v1/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    if symbol not in orderbooks:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return orderbooks[symbol]

@app.get("/api/v1/balance")
async def get_balance(api_key: str = "test_key"):
    if api_key not in balances:
        balances[api_key] = {
            "BTC": {"free": 0.0, "used": 0.0},
            "ETH": {"free": 0.0, "used": 0.0},
            "USDT": {"free": 1000.0, "used": 0.0}
        }
    return balances[api_key]

@app.post("/api/v1/order")
async def place_order(order_data: dict):
    order_id = f"order_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    order = {
        "id": order_id,
        "symbol": order_data["symbol"],
        "side": order_data["side"],
        "amount": order_data["amount"],
        "price": order_data.get("price"),
        "type": order_data.get("type", "market"),
        "status": "open",
        "filled": 0.0,
        "remaining": order_data["amount"],
        "timestamp": time.time()
    }
    
    orders[order_id] = order
    
    # Simulate partial or full fill
    if random.random() > 0.3:  # 70% chance of immediate fill
        fill_amount = order["amount"] * random.uniform(0.1, 1.0)
        order["filled"] = fill_amount
        order["remaining"] = order["amount"] - fill_amount
        
        if order["remaining"] < 0.001:
            order["status"] = "closed"
        else:
            order["status"] = "partially_filled"
        
        # Create trade record
        trade = {
            "id": f"trade_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "order_id": order_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "amount": fill_amount,
            "price": order["price"] or orderbooks[order["symbol"]]["asks"][0][0],
            "timestamp": time.time(),
            "fee": fill_amount * 0.001
        }
        trades.append(trade)
    
    return order

@app.get("/api/v1/order/{order_id}")
async def get_order(order_id: str):
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Order not found")
    return orders[order_id]

@app.get("/api/v1/trades")
async def get_trades(symbol: str = None, limit: int = 100):
    filtered_trades = trades
    if symbol:
        filtered_trades = [t for t in trades if t["symbol"] == symbol]
    return filtered_trades[-limit:]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
    
    @pytest.fixture
    def aws_credentials(self, localstack_container):
        """AWS credentials for LocalStack"""
        return {
            'aws_access_key_id': 'test',
            'aws_secret_access_key': 'test',
            'region_name': 'us-east-1',
            'endpoint_url': localstack_container.get_url()
        }
    
    @pytest.fixture
    def s3_client(self, aws_credentials):
        """S3 client for LocalStack"""
        return boto3.client('s3', **aws_credentials)
    
    @pytest.fixture
    def dynamodb_client(self, aws_credentials):
        """DynamoDB client for LocalStack"""
        return boto3.client('dynamodb', **aws_credentials)
    
    @pytest.fixture
    def sqs_client(self, aws_credentials):
        """SQS client for LocalStack"""
        return boto3.client('sqs', **aws_credentials)
    
    @contextmanager
    def exchange_environment(self, exchanges: List[str] = None):
        """Context manager for spinning up exchange test environment"""
        if exchanges is None:
            exchanges = ['binance', 'coinbase', 'kraken']
        
        containers = []
        
        try:
            # Start exchange simulator containers
            for exchange in exchanges:
                container = self._start_exchange_simulator(exchange)
                containers.append(container)
                time.sleep(2)  # Wait for startup
            
            yield {
                exchange: f"http://localhost:{self._get_container_port(container)}"
                for exchange, container in zip(exchanges, containers)
            }
        
        finally:
            # Cleanup containers
            for container in containers:
                try:
                    container.stop()
                    container.remove()
                except:
                    pass
    
    def _start_exchange_simulator(self, exchange_name: str):
        """Start exchange simulator container"""
        client = docker.from_env()
        
        # Use the exchange simulator image we built
        container = client.containers.run(
            "exchange-simulator:test",
            ports={'8080/tcp': None},
            detach=True,
            remove=True,
            environment={
                'EXCHANGE_NAME': exchange_name,
                'SIMULATION_MODE': 'true'
            }
        )
        
        return container
    
    def _get_container_port(self, container) -> int:
        """Get the mapped port for a container"""
        container.reload()
        port_info = container.attrs['NetworkSettings']['Ports']['8080/tcp']
        if port_info:
            return int(port_info[0]['HostPort'])
        return 8080
    
    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to be available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(1)
        
        return False
    
    def create_test_database_schema(self, postgres_container):
        """Create test database schema"""
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            port=postgres_container.get_exposed_port(5432),
            database="arbitrage_test",
            user="test",
            password="test"
        )
        
        cursor = conn.cursor()
        
        # Create tables for testing
        schema_sql = """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            amount DECIMAL(20, 8) NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exchange VARCHAR(50) NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS orders (
            id VARCHAR(100) PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            amount DECIMAL(20, 8) NOT NULL,
            price DECIMAL(20, 8),
            status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exchange VARCHAR(50) NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            buy_exchange VARCHAR(50) NOT NULL,
            sell_exchange VARCHAR(50) NOT NULL,
            buy_price DECIMAL(20, 8) NOT NULL,
            sell_price DECIMAL(20, 8) NOT NULL,
            profit_percentage DECIMAL(10, 4) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(schema_sql)
        conn.commit()
        cursor.close()
        conn.close()


class ExchangeTestEnvironment:
    """Manager for exchange test environments"""
    
    def __init__(self):
        self.containers = {}
        self.client = docker.from_env()
    
    def start_exchange_cluster(self, exchanges: List[str]) -> Dict[str, str]:
        """Start a cluster of exchange simulators"""
        endpoints = {}
        
        for exchange in exchanges:
            container = self._start_exchange_container(exchange)
            self.containers[exchange] = container
            
            # Wait for container to be ready
            port = self._get_container_port(container)
            endpoint = f"http://localhost:{port}"
            
            if self._wait_for_endpoint(endpoint):
                endpoints[exchange] = endpoint
            else:
                raise RuntimeError(f"Failed to start {exchange} simulator")
        
        return endpoints
    
    def _start_exchange_container(self, exchange_name: str):
        """Start individual exchange container"""
        return self.client.containers.run(
            "exchange-simulator:test",
            ports={'8080/tcp': None},
            detach=True,
            remove=True,
            environment={
                'EXCHANGE_NAME': exchange_name,
                'INITIAL_BALANCE': '100000',
                'PRICE_VOLATILITY': '0.01'
            }
        )
    
    def _get_container_port(self, container) -> int:
        """Get mapped port for container"""
        container.reload()
        port_info = container.attrs['NetworkSettings']['Ports']['8080/tcp']
        if port_info:
            return int(port_info[0]['HostPort'])
        return 8080
    
    def _wait_for_endpoint(self, endpoint: str, timeout: int = 30) -> bool:
        """Wait for endpoint to be available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{endpoint}/api/v1/balance", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def cleanup(self):
        """Stop and remove all containers"""
        for container in self.containers.values():
            try:
                container.stop()
                container.remove()
            except:
                pass
        self.containers.clear()

