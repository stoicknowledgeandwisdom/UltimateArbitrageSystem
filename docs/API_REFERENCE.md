# 📡 Ultimate Arbitrage Empire - API Reference

Complete API documentation for the Ultimate Arbitrage Empire system.

## 🎯 Table of Contents

1. [Core Classes](#core-classes)
2. [Data Structures](#data-structures)
3. [Maximum Income Optimizer](#maximum-income-optimizer)
4. [Advanced Arbitrage Engine](#advanced-arbitrage-engine)
5. [Predictive Market Intelligence](#predictive-market-intelligence)
6. [Quantum Optimizer](#quantum-optimizer)
7. [AI Strategy Engine](#ai-strategy-engine)
8. [Utility Functions](#utility-functions)
9. [Configuration](#configuration)
10. [Error Handling](#error-handling)

## 🔑 **Authentication**

### API Key Authentication
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.ultimatearbitrage.com/v1/status
```

### JWT Token Authentication
```bash
# Get token
curl -X POST https://api.ultimatearbitrage.com/v1/auth/login \
     -d '{"username":"user","password":"pass"}'

# Use token
curl -H "Authorization: Bearer JWT_TOKEN" \
     https://api.ultimatearbitrage.com/v1/portfolio
```

---

## 📊 **Core Endpoints**

### System Status

#### GET `/v1/status`
Retrieve system health and status information.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": 86400,
  "active_strategies": 12,
  "total_exchanges": 15,
  "latency_ms": 0.8
}
```

#### GET `/v1/metrics`
Get comprehensive system metrics.

**Response:**
```json
{
  "performance": {
    "orders_per_second": 1250,
    "avg_latency_ms": 0.6,
    "success_rate": 0.9987
  },
  "financial": {
    "daily_profit_usd": 2500.45,
    "total_volume_24h": 50000.00,
    "roi_percentage": 15.7
  }
}
```

---

## 💼 **Portfolio Management**

### GET `/v1/portfolio`
Retrieve current portfolio status.

**Response:**
```json
{
  "total_value_usd": 100000.00,
  "available_balance": 25000.00,
  "positions": [
    {
      "symbol": "BTC",
      "amount": 1.5,
      "value_usd": 75000.00,
      "exchange": "binance"
    }
  ]
}
```

### POST `/v1/portfolio/rebalance`
Trigger portfolio rebalancing.

**Request:**
```json
{
  "strategy": "conservative",
  "max_risk_percentage": 5.0,
  "target_allocation": {
    "BTC": 0.6,
    "ETH": 0.3,
    "USDT": 0.1
  }
}
```

---

## ⚡ **Trading Operations**

### GET `/v1/strategies`
List all available trading strategies.

**Response:**
```json
{
  "strategies": [
    {
      "id": "triangular_arbitrage",
      "name": "Triangular Arbitrage",
      "status": "active",
      "profit_24h": 1250.00,
      "risk_level": "low"
    }
  ]
}
```

### POST `/v1/strategies/{id}/start`
Start a specific trading strategy.

**Request:**
```json
{
  "capital_allocation": 10000.00,
  "risk_parameters": {
    "max_position_size": 0.1,
    "stop_loss_percentage": 2.0
  }
}
```

### POST `/v1/orders`
Place a manual order.

**Request:**
```json
{
  "exchange": "binance",
  "symbol": "BTCUSDT",
  "side": "buy",
  "type": "market",
  "quantity": 0.001,
  "strategy_id": "manual"
}
```

---

## 📈 **Market Data**

### GET `/v1/markets/prices`
Get real-time price data across exchanges.

**Query Parameters:**
- `symbols`: Comma-separated list of symbols (e.g., `BTC,ETH,ADA`)
- `exchanges`: Comma-separated list of exchanges

**Response:**
```json
{
  "timestamp": 1672531200000,
  "prices": {
    "BTC": {
      "binance": 50000.00,
      "coinbase": 50025.00,
      "kraken": 49980.00
    }
  }
}
```

### GET `/v1/markets/opportunities`
Get current arbitrage opportunities.

**Response:**
```json
{
  "opportunities": [
    {
      "symbol": "BTC",
      "buy_exchange": "kraken",
      "sell_exchange": "coinbase",
      "profit_percentage": 0.09,
      "profit_usd": 45.00,
      "confidence": 0.95
    }
  ]
}
```

---

## 🔍 **Analytics & Reporting**

### GET `/v1/analytics/performance`
Get performance analytics.

**Query Parameters:**
- `period`: `1h`, `24h`, `7d`, `30d`
- `strategy_id`: Filter by specific strategy

**Response:**
```json
{
  "period": "24h",
  "total_profit_usd": 2500.45,
  "total_trades": 1250,
  "win_rate": 0.87,
  "sharpe_ratio": 2.15,
  "max_drawdown": 0.03
}
```

### GET `/v1/analytics/risks`
Get risk analysis.

**Response:**
```json
{
  "overall_risk_score": 2.1,
  "var_95": 1500.00,
  "exposure_by_exchange": {
    "binance": 0.4,
    "coinbase": 0.35,
    "kraken": 0.25
  }
}
```

---

## 🚑 **Emergency Controls**

### POST `/v1/emergency/stop`
Emergency stop all trading activities.

**Request:**
```json
{
  "reason": "market_volatility",
  "close_positions": true,
  "cancel_orders": true
}
```

### POST `/v1/emergency/resume`
Resume trading after emergency stop.

**Request:**
```json
{
  "confirmation_code": "RESUME_TRADING_12345",
  "restart_strategies": ["triangular_arbitrage"]
}
```

---

## 📶 **WebSocket API**

### Connection
```javascript
const ws = new WebSocket('wss://api.ultimatearbitrage.com/v1/ws');

// Authentication
ws.send(JSON.stringify({
  "type": "auth",
  "token": "YOUR_JWT_TOKEN"
}));
```

### Subscriptions

#### Price Updates
```javascript
ws.send(JSON.stringify({
  "type": "subscribe",
  "channel": "prices",
  "symbols": ["BTC", "ETH"]
}));
```

#### Order Updates
```javascript
ws.send(JSON.stringify({
  "type": "subscribe",
  "channel": "orders"
}));
```

#### Arbitrage Opportunities
```javascript
ws.send(JSON.stringify({
  "type": "subscribe",
  "channel": "opportunities",
  "min_profit_percentage": 0.05
}));
```

### Message Format
```javascript
// Price update
{
  "type": "price_update",
  "timestamp": 1672531200000,
  "symbol": "BTC",
  "exchange": "binance",
  "price": 50000.00
}

// Order update
{
  "type": "order_update",
  "order_id": "12345",
  "status": "filled",
  "filled_quantity": 0.001,
  "avg_price": 50000.00
}
```

---

## 🛡️ **Rate Limits**

| Endpoint Category | Rate Limit | Burst Limit |
|------------------|------------|-------------|
| **Authentication** | 10/min | 20 |
| **Trading** | 100/min | 200 |
| **Market Data** | 1000/min | 2000 |
| **Analytics** | 60/min | 120 |
| **WebSocket** | 500 msgs/min | 1000 |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1672531260
```

---

## ⚡ **Error Handling**

### Standard Error Response
```json
{
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient balance for this operation",
    "details": {
      "required": 1000.00,
      "available": 500.00
    },
    "request_id": "req_12345"
  }
}
```

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **429**: Rate Limited
- **500**: Internal Server Error
- **503**: Service Unavailable

---

## 📚 **SDKs & Libraries**

### Python SDK
```bash
pip install ultimatearbitrage-sdk
```

```python
from ultimatearbitrage import Client

client = Client(api_key="your_key")
status = client.get_status()
print(f"System status: {status['status']}")
```

### JavaScript SDK
```bash
npm install @ultimatearbitrage/sdk
```

```javascript
const { UltimateArbitrageClient } = require('@ultimatearbitrage/sdk');

const client = new UltimateArbitrageClient({
  apiKey: 'your_key'
});

const portfolio = await client.getPortfolio();
console.log('Portfolio value:', portfolio.total_value_usd);
```

---

## 🔧 **Testing**

### Sandbox Environment
**Base URL**: `https://sandbox-api.ultimatearbitrage.com/v1`

### Test Credentials
```json
{
  "api_key": "test_key_12345",
  "secret": "test_secret_67890"
}
```

### Postman Collection
Download our [Postman Collection](./api-collection.json) for quick testing.

---

## 📞 **Support**

- **API Documentation**: [docs.ultimatearbitrage.com](https://docs.ultimatearbitrage.com)
- **Developer Support**: [dev-support@ultimatearbitrage.com](mailto:dev-support@ultimatearbitrage.com)
- **Status Page**: [status.ultimatearbitrage.com](https://status.ultimatearbitrage.com)
- **Discord Community**: [discord.gg/ultimatearbitrage](https://discord.gg/ultimatearbitrage)

---

*Last Updated: June 16, 2025*  
*API Version: v1.0*  
*Documentation Version: 2.0*

