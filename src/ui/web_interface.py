#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Arbitrage System - Web Interface
========================================

Beautiful, powerful, and intuitive web interface for managing the Ultimate Arbitrage System.
This UI provides complete control over all trading parameters, exchange credentials,
profit optimization, and real-time monitoring with maximum security and ease of use.

Features:
- Responsive dark theme design
- Real-time profit tracking
- Secure credential management
- Advanced strategy configuration
- Live performance analytics
- Risk management controls
- Multi-exchange monitoring
- DeFi integration interface
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import websockets

# Import our configuration manager
sys.path.append(str(Path(__file__).parent.parent / "core"))
from ultimate_config_manager import get_config_manager, UltimateConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with custom configuration
app = FastAPI(
    title="Ultimate Arbitrage System",
    description="The most advanced arbitrage trading system with maximum profit potential",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
active_connections: List[WebSocket] = []
config_manager = get_config_manager()

# Pydantic models for API
class ExchangeCredentials(BaseModel):
    exchange_id: str
    api_key: str
    api_secret: str
    api_passphrase: str = None

class TradingParametersUpdate(BaseModel):
    max_position_size_percent: float = None
    leverage_multiplier: float = None
    enable_arbitrage: bool = None
    enable_grid_trading: bool = None
    enable_scalping: bool = None
    stop_loss_percent: float = None
    take_profit_percent: float = None

class ProfitOptimizationUpdate(BaseModel):
    daily_profit_target: float = None
    enable_compound_growth: bool = None
    enable_defi_strategies: bool = None
    arbitrage_allocation: float = None
    leverage_multiplier: float = None

class MasterPasswordRequest(BaseModel):
    password: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, implement proper JWT token validation
    # For now, we'll use a simple token check
    if credentials.credentials != "ultimate_arbitrage_token":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return "authenticated_user"

# Main dashboard route
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with real-time trading interface"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Ultimate Arbitrage System",
        "version": "1.0.0"
    })

# Configuration page
@app.get("/config", response_class=HTMLResponse)
async def configuration_page(request: Request):
    """Configuration management interface"""
    return templates.TemplateResponse("configuration.html", {
        "request": request,
        "title": "System Configuration"
    })

# Analytics page
@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Advanced analytics and performance tracking"""
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "title": "Performance Analytics"
    })

# API Routes

@app.post("/api/auth/password")
async def set_master_password(password_request: MasterPasswordRequest):
    """Set or verify master password"""
    try:
        success = config_manager.set_master_password(password_request.password)
        if success:
            await manager.broadcast({
                "type": "notification",
                "message": "Master password set successfully",
                "level": "success"
            })
            return {"success": True, "message": "Master password set successfully"}
        else:
            return {"success": False, "message": "Failed to set master password"}
    except Exception as e:
        logger.error(f"Error setting master password: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/ui")
async def get_ui_configuration():
    """Get complete UI configuration"""
    try:
        config = config_manager.get_ui_configuration()
        return config
    except Exception as e:
        logger.error(f"Error getting UI configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/exchanges/credentials")
async def add_exchange_credentials(credentials: ExchangeCredentials):
    """Add or update exchange API credentials"""
    try:
        success = config_manager.add_exchange_credentials(
            credentials.exchange_id,
            credentials.api_key,
            credentials.api_secret,
            credentials.api_passphrase
        )
        
        if success:
            # Save configuration
            config_manager.save_configuration()
            
            # Broadcast update to all connected clients
            await manager.broadcast({
                "type": "exchange_updated",
                "exchange_id": credentials.exchange_id,
                "message": f"Credentials added for {credentials.exchange_id}"
            })
            
            return {"success": True, "message": "Credentials added successfully"}
        else:
            return {"success": False, "message": "Failed to add credentials"}
    except Exception as e:
        logger.error(f"Error adding exchange credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/trading/parameters")
async def update_trading_parameters(params: TradingParametersUpdate):
    """Update trading parameters"""
    try:
        # Convert to dict and filter None values
        updates = {k: v for k, v in params.dict().items() if v is not None}
        
        success = config_manager.update_trading_parameters(**updates)
        
        if success:
            config_manager.save_configuration()
            
            # Broadcast update
            await manager.broadcast({
                "type": "trading_parameters_updated",
                "updates": updates,
                "message": "Trading parameters updated successfully"
            })
            
            return {"success": True, "message": "Trading parameters updated"}
        else:
            return {"success": False, "message": "Failed to update parameters"}
    except Exception as e:
        logger.error(f"Error updating trading parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/profit/optimization")
async def update_profit_optimization(params: ProfitOptimizationUpdate):
    """Update profit optimization settings"""
    try:
        # Convert to dict and filter None values
        updates = {k: v for k, v in params.dict().items() if v is not None}
        
        success = config_manager.update_profit_optimization(**updates)
        
        if success:
            config_manager.save_configuration()
            
            # Broadcast update
            await manager.broadcast({
                "type": "profit_optimization_updated",
                "updates": updates,
                "message": "Profit optimization updated successfully"
            })
            
            return {"success": True, "message": "Profit optimization updated"}
        else:
            return {"success": False, "message": "Failed to update optimization"}
    except Exception as e:
        logger.error(f"Error updating profit optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/earnings-potential")
async def get_earnings_potential():
    """Get earnings potential analysis"""
    try:
        analysis = config_manager.get_earnings_potential_analysis()
        return analysis
    except Exception as e:
        logger.error(f"Error getting earnings potential: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/validation")
async def validate_configuration():
    """Validate current configuration"""
    try:
        issues = config_manager.validate_configuration()
        return issues
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/export")
async def export_configuration():
    """Export configuration as encrypted backup"""
    try:
        config_data = config_manager.export_configuration()
        return {"config_data": config_data, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error exporting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/import")
async def import_configuration(config_data: dict):
    """Import configuration from backup"""
    try:
        success = config_manager.import_configuration(config_data.get("config_data", ""))
        
        if success:
            # Broadcast configuration reload
            await manager.broadcast({
                "type": "configuration_imported",
                "message": "Configuration imported successfully"
            })
            
            return {"success": True, "message": "Configuration imported successfully"}
        else:
            return {"success": False, "message": "Failed to import configuration"}
    except Exception as e:
        logger.error(f"Error importing configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            data = {
                "type": "status_update",
                "timestamp": datetime.now().isoformat(),
                "earnings_potential": config_manager.get_earnings_potential_analysis(),
                "validation": config_manager.validate_configuration()
            }
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

# System status endpoint
@app.get("/api/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        enabled_exchanges = len([ex for ex in config_manager.exchanges.values() if ex.enabled])
        
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "exchanges": {
                "total": len(config_manager.exchanges),
                "enabled": enabled_exchanges,
                "configured": len([ex for ex in config_manager.exchanges.values() if ex.api_key])
            },
            "strategies": {
                "arbitrage": config_manager.trading_params.enable_arbitrage,
                "grid_trading": config_manager.trading_params.enable_grid_trading,
                "scalping": config_manager.trading_params.enable_scalping,
                "defi": config_manager.profit_config.enable_defi_strategies
            },
            "optimization_score": config_manager.get_earnings_potential_analysis()["optimization_score"]
        }
        
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance metrics simulation (would connect to real trading engine)
@app.get("/api/metrics/performance")
async def get_performance_metrics():
    """Get real-time performance metrics"""
    try:
        # Simulate real-time metrics (in production, this would come from trading engine)
        import random
        
        base_return = config_manager.profit_config.daily_profit_target
        current_return = base_return * (0.8 + random.random() * 0.4)  # Variation around target
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "daily_return": round(current_return, 3),
            "weekly_return": round(current_return * 7 * 0.9, 2),
            "monthly_return": round(current_return * 30 * 0.85, 2),
            "total_trades": random.randint(100, 500),
            "successful_trades": random.randint(80, 95),
            "arbitrage_opportunities": random.randint(10, 50),
            "active_positions": random.randint(5, 20),
            "portfolio_value": round(10000 * (1 + current_return/100), 2),
            "unrealized_pnl": round(random.uniform(-100, 500), 2),
            "realized_pnl": round(random.uniform(100, 1000), 2)
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Create static and template directories if they don't exist
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    logger.info("🚀 Starting Ultimate Arbitrage System Web Interface")
    logger.info("🌐 Access the interface at: http://localhost:8000")
    logger.info("📊 API Documentation: http://localhost:8000/api/docs")
    
    uvicorn.run(
        "web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

