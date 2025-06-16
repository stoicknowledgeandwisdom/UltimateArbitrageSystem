#!/usr/bin/env python3
"""
Direct API Server Launcher - No reload, immediate start
"""

import uvicorn
from api.simulation_api import app

if __name__ == "__main__":
    print("🚀 Starting Ultimate Investment Simulation API...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/{simulation_id}")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

