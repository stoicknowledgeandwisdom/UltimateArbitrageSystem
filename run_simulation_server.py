#!/usr/bin/env python3
"""
Ultimate Investment Simulation Server Launcher
Starts the FastAPI server for real-money simulation with quantum AI optimization
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("" + "="*60)
    print("🚀 ULTIMATE INVESTMENT SIMULATION SERVER")
    print("="*60)
    print()
    print("🎯 Features:")
    print("   ✅ Quantum Portfolio Optimization")
    print("   ✅ AI-Powered Trading Strategies")
    print("   ✅ Real-time Risk Management")
    print("   ✅ WebSocket Live Updates")
    print("   ✅ Comprehensive Analytics")
    print("   ✅ Multi-Asset Support")
    print("   ✅ Advanced Order Management")
    print()
    print("🌐 Server will be available at: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/{simulation_id}")
    print()
    print("💡 To test the API, run: python test_simulation_api.py")
    print()
    print("🎉 Starting server...")
    print("="*60)
    
    try:
        # Start the server with import string for reload functionality
        uvicorn.run(
            "api.simulation_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            access_log=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\n💡 Make sure all dependencies are installed:")
        print("   pip install fastapi uvicorn websockets pandas numpy")
        sys.exit(1)

if __name__ == "__main__":
    main()

