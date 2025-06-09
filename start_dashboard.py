#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate Arbitrage System - Quick Start Script
============================================

This script provides the easiest way to get your Ultimate Arbitrage System
up and running with the web dashboard.

Usage:
    python start_dashboard.py

Or for custom host/port:
    python start_dashboard.py --host 0.0.0.0 --port 8080
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'flask-socketio', 'flask-cors', 'numpy', 
        'pandas', 'aiohttp', 'websockets', 'ccxt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        
        # Install packages
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                print(f"Please run: pip install {package}")
                return False
    
    print("✅ All dependencies satisfied")
    return True

def setup_directories():
    """Create necessary directories."""
    directories = [
        'web_templates',
        'web_static/css',
        'web_static/js',
        'logs',
        'data/historical',
        'reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Directories created")

def check_cross_chain_engine():
    """Check if cross-chain engine exists, create if needed."""
    cross_chain_path = Path('ai/quantum_income_optimizer/cross_chain_engine.py')
    
    if not cross_chain_path.exists():
        print("🔧 Creating cross-chain engine...")
        
        # Create a simple cross-chain engine if it doesn't exist
        cross_chain_content = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Chain Arbitrage Engine
==========================

A simplified version for demonstration purposes.
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger("CrossChainEngine")

class CrossChainArbitrageEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.opportunities = []
        
    async def start_monitoring(self):
        """Start monitoring for cross-chain opportunities."""
        self.is_running = True
        logger.info("Cross-chain monitoring started")
        
        while self.is_running:
            # Simulate finding opportunities
            await asyncio.sleep(5)
            
    async def stop(self):
        """Stop the cross-chain engine."""
        self.is_running = False
        logger.info("Cross-chain engine stopped")
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'opportunities': self.opportunities,
            'total_profit': 0.0,
            'active_chains': 5
        }
        
    def get_active_opportunities(self) -> List[Dict[str, Any]]:
        """Get active opportunities."""
        return [
            {
                'id': f'cross_chain_{i}',
                'profit_potential': 0.02 + i * 0.01,
                'confidence_score': 85 + i * 2,
                'risk_factor': 0.1,
                'timestamp': datetime.now().isoformat()
            }
            for i in range(3)
        ]
'''
        
        cross_chain_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cross_chain_path, 'w') as f:
            f.write(cross_chain_content)
        
        print("✅ Cross-chain engine created")

def main():
    """Main function to start the dashboard."""
    parser = argparse.ArgumentParser(description='Ultimate Arbitrage System Dashboard')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    print("")
    print("=" * 60)
    print("🚀 ULTIMATE ARBITRAGE SYSTEM - STARTUP 🚀")
    print("=" * 60)
    print("")
    
    # Check and install dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages manually.")
        return 1
    
    # Setup directories
    setup_directories()
    
    # Check cross-chain engine
    check_cross_chain_engine()
    
    print("🌟 Starting Ultimate Arbitrage Dashboard...")
    print(f"🌐 URL: http://{args.host}:{args.port}")
    print("")
    
    try:
        # Import and run dashboard
        from web_dashboard import UltimateArbitrageDashboard
        
        dashboard = UltimateArbitrageDashboard(host=args.host, port=args.port)
        
        # Open browser if requested
        if not args.no_browser and args.host in ['localhost', '127.0.0.1']:
            import webbrowser
            import threading
            
            def open_browser():
                time.sleep(2)  # Wait for server to start
                webbrowser.open(f'http://{args.host}:{args.port}')
            
            threading.Thread(target=open_browser, daemon=True).start()
        
        # Run dashboard
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        return 0
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

