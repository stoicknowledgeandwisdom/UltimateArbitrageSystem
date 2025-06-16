#!/usr/bin/env python3
"""
Quick API Health Test
"""

import requests
import json
import sys
from datetime import datetime

def test_api_health():
    """Quick test of API health endpoint"""
    api_url = "http://localhost:8000"
    
    print("🚀 Quick API Test...")
    print(f"Testing: {api_url}")
    print()
    
    try:
        # Test health endpoint
        response = requests.get(f"{api_url}/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Health Check PASSED!")
            print(f"   Status: {health_data['status']}")
            print(f"   Active Simulations: {health_data['active_simulations']}")
            print(f"   WebSocket Connections: {health_data['websocket_connections']}")
            print(f"   Timestamp: {health_data['timestamp']}")
            return True
        else:
            print(f"❌ API Health Check FAILED: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: API server is not running")
        print("💡 Start the server first with: python start_api.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_test_simulation():
    """Create a test simulation"""
    api_url = "http://localhost:8000"
    
    print("\n📊 Creating test simulation...")
    
    config = {
        "name": "Quick Test Portfolio",
        "initial_capital": 50000.0,
        "risk_tolerance": 0.6,
        "quantum_enabled": True,
        "ai_strategies": ["momentum", "mean_reversion"],
        "assets": ["AAPL", "GOOGL", "MSFT"],
        "max_position_size": 0.2,
        "stop_loss": 0.05,
        "take_profit": 0.15
    }
    
    try:
        response = requests.post(f"{api_url}/simulations", json=config, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            sim_id = result['simulation_id']
            print(f"✅ Simulation created successfully!")
            print(f"   ID: {sim_id}")
            print(f"   Message: {result['message']}")
            return sim_id
        else:
            print(f"❌ Simulation creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating simulation: {e}")
        return None

def run_simulation_step(sim_id):
    """Run a single simulation step"""
    api_url = "http://localhost:8000"
    
    print(f"\n🔄 Running simulation step...")
    
    try:
        response = requests.post(f"{api_url}/simulations/{sim_id}/step", timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Simulation step completed!")
            print(f"   Portfolio Value: ${result['portfolio_value']:,.2f}")
            print(f"   Total Return: {result['total_return']:+.2f}%")
            print(f"   AI Confidence: {result['ai_confidence']:.1%}")
            print(f"   Step: {result['step']}")
            return True
        else:
            print(f"❌ Simulation step failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error running simulation step: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("🎯 QUICK API TEST")
    print("="*50)
    
    # Test API health
    if test_api_health():
        # Create simulation
        sim_id = create_test_simulation()
        
        if sim_id:
            # Run simulation step
            run_simulation_step(sim_id)
    
    print("\n🎉 Quick test completed!")

