#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Button Test Dashboard - Verify all buttons work correctly
"""

import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os

app = Flask(__name__, 
           template_folder='web_templates',
           static_folder='web_static',
           static_url_path='/static')
app.secret_key = 'test_key'

# Enable CORS and SocketIO
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Test state
test_state = {
    'system_running': False,
    'wallets': {},
    'strategies': {},
    'start_time': None
}

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/system/status')
def system_status():
    """Get current system status."""
    return jsonify({
        'running': test_state['system_running'],
        'uptime': get_uptime(),
        'components': {
            'arbitrage_system': True,
            'quantum_engine': True,
            'cross_chain_engine': True
        },
        'wallets_connected': len(test_state['wallets']),
        'active_strategies': len(test_state['strategies'])
    })

@app.route('/api/wallets/connect', methods=['POST'])
def connect_wallet():
    """Connect a new wallet."""
    data = request.json
    wallet_id = data.get('wallet_id')
    private_key = data.get('private_key')
    network = data.get('network', 'ethereum')
    
    if not wallet_id or not private_key:
        return jsonify({'error': 'Wallet ID and private key required'}), 400
    
    # Store wallet
    test_state['wallets'][wallet_id] = {
        'network': network,
        'private_key': private_key,
        'connected_at': datetime.now().isoformat(),
        'balance': 5000.0  # Test balance
    }
    
    print(f"✅ BUTTON TEST: Wallet '{wallet_id}' connected successfully!")
    
    return jsonify({
        'success': True,
        'wallet_id': wallet_id,
        'network': network,
        'balance': 5000.0
    })

@app.route('/api/wallets/list')
def list_wallets():
    """List all connected wallets."""
    wallet_list = []
    for wallet_id, wallet_data in test_state['wallets'].items():
        wallet_list.append({
            'wallet_id': wallet_id,
            'network': wallet_data['network'],
            'balance': wallet_data['balance'],
            'connected_at': wallet_data['connected_at']
        })
    return jsonify(wallet_list)

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the arbitrage system."""
    try:
        if test_state['system_running']:
            return jsonify({'error': 'System is already running'}), 400
        
        test_state['system_running'] = True
        test_state['start_time'] = datetime.now()
        
        print("🚀 BUTTON TEST: System started successfully!")
        
        return jsonify({
            'success': True,
            'message': 'Ultimate Arbitrage System started successfully',
            'started_at': test_state['start_time'].isoformat()
        })
        
    except Exception as e:
        print(f"❌ BUTTON TEST: Error starting system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Stop the arbitrage system."""
    try:
        if not test_state['system_running']:
            return jsonify({'error': 'System is not running'}), 400
        
        test_state['system_running'] = False
        
        print("🛑 BUTTON TEST: System stopped successfully!")
        
        return jsonify({
            'success': True,
            'message': 'System stopped successfully'
        })
        
    except Exception as e:
        print(f"❌ BUTTON TEST: Error stopping system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance/metrics')
def performance_metrics():
    """Get real-time performance metrics."""
    # Return test metrics
    profit = 150.75 if test_state['system_running'] else 0.0
    return jsonify({
        'total_profit': profit,
        'win_rate': 85.5,
        'active_opportunities': 3 if test_state['system_running'] else 0,
        'successful_trades': 12,
        'failed_trades': 2,
        'quantum_advantage': 2.3,
        'cross_chain_opportunities': 1
    })

@app.route('/api/opportunities/live')
def live_opportunities():
    """Get live arbitrage opportunities."""
    if not test_state['system_running']:
        return jsonify([])
    
    # Return test opportunities
    return jsonify([
        {
            'type': 'quantum',
            'profit_potential': 0.025,
            'confidence_score': 92.5,
            'risk_factor': 0.015
        },
        {
            'type': 'cross_chain',
            'profit_potential': 0.018,
            'confidence_score': 88.2,
            'risk_factor': 0.022
        }
    ])

@app.route('/api/strategies/deploy', methods=['POST'])
def deploy_strategy():
    """Deploy a new trading strategy."""
    data = request.json
    strategy_type = data.get('strategy_type')
    wallet_id = data.get('wallet_id')
    capital_amount = data.get('capital_amount', 0)
    
    if not strategy_type or not wallet_id:
        return jsonify({'error': 'Strategy type and wallet ID required'}), 400
    
    if wallet_id not in test_state['wallets']:
        return jsonify({'error': 'Wallet not connected'}), 400
    
    # Deploy strategy
    strategy_id = f"{strategy_type}_{int(time.time())}"
    test_state['strategies'][strategy_id] = {
        'type': strategy_type,
        'wallet_id': wallet_id,
        'capital': capital_amount,
        'deployed_at': datetime.now().isoformat(),
        'status': 'active',
        'profit': 25.50  # Test profit
    }
    
    print(f"🎯 BUTTON TEST: Strategy '{strategy_type}' deployed successfully!")
    
    return jsonify({
        'success': True,
        'strategy_id': strategy_id,
        'message': f'Strategy {strategy_type} deployed successfully'
    })

@app.route('/api/strategies/list')
def list_strategies():
    """List all active strategies."""
    return jsonify(list(test_state['strategies'].values()))

@socketio.on('connect')
def handle_connect():
    print('✅ BUTTON TEST: Client connected to real-time feed')
    emit('status', {'message': 'Connected to Ultimate Arbitrage System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ BUTTON TEST: Client disconnected from real-time feed')

def get_uptime():
    """Get system uptime."""
    if not test_state['start_time']:
        return "Not started"
    
    uptime = datetime.now() - test_state['start_time']
    hours, remainder = divmod(uptime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 BUTTON TEST DASHBOARD 🧪")
    print("="*60)
    print(f"\n🌐 Access test dashboard at: http://localhost:8080")
    print("\n📋 Button Test Instructions:")
    print("   1. Click 'Connect Wallet' - Should show success notification")
    print("   2. Click 'Deploy Strategies' - Should deploy automatically")
    print("   3. Click 'Start System' - Should show system online")
    print("   4. Click 'Stop System' - Should show system offline")
    print("\n🔍 Watch console for button test confirmations!")
    print("\n" + "="*60)
    
    try:
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        print(f"Error running test dashboard: {e}")
        raise

