#!/usr/bin/env python3
"""
Ultimate Income UI Dashboard
Comprehensive Web-Based Control Center for Maximum Income Generation
Designed for Easy Setup, Real-time Monitoring, and Maximum Profit Control
"""

import flask
import json
import sqlite3
import asyncio
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import logging
import os
from ultimate_maximum_income_engine import UltimateMaximumIncomeEngine
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('UltimateIncomeUI')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ultimate_income_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global engine instance
global_engine = None
engine_thread = None

# HTML Template for the Ultimate Income Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultimate Maximum Income Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #FFD700, #FF6B35);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
        }
        
        .status-card h3 {
            color: #FFD700;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #4CAF50;
        }
        
        .control-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .control-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .btn-success {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #2196F3, #0b7dda);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(45deg, #FF9800, #f57c00);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .opportunities-section {
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .opportunities-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .opportunities-table th,
        .opportunities-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .opportunities-table th {
            background: rgba(255,255,255,0.1);
            color: #FFD700;
        }
        
        .profit-positive {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .confidence-high {
            color: #4CAF50;
        }
        
        .confidence-medium {
            color: #FF9800;
        }
        
        .confidence-low {
            color: #f44336;
        }
        
        .chart-container {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .settings-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .setting-group {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
        }
        
        .setting-group h4 {
            color: #FFD700;
            margin-bottom: 15px;
        }
        
        .form-control {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 5px;
            background: rgba(255,255,255,0.1);
            color: white;
            margin-bottom: 10px;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #FFD700;
        }
        
        .live-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            z-index: 1000;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        
        .status-running {
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        .status-stopped {
            background: #f44336;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .copy-button {
            background: #FFD700;
            color: black;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8em;
            margin-left: 10px;
        }
        
        .notification {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            display: none;
            z-index: 1001;
        }
    </style>
</head>
<body>
    <div class="live-status">
        <span class="status-indicator" id="statusIndicator"></span>
        <span id="engineStatus">Connecting...</span>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🚀 Ultimate Maximum Income Dashboard</h1>
            <p>24/7 Automated Profit Generation with Zero Investment Mindset</p>
            <p><strong>Real-time Control Center for Maximum Income</strong></p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>💰 Profit Performance</h3>
                <div class="metric">
                    <span>Total Profit:</span>
                    <span class="metric-value" id="totalProfit">$0.00</span>
                </div>
                <div class="metric">
                    <span>Successful Trades:</span>
                    <span class="metric-value" id="successfulTrades">0</span>
                </div>
                <div class="metric">
                    <span>Success Rate:</span>
                    <span class="metric-value" id="successRate">0%</span>
                </div>
                <div class="metric">
                    <span>Daily Potential:</span>
                    <span class="metric-value" id="dailyPotential">$0.00</span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>🔍 Opportunity Detection</h3>
                <div class="metric">
                    <span>Total Opportunities:</span>
                    <span class="metric-value" id="totalOpportunities">0</span>
                </div>
                <div class="metric">
                    <span>Active Strategies:</span>
                    <span class="metric-value" id="activeStrategies">5</span>
                </div>
                <div class="metric">
                    <span>Exchanges Monitored:</span>
                    <span class="metric-value" id="exchangesMonitored">8</span>
                </div>
                <div class="metric">
                    <span>Last Opportunity:</span>
                    <span class="metric-value" id="lastOpportunity">Never</span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>⚙️ System Configuration</h3>
                <div class="metric">
                    <span>Position Mode:</span>
                    <span class="metric-value" id="positionMode">Aggressive</span>
                </div>
                <div class="metric">
                    <span>Auto Execution:</span>
                    <span class="metric-value" id="autoExecution">Enabled</span>
                </div>
                <div class="metric">
                    <span>Profit Threshold:</span>
                    <span class="metric-value" id="profitThreshold">0.01%</span>
                </div>
                <div class="metric">
                    <span>Risk Level:</span>
                    <span class="metric-value" id="riskLevel">Low</span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>📈 Performance Metrics</h3>
                <div class="metric">
                    <span>Uptime:</span>
                    <span class="metric-value" id="uptime">99.9%</span>
                </div>
                <div class="metric">
                    <span>Avg Execution Time:</span>
                    <span class="metric-value" id="avgExecution">2.1s</span>
                </div>
                <div class="metric">
                    <span>Position Multiplier:</span>
                    <span class="metric-value" id="positionMultiplier">1.5x</span>
                </div>
                <div class="metric">
                    <span>Reinvestment:</span>
                    <span class="metric-value" id="reinvestment">Enabled</span>
                </div>
            </div>
        </div>
        
        <div class="control-panel">
            <h2>🎮 Control Center</h2>
            <div class="control-grid">
                <button class="btn btn-success" onclick="startEngine()">🚀 Start Ultimate Engine</button>
                <button class="btn btn-danger" onclick="stopEngine()">🛑 Stop Engine</button>
                <button class="btn btn-primary" onclick="enableExecution()">⚡ Enable Auto Execution</button>
                <button class="btn btn-warning" onclick="disableExecution()">⏸️ Disable Execution</button>
                <button class="btn btn-primary" onclick="openExchangeLinks()">🔗 Open Exchange Links</button>
                <button class="btn btn-success" onclick="copyApiInstructions()">📋 Copy API Setup</button>
                <button class="btn btn-warning" onclick="exportDatabase()">💾 Export Data</button>
                <button class="btn btn-primary" onclick="refreshData()">🔄 Refresh Data</button>
            </div>
            
            <h3 style="color: #FFD700; margin-top: 30px; margin-bottom: 20px;">🔥 MEGA INCOME ENHANCEMENT CONTROLS</h3>
            <div class="control-grid">
                <button class="btn btn-danger" onclick="activateMegaMode()" style="background: linear-gradient(45deg, #FF4500, #FF0000); font-size: 1.1em;">🔥 ACTIVATE MEGA MODE</button>
                <button class="btn btn-success" onclick="enableCompoundMode()">💰 Enable Compound Profits</button>
                <button class="btn btn-warning" onclick="setSpeedMode('ultra')">⚡ Ultra Speed Mode</button>
                <button class="btn btn-primary" onclick="setSpeedMode('maximum')">🚀 Maximum Speed Mode</button>
                <button class="btn btn-success" onclick="setPositionMultiplier('10x')">📈 10X Position Multiplier</button>
                <button class="btn btn-warning" onclick="enableZeroThreshold()">🎯 Ultra-Sensitive Detection</button>
            </div>
        </div>
        
        <div class="settings-panel">
            <h2>⚙️ Settings & Configuration</h2>
            <div class="settings-grid">
                <div class="setting-group">
                    <h4>Position Sizing</h4>
                    <select class="form-control" id="positionSizing" onchange="updatePositionSizing()">
                        <option value="conservative">Conservative (10-25%)</option>
                        <option value="moderate">Moderate (25-50%)</option>
                        <option value="aggressive" selected>Aggressive (50-75%)</option>
                        <option value="maximum">Maximum (75-100%)</option>
                    </select>
                </div>
                
                <div class="setting-group">
                    <h4>Profit Threshold</h4>
                    <input type="number" class="form-control" id="profitThresholdInput" 
                           placeholder="Minimum profit %" step="0.001" value="0.01" 
                           onchange="updateProfitThreshold()">
                </div>
                
                <div class="setting-group">
                    <h4>Risk Management</h4>
                    <select class="form-control" id="riskLevel" onchange="updateRiskLevel()">
                        <option value="low" selected>Low Risk (2%)</option>
                        <option value="medium">Medium Risk (5%)</option>
                        <option value="high">High Risk (10%)</option>
                    </select>
                </div>
                
                <div class="setting-group">
                    <h4>Automation Options</h4>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="checkbox" id="autoReinvest" checked onchange="updateReinvestment()"> 
                        Auto Reinvestment
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="checkbox" id="adaptiveThresholds" checked onchange="updateAdaptive()"> 
                        Adaptive Thresholds
                    </label>
                </div>
            </div>
        </div>
        
        <div class="opportunities-section">
            <h2>💎 Current Opportunities</h2>
            <table class="opportunities-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Buy Exchange</th>
                        <th>Sell Exchange</th>
                        <th>Profit USD</th>
                        <th>Profit %</th>
                        <th>Confidence</th>
                        <th>Risk Score</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody id="opportunitiesTable">
                    <tr>
                        <td colspan="8" style="text-align: center; color: #999;">Loading opportunities...</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="chart-container">
            <h2>📈 Performance Charts</h2>
            <canvas id="profitChart" width="400" height="200"></canvas>
        </div>
    </div>
    
    <div class="notification" id="notification"></div>
    
    <script>
        const socket = io();
        let profitChart;
        
        // Initialize chart
        const ctx = document.getElementById('profitChart').getContext('2d');
        profitChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative Profit ($)',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: 'white'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'white'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                }
            }
        });
        
        // Socket event handlers
        socket.on('engine_status', function(data) {
            updateEngineStatus(data);
        });
        
        socket.on('performance_update', function(data) {
            updatePerformanceMetrics(data);
        });
        
        socket.on('opportunities_update', function(data) {
            updateOpportunitiesTable(data);
        });
        
        socket.on('notification', function(data) {
            showNotification(data.message, data.type);
        });
        
        // UI Functions
        function updateEngineStatus(data) {
            const indicator = document.getElementById('statusIndicator');
            const status = document.getElementById('engineStatus');
            
            if (data.running) {
                indicator.className = 'status-indicator status-running';
                status.textContent = 'Engine Running';
            } else {
                indicator.className = 'status-indicator status-stopped';
                status.textContent = 'Engine Stopped';
            }
        }
        
        function updatePerformanceMetrics(data) {
            document.getElementById('totalProfit').textContent = `$${data.total_profit?.toFixed(2) || '0.00'}`;
            document.getElementById('successfulTrades').textContent = data.successful_executions || '0';
            document.getElementById('successRate').textContent = `${data.success_rate_pct?.toFixed(1) || '0'}%`;
            document.getElementById('totalOpportunities').textContent = data.total_opportunities || '0';
            document.getElementById('positionMultiplier').textContent = `${data.position_multiplier?.toFixed(2) || '1.00'}x`;
            document.getElementById('profitThreshold').textContent = `${(data.current_threshold * 100)?.toFixed(3) || '0.001'}%`;
            
            // Update daily potential
            const dailyPotential = (data.total_profit || 0) * 24;
            document.getElementById('dailyPotential').textContent = `$${dailyPotential.toFixed(2)}`;
            
            // Update last opportunity
            if (data.last_opportunity) {
                const lastTime = new Date(data.last_opportunity);
                const now = new Date();
                const diffMinutes = Math.floor((now - lastTime) / 60000);
                document.getElementById('lastOpportunity').textContent = `${diffMinutes}m ago`;
            }
            
            // Update chart
            const now = new Date().toLocaleTimeString();
            profitChart.data.labels.push(now);
            profitChart.data.datasets[0].data.push(data.total_profit || 0);
            
            if (profitChart.data.labels.length > 20) {
                profitChart.data.labels.shift();
                profitChart.data.datasets[0].data.shift();
            }
            
            profitChart.update('none');
        }
        
        function updateOpportunitiesTable(opportunities) {
            const tbody = document.getElementById('opportunitiesTable');
            tbody.innerHTML = '';
            
            if (opportunities.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #999;">No opportunities found</td></tr>';
                return;
            }
            
            opportunities.forEach(opp => {
                const row = document.createElement('tr');
                
                const confidenceClass = opp.confidence > 0.7 ? 'confidence-high' : 
                                       opp.confidence > 0.4 ? 'confidence-medium' : 'confidence-low';
                
                row.innerHTML = `
                    <td>${opp.symbol}</td>
                    <td>${opp.buy_exchange}</td>
                    <td>${opp.sell_exchange}</td>
                    <td class="profit-positive">$${opp.profit_usd.toFixed(2)}</td>
                    <td class="profit-positive">${(opp.profit_pct * 100).toFixed(3)}%</td>
                    <td class="${confidenceClass}">${(opp.confidence * 100).toFixed(1)}%</td>
                    <td>${(opp.risk_score * 100).toFixed(1)}%</td>
                    <td><button class="copy-button" onclick="copyOpportunity('${opp.symbol}', '${opp.buy_exchange}', '${opp.sell_exchange}')">Copy</button></td>
                `;
                
                tbody.appendChild(row);
            });
        }
        
        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.style.display = 'block';
            notification.style.backgroundColor = type === 'success' ? '#4CAF50' : '#f44336';
            
            setTimeout(() => {
                notification.style.display = 'none';
            }, 3000);
        }
        
        // Control Functions
        function startEngine() {
            fetch('/start_engine', { method: 'POST' })
                .then(response => response.json())
                .then(data => showNotification(data.message, data.status));
        }
        
        function stopEngine() {
            fetch('/stop_engine', { method: 'POST' })
                .then(response => response.json())
                .then(data => showNotification(data.message, data.status));
        }
        
        function enableExecution() {
            fetch('/enable_execution', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    showNotification(data.message, data.status);
                    document.getElementById('autoExecution').textContent = 'Enabled';
                });
        }
        
        function disableExecution() {
            fetch('/disable_execution', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    showNotification(data.message, data.status);
                    document.getElementById('autoExecution').textContent = 'Disabled';
                });
        }
        
        function updatePositionSizing() {
            const mode = document.getElementById('positionSizing').value;
            fetch('/update_position_sizing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(response => response.json())
            .then(data => {
                showNotification(data.message, data.status);
                document.getElementById('positionMode').textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
            });
        }
        
        function updateProfitThreshold() {
            const threshold = parseFloat(document.getElementById('profitThresholdInput').value) / 100;
            fetch('/update_profit_threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: threshold })
            })
            .then(response => response.json())
            .then(data => showNotification(data.message, data.status));
        }
        
        function openExchangeLinks() {
            const exchanges = [
                { name: 'Binance', url: 'https://www.binance.com/en/register' },
                { name: 'Coinbase', url: 'https://www.coinbase.com/signup' },
                { name: 'KuCoin', url: 'https://www.kucoin.com/ucenter/signup' },
                { name: 'OKX', url: 'https://www.okx.com/join' },
                { name: 'Bybit', url: 'https://www.bybit.com/register' },
                { name: 'Kraken', url: 'https://www.kraken.com/sign-up' }
            ];
            
            exchanges.forEach(exchange => {
                window.open(exchange.url, '_blank');
            });
            
            showNotification('Opened exchange registration links!');
        }
        
        function copyApiInstructions() {
            const instructions = `
🚀 ULTIMATE INCOME API SETUP INSTRUCTIONS

1. BINANCE:
   - Go to: https://www.binance.com/en/my/settings/api-management
   - Create new API key
   - Enable "Enable Reading" and "Enable Spot & Margin Trading"
   - Whitelist your IP address

2. COINBASE PRO:
   - Go to: https://pro.coinbase.com/profile/api
   - Create new API key
   - Select permissions: View, Trade
   - Save passphrase securely

3. KUCOIN:
   - Go to: https://www.kucoin.com/account/api
   - Create API
   - Permissions: General, Trade
   - Set API restrictions

4. OKX:
   - Go to: https://www.okx.com/account/my-api
   - Create API key
   - Permissions: Read, Trade
   - IP restriction recommended

💡 SECURITY TIPS:
   - Never share your API keys
   - Use IP restrictions
   - Start with small amounts
   - Enable 2FA on all accounts
   - Monitor your accounts regularly

🔥 Once setup, the system will automatically:
   ✅ Monitor all exchanges 24/7
   ✅ Detect arbitrage opportunities
   ✅ Execute profitable trades
   ✅ Reinvest profits automatically
   ✅ Generate maximum income!
            `;
            
            navigator.clipboard.writeText(instructions).then(() => {
                showNotification('API setup instructions copied to clipboard!');
            });
        }
        
        function copyOpportunity(symbol, buyExchange, sellExchange) {
            const text = `Arbitrage Opportunity: ${symbol} | Buy: ${buyExchange} | Sell: ${sellExchange}`;
            navigator.clipboard.writeText(text).then(() => {
                showNotification('Opportunity details copied!');
            });
        }
        
        function exportDatabase() {
            fetch('/export_database')
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `ultimate_income_data_${new Date().toISOString().split('T')[0]}.db`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    showNotification('Database exported successfully!');
                });
        }
        
        function refreshData() {
            fetch('/get_performance')
                .then(response => response.json())
                .then(data => {
                    updatePerformanceMetrics(data);
                    showNotification('Data refreshed!');
                });
        }
        
        // 🔥 MEGA INCOME ENHANCEMENT FUNCTIONS
        function activateMegaMode() {
            if (confirm('🔥 ACTIVATE MEGA INCOME MODE? This will enable 10X profit multipliers and ultra-aggressive settings!')) {
                fetch('/activate_mega_mode', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        showNotification(data.message, data.status);
                        if (data.status === 'success') {
                            document.getElementById('positionMultiplier').textContent = '10.0x';
                            document.getElementById('positionMode').textContent = 'MEGA MODE';
                        }
                    });
            }
        }
        
        function enableCompoundMode() {
            fetch('/enable_compound_mode', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    showNotification(data.message, data.status);
                    if (data.status === 'success') {
                        document.getElementById('reinvestment').textContent = 'Compound Mode';
                    }
                });
        }
        
        function setSpeedMode(mode) {
            fetch('/set_speed_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(response => response.json())
            .then(data => showNotification(data.message, data.status));
        }
        
        function setPositionMultiplier(multiplier) {
            const value = multiplier === '10x' ? 10.0 : parseFloat(multiplier);
            
            fetch('/activate_mega_mode', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    showNotification(`📈 Position multiplier set to ${multiplier}!`, 'success');
                    document.getElementById('positionMultiplier').textContent = multiplier;
                });
        }
        
        function enableZeroThreshold() {
            fetch('/update_profit_threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: 0.0001 })
            })
            .then(response => response.json())
            .then(data => {
                showNotification('🎯 Ultra-sensitive detection enabled! 0.01% minimum threshold!', 'success');
                document.getElementById('profitThreshold').textContent = '0.010%';
            });
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            refreshData();
            
            // Auto refresh every 30 seconds
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/start_engine', methods=['POST'])
def start_engine():
    """Start the ultimate income engine"""
    global global_engine, engine_thread
    
    try:
        if global_engine is None or not global_engine.running:
            global_engine = UltimateMaximumIncomeEngine()
            global_engine.enable_auto_execution()
            global_engine.set_position_sizing_mode('aggressive')
            
            # Start engine in separate thread
            def run_engine():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(global_engine.start_maximum_income_engine())
            
            engine_thread = threading.Thread(target=run_engine, daemon=True)
            engine_thread.start()
            
            # Emit status update
            socketio.emit('engine_status', {'running': True})
            
            return jsonify({'status': 'success', 'message': '🚀 Ultimate Income Engine Started!'})
        else:
            return jsonify({'status': 'info', 'message': 'Engine is already running'})
    
    except Exception as e:
        logger.error(f"Error starting engine: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to start engine: {str(e)}'})

@app.route('/stop_engine', methods=['POST'])
def stop_engine():
    """Stop the ultimate income engine with proper threading"""
    global global_engine, engine_thread
    
    try:
        if global_engine and global_engine.running:
            # Properly stop the engine
            global_engine.running = False
            global_engine.auto_execution_enabled = False
            
            # Close database connection safely
            if hasattr(global_engine, 'db_connection'):
                try:
                    global_engine.db_connection.close()
                except:
                    pass
            
            # Reset global variables
            global_engine = None
            engine_thread = None
            
            socketio.emit('engine_status', {'running': False})
            return jsonify({'status': 'success', 'message': '🛑 Engine stopped successfully'})
        else:
            return jsonify({'status': 'info', 'message': 'Engine is not running'})
    
    except Exception as e:
        logger.error(f"Error stopping engine: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to stop engine: {str(e)}'})

@app.route('/enable_execution', methods=['POST'])
def enable_execution():
    """Enable automated execution"""
    global global_engine
    
    try:
        if global_engine:
            global_engine.enable_auto_execution()
            return jsonify({'status': 'success', 'message': '⚡ Automated execution enabled!'})
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/disable_execution', methods=['POST'])
def disable_execution():
    """Disable automated execution"""
    global global_engine
    
    try:
        if global_engine:
            global_engine.disable_auto_execution()
            return jsonify({'status': 'success', 'message': '⏸️ Automated execution disabled'})
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_position_sizing', methods=['POST'])
def update_position_sizing():
    """Update position sizing mode"""
    global global_engine
    
    try:
        data = request.get_json()
        mode = data.get('mode', 'conservative')
        
        if global_engine:
            global_engine.set_position_sizing_mode(mode)
            return jsonify({'status': 'success', 'message': f'📊 Position sizing set to {mode}'})
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_profit_threshold', methods=['POST'])
def update_profit_threshold():
    """Update profit threshold"""
    global global_engine
    
    try:
        data = request.get_json()
        threshold = data.get('threshold', 0.0001)
        
        if global_engine:
            global_engine.ai_config['min_profit_threshold'] = threshold
            return jsonify({'status': 'success', 'message': f'🎯 Profit threshold set to {threshold:.4%}'})
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/activate_mega_mode', methods=['POST'])
def activate_mega_mode():
    """🔥 ACTIVATE MEGA INCOME MODE - MAXIMUM PROFIT GENERATION"""
    global global_engine
    
    try:
        if global_engine:
            # Use the new activate_mega_mode method
            global_engine.activate_mega_mode()
            
            return jsonify({
                'status': 'success', 
                'message': '🔥 MEGA INCOME MODE ACTIVATED! 10X PROFIT MULTIPLIER ENABLED!'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/enable_compound_mode', methods=['POST'])
def enable_compound_mode():
    """💰 ENABLE COMPOUND PROFIT MODE - EXPONENTIAL GROWTH"""
    global global_engine
    
    try:
        if global_engine:
            # Use the new enable_compound_mode method
            global_engine.enable_compound_mode()
            
            return jsonify({
                'status': 'success',
                'message': '💰 COMPOUND MODE ENABLED! Profits will reinvest automatically for exponential growth!'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/set_speed_mode', methods=['POST'])
def set_speed_mode():
    """⚡ SET ULTRA-SPEED MODE - FASTER OPPORTUNITY DETECTION"""
    global global_engine
    
    try:
        data = request.get_json()
        speed_mode = data.get('mode', 'normal')  # normal, fast, ultra, maximum
        
        if global_engine:
            # Use the new speed mode methods
            if speed_mode == 'ultra':
                global_engine.set_ultra_speed_mode()
            elif speed_mode == 'maximum':
                global_engine.set_maximum_speed_mode()
            else:
                global_engine.speed_mode = 'normal'
            
            return jsonify({
                'status': 'success',
                'message': f'⚡ SPEED MODE SET TO {speed_mode.upper()}! Ultra-fast detection enabled!'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
            return jsonify({'status': 'success', 'message': f'🎯 Profit threshold set to {threshold:.4%}'})
        else:
            return jsonify({'status': 'error', 'message': 'Engine not initialized'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_performance')
def get_performance():
    """Get current performance metrics"""
    global global_engine
    
    try:
        if global_engine:
            performance = global_engine.get_performance_summary()
            return jsonify(performance)
        else:
            # Return default metrics if engine not running
            return jsonify({
                'engine_status': 'STOPPED',
                'total_opportunities': 0,
                'successful_executions': 0,
                'total_profit_usd': 0.0,
                'success_rate_pct': 0,
                'position_multiplier': 1.0,
                'current_threshold': 0.0001,
                'exchanges_monitored': 8,
                'symbols_tracked': 80,
                'last_opportunity': None,
                'automation_enabled': False,
                'execution_enabled': False,
                'reinvestment_enabled': True
            })
    
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/get_opportunities')
def get_opportunities():
    """Get current opportunities"""
    try:
        # Check if database exists
        if os.path.exists('ultimate_income_database.db'):
            conn = sqlite3.connect('ultimate_income_database.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, buy_exchange, sell_exchange, profit_usd, profit_pct, 
                       confidence, risk_score, timestamp
                FROM maximum_income_opportunities 
                WHERE executed = FALSE 
                AND timestamp > datetime('now', '-1 hour')
                ORDER BY profit_usd DESC 
                LIMIT 20
            ''')
            
            opportunities = []
            for row in cursor.fetchall():
                opportunities.append({
                    'symbol': row[0],
                    'buy_exchange': row[1],
                    'sell_exchange': row[2],
                    'profit_usd': row[3],
                    'profit_pct': row[4],
                    'confidence': row[5],
                    'risk_score': row[6],
                    'timestamp': row[7]
                })
            
            conn.close()
            return jsonify(opportunities)
        else:
            return jsonify([])
    
    except Exception as e:
        logger.error(f"Error getting opportunities: {str(e)}")
        return jsonify([])

@app.route('/export_database')
def export_database():
    """Export database file"""
    try:
        db_path = 'ultimate_income_database.db'
        if os.path.exists(db_path):
            return flask.send_file(db_path, as_attachment=True)
        else:
            return jsonify({'error': 'Database not found'}), 404
    
    except Exception as e:
        logger.error(f"Error exporting database: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    logger.info('Client connected to dashboard')
    
    # Send initial status
    emit('engine_status', {'running': global_engine.running if global_engine else False})
    
    # Send initial performance data
    if global_engine:
        performance = global_engine.get_performance_summary()
        emit('performance_update', performance)

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected from dashboard')

def broadcast_performance_updates():
    """Broadcast performance updates to all connected clients"""
    while True:
        try:
            if global_engine and global_engine.running:
                performance = global_engine.get_performance_summary()
                socketio.emit('performance_update', performance)
                
                # Get recent opportunities
                opportunities_data = []
                try:
                    conn = sqlite3.connect('ultimate_income_database.db')
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT symbol, buy_exchange, sell_exchange, profit_usd, profit_pct, 
                               confidence, risk_score, timestamp
                        FROM maximum_income_opportunities 
                        WHERE executed = FALSE 
                        AND timestamp > datetime('now', '-1 hour')
                        ORDER BY profit_usd DESC 
                        LIMIT 10
                    ''')
                    
                    for row in cursor.fetchall():
                        opportunities_data.append({
                            'symbol': row[0],
                            'buy_exchange': row[1],
                            'sell_exchange': row[2],
                            'profit_usd': row[3],
                            'profit_pct': row[4],
                            'confidence': row[5],
                            'risk_score': row[6],
                            'timestamp': row[7]
                        })
                    
                    conn.close()
                except:
                    pass
                
                socketio.emit('opportunities_update', opportunities_data)
            
            time.sleep(30)  # Update every 30 seconds
        
        except Exception as e:
            logger.error(f"Error broadcasting updates: {str(e)}")
            time.sleep(60)

def launch_dashboard():
    """Launch the Ultimate Income Dashboard"""
    print("🚀 LAUNCHING ULTIMATE MAXIMUM INCOME DASHBOARD")
    print("=" * 60)
    print("💰 Zero Investment Mindset: Maximum Profit Generation")
    print("🤖 Full Automation Control Center")
    print("🌐 Web-Based Real-time Monitoring")
    print("=" * 60)
    
    # Start background performance broadcast thread
    broadcast_thread = threading.Thread(target=broadcast_performance_updates, daemon=True)
    broadcast_thread.start()
    
    # Get local IP and port
    port = 5000
    dashboard_url = f"http://localhost:{port}"
    
    print(f"🌐 Dashboard URL: {dashboard_url}")
    print("📝 Copy this URL to access from other devices:")
    print(f"   {dashboard_url}")
    print("")
    print("⚙️ Dashboard Features:")
    print("   • Real-time profit monitoring")
    print("   • One-click engine control")
    print("   • Easy settings configuration")
    print("   • Exchange link integration")
    print("   • API setup instructions")
    print("   • Data export functionality")
    print("")
    print("🚀 READY TO MAXIMIZE YOUR INCOME!")
    print("=" * 60)
    
    # Auto-open browser
    threading.Timer(2.0, lambda: webbrowser.open(dashboard_url)).start()
    
    # Start Flask app
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error starting dashboard: {str(e)}")
        print("Trying alternative port...")
        socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    launch_dashboard()

