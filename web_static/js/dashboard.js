
// Ultimate Arbitrage System Dashboard JavaScript

class ArbitrageDashboard {
    constructor() {
        this.socket = io();
        this.isSystemRunning = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.chartInstances = {};
        this.performanceHistory = [];
        
        this.setupEventListeners();
        this.setupSocketListeners();
        this.initializeCharts();
        this.updateData();
        
        // Auto-refresh data every 5 seconds
        setInterval(() => this.updateData(), 5000);
        
        // Save performance data every minute
        setInterval(() => this.savePerformanceHistory(), 60000);
    }
    
    setupEventListeners() {
        document.getElementById('startSystemBtn').addEventListener('click', () => this.startSystem());
        document.getElementById('stopSystemBtn').addEventListener('click', () => this.stopSystem());
    }
    
    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to real-time feed');
            this.reconnectAttempts = 0;
            this.showNotification('Connected to real-time feed', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from real-time feed');
            this.showNotification('Connection lost. Attempting to reconnect...', 'warning');
            this.attemptReconnect();
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.attemptReconnect();
        });
        
        this.socket.on('system_update', (data) => {
            this.updateUI(data);
        });
        
        this.socket.on('new_opportunity', (opportunity) => {
            this.addOpportunityToTable(opportunity);
        });
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                this.socket.connect();
            }, 2000 * this.reconnectAttempts); // Exponential backoff
        } else {
            this.showNotification('Unable to reconnect. Please refresh the page.', 'error');
        }
    }
    
    initializeCharts() {
        // Initialize performance chart (using Chart.js if available)
        try {
            const ctx = document.getElementById('performanceChart');
            if (ctx && typeof Chart !== 'undefined') {
                this.chartInstances.performance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Total Profit ($)',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Profit Over Time'
                            }
                        }
                    }
                });
            }
        } catch (error) {
            console.log('Chart.js not available, charts disabled');
        }
    }
    
    savePerformanceHistory() {
        // Save current metrics to history
        if (this.isSystemRunning) {
            const now = new Date();
            this.performanceHistory.push({
                timestamp: now.toISOString(),
                profit: parseFloat(document.getElementById('totalProfit').textContent.replace('$', '')) || 0,
                winRate: parseFloat(document.getElementById('winRate').textContent.replace('%', '')) || 0,
                opportunities: parseInt(document.getElementById('activeOpportunities').textContent) || 0
            });
            
            // Keep only last 100 data points
            if (this.performanceHistory.length > 100) {
                this.performanceHistory.shift();
            }
            
            // Update chart if available
            this.updatePerformanceChart();
        }
    }
    
    updatePerformanceChart() {
        if (this.chartInstances.performance && this.performanceHistory.length > 0) {
            const chart = this.chartInstances.performance;
            chart.data.labels = this.performanceHistory.map(p => 
                new Date(p.timestamp).toLocaleTimeString()
            );
            chart.data.datasets[0].data = this.performanceHistory.map(p => p.profit);
            chart.update('none'); // No animation for real-time updates
        }
    }
    
    async startSystem() {
        const startBtn = document.getElementById('startSystemBtn');
        const originalText = startBtn.innerHTML;
        
        try {
            // Show loading state
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
            
            const response = await fetch('/api/system/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isSystemRunning = true;
                this.updateSystemStatus(true);
                this.showNotification('🚀 System started successfully! Monitoring opportunities...', 'success');
            } else {
                this.showNotification('Failed to start system: ' + result.error, 'error');
                startBtn.disabled = false;
                startBtn.innerHTML = originalText;
            }
        } catch (error) {
            this.showNotification('Error starting system: ' + error.message, 'error');
            startBtn.disabled = false;
            startBtn.innerHTML = originalText;
        }
    }
    
    async stopSystem() {
        // Show confirmation dialog
        if (!confirm('Are you sure you want to stop the arbitrage system? This will halt all active trading.')) {
            return;
        }
        
        const stopBtn = document.getElementById('stopSystemBtn');
        const originalText = stopBtn.innerHTML;
        
        try {
            // Show loading state
            stopBtn.disabled = true;
            stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
            
            const response = await fetch('/api/system/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isSystemRunning = false;
                this.updateSystemStatus(false);
                this.showNotification('🛑 System stopped successfully!', 'info');
            } else {
                this.showNotification('Failed to stop system: ' + result.error, 'error');
                stopBtn.disabled = false;
                stopBtn.innerHTML = originalText;
            }
        } catch (error) {
            this.showNotification('Error stopping system: ' + error.message, 'error');
            stopBtn.disabled = false;
            stopBtn.innerHTML = originalText;
        }
    }
    
    updateSystemStatus(isRunning) {
        const statusElement = document.getElementById('systemStatus');
        const startBtn = document.getElementById('startSystemBtn');
        const stopBtn = document.getElementById('stopSystemBtn');
        
        if (isRunning) {
            statusElement.textContent = 'Online';
            statusElement.className = 'badge bg-success pulse';
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            statusElement.textContent = 'Offline';
            statusElement.className = 'badge bg-secondary';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    }
    
    async updateData() {
        try {
            // Update performance metrics
            const metricsResponse = await fetch('/api/performance/metrics');
            const metrics = await metricsResponse.json();
            this.updateMetrics(metrics);
            
            // Update opportunities
            const oppsResponse = await fetch('/api/opportunities/live');
            const opportunities = await oppsResponse.json();
            this.updateOpportunities(opportunities);
            
            // Update wallets
            const walletsResponse = await fetch('/api/wallets/list');
            const wallets = await walletsResponse.json();
            this.updateWallets(wallets);
            
            // Update strategies
            const strategiesResponse = await fetch('/api/strategies/list');
            const strategies = await strategiesResponse.json();
            this.updateStrategies(strategies);
            
        } catch (error) {
            console.error('Error updating data:', error);
        }
    }
    
    updateMetrics(metrics) {
        document.getElementById('totalProfit').textContent = '$' + metrics.total_profit.toFixed(2);
        document.getElementById('winRate').textContent = metrics.win_rate.toFixed(1) + '%';
        document.getElementById('activeOpportunities').textContent = metrics.active_opportunities;
        document.getElementById('quantumAdvantage').textContent = metrics.quantum_advantage.toFixed(1) + 'x';
    }
    
    updateOpportunities(opportunities) {
        const tableBody = document.getElementById('opportunitiesTable');
        
        if (opportunities.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No opportunities detected yet.</td></tr>';
            return;
        }
        
        tableBody.innerHTML = opportunities.map(opp => `
            <tr class="opportunity-${this.getOpportunityClass(opp.profit_potential)}">
                <td><span class="badge bg-primary">${opp.type}</span></td>
                <td class="profit-positive">${(opp.profit_potential * 100).toFixed(2)}%</td>
                <td>${opp.confidence_score ? opp.confidence_score.toFixed(1) + '%' : 'N/A'}</td>
                <td>${opp.risk_factor ? (opp.risk_factor * 100).toFixed(1) + '%' : 'Low'}</td>
                <td><span class="badge bg-success">Active</span></td>
                <td><button class="btn btn-sm btn-success">Execute</button></td>
            </tr>
        `).join('');
    }
    
    updateWallets(wallets) {
        const walletsList = document.getElementById('walletsList');
        
        if (wallets.length === 0) {
            walletsList.innerHTML = '<p class="text-muted">No wallets connected. Click "Add Wallet" to get started.</p>';
            return;
        }
        
        walletsList.innerHTML = wallets.map(wallet => `
            <div class="wallet-card">
                <h6>${wallet.wallet_id}</h6>
                <p class="mb-1">Network: ${wallet.network}</p>
                <p class="mb-0">Balance: $${wallet.balance.toFixed(2)}</p>
            </div>
        `).join('');
        
        // Update strategy wallet dropdown
        const strategyWalletSelect = document.getElementById('strategyWallet');
        strategyWalletSelect.innerHTML = '<option value="">Select a wallet...</option>' +
            wallets.map(wallet => `<option value="${wallet.wallet_id}">${wallet.wallet_id} (${wallet.network})</option>`).join('');
    }
    
    updateStrategies(strategies) {
        const strategiesList = document.getElementById('strategiesList');
        
        if (strategies.length === 0) {
            strategiesList.innerHTML = '<p class="text-muted">No strategies deployed. Deploy strategies to start earning.</p>';
            return;
        }
        
        strategiesList.innerHTML = strategies.map(strategy => `
            <div class="strategy-card">
                <h6>${strategy.type}</h6>
                <p class="mb-1">Wallet: ${strategy.wallet_id}</p>
                <p class="mb-1">Capital: $${strategy.capital}</p>
                <p class="mb-0">Profit: <span class="${strategy.profit >= 0 ? 'profit-positive' : 'profit-negative'}">$${strategy.profit.toFixed(2)}</span></p>
            </div>
        `).join('');
    }
    
    getOpportunityClass(profit) {
        if (profit > 0.02) return 'high';
        if (profit > 0.01) return 'medium';
        return 'low';
    }
    
    showNotification(message, type = 'info') {
        // Create a simple notification (you can enhance this with a proper notification library)
        const alertClass = type === 'error' ? 'alert-danger' : 
                          type === 'success' ? 'alert-success' : 
                          type === 'warning' ? 'alert-warning' : 'alert-info';
        
        const notification = document.createElement('div');
        notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
}

// Wallet connection functionality
function showWalletModal() {
    const modal = new bootstrap.Modal(document.getElementById('walletModal'));
    modal.show();
}

async function connectWallet() {
    const walletId = document.getElementById('walletId').value.trim();
    const privateKey = document.getElementById('privateKey').value.trim();
    const network = document.getElementById('network').value;
    const connectBtn = document.querySelector('#walletModal .btn-primary');
    
    // Enhanced form validation
    if (!validateWalletForm(walletId, privateKey)) {
        return;
    }
    
    // Show loading state
    connectBtn.disabled = true;
    const originalText = connectBtn.innerHTML;
    connectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
    
    try {
        const response = await fetch('/api/wallets/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ wallet_id: walletId, private_key: privateKey, network: network })
        });
        
        const result = await response.json();
        
        if (result.success) {
            dashboard.showNotification('Wallet connected successfully!', 'success');
            bootstrap.Modal.getInstance(document.getElementById('walletModal')).hide();
            dashboard.updateData();
            
            // Clear form
            document.getElementById('walletForm').reset();
        } else {
            dashboard.showNotification('Failed to connect wallet: ' + result.error, 'error');
        }
    } catch (error) {
        dashboard.showNotification('Error connecting wallet: ' + error.message, 'error');
    } finally {
        // Restore button state
        connectBtn.disabled = false;
        connectBtn.innerHTML = originalText;
    }
}

// Enhanced form validation functions
function validateWalletForm(walletId, privateKey) {
    // Clear previous validation errors
    clearValidationErrors();
    
    let isValid = true;
    
    // Validate wallet ID
    if (!walletId) {
        showFieldError('walletId', 'Wallet name is required');
        isValid = false;
    } else if (walletId.length < 3) {
        showFieldError('walletId', 'Wallet name must be at least 3 characters');
        isValid = false;
    }
    
    // Validate private key
    if (!privateKey) {
        showFieldError('privateKey', 'Private key is required');
        isValid = false;
    } else if (privateKey.length < 32) {
        showFieldError('privateKey', 'Private key appears to be invalid (too short)');
        isValid = false;
    } else if (!/^[a-fA-F0-9]+$/.test(privateKey.replace('0x', ''))) {
        showFieldError('privateKey', 'Private key must be a valid hexadecimal string');
        isValid = false;
    }
    
    if (!isValid) {
        dashboard.showNotification('Please fix the validation errors', 'error');
    }
    
    return isValid;
}

function validateStrategyForm(strategyType, walletId, capitalAmount) {
    clearValidationErrors();
    
    let isValid = true;
    
    if (!strategyType) {
        showFieldError('strategyType', 'Please select a strategy type');
        isValid = false;
    }
    
    if (!walletId) {
        showFieldError('strategyWallet', 'Please select a wallet');
        isValid = false;
    }
    
    if (capitalAmount <= 0) {
        showFieldError('capitalAmount', 'Capital amount must be greater than 0');
        isValid = false;
    } else if (capitalAmount > 1000000) {
        showFieldError('capitalAmount', 'Capital amount seems unusually high. Please verify.');
        isValid = false;
    }
    
    if (!isValid) {
        dashboard.showNotification('Please fix the validation errors', 'error');
    }
    
    return isValid;
}

function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    field.classList.add('is-invalid');
    
    // Create or update error message
    let errorDiv = field.parentNode.querySelector('.invalid-feedback');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        field.parentNode.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
}

function clearValidationErrors() {
    document.querySelectorAll('.is-invalid').forEach(field => {
        field.classList.remove('is-invalid');
    });
    document.querySelectorAll('.invalid-feedback').forEach(error => {
        error.remove();
    });
}

// Strategy deployment functionality
function showStrategyModal() {
    const modal = new bootstrap.Modal(document.getElementById('strategyModal'));
    modal.show();
}

async function deployStrategy() {
    const strategyType = document.getElementById('strategyType').value;
    const walletId = document.getElementById('strategyWallet').value;
    const capitalAmount = parseFloat(document.getElementById('capitalAmount').value) || 0;
    const deployBtn = document.querySelector('#strategyModal .btn-success');
    
    // Enhanced form validation
    if (!validateStrategyForm(strategyType, walletId, capitalAmount)) {
        return;
    }
    
    // Show confirmation for large amounts
    if (capitalAmount > 10000) {
        if (!confirm(`You are about to deploy $${capitalAmount.toLocaleString()} in capital. Are you sure?`)) {
            return;
        }
    }
    
    // Show loading state
    deployBtn.disabled = true;
    const originalText = deployBtn.innerHTML;
    deployBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deploying...';
    
    try {
        const response = await fetch('/api/strategies/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                strategy_type: strategyType, 
                wallet_id: walletId, 
                capital_amount: capitalAmount 
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            dashboard.showNotification('Strategy deployed successfully!', 'success');
            bootstrap.Modal.getInstance(document.getElementById('strategyModal')).hide();
            dashboard.updateData();
            
            // Clear form
            document.getElementById('strategyForm').reset();
        } else {
            dashboard.showNotification('Failed to deploy strategy: ' + result.error, 'error');
        }
    } catch (error) {
        dashboard.showNotification('Error deploying strategy: ' + error.message, 'error');
    } finally {
        // Restore button state
        deployBtn.disabled = false;
        deployBtn.innerHTML = originalText;
    }
}

// Quick start functions
function deployDefaultStrategies() {
    dashboard.showNotification('Deploying recommended strategies automatically...', 'info');
    
    // Auto-deploy quantum arbitrage strategy if wallet is connected
    setTimeout(() => {
        const walletSelect = document.getElementById('strategyWallet');
        if (walletSelect.options.length > 1) {
            document.getElementById('strategyType').value = 'quantum_arbitrage';
            document.getElementById('strategyWallet').value = walletSelect.options[1].value;
            document.getElementById('capitalAmount').value = '1000';
            deployStrategy();
        } else {
            dashboard.showNotification('Please connect a wallet first', 'warning');
        }
    }, 1000);
}

function startEarning() {
    dashboard.startSystem();
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new ArbitrageDashboard();
});
        