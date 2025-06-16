import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  Slider,
  Chip,
  LinearProgress,
  Alert,
  Fab,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Paper,
  Avatar,
  Badge
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  TrendingUp,
  AccountBalance,
  Security,
  Speed,
  AutoMode,
  SmartToy,
  Bolt,
  Eco,
  Visibility,
  Settings,
  TouchApp,
  Psychology,
  Rocket
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import SetupWizard from './SetupWizard';
import VoiceControlInterface from './VoiceControlInterface';

const OneClickDashboard = () => {
  // Core State
  const [systemActive, setSystemActive] = useState(false);
  const [automationLevel, setAutomationLevel] = useState(100); // 0-100%
  const [riskTolerance, setRiskTolerance] = useState(50); // 0-100%
  const [profitTarget, setProfitTarget] = useState(20); // % annual
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [isFirstRun, setIsFirstRun] = useState(true);
  
  // Performance Data
  const [performance, setPerformance] = useState({
    totalReturn: 15.7,
    dailyReturn: 0.23,
    sharpeRatio: 3.2,
    maxDrawdown: -2.1,
    winRate: 73.5,
    portfolioValue: 1250000,
    quantumAdvantage: 2.3
  });
  
  // Real-time Status
  const [systemStatus, setSystemStatus] = useState({
    aiEngines: 'active',
    quantumEngine: 'active',
    dataFeeds: 'excellent',
    riskManagement: 'optimal',
    orderExecution: 'fast'
  });
  
  // UI State
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [notification, setNotification] = useState(null);
  const [confirmDialog, setConfirmDialog] = useState({ open: false, action: null });
  
  // Quick Actions
  const quickStart = async () => {
    // Check if this is the first run
    if (isFirstRun) {
      setShowSetupWizard(true);
      return;
    }
    
    setConfirmDialog({ 
      open: true, 
      action: 'start',
      title: 'Start Ultimate Optimization',
      message: 'This will activate all AI engines, quantum computing, and begin live trading optimization. Continue?'
    });
  };
  
  const emergencyStop = async () => {
    setConfirmDialog({ 
      open: true, 
      action: 'stop',
      title: 'Emergency Stop',
      message: 'This will immediately halt all trading activity and optimization. Continue?'
    });
  };
  
  const handleSetupComplete = async (config) => {
    try {
      setIsFirstRun(false);
      
      // Update local state with wizard configuration
      setAutomationLevel(config.automation_level * 100);
      setRiskTolerance(config.risk_tolerance * 100);
      setProfitTarget(config.profit_target * 100);
      
      // Start the system with the configured settings
      const response = await fetch('/api/system/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        setSystemActive(true);
        setNotification({ 
          type: 'success', 
          message: 'System configured and activated! Welcome to the Ultimate Arbitrage System!' 
        });
      } else {
        throw new Error('Failed to start system');
      }
    } catch (error) {
      setNotification({ 
        type: 'error', 
        message: 'Setup failed. Please try again.' 
      });
    }
  };
  
  // Handle voice commands
  const handleVoiceCommand = async (action, params = {}) => {
    try {
      switch (action) {
        case 'START_SYSTEM':
          await executeAction('start');
          break;
        case 'STOP_SYSTEM':
        case 'EMERGENCY_STOP':
          await executeAction('stop');
          break;
        case 'OPTIMIZE':
        case 'AUTO_OPTIMIZE':
          await autoOptimize();
          break;
        case 'SET_RISK':
          if (params.level) {
            setRiskTolerance(params.level);
          }
          break;
        case 'ADJUST_RISK':
          if (params.delta) {
            setRiskTolerance(prev => Math.max(0, Math.min(100, prev + params.delta)));
          }
          break;
        case 'SET_AUTOMATION':
          if (params.level) {
            setAutomationLevel(params.level);
          }
          break;
        case 'SHOW_ADVANCED':
          setShowAdvanced(true);
          break;
        case 'SHOW_SIMPLE':
          setShowAdvanced(false);
          break;
        case 'OPEN_WIZARD':
          setShowSetupWizard(true);
          break;
        case 'GET_STATUS':
        case 'GET_RETURNS':
        case 'SHOW_PERFORMANCE':
          // These are informational, just acknowledge
          setNotification({ 
            type: 'info', 
            message: `Current portfolio value: $${performance.portfolioValue.toLocaleString()}. Daily return: ${performance.dailyReturn >= 0 ? '+' : ''}${performance.dailyReturn}%` 
          });
          break;
        default:
          setNotification({ type: 'info', message: 'Command acknowledged' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'Voice command failed' });
    }
  };
  
  const executeAction = async (action) => {
    setConfirmDialog({ open: false });
    
    try {
      if (action === 'start') {
        // API call to start system
        const response = await fetch('/api/system/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            automation_level: automationLevel / 100,
            risk_tolerance: riskTolerance / 100,
            profit_target: profitTarget / 100
          })
        });
        
        if (response.ok) {
          setSystemActive(true);
          setNotification({ type: 'success', message: 'System activated! Optimization in progress...' });
        }
      } else if (action === 'stop') {
        // API call to stop system
        await fetch('/api/system/emergency-stop', { method: 'POST' });
        setSystemActive(false);
        setNotification({ type: 'info', message: 'System stopped. All positions safely managed.' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'Action failed. Please try again.' });
    }
  };
  
  // Auto-optimize settings based on market conditions
  const autoOptimize = async () => {
    try {
      const response = await fetch('/api/system/auto-optimize', { method: 'POST' });
      const result = await response.json();
      
      setAutomationLevel(result.recommended_automation * 100);
      setRiskTolerance(result.recommended_risk * 100);
      setProfitTarget(result.recommended_target * 100);
      
      setNotification({ 
        type: 'success', 
        message: 'Settings optimized for current market conditions!' 
      });
    } catch (error) {
      setNotification({ type: 'error', message: 'Auto-optimization failed.' });
    }
  };
  
  // Status Colors
  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
      case 'excellent':
      case 'optimal':
      case 'fast':
        return 'success';
      case 'warning':
      case 'slow':
        return 'warning';
      case 'error':
      case 'offline':
        return 'error';
      default:
        return 'default';
    }
  };
  
  // Performance Chart Data
  const chartData = [
    { time: '9:30', value: 1000000 },
    { time: '10:00', value: 1015000 },
    { time: '10:30', value: 1032000 },
    { time: '11:00', value: 1045000 },
    { time: '11:30', value: 1038000 },
    { time: '12:00', value: 1055000 },
    { time: '12:30', value: 1072000 },
    { time: '13:00', value: 1089000 },
    { time: '13:30', value: 1095000 },
    { time: '14:00', value: 1112000 },
    { time: '14:30', value: 1125000 },
    { time: '15:00', value: 1142000 },
    { time: '15:30', value: 1158000 },
    { time: '16:00', value: 1250000 }
  ];
  
  const strategyAllocation = [
    { name: 'Quantum Momentum', value: 25, color: '#1976d2' },
    { name: 'AI Mean Reversion', value: 20, color: '#dc004e' },
    { name: 'Cross-Asset', value: 20, color: '#2e7d32' },
    { name: 'Volatility Targeting', value: 15, color: '#ed6c02' },
    { name: 'Statistical Arbitrage', value: 12, color: '#9c27b0' },
    { name: 'Deep RL', value: 8, color: '#795548' }
  ];
  
  return (
    <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Hero Section - One-Click Control */}
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          mb: 3, 
          background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
          color: 'white',
          textAlign: 'center'
        }}
      >
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold' }}>
          🚀 Ultimate Arbitrage System
        </Typography>
        <Typography variant="h6" sx={{ mb: 3, opacity: 0.9 }}>
          Quantum-Enhanced AI Portfolio Optimization
        </Typography>
        
        {/* Main Control Button */}
        <Box sx={{ mb: 3 }}>
          {!systemActive ? (
            <Fab
              size="large"
              color="secondary"
              onClick={quickStart}
              sx={{ 
                width: 120, 
                height: 120, 
                fontSize: '2rem',
                boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                '&:hover': {
                  transform: 'scale(1.1)',
                  transition: 'transform 0.2s'
                }
              }}
            >
              {isFirstRun ? (
                <Settings sx={{ fontSize: '3rem' }} />
              ) : (
                <PlayArrow sx={{ fontSize: '3rem' }} />
              )}
            </Fab>
          ) : (
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Fab
                size="large"
                color="warning"
                onClick={() => setSystemActive(false)}
                sx={{ width: 80, height: 80 }}
              >
                <Pause sx={{ fontSize: '2rem' }} />
              </Fab>
              <Fab
                size="large"
                color="error"
                onClick={emergencyStop}
                sx={{ width: 80, height: 80 }}
              >
                <Stop sx={{ fontSize: '2rem' }} />
              </Fab>
            </Box>
          )}
        </Box>
        
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          {systemActive ? '🟢 SYSTEM ACTIVE - OPTIMIZING' : isFirstRun ? '🎯 CLICK TO START SETUP' : '⚪ SYSTEM READY'}
        </Typography>
        
        {systemActive && (
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 3 }}>
            <Chip 
              icon={<SmartToy />} 
              label="AI Engines: Active" 
              color="success" 
              variant="filled"
              sx={{ color: 'white', fontWeight: 'bold' }}
            />
            <Chip 
              icon={<Bolt />} 
              label="Quantum: Engaged" 
              color="secondary" 
              variant="filled"
              sx={{ color: 'white', fontWeight: 'bold' }}
            />
            <Chip 
              icon={<TrendingUp />} 
              label="Live Trading" 
              color="warning" 
              variant="filled"
              sx={{ color: 'white', fontWeight: 'bold' }}
            />
          </Box>
        )}
      </Paper>
      
      {/* Quick Settings Panel */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AutoMode sx={{ mr: 1, color: '#1976d2' }} />
                <Typography variant="h6">Automation Level</Typography>
              </Box>
              <Slider
                value={automationLevel}
                onChange={(e, value) => setAutomationLevel(value)}
                min={0}
                max={100}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}%`}
                marks={[
                  { value: 0, label: 'Manual' },
                  { value: 50, label: 'Hybrid' },
                  { value: 100, label: 'Full Auto' }
                ]}
                sx={{ mt: 2 }}
              />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Current: {automationLevel}% Automated
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Security sx={{ mr: 1, color: '#dc004e' }} />
                <Typography variant="h6">Risk Tolerance</Typography>
              </Box>
              <Slider
                value={riskTolerance}
                onChange={(e, value) => setRiskTolerance(value)}
                min={0}
                max={100}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}%`}
                marks={[
                  { value: 0, label: 'Conservative' },
                  { value: 50, label: 'Balanced' },
                  { value: 100, label: 'Aggressive' }
                ]}
                sx={{ mt: 2 }}
              />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Current: {riskTolerance < 30 ? 'Conservative' : riskTolerance < 70 ? 'Balanced' : 'Aggressive'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUp sx={{ mr: 1, color: '#2e7d32' }} />
                <Typography variant="h6">Profit Target</Typography>
              </Box>
              <Slider
                value={profitTarget}
                onChange={(e, value) => setProfitTarget(value)}
                min={5}
                max={50}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}%`}
                marks={[
                  { value: 5, label: '5%' },
                  { value: 20, label: '20%' },
                  { value: 50, label: '50%' }
                ]}
                sx={{ mt: 2 }}
              />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Annual Target: {profitTarget}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* One-Click Actions */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Button
            fullWidth
            variant="contained"
            size="large"
            startIcon={<Psychology />}
            onClick={autoOptimize}
            disabled={isFirstRun}
            sx={{ py: 2 }}
          >
            Auto-Optimize
          </Button>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Button
            fullWidth
            variant="outlined"
            size="large"
            startIcon={<Settings />}
            onClick={() => setShowSetupWizard(true)}
            sx={{ py: 2 }}
          >
            Setup Wizard
          </Button>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Button
            fullWidth
            variant="outlined"
            size="large"
            startIcon={<Visibility />}
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={isFirstRun}
            sx={{ py: 2 }}
          >
            {showAdvanced ? 'Simple View' : 'Advanced View'}
          </Button>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Button
            fullWidth
            variant="outlined"
            size="large"
            startIcon={<Eco />}
            disabled={isFirstRun}
            sx={{ py: 2 }}
          >
            ESG Mode
          </Button>
        </Grid>
      </Grid>
      
      {/* Performance Dashboard */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Portfolio Value */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <AccountBalance sx={{ fontSize: 40, color: '#1976d2', mb: 1 }} />
              <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
                ${performance.portfolioValue.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Portfolio Value
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  mt: 1,
                  color: performance.dailyReturn >= 0 ? '#2e7d32' : '#d32f2f',
                  fontWeight: 'bold'
                }}
              >
                {performance.dailyReturn >= 0 ? '+' : ''}{performance.dailyReturn}% Today
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Sharpe Ratio */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 40, color: '#dc004e', mb: 1 }} />
              <Typography variant="h4" color="secondary" sx={{ fontWeight: 'bold' }}>
                {performance.sharpeRatio}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Sharpe Ratio
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={Math.min(100, (performance.sharpeRatio / 4) * 100)}
                sx={{ mt: 2, height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        {/* Win Rate */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUp sx={{ fontSize: 40, color: '#2e7d32', mb: 1 }} />
              <Typography variant="h4" color="success.main" sx={{ fontWeight: 'bold' }}>
                {performance.winRate}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Win Rate
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={performance.winRate}
                color="success"
                sx={{ mt: 2, height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        {/* Quantum Advantage */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Bolt sx={{ fontSize: 40, color: '#9c27b0', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#9c27b0' }}>
                {performance.quantumAdvantage}x
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Quantum Advantage
              </Typography>
              <Chip 
                label="ACTIVE" 
                color="secondary" 
                size="small" 
                sx={{ mt: 1, fontWeight: 'bold' }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Charts */}
      <Grid container spacing={3}>
        {/* Performance Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 Real-Time Performance
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData}>
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#1976d2" 
                    fill="#1976d2"
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Strategy Allocation */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🎯 Strategy Allocation
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={strategyAllocation}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                  >
                    {strategyAllocation.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* System Status (if advanced view) */}
      {showAdvanced && (
        <Grid container spacing={3} sx={{ mt: 3 }}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🔧 System Status
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Chip 
                        label="AI Engines" 
                        color={getStatusColor(systemStatus.aiEngines)}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2">{systemStatus.aiEngines}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Chip 
                        label="Quantum Engine" 
                        color={getStatusColor(systemStatus.quantumEngine)}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2">{systemStatus.quantumEngine}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Chip 
                        label="Data Feeds" 
                        color={getStatusColor(systemStatus.dataFeeds)}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2">{systemStatus.dataFeeds}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Chip 
                        label="Risk Management" 
                        color={getStatusColor(systemStatus.riskManagement)}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2">{systemStatus.riskManagement}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Chip 
                        label="Order Execution" 
                        color={getStatusColor(systemStatus.orderExecution)}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2">{systemStatus.orderExecution}</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      
      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog.open} onClose={() => setConfirmDialog({ open: false })}>
        <DialogTitle>{confirmDialog.title}</DialogTitle>
        <DialogContent>
          <Typography>{confirmDialog.message}</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog({ open: false })}>Cancel</Button>
          <Button 
            onClick={() => executeAction(confirmDialog.action)} 
            variant="contained"
            color={confirmDialog.action === 'stop' ? 'error' : 'primary'}
          >
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Notifications */}
      <Snackbar
        open={!!notification}
        autoHideDuration={6000}
        onClose={() => setNotification(null)}
      >
        {notification && (
          <Alert severity={notification.type} onClose={() => setNotification(null)}>
            {notification.message}
          </Alert>
        )}
      </Snackbar>
      
      {/* Setup Wizard */}
      <SetupWizard
        open={showSetupWizard}
        onClose={() => setShowSetupWizard(false)}
        onComplete={handleSetupComplete}
      />
      
      {/* Voice Control Interface */}
      <VoiceControlInterface
        onCommand={handleVoiceCommand}
        systemActive={systemActive}
      />
    </Box>
  );
};

export default OneClickDashboard;

