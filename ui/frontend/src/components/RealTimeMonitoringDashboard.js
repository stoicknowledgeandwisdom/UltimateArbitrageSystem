import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Switch,
  FormControlLabel,
  Alert,
  Tooltip,
  Badge,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  Settings,
  Warning,
  CheckCircle,
  Error,
  ExpandMore,
  Speed,
  Timeline,
  AccountBalance
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar
} from 'recharts';

const RealTimeMonitoringDashboard = ({ 
  portfolioData, 
  isOptimizationRunning, 
  onToggleOptimization,
  onEmergencyStop
}) => {
  // State for real-time data
  const [liveData, setLiveData] = useState({
    performance: [],
    positions: [],
    orders: [],
    riskMetrics: {},
    marketData: {},
    systemHealth: {},
    alerts: []
  });
  
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(1000); // 1 second
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [dataQuality, setDataQuality] = useState('excellent');
  
  const wsRef = useRef(null);
  const refreshTimerRef = useRef(null);
  
  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/monitoring';
        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onopen = () => {
          console.log('WebSocket connected');
          setConnectionStatus('connected');
        };
        
        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleRealTimeUpdate(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        wsRef.current.onclose = () => {
          console.log('WebSocket disconnected');
          setConnectionStatus('disconnected');
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };
        
        wsRef.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnectionStatus('error');
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setConnectionStatus('error');
      }
    };
    
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);
  
  // Handle real-time data updates
  const handleRealTimeUpdate = useCallback((data) => {
    setLiveData(prevData => {
      const newData = { ...prevData };
      
      // Update performance data
      if (data.performance) {
        newData.performance = [
          ...prevData.performance.slice(-49), // Keep last 50 points
          {
            timestamp: new Date().toLocaleTimeString(),
            value: data.performance.portfolio_value,
            returns: data.performance.returns,
            sharpe: data.performance.sharpe_ratio,
            drawdown: data.performance.max_drawdown
          }
        ];
      }
      
      // Update positions
      if (data.positions) {
        newData.positions = data.positions;
      }
      
      // Update orders
      if (data.orders) {
        newData.orders = data.orders;
      }
      
      // Update risk metrics
      if (data.risk_metrics) {
        newData.riskMetrics = data.risk_metrics;
      }
      
      // Update market data
      if (data.market_data) {
        newData.marketData = data.market_data;
      }
      
      // Update system health
      if (data.system_health) {
        newData.systemHealth = data.system_health;
      }
      
      // Update alerts
      if (data.alerts) {
        newData.alerts = data.alerts.slice(-10); // Keep last 10 alerts
      }
      
      return newData;
    });
    
    // Update data quality based on latency
    const latency = data.latency_ms || 0;
    if (latency < 100) {
      setDataQuality('excellent');
    } else if (latency < 500) {
      setDataQuality('good');
    } else if (latency < 1000) {
      setDataQuality('fair');
    } else {
      setDataQuality('poor');
    }
  }, []);
  
  // Auto-refresh timer
  useEffect(() => {
    if (autoRefresh && connectionStatus !== 'connected') {
      refreshTimerRef.current = setInterval(() => {
        fetchLatestData();
      }, refreshInterval);
    } else {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    }
    
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, connectionStatus]);
  
  const fetchLatestData = async () => {
    try {
      const response = await fetch('/api/monitoring/live-data');
      const data = await response.json();
      handleRealTimeUpdate(data);
    } catch (error) {
      console.error('Failed to fetch live data:', error);
    }
  };
  
  // Status indicator component
  const StatusIndicator = ({ status, label }) => {
    const getStatusColor = (status) => {
      switch (status) {
        case 'healthy':
        case 'connected':
        case 'excellent':
          return 'success';
        case 'warning':
        case 'degraded':
        case 'good':
          return 'warning';
        case 'error':
        case 'disconnected':
        case 'poor':
          return 'error';
        default:
          return 'default';
      }
    };
    
    return (
      <Chip
        icon={status === 'healthy' || status === 'connected' ? <CheckCircle /> : <Warning />}
        label={label}
        color={getStatusColor(status)}
        size="small"
        variant="outlined"
      />
    );
  };
  
  // Performance chart colors
  const chartColors = {
    primary: '#1976d2',
    secondary: '#dc004e',
    success: '#2e7d32',
    warning: '#ed6c02',
    error: '#d32f2f'
  };
  
  return (
    <Box sx={{ p: 3 }}>
      {/* Header with system status */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>}
        <Typography variant="h4" component="h1">
          Real-Time Monitoring Dashboard
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>}
          <StatusIndicator status={connectionStatus} label={`Connection: ${connectionStatus}`} />
          <StatusIndicator status={dataQuality} label={`Data Quality: ${dataQuality}`} />
          
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          
          <IconButton 
            onClick={fetchLatestData}
            disabled={connectionStatus === 'connected'}
            color="primary"
          >
            <Refresh />
          </IconButton>
        </Box>
      </Box>
      
      {/* Alerts Section */}
      {liveData.alerts && liveData.alerts.length > 0 && (
        <Box sx={{ mb: 3 }}>}
          {liveData.alerts.map((alert, index) => (
            <Alert 
              key={index} 
              severity={alert.severity || 'info'}
              sx={{ mb: 1 }}
            >
              <strong>{alert.title}:</strong> {alert.message}
            </Alert>
          ))}
        </Box>
      )}
      
      {/* Key Metrics Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>}
                <AccountBalance sx={{ mr: 1, color: chartColors.primary }} />
                <Typography variant="h6">Portfolio Value</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                ${(liveData.riskMetrics.portfolio_value || 0).toLocaleString()}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>}
                {(liveData.riskMetrics.daily_return || 0) >= 0 ? 
                  <TrendingUp sx={{ color: chartColors.success, mr: 0.5 }} /> :
                  <TrendingDown sx={{ color: chartColors.error, mr: 0.5 }} />
                }
                <Typography 
                  variant="body2" 
                  color={(liveData.riskMetrics.daily_return || 0) >= 0 ? chartColors.success : chartColors.error}
                >
                  {((liveData.riskMetrics.daily_return || 0) * 100).toFixed(2)}% today
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>}
                <Speed sx={{ mr: 1, color: chartColors.secondary }} />
                <Typography variant="h6">Sharpe Ratio</Typography>
              </Box>
              <Typography variant="h4" color="secondary">
                {(liveData.riskMetrics.sharpe_ratio || 0).toFixed(2)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Risk-adjusted returns
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>}
                <Timeline sx={{ mr: 1, color: chartColors.warning }} />
                <Typography variant="h6">Max Drawdown</Typography>
              </Box>
              <Typography variant="h4" color="textSecondary">
                {((liveData.riskMetrics.max_drawdown || 0) * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Peak to trough decline
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>}
                <Warning sx={{ mr: 1, color: chartColors.error }} />
                <Typography variant="h6">Active Alerts</Typography>
              </Box>
              <Typography variant="h4">
                <Badge badgeContent={liveData.alerts.length} color="error">
                  {liveData.alerts.length}
                </Badge>
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Requiring attention
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Performance Chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Performance (Real-Time)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={liveData.performance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <RechartsTooltip />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke={chartColors.primary} 
                    fill={chartColors.primary}
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Metrics
              </Typography>
              <Box sx={{ mb: 2 }}>}
                <Typography variant="body2" color="textSecondary">Volatility</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(liveData.riskMetrics.volatility || 0) * 100} 
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption">
                  {((liveData.riskMetrics.volatility || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>}
                <Typography variant="body2" color="textSecondary">VaR (95%)</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.abs(liveData.riskMetrics.var_95 || 0) * 100} 
                  color="error"
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption">
                  {((liveData.riskMetrics.var_95 || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>}
                <Typography variant="body2" color="textSecondary">Portfolio Concentration</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(liveData.riskMetrics.concentration || 0) * 100} 
                  color="warning"
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption">
                  {((liveData.riskMetrics.concentration || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Positions and Orders */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Current Positions</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Market Value</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell align="right">Weight</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {liveData.positions.map((position, index) => (
                      <TableRow key={index}>
                        <TableCell>{position.symbol}</TableCell>
                        <TableCell align="right">{position.quantity}</TableCell>
                        <TableCell align="right">$${position.market_value?.toLocaleString()}</TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: position.unrealized_pnl >= 0 ? chartColors.success : chartColors.error 
                          }}
                        >
                          ${position.unrealized_pnl?.toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          {(position.weight * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        </Grid>
        
        <Grid item xs={12} lg={6}>
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Active Orders</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {liveData.orders.map((order, index) => (
                      <TableRow key={index}>
                        <TableCell>{order.symbol}</TableCell>
                        <TableCell>
                          <Chip 
                            label={order.side} 
                            color={order.side === 'BUY' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{order.type}</TableCell>
                        <TableCell align="right">{order.quantity}</TableCell>
                        <TableCell align="right">${order.price}</TableCell>
                        <TableCell>
                          <Chip 
                            label={order.status} 
                            color={
                              order.status === 'FILLED' ? 'success' :
                              order.status === 'PENDING' ? 'warning' : 'default'
                            }
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>
      
      {/* Emergency Controls */}
      <Box sx={{ mt: 3, p: 2, border: '2px solid', borderColor: 'error.main', borderRadius: 2 }}>}
        <Typography variant="h6" color="error" gutterBottom>
          Emergency Controls
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>}
          <IconButton
            onClick={onToggleOptimization}
            color={isOptimizationRunning ? "warning" : "success"}
            size="large"
          >
            {isOptimizationRunning ? <Pause /> : <PlayArrow />}
          </IconButton>
          
          <IconButton
            onClick={onEmergencyStop}
            color="error"
            size="large"
          >
            <Stop />
          </IconButton>
          
          <Box sx={{ ml: 2 }}>}
            <Typography variant="body2" color="textSecondary">
              {isOptimizationRunning ? 
                'Click pause to temporarily stop optimization' :
                'Click play to resume optimization'
              }
            </Typography>
            <Typography variant="body2" color="error">
              Emergency stop will halt all trading immediately
            </Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default RealTimeMonitoringDashboard;

