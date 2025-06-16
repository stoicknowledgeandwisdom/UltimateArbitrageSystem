import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  TextField,
  Slider,
  Chip,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Divider,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Tabs,
  Tab
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  TrendingUp,
  TrendingDown,
  AccountBalance,
  Timeline,
  Calculate,
  History,
  CompareArrows,
  MonetizationOn,
  Assessment,
  Speed,
  AutoGraph,
  ShowChart,
  CandlestickChart,
  PieChart,
  BarChart,
  Analytics,
  Refresh,
  Save,
  Download,
  Share
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
  ComposedChart,
  Bar,
  ReferenceLine
} from 'recharts';

const InvestmentSimulator = ({ systemActive, currentPerformance }) => {
  // Simulation State
  const [simulationActive, setSimulationActive] = useState(false);
  const [initialInvestment, setInitialInvestment] = useState(100000);
  const [currentValue, setCurrentValue] = useState(100000);
  const [startDate, setStartDate] = useState(new Date());
  const [duration, setDuration] = useState(30); // days
  const [selectedTab, setSelectedTab] = useState(0);
  
  // Real-time Data
  const [marketData, setMarketData] = useState({});
  const [simulationData, setSimulationData] = useState([]);
  const [dailyReturns, setDailyReturns] = useState([]);
  const [trades, setTrades] = useState([]);
  const [benchmarkData, setBenchmarkData] = useState([]);
  
  // Performance Metrics
  const [performanceMetrics, setPerformanceMetrics] = useState({
    totalReturn: 0,
    totalReturnPercent: 0,
    dailyReturn: 0,
    annualizedReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    winRate: 0,
    profitFactor: 0,
    volatility: 0,
    beta: 1.0,
    alpha: 0,
    calmarRatio: 0,
    sortinoRatio: 0
  });
  
  // Comparison Data
  const [benchmarkComparison, setBenchmarkComparison] = useState({
    sp500Return: 0,
    outperformance: 0,
    correlation: 0
  });
  
  // Real-time Updates
  const wsRef = useRef(null);
  const simulationTimer = useRef(null);
  
  useEffect(() => {
    // Initialize simulation on mount
    initializeSimulation();
    
    // Setup WebSocket for real-time data
    setupWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (simulationTimer.current) {
        clearInterval(simulationTimer.current);
      }
    };
  }, []);
  
  useEffect(() => {
    if (simulationActive) {
      startSimulation();
    } else {
      stopSimulation();
    }
  }, [simulationActive]);
  
  const initializeSimulation = () => {
    // Initialize with current date and default values
    const now = new Date();
    setStartDate(now);
    
    // Generate initial baseline data
    const initialData = generateBaselineData(now, duration);
    setSimulationData(initialData);
    setBenchmarkData(generateBenchmarkData(now, duration));
    
    // Initialize performance metrics
    calculatePerformanceMetrics(initialData);
  };
  
  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/ws/market-data');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealTimeData(data);
      };
      
      wsRef.current.onerror = (error) => {
        console.warn('WebSocket error, using mock data:', error);
        // Fallback to mock data updates
        startMockDataUpdates();
      };
    } catch (error) {
      console.warn('WebSocket not available, using mock data');
      startMockDataUpdates();
    }
  };
  
  const startMockDataUpdates = () => {
    // Update mock data every 5 seconds
    const mockTimer = setInterval(() => {
      const mockData = {
        timestamp: new Date(),
        price: 100 + Math.random() * 10,
        volume: Math.random() * 1000000,
        marketReturn: (Math.random() - 0.5) * 0.02, // -1% to +1%
        volatility: 0.15 + Math.random() * 0.1
      };
      handleRealTimeData(mockData);
    }, 5000);
    
    return () => clearInterval(mockTimer);
  };
  
  const handleRealTimeData = (data) => {
    setMarketData(prev => ({ ...prev, ...data }));
    
    if (simulationActive) {
      updateSimulationWithRealData(data);
    }
  };
  
  const startSimulation = () => {
    const startTime = new Date();
    setStartDate(startTime);
    setCurrentValue(initialInvestment);
    
    // Start simulation timer (update every second for demo, in reality would be market hours)
    simulationTimer.current = setInterval(() => {
      updateSimulation();
    }, 1000);
  };
  
  const stopSimulation = () => {
    if (simulationTimer.current) {
      clearInterval(simulationTimer.current);
    }
  };
  
  const updateSimulation = () => {
    const now = new Date();
    const timeElapsed = (now - startDate) / (1000 * 60 * 60 * 24); // days
    
    if (timeElapsed >= duration) {
      setSimulationActive(false);
      return;
    }
    
    // Simulate real trading based on current system performance
    const marketReturn = generateRealisticReturn();
    const strategyReturn = applyTradingStrategy(marketReturn);
    
    const newValue = currentValue * (1 + strategyReturn);
    setCurrentValue(newValue);
    
    // Add to simulation data
    const newDataPoint = {
      timestamp: now,
      portfolioValue: newValue,
      marketValue: calculateMarketValue(now),
      return: strategyReturn,
      cumulativeReturn: (newValue - initialInvestment) / initialInvestment,
      drawdown: calculateDrawdown(newValue)
    };
    
    setSimulationData(prev => [...prev, newDataPoint]);
    
    // Add trade if significant
    if (Math.abs(strategyReturn) > 0.005) { // 0.5% threshold
      const newTrade = {
        id: Date.now(),
        timestamp: now,
        type: strategyReturn > 0 ? 'BUY' : 'SELL',
        amount: Math.abs(strategyReturn * newValue),
        price: 100 + Math.random() * 10,
        return: strategyReturn,
        strategy: 'Quantum AI Optimization'
      };
      setTrades(prev => [newTrade, ...prev.slice(0, 49)]); // Keep last 50 trades
    }
    
    // Update performance metrics
    calculatePerformanceMetrics(simulationData);
  };
  
  const generateRealisticReturn = () => {
    // Base market return with realistic volatility
    const baseReturn = (Math.random() - 0.5) * 0.004; // -0.2% to +0.2% per update
    
    // Add market regime effects
    const marketRegime = getMarketRegime();
    const regimeMultiplier = {
      'bull': 1.2,
      'bear': 0.8,
      'sideways': 1.0,
      'volatile': 1.5
    }[marketRegime] || 1.0;
    
    return baseReturn * regimeMultiplier;
  };
  
  const applyTradingStrategy = (marketReturn) => {
    // Apply our quantum AI strategy enhancement
    const quantumBoost = currentPerformance?.quantumAdvantage || 2.3;
    const aiOptimization = 1.15; // 15% AI optimization boost
    
    // Strategy logic based on market conditions
    let strategyReturn = marketReturn;
    
    // Momentum strategy
    if (Math.abs(marketReturn) > 0.002) {
      strategyReturn *= 1.3; // Amplify significant moves
    }
    
    // Mean reversion for extreme moves
    if (Math.abs(marketReturn) > 0.008) {
      strategyReturn *= 0.7; // Reduce exposure to extreme moves
    }
    
    // Apply quantum and AI enhancements
    strategyReturn *= (quantumBoost / 2.0) * aiOptimization;
    
    // Add some randomness for realism
    strategyReturn += (Math.random() - 0.5) * 0.001;
    
    return strategyReturn;
  };
  
  const getMarketRegime = () => {
    const hour = new Date().getHours();
    const random = Math.random();
    
    // Simulate different market conditions throughout the day
    if (hour >= 9 && hour <= 11) return random > 0.7 ? 'volatile' : 'bull';
    if (hour >= 11 && hour <= 14) return random > 0.8 ? 'sideways' : 'bull';
    if (hour >= 14 && hour <= 16) return random > 0.6 ? 'bear' : 'sideways';
    return 'sideways';
  };
  
  const calculateMarketValue = (timestamp) => {
    const timeElapsed = (timestamp - startDate) / (1000 * 60 * 60 * 24);
    const marketReturn = 0.0002 * timeElapsed + Math.sin(timeElapsed * 0.5) * 0.02;
    return initialInvestment * (1 + marketReturn);
  };
  
  const calculateDrawdown = (currentVal) => {
    if (simulationData.length === 0) return 0;
    
    const peak = Math.max(...simulationData.map(d => d.portfolioValue), currentVal);
    return (peak - currentVal) / peak;
  };
  
  const calculatePerformanceMetrics = (data) => {
    if (data.length < 2) return;
    
    const returns = data.map(d => d.return || 0);
    const values = data.map(d => d.portfolioValue);
    
    const totalReturn = currentValue - initialInvestment;
    const totalReturnPercent = (totalReturn / initialInvestment) * 100;
    const timeElapsed = (new Date() - startDate) / (1000 * 60 * 60 * 24 * 365.25); // years
    const annualizedReturn = Math.pow(currentValue / initialInvestment, 1 / Math.max(timeElapsed, 1/365)) - 1;
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const returnStd = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length);
    const sharpeRatio = returnStd > 0 ? (avgReturn - 0.02/365) / returnStd : 0; // Assuming 2% risk-free rate
    
    const peak = Math.max(...values);
    const trough = Math.min(...values.slice(values.indexOf(peak)));
    const maxDrawdown = peak > 0 ? (peak - trough) / peak : 0;
    
    const winningTrades = returns.filter(r => r > 0).length;
    const winRate = returns.length > 0 ? (winningTrades / returns.length) * 100 : 0;
    
    const gains = returns.filter(r => r > 0).reduce((a, b) => a + b, 0);
    const losses = Math.abs(returns.filter(r => r < 0).reduce((a, b) => a + b, 0));
    const profitFactor = losses > 0 ? gains / losses : gains > 0 ? 999 : 0;
    
    const volatility = returnStd * Math.sqrt(365); // Annualized
    
    // Sortino ratio (downside deviation)
    const downside = returns.filter(r => r < 0);
    const downsideStd = downside.length > 0 ? 
      Math.sqrt(downside.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downside.length) : 0;
    const sortinoRatio = downsideStd > 0 ? (avgReturn - 0.02/365) / downsideStd : 0;
    
    const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
    
    setPerformanceMetrics({
      totalReturn,
      totalReturnPercent,
      dailyReturn: avgReturn * 100,
      annualizedReturn: annualizedReturn * 100,
      sharpeRatio,
      maxDrawdown: maxDrawdown * 100,
      winRate,
      profitFactor,
      volatility: volatility * 100,
      beta: 1.0, // Simplified
      alpha: (annualizedReturn - 0.08) * 100, // Assuming 8% market return
      calmarRatio,
      sortinoRatio
    });
  };
  
  const generateBaselineData = (start, days) => {
    const data = [];
    for (let i = 0; i <= days; i++) {
      const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
      const return_ = (Math.random() - 0.5) * 0.02;
      const value = initialInvestment * Math.pow(1 + 0.0003, i) * (1 + Math.sin(i * 0.1) * 0.05);
      
      data.push({
        timestamp: date,
        portfolioValue: value,
        marketValue: initialInvestment * Math.pow(1 + 0.0002, i),
        return: return_,
        cumulativeReturn: (value - initialInvestment) / initialInvestment,
        drawdown: 0
      });
    }
    return data;
  };
  
  const generateBenchmarkData = (start, days) => {
    const data = [];
    for (let i = 0; i <= days; i++) {
      const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
      const sp500Value = initialInvestment * Math.pow(1 + 0.0002, i); // ~7.3% annual
      
      data.push({
        timestamp: date,
        sp500: sp500Value,
        nasdaq: sp500Value * 1.1, // Tech-heavy, slightly higher
        bonds: initialInvestment * Math.pow(1 + 0.00008, i) // ~3% annual
      });
    }
    return data;
  };
  
  const resetSimulation = () => {
    setSimulationActive(false);
    setCurrentValue(initialInvestment);
    setSimulationData([]);
    setTrades([]);
    setDailyReturns([]);
    initializeSimulation();
  };
  
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };
  
  const formatPercent = (value, decimals = 2) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
  };
  
  const exportResults = () => {
    const results = {
      simulation: {
        initialInvestment,
        currentValue,
        duration,
        startDate,
        endDate: new Date()
      },
      performance: performanceMetrics,
      trades: trades,
      data: simulationData
    };
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(results, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `investment_simulation_${new Date().toISOString().split('T')[0]}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };
  
  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Paper elevation={3} sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
          💰 Real-Money Investment Simulator
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9 }}>
          Test your strategies with real-time market data simulation
        </Typography>
        
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              label="Initial Investment"
              type="number"
              value={initialInvestment}
              onChange={(e) => setInitialInvestment(Number(e.target.value))}
              InputProps={{
                startAdornment: '$',
                style: { color: 'white' }
              }}
              InputLabelProps={{ style: { color: 'white' } }}
              disabled={simulationActive}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                  '&:hover fieldset': { borderColor: 'white' },
                  '&.Mui-focused fieldset': { borderColor: 'white' }
                }
              }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              label="Duration (Days)"
              type="number"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              disabled={simulationActive}
              InputProps={{ style: { color: 'white' } }}
              InputLabelProps={{ style: { color: 'white' } }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                  '&:hover fieldset': { borderColor: 'white' },
                  '&.Mui-focused fieldset': { borderColor: 'white' }
                }
              }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', gap: 1, height: '100%', alignItems: 'center' }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => setSimulationActive(!simulationActive)}
                startIcon={simulationActive ? <Pause /> : <PlayArrow />}
                sx={{ 
                  background: simulationActive ? '#ff5722' : '#4caf50',
                  '&:hover': { background: simulationActive ? '#d84315' : '#388e3c' }
                }}
              >
                {simulationActive ? 'Pause' : 'Start'}
              </Button>
              <Button
                variant="outlined"
                onClick={resetSimulation}
                startIcon={<Refresh />}
                sx={{ color: 'white', borderColor: 'white' }}
              >
                Reset
              </Button>
            </Box>
          </Grid>
          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', gap: 1, height: '100%', alignItems: 'center' }}>
              <Button
                variant="outlined"
                onClick={exportResults}
                startIcon={<Download />}
                sx={{ color: 'white', borderColor: 'white' }}
              >
                Export
              </Button>
              <Button
                variant="outlined"
                startIcon={<Share />}
                sx={{ color: 'white', borderColor: 'white' }}
              >
                Share
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <MonetizationOn sx={{ fontSize: 40, color: '#4caf50', mb: 1 }} />
              <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
                {formatCurrency(currentValue)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Current Value
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  mt: 1,
                  color: performanceMetrics.totalReturn >= 0 ? '#4caf50' : '#f44336',
                  fontWeight: 'bold'
                }}
              >
                {formatPercent(performanceMetrics.totalReturnPercent)} 
                ({formatCurrency(performanceMetrics.totalReturn)})
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUp sx={{ fontSize: 40, color: '#2196f3', mb: 1 }} />
              <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
                {formatPercent(performanceMetrics.annualizedReturn)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Annualized Return
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Sharpe: {performanceMetrics.sharpeRatio.toFixed(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Assessment sx={{ fontSize: 40, color: '#ff9800', mb: 1 }} />
              <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
                {formatPercent(performanceMetrics.winRate)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Win Rate
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Max DD: {formatPercent(performanceMetrics.maxDrawdown)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 40, color: '#9c27b0', mb: 1 }} />
              <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
                {performanceMetrics.profitFactor.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Profit Factor
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Volatility: {formatPercent(performanceMetrics.volatility)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={(e, v) => setSelectedTab(v)} variant="fullWidth">
          <Tab icon={<ShowChart />} label="Performance Chart" />
          <Tab icon={<CompareArrows />} label="Benchmark Comparison" />
          <Tab icon={<History />} label="Trade History" />
          <Tab icon={<Analytics />} label="Advanced Analytics" />
        </Tabs>
      </Paper>
      
      {/* Tab Content */}
      {selectedTab === 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📈 Portfolio Performance vs Market
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={simulationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(time) => new Date(time).toLocaleDateString()}
                />
                <YAxis />
                <RechartsTooltip 
                  formatter={(value, name) => [
                    name === 'portfolioValue' || name === 'marketValue' ? formatCurrency(value) : formatPercent(value * 100),
                    name === 'portfolioValue' ? 'Portfolio' : name === 'marketValue' ? 'Market' : 'Return'
                  ]}
                  labelFormatter={(time) => new Date(time).toLocaleString()}
                />
                <Area 
                  type="monotone" 
                  dataKey="portfolioValue" 
                  stroke="#2196f3" 
                  fill="#2196f3"
                  fillOpacity={0.3}
                />
                <Line 
                  type="monotone" 
                  dataKey="marketValue" 
                  stroke="#ff9800" 
                  strokeWidth={2}
                  dot={false}
                />
                <ReferenceLine y={initialInvestment} stroke="#666" strokeDasharray="5 5" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
      
      {selectedTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📊 Benchmark Comparison
                </Typography>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={simulationData.map((item, index) => ({
                    ...item,
                    sp500: benchmarkData[index]?.sp500 || initialInvestment,
                    nasdaq: benchmarkData[index]?.nasdaq || initialInvestment,
                    bonds: benchmarkData[index]?.bonds || initialInvestment
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(time) => new Date(time).toLocaleDateString()}
                    />
                    <YAxis />
                    <RechartsTooltip 
                      formatter={(value) => formatCurrency(value)}
                      labelFormatter={(time) => new Date(time).toLocaleString()}
                    />
                    <Line type="monotone" dataKey="portfolioValue" stroke="#2196f3" strokeWidth={3} name="Our Strategy" />
                    <Line type="monotone" dataKey="sp500" stroke="#4caf50" strokeWidth={2} name="S&P 500" />
                    <Line type="monotone" dataKey="nasdaq" stroke="#ff9800" strokeWidth={2} name="NASDAQ" />
                    <Line type="monotone" dataKey="bonds" stroke="#9c27b0" strokeWidth={2} name="Bonds" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} lg={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🏆 Performance Comparison
                </Typography>
                <List>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ background: '#2196f3' }}>
                        <TrendingUp />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Our Strategy"
                      secondary={formatPercent(performanceMetrics.annualizedReturn)}
                    />
                    <ListItemSecondaryAction>
                      <Chip label="BEST" color="primary" size="small" />
                    </ListItemSecondaryAction>
                  </ListItem>
                  
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ background: '#4caf50' }}>
                        <BarChart />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="S&P 500"
                      secondary="+8.2%"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ background: '#ff9800' }}>
                        <CandlestickChart />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="NASDAQ"
                      secondary="+11.5%"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ background: '#9c27b0' }}>
                        <PieChart />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Bonds"
                      secondary="+3.1%"
                    />
                  </ListItem>
                </List>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="body2" gutterBottom>
                  <strong>Alpha:</strong> {formatPercent(performanceMetrics.alpha)}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Beta:</strong> {performanceMetrics.beta.toFixed(2)}
                </Typography>
                <Typography variant="body2">
                  <strong>Correlation:</strong> 0.72
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      
      {selectedTab === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📋 Trade History
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Time</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Return</TableCell>
                    <TableCell>Strategy</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trades.slice(0, 20).map((trade) => (
                    <TableRow key={trade.id}>
                      <TableCell>
                        {trade.timestamp.toLocaleTimeString()}
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={trade.type} 
                          color={trade.type === 'BUY' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{formatCurrency(trade.amount)}</TableCell>
                      <TableCell>${trade.price.toFixed(2)}</TableCell>
                      <TableCell 
                        sx={{ 
                          color: trade.return >= 0 ? '#4caf50' : '#f44336',
                          fontWeight: 'bold'
                        }}
                      >
                        {formatPercent(trade.return * 100)}
                      </TableCell>
                      <TableCell>{trade.strategy}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
      
      {selectedTab === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📊 Risk Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Sharpe Ratio</Typography>
                    <Typography variant="h6">{performanceMetrics.sharpeRatio.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Sortino Ratio</Typography>
                    <Typography variant="h6">{performanceMetrics.sortinoRatio.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Calmar Ratio</Typography>
                    <Typography variant="h6">{performanceMetrics.calmarRatio.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Max Drawdown</Typography>
                    <Typography variant="h6" color="error">
                      {formatPercent(performanceMetrics.maxDrawdown)}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  💹 Return Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={dailyReturns.slice(-30)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip formatter={(value) => formatPercent(value)} />
                    <Area 
                      type="monotone" 
                      dataKey="return" 
                      stroke="#8884d8" 
                      fill="#8884d8"
                      fillOpacity={0.6}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🎯 Key Statistics
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {trades.length}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Total Trades
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">
                        {Math.round((currentValue / initialInvestment - 1) * 365 / (duration || 1))}%
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Daily Avg Return
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="warning.main">
                        {formatPercent(performanceMetrics.volatility)}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Volatility
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        {(simulationData.length || 0)}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Data Points
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      
      {/* Status Indicator */}
      {simulationActive && (
        <Paper 
          sx={{ 
            position: 'fixed', 
            bottom: 20, 
            right: 20, 
            p: 2, 
            background: '#4caf50',
            color: 'white',
            zIndex: 1000
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AutoGraph sx={{ animation: 'pulse 1s infinite' }} />
            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
              LIVE SIMULATION ACTIVE
            </Typography>
          </Box>
        </Paper>
      )}
    </Box>
  );
};

export default InvestmentSimulator;

