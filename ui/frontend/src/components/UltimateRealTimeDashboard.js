import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Alert,
  Chip,
  LinearProgress,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Fab,
  Tooltip,
  Avatar,
  Badge,
  Snackbar,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Stepper,
  Step,
  StepLabel,
  StepContent
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Pause,
  AutoMode,
  TrendingUp,
  TrendingDown,
  Settings,
  Psychology,
  Speed,
  Bolt,
  MonetizationOn,
  Analytics,
  Security,
  AccountBalance,
  Timeline,
  SmartToy,
  Rocket,
  ExpandMore,
  Notifications,
  VolumeUp,
  Refresh,
  Download,
  Share,
  Visibility,
  Star,
  FlashOn,
  ShowChart,
  AccountBalanceWallet,
  Warning,
  CheckCircle,
  Error,
  Info,
  Whatshot,
  LocalFireDepartment,
  TrendingFlat,
  AttachMoney,
  CurrencyBitcoin,
  Diamond,
  Celebration,
  EmojiEvents,
  Casino,
  MoneyOff,
  Money,
  Paid,
  Savings
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadialBarChart,
  RadialBar,
  ComposedChart,
  Scatter,
  ScatterChart,
  ReferenceLine
} from 'recharts';
import { keyframes } from '@emotion/react';
import { styled } from '@mui/material/styles';

// Advanced Animations for Maximum Visual Impact
const profitPulse = keyframes`
  0% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
  }
  50% {
    transform: scale(1.05);
    box-shadow: 0 0 0 15px rgba(76, 175, 80, 0);
  }
  100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
  }
`;

const quantumGlow = keyframes`
  0% {
    box-shadow: 0 0 10px rgba(147, 51, 234, 0.5);
  }
  50% {
    box-shadow: 0 0 30px rgba(147, 51, 234, 0.8), 0 0 40px rgba(79, 70, 229, 0.6);
  }
  100% {
    box-shadow: 0 0 10px rgba(147, 51, 234, 0.5);
  }
`;

const fireAnimation = keyframes`
  0% {
    transform: rotate(-1deg) scale(1);
    filter: hue-rotate(0deg);
  }
  25% {
    transform: rotate(1deg) scale(1.02);
    filter: hue-rotate(5deg);
  }
  50% {
    transform: rotate(-0.5deg) scale(1.01);
    filter: hue-rotate(0deg);
  }
  75% {
    transform: rotate(0.5deg) scale(1.02);
    filter: hue-rotate(-5deg);
  }
  100% {
    transform: rotate(-1deg) scale(1);
    filter: hue-rotate(0deg);
  }
`;

const moneyRain = keyframes`
  0% {
    transform: translateY(-100vh) rotate(0deg);
    opacity: 1;
  }
  100% {
    transform: translateY(100vh) rotate(360deg);
    opacity: 0;
  }
`;

// Styled Components for Enhanced UI
const ProfitCard = styled(Card)(({ theme, isprofitable }) => ({
  background: isprofitable === 'true' ? 
    'linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%)' :
    'linear-gradient(135deg, #f44336 0%, #e57373 100%)',
  color: 'white',
  animation: isprofitable === 'true' ? `${profitPulse} 2s infinite` : 'none',
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 10px 30px rgba(0,0,0,0.3)'
  }
}));

const QuantumCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  color: 'white',
  animation: `${quantumGlow} 3s infinite`,
  border: '2px solid rgba(147, 51, 234, 0.5)'
}));

const FireCard = styled(Card)(({ theme, isactive }) => ({
  background: isactive === 'true' ? 
    'linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%)' :
    'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
  animation: isactive === 'true' ? `${fireAnimation} 0.5s infinite` : 'none'
}));

const MoneyIcon = styled(AttachMoney)(({ theme }) => ({
  animation: `${moneyRain} 3s linear infinite`,
  position: 'absolute',
  fontSize: '2rem',
  color: '#4CAF50',
  zIndex: 1000
}));

const UltimateRealTimeDashboard = () => {
  // State Management for Ultimate Dashboard
  const [systemActive, setSystemActive] = useState(false);
  const [quantumMode, setQuantumMode] = useState(false);
  const [aggressionLevel, setAggressionLevel] = useState(75);
  const [autoRebalance, setAutoRebalance] = useState(true);
  const [showMoneyRain, setShowMoneyRain] = useState(false);
  const [emergencyStop, setEmergencyStop] = useState(false);
  
  // Performance Data State
  const [performanceData, setPerformanceData] = useState({
    totalProfit: 847523.45,
    dailyProfit: 23847.12,
    hourlyProfit: 1243.58,
    minuteProfit: 20.73,
    winRate: 87.3,
    sharpeRatio: 3.24,
    maxDrawdown: -2.1,
    quantumAdvantage: 2.87,
    activeTrades: 247,
    successfulTrades: 1846,
    failedTrades: 263,
    averageTradeTime: 0.34,
    portfolioValue: 5847523.45,
    availableCapital: 1234567.89,
    allocatedCapital: 4612955.56,
    roi: 169.5,
    monthlyTarget: 50000,
    yearlyTarget: 600000,
    riskScore: 3.2,
    volatilityIndex: 0.18,
    correlationIndex: 0.65
  });
  
  // Strategy Performance Data
  const [strategyData, setStrategyData] = useState([
    {
      name: 'Quantum Arbitrage',
      allocation: 25.4,
      profit: 45623.12,
      trades: 89,
      winRate: 94.4,
      sharpe: 4.2,
      status: 'active',
      risk: 'low'
    },
    {
      name: 'Cross-Chain MEV',
      allocation: 18.7,
      profit: 38921.45,
      trades: 156,
      winRate: 82.1,
      sharpe: 3.8,
      status: 'active',
      risk: 'medium'
    },
    {
      name: 'AI Momentum',
      allocation: 15.2,
      profit: 29847.33,
      trades: 234,
      winRate: 76.5,
      sharpe: 3.1,
      status: 'active',
      risk: 'medium'
    },
    {
      name: 'Flash Loan Arbitrage',
      allocation: 12.8,
      profit: 21456.78,
      trades: 67,
      winRate: 89.6,
      sharpe: 3.9,
      status: 'active',
      risk: 'high'
    },
    {
      name: 'Triangular Arbitrage',
      allocation: 10.3,
      profit: 18394.21,
      trades: 312,
      winRate: 71.2,
      sharpe: 2.8,
      status: 'active',
      risk: 'low'
    },
    {
      name: 'Volatility Harvesting',
      allocation: 8.9,
      profit: 15672.89,
      trades: 445,
      winRate: 68.9,
      sharpe: 2.5,
      status: 'active',
      risk: 'medium'
    },
    {
      name: 'Options Arbitrage',
      allocation: 5.4,
      profit: 9823.45,
      trades: 78,
      winRate: 85.9,
      sharpe: 3.6,
      status: 'paused',
      risk: 'high'
    },
    {
      name: 'Social Sentiment',
      allocation: 3.3,
      profit: 7245.67,
      trades: 189,
      winRate: 63.5,
      sharpe: 2.1,
      status: 'active',
      risk: 'medium'
    }
  ]);
  
  // Real-time profit chart data
  const [chartData, setChartData] = useState([
    { time: '09:30', profit: 0, cumulative: 0 },
    { time: '09:45', profit: 1245, cumulative: 1245 },
    { time: '10:00', profit: 2341, cumulative: 3586 },
    { time: '10:15', profit: 1876, cumulative: 5462 },
    { time: '10:30', profit: 3421, cumulative: 8883 },
    { time: '10:45', profit: 2198, cumulative: 11081 },
    { time: '11:00', profit: 4567, cumulative: 15648 },
    { time: '11:15', profit: 3234, cumulative: 18882 },
    { time: '11:30', profit: 2987, cumulative: 21869 },
    { time: '11:45', profit: 1954, cumulative: 23823 }
  ]);
  
  // Opportunities data
  const [opportunities, setOpportunities] = useState([
    {
      id: 1,
      type: 'Quantum Arbitrage',
      pair: 'BTC/USDT',
      profit: 2.34,
      confidence: 96.8,
      timeLeft: 12,
      status: 'executing'
    },
    {
      id: 2,
      type: 'Cross-Chain MEV',
      pair: 'ETH/USDC',
      profit: 1.87,
      confidence: 94.2,
      timeLeft: 8,
      status: 'pending'
    },
    {
      id: 3,
      type: 'Flash Loan',
      pair: 'LINK/USDT',
      profit: 4.21,
      confidence: 91.5,
      timeLeft: 5,
      status: 'analyzing'
    },
    {
      id: 4,
      type: 'Triangular',
      pair: 'ADA/BTC/USDT',
      profit: 1.23,
      confidence: 88.7,
      timeLeft: 18,
      status: 'ready'
    }
  ]);
  
  // Risk metrics
  const [riskMetrics, setRiskMetrics] = useState({
    overallRisk: 2.8,
    portfolioVar: 0.024,
    expectedShortfall: 0.035,
    stressTestResults: {
      marketCrash: -5.2,
      liquidityCrisis: -8.1,
      regulatoryShock: -3.7,
      techFailure: -12.4
    }
  });
  
  // Real-time updates simulation
  useEffect(() => {
    const interval = setInterval(() => {
      if (systemActive) {
        // Update performance data
        setPerformanceData(prev => ({
          ...prev,
          minuteProfit: +(Math.random() * 50 - 5).toFixed(2),
          hourlyProfit: prev.hourlyProfit + +(Math.random() * 100 - 10).toFixed(2),
          dailyProfit: prev.dailyProfit + +(Math.random() * 200 - 20).toFixed(2),
          totalProfit: prev.totalProfit + +(Math.random() * 300 - 30).toFixed(2),
          activeTrades: Math.floor(Math.random() * 50) + 200,
          winRate: +(Math.random() * 10 + 80).toFixed(1)
        }));
        
        // Trigger money rain on big profits
        if (Math.random() < 0.1) {
          setShowMoneyRain(true);
          setTimeout(() => setShowMoneyRain(false), 3000);
        }
        
        // Update chart data
        const currentTime = new Date();
        const timeStr = currentTime.toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit' 
        });
        
        setChartData(prev => {
          const newProfit = +(Math.random() * 5000).toFixed(2);
          const lastCumulative = prev[prev.length - 1]?.cumulative || 0;
          
          const newData = [...prev.slice(-19), {
            time: timeStr,
            profit: newProfit,
            cumulative: lastCumulative + newProfit
          }];
          
          return newData;
        });
      }
    }, 5000); // Update every 5 seconds
    
    return () => clearInterval(interval);
  }, [systemActive]);
  
  // Handle system start/stop
  const handleSystemToggle = () => {
    setSystemActive(!systemActive);
    if (!systemActive) {
      setQuantumMode(true);
    }
  };
  
  // Handle emergency stop
  const handleEmergencyStop = () => {
    setSystemActive(false);
    setQuantumMode(false);
    setEmergencyStop(true);
    setTimeout(() => setEmergencyStop(false), 5000);
  };
  
  // Get profit color based on value
  const getProfitColor = (profit) => {
    if (profit > 0) return '#4CAF50';
    if (profit < 0) return '#f44336';
    return '#FFC107';
  };
  
  // Get strategy status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'success';
      case 'paused': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };
  
  // Get risk color
  const getRiskColor = (risk) => {
    switch (risk) {
      case 'low': return '#4CAF50';
      case 'medium': return '#FF9800';
      case 'high': return '#f44336';
      default: return '#9E9E9E';
    }
  };
  
  // Calculate profit percentage
  const profitPercentage = ((performanceData.totalProfit / 1000000) * 100).toFixed(2);
  
  return (
    <Box sx={{ p: 3, backgroundColor: '#0a0a0a', minHeight: '100vh', color: 'white', position: 'relative' }}>
      {/* Money Rain Effect */}
      {showMoneyRain && (
        <Box sx={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 9999 }}>
          {[...Array(20)].map((_, i) => (
            <MoneyIcon 
              key={i} 
              sx={{ 
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 2}s`,
                animationDuration: `${2 + Math.random() * 2}s`
              }} 
            />
          ))}
        </Box>
      )}
      
      {/* Emergency Stop Alert */}
      {emergencyStop && (
        <Alert 
          severity="error" 
          sx={{ 
            position: 'fixed', 
            top: 20, 
            left: '50%', 
            transform: 'translateX(-50%)', 
            zIndex: 10000,
            fontSize: '1.5rem',
            fontWeight: 'bold'
          }}
        >
          🚨 EMERGENCY STOP ACTIVATED - ALL TRADING HALTED 🚨
        </Alert>
      )}
      
      {/* Header Section */}
      <Paper 
        elevation={6}
        sx={{
          p: 4,
          mb: 3,
          background: systemActive ? 
            'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' :
            'linear-gradient(135deg, #232526 0%, #414345 100%)',
          color: 'white',
          textAlign: 'center',
          animation: systemActive ? `${quantumGlow} 3s infinite` : 'none'
        }}
      >
        <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          🚀 ULTIMATE ARBITRAGE SYSTEM 🚀
        </Typography>
        <Typography variant="h5" sx={{ mb: 3, opacity: 0.9 }}>
          {systemActive ? '⚡ QUANTUM-AI PROFIT MAXIMIZATION ACTIVE ⚡' : '⏸️ SYSTEM STANDBY MODE'}
        </Typography>
        
        {/* Main Control Buttons */}
        <Box sx={{ display: 'flex', gap: 3, justifyContent: 'center', mb: 3 }}>
          <Fab
            size="large"
            color={systemActive ? "secondary" : "primary"}
            onClick={handleSystemToggle}
            sx={{ 
              width: 120, 
              height: 120, 
              fontSize: '3rem',
              animation: systemActive ? `${profitPulse} 2s infinite` : 'none'
            }}
          >
            {systemActive ? <Pause sx={{ fontSize: '3rem' }} /> : <PlayArrow sx={{ fontSize: '3rem' }} />}
          </Fab>
          
          <Fab
            size="large"
            color="error"
            onClick={handleEmergencyStop}
            sx={{ width: 100, height: 100 }}
          >
            <Stop sx={{ fontSize: '2.5rem' }} />
          </Fab>
        </Box>
        
        {/* Status Indicators */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Chip 
            icon={<SmartToy />} 
            label={`AI Engines: ${systemActive ? 'ACTIVE' : 'STANDBY'}`}
            color={systemActive ? "success" : "default"}
            variant="filled"
            sx={{ fontSize: '1rem', p: 1 }}
          />
          <Chip 
            icon={<Bolt />} 
            label={`Quantum Core: ${quantumMode ? 'ENGAGED' : 'OFFLINE'}`}
            color={quantumMode ? "secondary" : "default"}
            variant="filled"
            sx={{ fontSize: '1rem', p: 1 }}
          />
          <Chip 
            icon={<TrendingUp />} 
            label={`Live Trading: ${systemActive ? 'EXECUTING' : 'PAUSED'}`}
            color={systemActive ? "warning" : "default"}
            variant="filled"
            sx={{ fontSize: '1rem', p: 1 }}
          />
        </Box>
      </Paper>
      
      {/* Performance Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <ProfitCard isprofitable={performanceData.dailyProfit > 0 ? 'true' : 'false'}>
            <CardContent sx={{ textAlign: 'center' }}>
              <AttachMoney sx={{ fontSize: 60, mb: 1 }} />
              <Typography variant="h3" sx={{ fontWeight: 'bold', mb: 1 }}>
                ${performanceData.totalProfit.toLocaleString()}
              </Typography>
              <Typography variant="h6">Total Profit</Typography>
              <Typography variant="body1" sx={{ mt: 1 }}>
                {profitPercentage > 0 ? '+' : ''}{profitPercentage}% ROI
              </Typography>
            </CardContent>
          </ProfitCard>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <QuantumCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <Bolt sx={{ fontSize: 60, mb: 1 }} />
              <Typography variant="h3" sx={{ fontWeight: 'bold', mb: 1 }}>
                {performanceData.quantumAdvantage}x
              </Typography>
              <Typography variant="h6">Quantum Advantage</Typography>
              <Typography variant="body1" sx={{ mt: 1 }}>
                {(performanceData.quantumAdvantage * 100 - 100).toFixed(1)}% boost
              </Typography>
            </CardContent>
          </QuantumCard>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <FireCard isactive={performanceData.winRate > 85 ? 'true' : 'false'}>
            <CardContent sx={{ textAlign: 'center' }}>
              <EmojiEvents sx={{ fontSize: 60, mb: 1, color: '#FFD700' }} />
              <Typography variant="h3" sx={{ fontWeight: 'bold', mb: 1, color: '#333' }}>
                {performanceData.winRate}%
              </Typography>
              <Typography variant="h6" sx={{ color: '#333' }}>Win Rate</Typography>
              <Typography variant="body1" sx={{ mt: 1, color: '#333' }}>
                {performanceData.successfulTrades} / {performanceData.successfulTrades + performanceData.failedTrades} trades
              </Typography>
            </CardContent>
          </FireCard>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 60, mb: 1 }} />
              <Typography variant="h3" sx={{ fontWeight: 'bold', mb: 1 }}>
                {performanceData.sharpeRatio}
              </Typography>
              <Typography variant="h6">Sharpe Ratio</Typography>
              <Typography variant="body1" sx={{ mt: 1 }}>
                Risk-adjusted returns
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Control Panel */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', backgroundColor: '#1a1a1a', color: 'white' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ⚙️ System Controls
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={quantumMode}
                    onChange={(e) => setQuantumMode(e.target.checked)}
                    color="secondary"
                  />
                }
                label="Quantum Mode"
                sx={{ mb: 2, display: 'block' }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRebalance}
                    onChange={(e) => setAutoRebalance(e.target.checked)}
                    color="primary"
                  />
                }
                label="Auto Rebalancing"
                sx={{ mb: 3, display: 'block' }}
              />
              
              <Typography gutterBottom>Aggression Level: {aggressionLevel}%</Typography>
              <Slider
                value={aggressionLevel}
                onChange={(e, value) => setAggressionLevel(value)}
                min={0}
                max={100}
                marks={[
                  { value: 0, label: 'Conservative' },
                  { value: 50, label: 'Balanced' },
                  { value: 100, label: 'Maximum' }
                ]}
                sx={{ color: '#4CAF50' }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', backgroundColor: '#1a1a1a', color: 'white' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Real-Time Metrics
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="#4CAF50">
                  Minute Profit: ${performanceData.minuteProfit}
                </Typography>
                <Typography variant="body2" color="#2196F3">
                  Hour Profit: ${performanceData.hourlyProfit.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="#FF9800">
                  Daily Profit: ${performanceData.dailyProfit.toLocaleString()}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2">Active Trades: {performanceData.activeTrades}</Typography>
                <Typography variant="body2">Avg Trade Time: {performanceData.averageTradeTime}s</Typography>
                <Typography variant="body2">Portfolio Value: ${performanceData.portfolioValue.toLocaleString()}</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', backgroundColor: '#1a1a1a', color: 'white' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🎯 Profit Targets
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2">Monthly Target</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(performanceData.dailyProfit / performanceData.monthlyTarget) * 100}
                  sx={{ mb: 1, height: 8, borderRadius: 4 }}
                  color="success"
                />
                <Typography variant="caption">
                  ${performanceData.dailyProfit.toLocaleString()} / ${performanceData.monthlyTarget.toLocaleString()}
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="body2">Yearly Target</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(performanceData.totalProfit / performanceData.yearlyTarget) * 100}
                  sx={{ mb: 1, height: 8, borderRadius: 4 }}
                  color="primary"
                />
                <Typography variant="caption">
                  ${performanceData.totalProfit.toLocaleString()} / ${performanceData.yearlyTarget.toLocaleString()}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={8}>
          <Card sx={{ backgroundColor: '#1a1a1a', color: 'white' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 Real-Time Profit Chart
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="time" stroke="#fff" />
                  <YAxis stroke="#fff" />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#333', border: '1px solid #555' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar dataKey="profit" fill="#4CAF50" name="Period Profit" />
                  <Line 
                    type="monotone" 
                    dataKey="cumulative" 
                    stroke="#2196F3" 
                    strokeWidth={3}
                    name="Cumulative Profit"
                    dot={{ fill: '#2196F3', strokeWidth: 2, r: 4 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Card sx={{ backgroundColor: '#1a1a1a', color: 'white', height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🎪 Strategy Allocation
              </Typography>
              <ResponsiveContainer width="100%" height={350}>
                <PieChart>
                  <Pie
                    data={strategyData.slice(0, 6)}
                    cx="50%"
                    cy="50%"
                    outerRadius={120}
                    dataKey="allocation"
                    label={({ name, allocation }) => `${name}: ${allocation}%`}
                  >
                    {strategyData.slice(0, 6).map((entry, index) => {
                      const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4'];
                      return <Cell key={`cell-${index}`} fill={colors[index]} />;
                    })}
                  </Pie>
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#333', border: '1px solid #555' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Live Opportunities */}
      <Card sx={{ mb: 3, backgroundColor: '#1a1a1a', color: 'white' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔥 Live Arbitrage Opportunities
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Type</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Pair</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Profit %</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Confidence</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Time Left</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Status</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {opportunities.map((opp) => (
                  <TableRow key={opp.id} sx={{ '&:hover': { backgroundColor: '#333' } }}>
                    <TableCell sx={{ color: 'white' }}>
                      <Chip label={opp.type} size="small" color="primary" />
                    </TableCell>
                    <TableCell sx={{ color: 'white' }}>{opp.pair}</TableCell>
                    <TableCell sx={{ color: getProfitColor(opp.profit), fontWeight: 'bold' }}>
                      +{opp.profit}%
                    </TableCell>
                    <TableCell sx={{ color: 'white' }}>{opp.confidence}%</TableCell>
                    <TableCell sx={{ color: 'white' }}>{opp.timeLeft}s</TableCell>
                    <TableCell>
                      <Chip 
                        label={opp.status} 
                        size="small" 
                        color={getStatusColor(opp.status)}
                      />
                    </TableCell>
                    <TableCell>
                      <Button 
                        variant="contained" 
                        size="small" 
                        color="success"
                        disabled={opp.status === 'executing'}
                      >
                        {opp.status === 'executing' ? 'EXECUTING' : 'EXECUTE'}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
      
      {/* Strategy Performance Table */}
      <Card sx={{ backgroundColor: '#1a1a1a', color: 'white' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🏆 Strategy Performance Leaderboard
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Strategy</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Allocation</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Profit</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Trades</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Win Rate</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Sharpe</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Risk</TableCell>
                  <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {strategyData.map((strategy, index) => (
                  <TableRow key={index} sx={{ '&:hover': { backgroundColor: '#333' } }}>
                    <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>
                      {index === 0 && <EmojiEvents sx={{ color: '#FFD700', mr: 1 }} />}
                      {strategy.name}
                    </TableCell>
                    <TableCell sx={{ color: 'white' }}>{strategy.allocation}%</TableCell>
                    <TableCell sx={{ color: getProfitColor(strategy.profit), fontWeight: 'bold' }}>
                      ${strategy.profit.toLocaleString()}
                    </TableCell>
                    <TableCell sx={{ color: 'white' }}>{strategy.trades}</TableCell>
                    <TableCell sx={{ color: 'white' }}>{strategy.winRate}%</TableCell>
                    <TableCell sx={{ color: 'white' }}>{strategy.sharpe}</TableCell>
                    <TableCell>
                      <Chip 
                        label={strategy.risk} 
                        size="small"
                        sx={{ 
                          backgroundColor: getRiskColor(strategy.risk),
                          color: 'white'
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={strategy.status} 
                        size="small" 
                        color={getStatusColor(strategy.status)}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default UltimateRealTimeDashboard;

