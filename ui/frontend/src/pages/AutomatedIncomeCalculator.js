import React, { useState, useEffect, useRef } from 'react';
import {
  Typography,
  Container,
  Grid,
  Paper,
  TextField,
  Button,
  Card,
  CardContent,
  Slider,
  Box,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip
} from '@mui/material';
import {
  TrendingUp,
  AutoMode,
  Assessment,
  AccountBalance,
  Timer,
  Speed,
  Security,
  CompareArrows,
  MonetizationOn,
  ShowChart,
  Psychology,
  Lightbulb
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { useMarketData } from '../contexts/MarketDataContext';

const AutomatedIncomeCalculator = () => {
  // State management
  const [investmentAmount, setInvestmentAmount] = useState(10000);
  const [selectedTab, setSelectedTab] = useState(0);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [earningsProjection, setEarningsProjection] = useState(null);
  const [realTimeEarnings, setRealTimeEarnings] = useState(null);
  const [automationLevel, setAutomationLevel] = useState(95);
  const [competitorComparison, setCompetitorComparison] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isCalculating, setIsCalculating] = useState(false);
  const [earningsHistory, setEarningsHistory] = useState([]);
  const [profitSimulation, setProfitSimulation] = useState([]);
  
  // Market data context
  const { marketData, isLoading } = useMarketData();
  
  // Refs for real-time updates
  const intervalRef = useRef(null);
  const chartRef = useRef(null);
  
  // Investment tiers configuration
  const investmentTiers = {
    starter: { min: 100, max: 1000, color: '#4caf50', automation: 80 },
    growth: { min: 1000, max: 10000, color: '#2196f3', automation: 90 },
    professional: { min: 10000, max: 100000, color: '#ff9800', automation: 95 },
    enterprise: { min: 100000, max: 10000000, color: '#9c27b0', automation: 98 }
  };
  
  // Get current tier based on investment amount
  const getCurrentTier = (amount) => {
    if (amount >= 100000) return 'enterprise';
    if (amount >= 10000) return 'professional';
    if (amount >= 1000) return 'growth';
    return 'starter';
  };
  
  // Calculate earnings projection
  const calculateEarningsProjection = async (amount) => {
    setIsCalculating(true);
    try {
      // Simulate API call to backend calculator
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const tier = getCurrentTier(amount);
      const tierConfig = investmentTiers[tier];
      
      // Base ROI calculations based on tier
      const baseROI = {
        starter: 0.02,        // 2% daily
        growth: 0.035,        // 3.5% daily
        professional: 0.05,   // 5% daily
        enterprise: 0.08      // 8% daily
      };
      
      const dailyROI = baseROI[tier];
      const automationBonus = (tierConfig.automation / 100) * 0.1; // Up to 10% bonus
      const adjustedDailyROI = dailyROI + automationBonus;
      
      const dailyProfit = amount * adjustedDailyROI;
      const weeklyProfit = dailyProfit * 7;
      const monthlyProfit = dailyProfit * 30;
      const yearlyROI = Math.pow(1 + adjustedDailyROI, 365) - 1;
      const yearlyProfit = amount * yearlyROI;
      
      const projection = {
        investmentAmount: amount,
        dailyProfit: dailyProfit,
        weeklyProfit: weeklyProfit,
        monthlyProfit: monthlyProfit,
        yearlyProfit: yearlyProfit,
        roiDaily: adjustedDailyROI * 100,
        roiWeekly: (weeklyProfit / amount) * 100,
        roiMonthly: (monthlyProfit / amount) * 100,
        roiYearly: yearlyROI * 100,
        automationLevel: tierConfig.automation,
        tier: tier,
        riskLevel: dailyROI > 0.06 ? 'High' : dailyROI > 0.04 ? 'Medium' : 'Low',
        confidenceScore: 0.85 + (tierConfig.automation / 100) * 0.1,
        strategiesActive: getStrategiesForTier(tier)
      };
      
      setEarningsProjection(projection);
      generateProfitSimulation(projection);
      
    } catch (error) {
      console.error('Error calculating earnings projection:', error);
    } finally {
      setIsCalculating(false);
    }
  };
  
  // Get strategies for tier
  const getStrategiesForTier = (tier) => {
    const strategies = {
      starter: ['Triangular Arbitrage', 'Cross-Exchange Arbitrage'],
      growth: ['Triangular Arbitrage', 'Cross-Exchange Arbitrage', 'Flash Loan Arbitrage', 'Market Making'],
      professional: ['Triangular Arbitrage', 'Cross-Exchange Arbitrage', 'Flash Loan Arbitrage', 'Market Making', 'AI Trading', 'DeFi Yield Farming'],
      enterprise: ['All Strategies', 'Quantum Trading', 'Institutional Arbitrage', 'Cross-Chain Arbitrage', 'MEV Extraction']
    };
    return strategies[tier] || [];
  };
  
  // Generate profit simulation data
  const generateProfitSimulation = (projection) => {
    const data = [];
    let cumulativeProfit = 0;
    
    for (let day = 0; day <= 30; day++) {
      if (day === 0) {
        data.push({ day, profit: 0, cumulative: projection.investmentAmount, dailyReturn: 0 });
      } else {
        const dailyProfit = projection.dailyProfit * (0.8 + Math.random() * 0.4); // Add some variance
        cumulativeProfit += dailyProfit;
        data.push({
          day,
          profit: dailyProfit,
          cumulative: projection.investmentAmount + cumulativeProfit,
          dailyReturn: (dailyProfit / projection.investmentAmount) * 100
        });
      }
    }
    
    setProfitSimulation(data);
  };
  
  // Simulate real-time earnings
  const simulateRealTimeEarnings = () => {
    if (!earningsProjection) return;
    
    const currentTime = new Date();
    const minutesIntoDay = currentTime.getHours() * 60 + currentTime.getMinutes();
    const progressThroughDay = minutesIntoDay / (24 * 60);
    
    const expectedDailyProfit = earningsProjection.dailyProfit;
    const currentProfit = expectedDailyProfit * progressThroughDay * (0.8 + Math.random() * 0.4);
    
    const earnings = {
      currentProfit: currentProfit,
      profitRatePerHour: currentProfit / (minutesIntoDay / 60 || 1),
      profitRatePerMinute: currentProfit / (minutesIntoDay || 1),
      activeTrades: Math.floor(Math.random() * 15) + 5,
      successfulTrades: Math.floor(Math.random() * 50) + 20,
      failedTrades: Math.floor(Math.random() * 5) + 2,
      winRate: 85 + Math.random() * 10,
      totalVolume: Math.random() * 100000 + 50000,
      automationPercentage: earningsProjection.automationLevel,
      strategiesRunning: earningsProjection.strategiesActive,
      timestamp: currentTime
    };
    
    setRealTimeEarnings(earnings);
    
    // Add to history
    setEarningsHistory(prev => {
      const newHistory = [...prev, { time: currentTime.toLocaleTimeString(), profit: currentProfit }];
      return newHistory.slice(-20); // Keep last 20 data points
    });
  };
  
  // Generate recommendations
  const generateRecommendations = async (availableCapital) => {
    const options = [0.1, 0.25, 0.5, 0.75, 0.9];
    const recs = [];
    
    for (const percentage of options) {
      const amount = availableCapital * percentage;
      if (amount < 100) continue;
      
      const tier = getCurrentTier(amount);
      const tierConfig = investmentTiers[tier];
      
      const baseROI = {
        starter: 0.02,
        growth: 0.035,
        professional: 0.05,
        enterprise: 0.08
      };
      
      const dailyROI = baseROI[tier];
      const dailyProfit = amount * dailyROI;
      const monthlyProfit = dailyProfit * 30;
      
      recs.push({
        investmentAmount: amount,
        percentage: percentage * 100,
        expectedDailyProfit: dailyProfit,
        expectedMonthlyProfit: monthlyProfit,
        dailyROIPercentage: dailyROI * 100,
        automationLevel: tierConfig.automation,
        riskLevel: dailyROI > 0.06 ? 'High' : dailyROI > 0.04 ? 'Medium' : 'Low',
        tier: tier,
        score: (dailyROI * 100) * (tierConfig.automation / 100)
      });
    }
    
    // Sort by score and mark best as recommended
    recs.sort((a, b) => b.score - a.score);
    if (recs.length > 0) {
      recs[0].recommended = true;
    }
    
    setRecommendations(recs);
  };
  
  // Load competitor comparison
  const loadCompetitorComparison = () => {
    const comparison = {
      ourSystem: {
        automation: automationLevel,
        dailyROI: earningsProjection ? earningsProjection.roiDaily : 4.5,
        strategies: 15
      },
      competitors: {
        'TradingView': { automation: 45, dailyROI: 1.2, strategies: 3 },
        '3Commas': { automation: 60, dailyROI: 1.8, strategies: 5 },
        'Cryptohopper': { automation: 70, dailyROI: 2.1, strategies: 7 },
        'Gunbot': { automation: 75, dailyROI: 2.5, strategies: 8 },
        'HaasOnline': { automation: 80, dailyROI: 2.8, strategies: 10 }
      }
    };
    
    const maxAutomation = Math.max(...Object.values(comparison.competitors).map(c => c.automation));
    const maxROI = Math.max(...Object.values(comparison.competitors).map(c => c.dailyROI));
    const maxStrategies = Math.max(...Object.values(comparison.competitors).map(c => c.strategies));
    
    const advantages = [];
    if (comparison.ourSystem.automation > maxAutomation) {
      advantages.push(`Highest automation: ${comparison.ourSystem.automation}%`);
    }
    if (comparison.ourSystem.dailyROI > maxROI) {
      advantages.push(`Superior ROI: ${comparison.ourSystem.dailyROI.toFixed(1)}%`);
    }
    if (comparison.ourSystem.strategies > maxStrategies) {
      advantages.push(`Most strategies: ${comparison.ourSystem.strategies}`);
    }
    
    comparison.advantages = advantages;
    comparison.marketPosition = advantages.length >= 2 ? 'Leading' : 'Competitive';
    
    setCompetitorComparison(comparison);
  };
  
  // Effects
  useEffect(() => {
    calculateEarningsProjection(investmentAmount);
    generateRecommendations(50000); // Default available capital
    loadCompetitorComparison();
  }, [investmentAmount]);
  
  useEffect(() => {
    if (realTimeMode && earningsProjection) {
      simulateRealTimeEarnings();
      intervalRef.current = setInterval(simulateRealTimeEarnings, 2000);
      
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [realTimeMode, earningsProjection]);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };
  
  // Handle investment amount change
  const handleInvestmentChange = (event, newValue) => {
    setInvestmentAmount(newValue);
  };
  
  // Format currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };
  
  // Format percentage
  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };
  
  // Get tier color
  const getTierColor = (tier) => {
    return investmentTiers[tier]?.color || '#666';
  };
  
  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];
  
  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>Loading Automated Income Calculator...</Typography>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          <AutoMode sx={{ fontSize: 40, mr: 2, verticalAlign: 'middle' }} />
          Automated Income Stream Calculator
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph>
          Calculate real-time earnings potential with fully automated trading strategies
        </Typography>
        
        {/* Real-time toggle */}
        <FormControlLabel
          control={
            <Switch
              checked={realTimeMode}
              onChange={(e) => setRealTimeMode(e.target.checked)}
              color="primary"
            />
          }
          label="Real-time Mode"
        />
      </Box>
      
      {/* Investment Amount Slider */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <MonetizationOn sx={{ mr: 1, verticalAlign: 'middle' }} />
          Investment Amount
        </Typography>
        
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={8}>
            <Slider
              value={investmentAmount}
              onChange={handleInvestmentChange}
              min={100}
              max={1000000}
              step={100}
              scale={(x) => x}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => formatCurrency(value)}
              sx={{ height: 8 }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="Investment Amount"
              type="number"
              value={investmentAmount}
              onChange={(e) => setInvestmentAmount(Number(e.target.value))}
              InputProps={{ startAdornment: '$' }}
              fullWidth
            />
          </Grid>
        </Grid>
        
        {/* Tier indicator */}
        {earningsProjection && (
          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
            <Chip
              label={`${earningsProjection.tier.toUpperCase()} TIER`}
              sx={{
                backgroundColor: getTierColor(earningsProjection.tier),
                color: 'white',
                fontWeight: 'bold'
              }}
            />
            <Typography variant="body2" sx={{ ml: 2 }}>
              {earningsProjection.automationLevel}% Automation Level
            </Typography>
          </Box>
        )}
      </Paper>
      
      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={handleTabChange} variant="fullWidth">
          <Tab label="Earnings Projection" icon={<TrendingUp />} />
          <Tab label="Real-time Performance" icon={<Speed />} />
          <Tab label="Investment Recommendations" icon={<Lightbulb />} />
          <Tab label="Competitor Analysis" icon={<CompareArrows />} />
        </Tabs>
      </Paper>
      
      {/* Tab Content */}
      {selectedTab === 0 && (
        <Grid container spacing={3}>
          {/* Earnings Summary Cards */}
          {earningsProjection && (
            <>
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Daily Profit</Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                      {formatCurrency(earningsProjection.dailyProfit)}
                    </Typography>
                    <Typography variant="body2">
                      {formatPercentage(earningsProjection.roiDaily)} ROI
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Monthly Profit</Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                      {formatCurrency(earningsProjection.monthlyProfit)}
                    </Typography>
                    <Typography variant="body2">
                      {formatPercentage(earningsProjection.roiMonthly)} ROI
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Yearly Profit</Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                      {formatCurrency(earningsProjection.yearlyProfit)}
                    </Typography>
                    <Typography variant="body2">
                      {formatPercentage(earningsProjection.roiYearly)} ROI
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)', color: 'white' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Automation</Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                      {earningsProjection.automationLevel}%
                    </Typography>
                    <Typography variant="body2">
                      Fully Automated
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}
          
          {/* Profit Simulation Chart */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                <ShowChart sx={{ mr: 1, verticalAlign: 'middle' }} />
                30-Day Profit Simulation
              </Typography>
              
              {profitSimulation.length > 0 && (
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={profitSimulation}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <RechartsTooltip formatter={(value, name) => [formatCurrency(value), name]} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="cumulative" 
                      stroke="#8884d8" 
                      name="Cumulative Value"
                      strokeWidth={3}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="profit" 
                      stroke="#82ca9d" 
                      name="Daily Profit"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Paper>
          </Grid>
          
          {/* Strategy Details */}
          {earningsProjection && (
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  <Psychology sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Active Strategies
                </Typography>
                
                <List>
                  {earningsProjection.strategiesActive.map((strategy, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <TrendingUp color="primary" />
                      </ListItemIcon>
                      <ListItemText primary={strategy} />
                    </ListItem>
                  ))}
                </List>
                
                <Divider sx={{ my: 2 }} />
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2">Risk Level</Typography>
                  <Chip 
                    label={earningsProjection.riskLevel}
                    color={earningsProjection.riskLevel === 'Low' ? 'success' : earningsProjection.riskLevel === 'Medium' ? 'warning' : 'error'}
                    size="small"
                  />
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2">Confidence Score</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                    {formatPercentage(earningsProjection.confidenceScore * 100)}
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          )}
          
          {/* Loading indicator */}
          {isCalculating && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <CircularProgress sx={{ mb: 2 }} />
                <Typography variant="h6">Calculating Earnings Projection...</Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      )}
      
      {selectedTab === 1 && realTimeEarnings && (
        <Grid container spacing={3}>
          {/* Real-time metrics */}
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Current Profit
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                  {formatCurrency(realTimeEarnings.currentProfit)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {formatCurrency(realTimeEarnings.profitRatePerHour)}/hour
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Active Trades
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                  {realTimeEarnings.activeTrades}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Currently executing
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Win Rate
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                  {formatPercentage(realTimeEarnings.winRate)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {realTimeEarnings.successfulTrades} successful
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Automation
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'info.main' }}>
                  {formatPercentage(realTimeEarnings.automationPercentage)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Fully automated
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Real-time chart */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                <Timer sx={{ mr: 1, verticalAlign: 'middle' }} />
                Real-time Earnings
              </Typography>
              
              {earningsHistory.length > 0 && (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={earningsHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <RechartsTooltip formatter={(value) => [formatCurrency(value), 'Profit']} />
                    <Line 
                      type="monotone" 
                      dataKey="profit" 
                      stroke="#00C49F" 
                      strokeWidth={3}
                      dot={{ fill: '#00C49F', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Paper>
          </Grid>
          
          {/* Running strategies */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Running Strategies
              </Typography>
              <Grid container spacing={2}>
                {realTimeEarnings.strategiesRunning.map((strategy, index) => (
                  <Grid item key={index}>
                    <Chip 
                      label={strategy}
                      color="primary"
                      variant="outlined"
                      icon={<AutoMode />}
                    />
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      )}
      
      {selectedTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                <Lightbulb sx={{ mr: 1, verticalAlign: 'middle' }} />
                Investment Recommendations
              </Typography>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Investment</TableCell>
                      <TableCell>Tier</TableCell>
                      <TableCell>Daily Profit</TableCell>
                      <TableCell>Monthly Profit</TableCell>
                      <TableCell>ROI %</TableCell>
                      <TableCell>Automation</TableCell>
                      <TableCell>Risk</TableCell>
                      <TableCell>Recommended</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {recommendations.map((rec, index) => (
                      <TableRow key={index} sx={{ backgroundColor: rec.recommended ? 'action.selected' : 'inherit' }}>
                        <TableCell>
                          <Box>
                            <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                              {formatCurrency(rec.investmentAmount)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {rec.percentage.toFixed(0)}% of capital
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={rec.tier.toUpperCase()}
                            sx={{ backgroundColor: getTierColor(rec.tier), color: 'white' }}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{formatCurrency(rec.expectedDailyProfit)}</TableCell>
                        <TableCell>{formatCurrency(rec.expectedMonthlyProfit)}</TableCell>
                        <TableCell>{formatPercentage(rec.dailyROIPercentage)}</TableCell>
                        <TableCell>{rec.automationLevel}%</TableCell>
                        <TableCell>
                          <Chip
                            label={rec.riskLevel}
                            color={rec.riskLevel === 'Low' ? 'success' : rec.riskLevel === 'Medium' ? 'warning' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {rec.recommended && (
                            <Chip label="RECOMMENDED" color="primary" variant="filled" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      )}
      
      {selectedTab === 3 && competitorComparison && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Alert severity="success" sx={{ mb: 3 }}>
              <Typography variant="h6">Market Position: {competitorComparison.marketPosition}</Typography>
              <Typography variant="body2">
                {competitorComparison.advantages.join(' • ')}
              </Typography>
            </Alert>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Automation Comparison
              </Typography>
              
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { name: 'Our System', value: competitorComparison.ourSystem.automation, fill: '#8884d8' },
                  ...Object.entries(competitorComparison.competitors).map(([name, data]) => ({
                    name,
                    value: data.automation,
                    fill: '#82ca9d'
                  }))
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <RechartsTooltip formatter={(value) => [`${value}%`, 'Automation Level']} />
                  <Bar dataKey="value" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                ROI Comparison
              </Typography>
              
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { name: 'Our System', value: competitorComparison.ourSystem.dailyROI, fill: '#8884d8' },
                  ...Object.entries(competitorComparison.competitors).map(([name, data]) => ({
                    name,
                    value: data.dailyROI,
                    fill: '#ffc658'
                  }))
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <RechartsTooltip formatter={(value) => [`${value}%`, 'Daily ROI']} />
                  <Bar dataKey="value" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Detailed Comparison
              </Typography>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Platform</TableCell>
                      <TableCell>Automation %</TableCell>
                      <TableCell>Daily ROI %</TableCell>
                      <TableCell>Strategies</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow sx={{ backgroundColor: 'primary.light', '& td': { fontWeight: 'bold' } }}>
                      <TableCell>Our System</TableCell>
                      <TableCell>{competitorComparison.ourSystem.automation}%</TableCell>
                      <TableCell>{competitorComparison.ourSystem.dailyROI}%</TableCell>
                      <TableCell>{competitorComparison.ourSystem.strategies}</TableCell>
                      <TableCell>
                        <Chip label="LEADER" color="primary" />
                      </TableCell>
                    </TableRow>
                    {Object.entries(competitorComparison.competitors).map(([name, data]) => (
                      <TableRow key={name}>
                        <TableCell>{name}</TableCell>
                        <TableCell>{data.automation}%</TableCell>
                        <TableCell>{data.dailyROI}%</TableCell>
                        <TableCell>{data.strategies}</TableCell>
                        <TableCell>
                          <Chip label="Competitor" variant="outlined" />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default AutomatedIncomeCalculator;

