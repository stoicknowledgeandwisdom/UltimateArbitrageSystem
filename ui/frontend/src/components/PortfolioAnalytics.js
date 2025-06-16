import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  Chip,
  Tooltip,
  IconButton,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Assessment,
  TimerSharp,
  Refresh,
  ArrowUpward,
  ArrowDownward,
} from '@mui/icons-material';

const PortfolioAnalytics = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      const mockData = {
        total_portfolio_value: 5847523.45,
        total_profit: 847523.45,
        profit_percentage: 16.95,
        daily_profit: 23847.12,
        positions: [
          {
            strategy: 'Quantum Arbitrage',
            profit: 45623.12,
            roi: 94.4,
            active_positions: 89,
            risk_score: 2.1
          },
          {
            strategy: 'Cross-Chain MEV',
            profit: 38921.45,
            roi: 82.1,
            active_positions: 156,
            risk_score: 3.8
          },
          {
            strategy: 'Flash Loan Arbitrage',
            profit: 29847.33,
            roi: 76.5,
            active_positions: 234,
            risk_score: 1.2
          }
        ],
        metrics: {
          sharpe_ratio: 3.24,
          sortino_ratio: 4.12,
          max_drawdown: -2.1,
          win_rate: 87.3,
          profit_factor: 3.45,
          avg_trade_duration: '00:02:15'
        },
        chain_distribution: [
          { chain: 'Ethereum', value: 1234567.89, percentage: 25.4 },
          { chain: 'BSC', value: 987654.32, percentage: 18.7 },
          { chain: 'Polygon', value: 756432.10, percentage: 15.2 },
          { chain: 'Arbitrum', value: 543210.98, percentage: 12.1 }
        ]
      };

      setData(mockData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching portfolio data:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  if (loading && !data) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6">
            Portfolio Analytics
          </Typography>
          <Box>
            <Tooltip title="Last updated">
              <Typography variant="caption" color="text.secondary" sx={{ mr: 2 }}>
                {lastUpdate?.toLocaleTimeString()}
              </Typography>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton onClick={fetchData} size="small">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Summary Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Total Portfolio Value
                </Typography>
                <Typography variant="h4">
                  {formatCurrency(data.total_portfolio_value)}
                </Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <TrendingUp color="success" sx={{ mr: 1 }} />
                  <Typography variant="body2" color="success.main">
                    {formatPercentage(data.profit_percentage)} total return
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Total Profit
                </Typography>
                <Typography variant="h4" color="success.main">
                  {formatCurrency(data.total_profit)}
                </Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <ShowChart color="primary" sx={{ mr: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    {formatCurrency(data.daily_profit)} today
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Performance Metrics
                </Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <Assessment color="info" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    Sharpe: {data.metrics.sharpe_ratio.toFixed(2)}
                  </Typography>
                </Box>
                <Box display="flex" alignItems="center" mt={1}>
                  <TimerSharp color="warning" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    Win Rate: {formatPercentage(data.metrics.win_rate)}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Risk Metrics
                </Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <TrendingDown color="error" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    Max Drawdown: {formatPercentage(data.metrics.max_drawdown)}
                  </Typography>
                </Box>
                <Box display="flex" alignItems="center" mt={1}>
                  <AccountBalance color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    Profit Factor: {data.metrics.profit_factor.toFixed(2)}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Strategy Performance Table */}
        <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Strategy</TableCell>
                <TableCell align="right">Profit</TableCell>
                <TableCell align="right">ROI</TableCell>
                <TableCell align="right">Active Positions</TableCell>
                <TableCell align="right">Risk Score</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.positions.map((position, index) => (
                <TableRow key={index}>
                  <TableCell>{position.strategy}</TableCell>
                  <TableCell align="right">
                    <Box display="flex" alignItems="center" justifyContent="flex-end">
                      {position.profit >= 0 ? (
                        <ArrowUpward fontSize="small" color="success" sx={{ mr: 1 }} />
                      ) : (
                        <ArrowDownward fontSize="small" color="error" sx={{ mr: 1 }} />
                      )}
                      {formatCurrency(position.profit)}
                    </Box>
                  </TableCell>
                  <TableCell align="right">{formatPercentage(position.roi)}</TableCell>
                  <TableCell align="right">{position.active_positions}</TableCell>
                  <TableCell align="right">
                    <Chip
                      label={position.risk_score.toFixed(1)}
                      size="small"
                      color={position.risk_score < 3 ? 'success' : position.risk_score < 5 ? 'warning' : 'error'}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Chain Distribution */}
        <Typography variant="subtitle2" gutterBottom>
          Chain Distribution
        </Typography>
        <Grid container spacing={1}>
          {data.chain_distribution.map((chain, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2">
                    {chain.chain}
                  </Typography>
                  <Typography variant="h6">
                    {formatCurrency(chain.value)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatPercentage(chain.percentage)} of portfolio
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default PortfolioAnalytics;

