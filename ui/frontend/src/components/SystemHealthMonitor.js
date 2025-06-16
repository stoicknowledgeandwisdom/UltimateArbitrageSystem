import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  LinearProgress,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  Button,
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Memory,
  Speed,
  Timeline,
  CloudQueue,
  Storage,
  Refresh,
  Info,
} from '@mui/icons-material';

const SystemHealthMonitor = () => {
  const [loading, setLoading] = useState(true);
  const [healthData, setHealthData] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchHealthData = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      const mockData = {
        quantum_system: {
          status: 'optimal',
          performance: 96,
          message: 'Quantum enhancement active',
          last_optimization: '2 minutes ago'
        },
        market_data: {
          status: 'good',
          performance: 89,
          message: 'Real-time data streaming',
          connected_feeds: 24
        },
        execution_engine: {
          status: 'optimal',
          performance: 98,
          message: 'Sub-millisecond execution',
          active_orders: 156
        },
        risk_management: {
          status: 'warning',
          performance: 78,
          message: 'Elevated market volatility',
          risk_score: 6.2
        },
        infrastructure: {
          status: 'good',
          performance: 92,
          message: 'All systems operational',
          node_count: 12
        },
        memory_usage: {
          status: 'good',
          performance: 85,
          message: 'Optimal memory allocation',
          used_gb: 24
        },
        network: {
          status: 'optimal',
          performance: 95,
          message: 'Low latency connections',
          ping_ms: 12
        },
        storage: {
          status: 'good',
          performance: 88,
          message: 'Fast data access',
          used_percent: 45
        }
      };

      setHealthData(mockData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching health data:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchHealthData();
    const interval = setInterval(fetchHealthData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'optimal':
        return <CheckCircle color="success" />;
      case 'good':
        return <CheckCircle color="primary" />;
      case 'warning':
        return <Warning color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <Info color="disabled" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'optimal': return 'success';
      case 'good': return 'primary';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const systemComponents = [
    { key: 'quantum_system', label: 'Quantum System', icon: <Memory /> },
    { key: 'market_data', label: 'Market Data', icon: <Timeline /> },
    { key: 'execution_engine', label: 'Execution Engine', icon: <Speed /> },
    { key: 'risk_management', label: 'Risk Management', icon: <Warning /> },
    { key: 'infrastructure', label: 'Infrastructure', icon: <CloudQueue /> },
    { key: 'memory_usage', label: 'Memory Usage', icon: <Storage /> },
    { key: 'network', label: 'Network', icon: <CloudQueue /> },
    { key: 'storage', label: 'Storage', icon: <Storage /> }
  ];

  if (loading && !healthData) {
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
            System Health Monitor
          </Typography>
          <Box>
            <Tooltip title="Last updated">
              <Typography variant="caption" color="text.secondary" sx={{ mr: 2 }}>
                {lastUpdate?.toLocaleTimeString()}
              </Typography>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton onClick={fetchHealthData} size="small">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <Grid container spacing={2}>
          {systemComponents.map(component => {
            const data = healthData?.[component.key];
            if (!data) return null;

            return (
              <Grid item xs={12} sm={6} md={3} key={component.key}>
                <Card
                  variant="outlined"
                  sx={{
                    height: '100%',
                    borderColor: theme => 
                      theme.palette[getStatusColor(data.status)].main
                  }}
                >
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Box display="flex" alignItems="center">
                        {component.icon}
                        <Typography variant="subtitle2" sx={{ ml: 1 }}>
                          {component.label}
                        </Typography>
                      </Box>
                      <Tooltip title={data.message}>
                        {getStatusIcon(data.status)}
                      </Tooltip>
                    </Box>

                    <Box position="relative" mb={1}>
                      <LinearProgress
                        variant="determinate"
                        value={data.performance}
                        color={getStatusColor(data.status)}
                      />
                      <Typography
                        variant="caption"
                        sx={{
                          position: 'absolute',
                          right: 0,
                          top: -18,
                          color: theme => 
                            theme.palette[getStatusColor(data.status)].main
                        }}
                      >
                        {data.performance}%
                      </Typography>
                    </Box>

                    <Typography variant="body2" color="text.secondary">
                      {data.message}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SystemHealthMonitor;

