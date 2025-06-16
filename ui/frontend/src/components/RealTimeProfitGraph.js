import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  ButtonGroup,
  Button,
  IconButton,
  Tooltip,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  Timeline,
  Refresh,
  ZoomIn,
  ZoomOut,
  PlayArrow,
  Pause,
} from '@mui/icons-material';

// Import Chart.js
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

const RealTimeProfitGraph = () => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [timeframe, setTimeframe] = useState('1H');
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);
  const updateInterval = useRef(null);

  const timeframes = ['1M', '5M', '15M', '1H', '4H', '1D'];

  const generateMockData = (points) => {
    const now = Date.now();
    let lastValue = 1000000;
    const data = [];

    for (let i = points; i >= 0; i--) {
      const timestamp = now - i * 60000; // 1 minute intervals
      const change = (Math.random() - 0.48) * 1000; // Bias towards positive
      lastValue += change;
      data.push({
        timestamp,
        value: lastValue,
        profit: change,
      });
    }

    return data;
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      const mockData = generateMockData(60);
      setData(mockData);
    } catch (error) {
      console.error('Error fetching profit data:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
    };
  }, []);

  useEffect(() => {
    if (updateInterval.current) {
      clearInterval(updateInterval.current);
    }

    if (autoUpdate) {
      updateInterval.current = setInterval(() => {
        const newPoint = {
          timestamp: Date.now(),
          value: (data[data.length - 1]?.value || 1000000) + (Math.random() - 0.48) * 1000,
          profit: Math.random() * 1000,
        };
        setData(prev => [...prev.slice(1), newPoint]);
      }, 1000);
    }

    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
    };
  }, [autoUpdate, data]);

  const chartData = {
    labels: data?.map(d => new Date(d.timestamp).toLocaleTimeString()) || [],
    datasets: [
      {
        label: 'Portfolio Value',
        data: data?.map(d => d.value) || [],
        borderColor: theme.palette.primary.main,
        backgroundColor: `${theme.palette.primary.main}20`,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: (context) => {
            const value = context.raw;
            return `Portfolio Value: $${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
          drawBorder: false,
        },
        ticks: {
          maxTicksLimit: 8,
          color: theme.palette.text.secondary,
        },
      },
      y: {
        grid: {
          color: theme.palette.divider,
        },
        ticks: {
          color: theme.palette.text.secondary,
          callback: (value) => `$${value.toLocaleString()}`,
        },
      },
    },
    interaction: {
      intersect: false,
      mode: 'index',
    },
    animation: {
      duration: 0,
    },
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6" display="flex" alignItems="center">
            <Timeline sx={{ mr: 1 }} />
            Real-Time Profit Graph
          </Typography>
          <Box display="flex" gap={2} alignItems="center">
            <ButtonGroup size="small">
              {timeframes.map((tf) => (
                <Button
                  key={tf}
                  variant={timeframe === tf ? 'contained' : 'outlined'}
                  onClick={() => setTimeframe(tf)}
                >
                  {tf}
                </Button>
              ))}
            </ButtonGroup>
            <ButtonGroup size="small">
              <Tooltip title="Zoom Out">
                <IconButton
                  onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.1))}
                  size="small"
                >
                  <ZoomOut />
                </IconButton>
              </Tooltip>
              <Tooltip title="Zoom In">
                <IconButton
                  onClick={() => setZoomLevel(Math.min(2, zoomLevel + 0.1))}
                  size="small"
                >
                  <ZoomIn />
                </IconButton>
              </Tooltip>
            </ButtonGroup>
            <Tooltip title={autoUpdate ? 'Pause' : 'Play'}>
              <IconButton
                onClick={() => setAutoUpdate(!autoUpdate)}
                color={autoUpdate ? 'primary' : 'default'}
                size="small"
              >
                {autoUpdate ? <Pause /> : <PlayArrow />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton onClick={fetchData} size="small">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {loading ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={400}>
            <CircularProgress />
          </Box>
        ) : (
          <Box height={400}>
            <Line data={chartData} options={chartOptions} />
          </Box>
        )}

        <Box display="flex" justifyContent="space-between" mt={2}>
          <Typography variant="body2" color="text.secondary">
            Last Update: {new Date().toLocaleTimeString()}
          </Typography>
          <Typography variant="body2" color="success.main">
            +${(data?.[data.length - 1]?.profit || 0).toFixed(2)} in last minute
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RealTimeProfitGraph;

