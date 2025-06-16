import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  IconButton,
  Tooltip,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
} from '@mui/material';
import {
  Bolt,
  SwapHoriz,
  AddCircle,
  SyncAlt,
  Speed,
  Timeline,
  WaterfallChart,
  AccountBalanceWallet,
  ShowChart,
  Autorenew,
  Shield,
  Calculate,
} from '@mui/icons-material';

const QuickActionsPanel = () => {
  const [openDialog, setOpenDialog] = React.useState(false);
  const [dialogType, setDialogType] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [success, setSuccess] = React.useState(false);

  const handleQuickAction = async (action) => {
    setDialogType(action);
    setOpenDialog(true);
  };

  const handleClose = () => {
    setOpenDialog(false);
    setSuccess(false);
  };

  const handleExecute = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      setSuccess(true);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const quickActions = [
    {
      title: 'Flash Rebalance',
      icon: <Bolt color="warning" />,
      description: 'Instantly rebalance all portfolios',
      action: 'rebalance'
    },
    {
      title: 'Cross-Chain Swap',
      icon: <SwapHoriz color="info" />,
      description: 'Execute cross-chain token swaps',
      action: 'swap'
    },
    {
      title: 'Quick Deploy',
      icon: <AddCircle color="success" />,
      description: 'Deploy new strategy instance',
      action: 'deploy'
    },
    {
      title: 'Sync All',
      icon: <SyncAlt />,
      description: 'Synchronize all wallets and exchanges',
      action: 'sync'
    },
    {
      title: 'Performance Boost',
      icon: <Speed color="error" />,
      description: 'Optimize execution parameters',
      action: 'optimize'
    },
    {
      title: 'Risk Scan',
      icon: <Shield color="secondary" />,
      description: 'Run full risk assessment',
      action: 'risk'
    },
    {
      title: 'Profit Lock',
      icon: <Calculate color="primary" />,
      description: 'Lock in current profits',
      action: 'lock'
    },
    {
      title: 'Auto-Scale',
      icon: <Autorenew />,
      description: 'Scale positions based on performance',
      action: 'scale'
    }
  ];

  const renderDialog = () => {
    const getDialogContent = () => {
      switch (dialogType) {
        case 'rebalance':
          return (
            <>
              <Alert severity="info" sx={{ mb: 2 }}>
                This will rebalance all portfolios using quantum-optimized allocation
              </Alert>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Rebalance Strategy</InputLabel>
                <Select defaultValue="optimal">
                  <MenuItem value="optimal">Optimal Returns</MenuItem>
                  <MenuItem value="conservative">Risk Minimization</MenuItem>
                  <MenuItem value="aggressive">Maximum Growth</MenuItem>
                </Select>
              </FormControl>
              <TextField
                fullWidth
                label="Max Slippage (%)"
                type="number"
                defaultValue="0.5"
              />
            </>
          );

        case 'swap':
          return (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Source Chain</InputLabel>
                  <Select defaultValue="eth">
                    <MenuItem value="eth">Ethereum</MenuItem>
                    <MenuItem value="bsc">BSC</MenuItem>
                    <MenuItem value="polygon">Polygon</MenuItem>
                    <MenuItem value="arbitrum">Arbitrum</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Destination Chain</InputLabel>
                  <Select defaultValue="arbitrum">
                    <MenuItem value="eth">Ethereum</MenuItem>
                    <MenuItem value="bsc">BSC</MenuItem>
                    <MenuItem value="polygon">Polygon</MenuItem>
                    <MenuItem value="arbitrum">Arbitrum</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Amount"
                  type="number"
                  placeholder="Enter amount"
                />
              </Grid>
            </Grid>
          );

        default:
          return (
            <Alert severity="info">
              Confirm execution of {dialogType} action?
            </Alert>
          );
      }
    };

    return (
      <Dialog open={openDialog} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>
          {quickActions.find(a => a.action === dialogType)?.title || 'Execute Action'}
        </DialogTitle>
        <DialogContent>
          {success ? (
            <Alert severity="success" sx={{ mt: 2 }}>
              Action completed successfully!
            </Alert>
          ) : (
            getDialogContent()
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleExecute}
            disabled={loading || success}
          >
            {loading ? 'Executing...' : 'Execute'}
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Quick Actions
        </Typography>
        <Grid container spacing={2}>
          {quickActions.map((action, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card
                sx={{
                  height: '100%',
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4
                  }
                }}
                onClick={() => handleQuickAction(action.action)}
              >
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    {action.icon}
                    <Typography variant="subtitle1" sx={{ ml: 1 }}>
                      {action.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {action.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </CardContent>
      {renderDialog()}
    </Card>
  );
};

export default QuickActionsPanel;

