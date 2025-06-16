import React, { useState, useEffect } from 'react';
import {
  Card, CardContent, CardHeader,
  Button, TextField, Select, MenuItem, FormControl, InputLabel,
  Typography, Grid, Box, Chip, Alert, Dialog, DialogTitle, DialogContent,
  DialogActions, Accordion, AccordionSummary, AccordionDetails,
  List, ListItem, ListItemText, ListItemSecondaryAction, IconButton,
  Switch, FormControlLabel, Tooltip, LinearProgress, Divider
} from '@mui/material';
import {
  ExpandMore, Add, Edit, Delete, Visibility, VisibilityOff,
  AccountBalanceWallet, Security, Speed, TrendingUp, Info,
  CheckCircle, Warning, Error as ErrorIcon, Settings
} from '@mui/icons-material';

const WalletConfigPanel = ({ onConfigChange }) => {
  const [wallets, setWallets] = useState([]);
  const [exchanges, setExchanges] = useState([]);
  const [strategies, setStrategies] = useState([]);
  const [availableStrategies, setAvailableStrategies] = useState([]);
  const [configSummary, setConfigSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [showAddWallet, setShowAddWallet] = useState(false);
  const [showAddExchange, setShowAddExchange] = useState(false);
  const [newWallet, setNewWallet] = useState({
    network: '',
    address: '',
    privateKey: '',
    balance: 0
  });
  const [newExchange, setNewExchange] = useState({
    exchange: '',
    apiKey: '',
    apiSecret: '',
    passphrase: '',
    sandboxMode: true
  });
  const [showPrivateKeys, setShowPrivateKeys] = useState(false);
  const [validationResults, setValidationResults] = useState({});

  // Supported networks and exchanges
  const networks = [
    'ethereum', 'binance_smart_chain', 'polygon', 'arbitrum', 'optimism',
    'avalanche', 'fantom', 'solana', 'cardano', 'polkadot', 'cosmos',
    'near', 'terra', 'bitcoin', 'litecoin', 'dogecoin', 'ripple',
    'stellar', 'monero', 'zcash'
  ];

  const exchangeTypes = [
    'binance', 'coinbase_pro', 'kraken', 'bybit', 'bitget', 'okx',
    'gate_io', 'huobi', 'kucoin', 'bitfinex', 'gemini', 'crypto_com',
    'mexc', 'ascendex', 'bitrue', 'bitstamp', 'dydx', 'uniswap',
    'sushiswap', 'pancakeswap', 'curve', 'balancer', 'yearn',
    'compound', 'aave', 'maker', 'synthetix'
  ];

  useEffect(() => {
    loadConfigurations();
    loadAvailableStrategies();
  }, []);

  const loadConfigurations = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/wallet-config/summary');
      const data = await response.json();
      setConfigSummary(data);
      
      // Load detailed configurations
      const walletsResponse = await fetch('/api/wallet-config/wallets');
      const walletsData = await walletsResponse.json();
      setWallets(walletsData);
      
      const exchangesResponse = await fetch('/api/wallet-config/exchanges');
      const exchangesData = await exchangesResponse.json();
      setExchanges(exchangesData);
      
      const strategiesResponse = await fetch('/api/wallet-config/strategies');
      const strategiesData = await strategiesResponse.json();
      setStrategies(strategiesData);
      
    } catch (error) {
      console.error('Error loading configurations:', error);
    }
    setLoading(false);
  };

  const loadAvailableStrategies = async () => {
    try {
      const response = await fetch('/api/wallet-config/available-strategies');
      const data = await response.json();
      setAvailableStrategies(data);
    } catch (error) {
      console.error('Error loading available strategies:', error);
    }
  };

  const addWallet = async () => {
    try {
      const response = await fetch('/api/wallet-config/wallets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newWallet)
      });
      
      if (response.ok) {
        setShowAddWallet(false);
        setNewWallet({ network: '', address: '', privateKey: '', balance: 0 });
        loadConfigurations();
      }
    } catch (error) {
      console.error('Error adding wallet:', error);
    }
  };

  const addExchange = async () => {
    try {
      const response = await fetch('/api/wallet-config/exchanges', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newExchange)
      });
      
      if (response.ok) {
        setShowAddExchange(false);
        setNewExchange({ 
          exchange: '', apiKey: '', apiSecret: '', 
          passphrase: '', sandboxMode: true 
        });
        loadConfigurations();
      }
    } catch (error) {
      console.error('Error adding exchange:', error);
    }
  };

  const validateStrategy = async (strategyId) => {
    try {
      const response = await fetch(`/api/wallet-config/strategies/${strategyId}/validate`);
      const validation = await response.json();
      setValidationResults(prev => ({
        ...prev,
        [strategyId]: validation
      }));
      return validation;
    } catch (error) {
      console.error('Error validating strategy:', error);
      return { valid: false, error: 'Validation failed' };
    }
  };

  const autoAssignStrategy = async (strategyId) => {
    try {
      const response = await fetch(`/api/wallet-config/strategies/${strategyId}/auto-assign`, {
        method: 'POST'
      });
      
      if (response.ok) {
        loadConfigurations();
      }
    } catch (error) {
      console.error('Error auto-assigning strategy:', error);
    }
  };

  const toggleStrategy = async (strategyId, enabled) => {
    try {
      const response = await fetch(`/api/wallet-config/strategies/${strategyId}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      
      if (response.ok) {
        loadConfigurations();
      }
    } catch (error) {
      console.error('Error toggling strategy:', error);
    }
  };

  const getStrategyStatusIcon = (strategy) => {
    const validation = validationResults[strategy.id];
    if (!validation) return <Info color="disabled" />;
    
    if (validation.valid && strategy.is_enabled) {
      return <CheckCircle color="success" />;
    } else if (validation.valid && !strategy.is_enabled) {
      return <Warning color="warning" />;
    } else {
      return <ErrorIcon color="error" />;
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const StrategyExplanationDialog = ({ strategy, open, onClose }) => (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <TrendingUp color="primary" />
          {strategy?.name}
        </Box>
      </DialogTitle>
      <DialogContent>
        <Typography variant="h6" gutterBottom>
          How This Strategy Works:
        </Typography>
        <Typography 
          variant="body1" 
          component="pre" 
          sx={{ 
            whiteSpace: 'pre-wrap', 
            fontFamily: 'monospace',
            backgroundColor: 'grey.100',
            p: 2,
            borderRadius: 1
          }}
        >
          {strategy?.explanation}
        </Typography>
        
        <Box mt={3}>
          <Typography variant="h6" gutterBottom>
            Requirements:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="subtitle1">Required Wallets:</Typography>
              <List dense>
                {strategy?.required_wallets?.map((wallet, index) => (
                  <ListItem key={index}>
                    <Chip label={wallet} size="small" />
                  </ListItem>
                ))}
              </List>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="subtitle1">Required Exchanges:</Typography>
              <List dense>
                {strategy?.required_exchanges?.map((exchange, index) => (
                  <ListItem key={index}>
                    <Chip label={exchange} size="small" />
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
          
          <Box mt={2}>
            <Typography variant="body2">
              <strong>Minimum Capital:</strong> {formatCurrency(strategy?.min_capital_usd || 0)}
            </Typography>
            <Typography variant="body2">
              <strong>Profit Potential:</strong> {strategy?.profit_potential || 0}% per execution
            </Typography>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button 
          variant="contained" 
          onClick={() => {
            onClose();
            // Auto-assign this strategy
            autoAssignStrategy(strategy?.type);
          }}
        >
          Configure This Strategy
        </Button>
      </DialogActions>
    </Dialog>
  );

  if (loading) {
    return (
      <Box p={3}>
        <LinearProgress />
        <Typography align="center" mt={2}>Loading configurations...</Typography>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        🚀 Ultimate Wallet & Strategy Configuration
      </Typography>
      
      {/* Configuration Summary */}
      {configSummary && (
        <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
          <CardContent>
            <Typography variant="h6" color="white" gutterBottom>
              Configuration Overview
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={4}>
                <Box textAlign="center">
                  <AccountBalanceWallet sx={{ fontSize: 40, color: 'white' }} />
                  <Typography variant="h4" color="white">
                    {configSummary.wallets?.total || 0}
                  </Typography>
                  <Typography color="white">Wallets</Typography>
                  <Typography variant="body2" color="white">
                    {formatCurrency(configSummary.wallets?.total_balance_usd || 0)}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box textAlign="center">
                  <Security sx={{ fontSize: 40, color: 'white' }} />
                  <Typography variant="h4" color="white">
                    {configSummary.exchanges?.active || 0}
                  </Typography>
                  <Typography color="white">Active Exchanges</Typography>
                  <Typography variant="body2" color="white">
                    {configSummary.exchanges?.total || 0} total configured
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box textAlign="center">
                  <Speed sx={{ fontSize: 40, color: 'white' }} />
                  <Typography variant="h4" color="white">
                    {configSummary.strategies?.enabled || 0}
                  </Typography>
                  <Typography color="white">Active Strategies</Typography>
                  <Typography variant="body2" color="white">
                    {configSummary.strategies?.fully_configured || 0} fully configured
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Available Strategies */}
      <Accordion defaultExpanded sx={{ mb: 3 }}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">
            <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
            Available Trading Strategies ({availableStrategies.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {availableStrategies.map((strategy, index) => (
              <Grid item xs={12} md={6} lg={4} key={index}>
                <Card 
                  sx={{ 
                    height: '100%',
                    '&:hover': { boxShadow: 6 },
                    transition: 'box-shadow 0.3s'
                  }}
                >
                  <CardHeader
                    title={strategy.name}
                    subheader={`Min Capital: ${formatCurrency(strategy.min_capital_usd)}`}
                    action={
                      <Tooltip title="View Explanation">
                        <IconButton 
                          onClick={() => {
                            setSelectedStrategy(strategy);
                            setShowExplanation(true);
                          }}
                        >
                          <Info />
                        </IconButton>
                      </Tooltip>
                    }
                  />
                  <CardContent>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {strategy.description}
                    </Typography>
                    
                    <Box mt={2}>
                      <Typography variant="caption" display="block">
                        Profit Potential: <strong>{strategy.profit_potential}%</strong>
                      </Typography>
                      <Typography variant="caption" display="block">
                        Networks: {strategy.required_wallets.length}
                      </Typography>
                      <Typography variant="caption" display="block">
                        Exchanges: {strategy.required_exchanges.length}
                      </Typography>
                    </Box>
                    
                    <Box mt={2}>
                      <Button
                        variant="contained"
                        size="small"
                        fullWidth
                        onClick={() => {
                          setSelectedStrategy(strategy);
                          setShowExplanation(true);
                        }}
                      >
                        Configure Strategy
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Wallet Management */}
      <Accordion sx={{ mb: 3 }}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">
            <AccountBalanceWallet sx={{ mr: 1, verticalAlign: 'middle' }} />
            Wallet Management ({wallets.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box mb={2}>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setShowAddWallet(true)}
            >
              Add Wallet
            </Button>
          </Box>
          
          <Grid container spacing={2}>
            {wallets.map((wallet, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="h6">
                        {wallet.network.toUpperCase()}
                      </Typography>
                      <Chip 
                        label={wallet.is_active ? 'Active' : 'Inactive'}
                        color={wallet.is_active ? 'success' : 'default'}
                        size="small"
                      />
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary">
                      Address: {wallet.address.substring(0, 10)}...
                      {wallet.address.substring(wallet.address.length - 6)}
                    </Typography>
                    
                    <Typography variant="body1" mt={1}>
                      Balance: <strong>{formatCurrency(wallet.balance_usd)}</strong>
                    </Typography>
                    
                    <Box mt={2} display="flex" gap={1}>
                      <Button size="small" startIcon={<Edit />}>
                        Edit
                      </Button>
                      <Button size="small" startIcon={<Delete />} color="error">
                        Remove
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Exchange Management */}
      <Accordion sx={{ mb: 3 }}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">
            <Security sx={{ mr: 1, verticalAlign: 'middle' }} />
            Exchange API Configuration ({exchanges.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box mb={2}>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setShowAddExchange(true)}
            >
              Add Exchange
            </Button>
          </Box>
          
          <Grid container spacing={2}>
            {exchanges.map((exchange, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="h6">
                        {exchange.exchange.toUpperCase()}
                      </Typography>
                      <Box>
                        <Chip 
                          label={exchange.sandbox_mode ? 'Sandbox' : 'Live'}
                          color={exchange.sandbox_mode ? 'warning' : 'success'}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Chip 
                          label={exchange.is_active ? 'Active' : 'Inactive'}
                          color={exchange.is_active ? 'success' : 'default'}
                          size="small"
                        />
                      </Box>
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary">
                      API Key: {exchange.api_key.substring(0, 8)}...
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary">
                      Daily Limit: {formatCurrency(exchange.daily_trading_limit_usd)}
                    </Typography>
                    
                    <Box mt={2} display="flex" gap={1}>
                      <Button size="small" startIcon={<Edit />}>
                        Edit
                      </Button>
                      <Button size="small" startIcon={<Settings />}>
                        Test Connection
                      </Button>
                      <Button size="small" startIcon={<Delete />} color="error">
                        Remove
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Strategy Configuration */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">
            <Speed sx={{ mr: 1, verticalAlign: 'middle' }} />
            Configured Strategies ({strategies.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {strategies.map((strategy, index) => {
              const validation = validationResults[strategy.id];
              return (
                <Grid item xs={12} key={index}>
                  <Card>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Box display="flex" alignItems="center" gap={1}>
                          {getStrategyStatusIcon(strategy)}
                          <Typography variant="h6">
                            {strategy.name}
                          </Typography>
                        </Box>
                        
                        <Box display="flex" alignItems="center" gap={1}>
                          <FormControlLabel
                            control={
                              <Switch
                                checked={strategy.is_enabled}
                                onChange={(e) => toggleStrategy(strategy.id, e.target.checked)}
                              />
                            }
                            label="Enabled"
                          />
                          <Button
                            size="small"
                            onClick={() => validateStrategy(strategy.id)}
                          >
                            Validate
                          </Button>
                          <Button
                            size="small"
                            onClick={() => autoAssignStrategy(strategy.id)}
                          >
                            Auto-Assign
                          </Button>
                        </Box>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {strategy.description}
                      </Typography>
                      
                      {validation && (
                        <Box mt={1}>
                          {validation.valid ? (
                            <Alert severity="success" size="small">
                              Strategy is fully configured and ready to run!
                            </Alert>
                          ) : (
                            <Alert severity="warning" size="small">
                              Missing: {validation.missing_wallets?.join(', ')} {validation.missing_exchanges?.join(', ')}
                            </Alert>
                          )}
                        </Box>
                      )}
                      
                      <Box mt={2}>
                        <Typography variant="body2">
                          <strong>Capital Range:</strong> {formatCurrency(strategy.min_capital_usd)} - {formatCurrency(strategy.max_capital_usd)}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Profit Target:</strong> {strategy.profit_target_percent}%
                        </Typography>
                        <Typography variant="body2">
                          <strong>Execution:</strong> Every {strategy.execution_frequency_minutes} minutes
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Add Wallet Dialog */}
      <Dialog open={showAddWallet} onClose={() => setShowAddWallet(false)}>
        <DialogTitle>Add New Wallet</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Network</InputLabel>
                <Select
                  value={newWallet.network}
                  onChange={(e) => setNewWallet({ ...newWallet, network: e.target.value })}
                >
                  {networks.map((network) => (
                    <MenuItem key={network} value={network}>
                      {network.replace('_', ' ').toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Wallet Address"
                value={newWallet.address}
                onChange={(e) => setNewWallet({ ...newWallet, address: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Private Key (Optional)"
                type={showPrivateKeys ? 'text' : 'password'}
                value={newWallet.privateKey}
                onChange={(e) => setNewWallet({ ...newWallet, privateKey: e.target.value })}
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowPrivateKeys(!showPrivateKeys)}
                    >
                      {showPrivateKeys ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  )
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Current Balance (USD)"
                type="number"
                value={newWallet.balance}
                onChange={(e) => setNewWallet({ ...newWallet, balance: parseFloat(e.target.value) })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAddWallet(false)}>Cancel</Button>
          <Button variant="contained" onClick={addWallet}>Add Wallet</Button>
        </DialogActions>
      </Dialog>

      {/* Add Exchange Dialog */}
      <Dialog open={showAddExchange} onClose={() => setShowAddExchange(false)}>
        <DialogTitle>Add Exchange API Configuration</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Exchange</InputLabel>
                <Select
                  value={newExchange.exchange}
                  onChange={(e) => setNewExchange({ ...newExchange, exchange: e.target.value })}
                >
                  {exchangeTypes.map((exchange) => (
                    <MenuItem key={exchange} value={exchange}>
                      {exchange.replace('_', ' ').toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="API Key"
                value={newExchange.apiKey}
                onChange={(e) => setNewExchange({ ...newExchange, apiKey: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="API Secret"
                type={showPrivateKeys ? 'text' : 'password'}
                value={newExchange.apiSecret}
                onChange={(e) => setNewExchange({ ...newExchange, apiSecret: e.target.value })}
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowPrivateKeys(!showPrivateKeys)}
                    >
                      {showPrivateKeys ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  )
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Passphrase (Optional)"
                value={newExchange.passphrase}
                onChange={(e) => setNewExchange({ ...newExchange, passphrase: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={newExchange.sandboxMode}
                    onChange={(e) => setNewExchange({ ...newExchange, sandboxMode: e.target.checked })}
                  />
                }
                label="Sandbox Mode (Recommended for testing)"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAddExchange(false)}>Cancel</Button>
          <Button variant="contained" onClick={addExchange}>Add Exchange</Button>
        </DialogActions>
      </Dialog>

      {/* Strategy Explanation Dialog */}
      <StrategyExplanationDialog
        strategy={selectedStrategy}
        open={showExplanation}
        onClose={() => setShowExplanation(false)}
      />
    </Box>
  );
};

export default WalletConfigPanel;

