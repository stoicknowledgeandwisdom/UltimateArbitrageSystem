import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Slider,
  TextField,
  Chip,
  Alert,
  LinearProgress,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Switch,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Psychology,
  AccountBalance,
  Security,
  TrendingUp,
  Speed,
  Bolt,
  SmartToy,
  CheckCircle,
  Warning,
  Info,
  Settings,
  Eco,
  Rocket,
  AutoMode
} from '@mui/icons-material';

const SetupWizard = ({ open, onClose, onComplete }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [completed, setCompleted] = useState(new Set());
  const [loading, setLoading] = useState(false);
  
  // Configuration State
  const [config, setConfig] = useState({
    // Step 1: Experience Level
    userType: 'intermediate', // beginner, intermediate, expert
    experience: 50, // 0-100
    
    // Step 2: Investment Profile
    investmentGoal: 'balanced_growth', // capital_preservation, balanced_growth, aggressive_growth
    riskTolerance: 50, // 0-100
    timeHorizon: 'medium', // short, medium, long
    portfolioSize: 1000000, // USD
    
    // Step 3: Strategy Preferences
    strategies: {
      quantum_momentum: true,
      ai_mean_reversion: true,
      cross_asset: true,
      volatility_targeting: false,
      statistical_arbitrage: false,
      deep_rl: false
    },
    
    // Step 4: Automation & Risk
    automationLevel: 80, // 0-100
    maxDrawdown: 5, // %
    profitTarget: 20, // % annual
    
    // Step 5: Advanced Settings
    useQuantum: true,
    useAI: true,
    esgMode: false,
    turboMode: false,
    paperTrading: true
  });
  
  const steps = [
    {
      label: 'Experience Level',
      description: 'Tell us about your trading experience',
      icon: <Psychology />
    },
    {
      label: 'Investment Profile',
      description: 'Define your investment goals and risk profile',
      icon: <AccountBalance />
    },
    {
      label: 'Strategy Selection',
      description: 'Choose your preferred trading strategies',
      icon: <TrendingUp />
    },
    {
      label: 'Automation & Risk',
      description: 'Configure automation and risk parameters',
      icon: <AutoMode />
    },
    {
      label: 'Advanced Settings',
      description: 'Fine-tune advanced features',
      icon: <Settings />
    },
    {
      label: 'Review & Launch',
      description: 'Review your configuration and start optimizing',
      icon: <Rocket />
    }
  ];
  
  const userTypeOptions = [
    {
      value: 'beginner',
      label: 'Beginner',
      description: 'New to algorithmic trading',
      icon: '🌱',
      defaults: { automationLevel: 95, riskTolerance: 30, profitTarget: 15 }
    },
    {
      value: 'intermediate',
      label: 'Intermediate',
      description: 'Some trading experience',
      icon: '📈',
      defaults: { automationLevel: 80, riskTolerance: 50, profitTarget: 20 }
    },
    {
      value: 'expert',
      label: 'Expert',
      description: 'Advanced trader with deep knowledge',
      icon: '🎯',
      defaults: { automationLevel: 60, riskTolerance: 70, profitTarget: 30 }
    }
  ];
  
  const investmentGoals = [
    {
      value: 'capital_preservation',
      label: 'Capital Preservation',
      description: 'Protect capital with minimal risk',
      icon: '🛡️',
      risk: 20,
      target: 8
    },
    {
      value: 'balanced_growth',
      label: 'Balanced Growth',
      description: 'Steady growth with moderate risk',
      icon: '⚖️',
      risk: 50,
      target: 20
    },
    {
      value: 'aggressive_growth',
      label: 'Aggressive Growth',
      description: 'Maximum returns with higher risk',
      icon: '🚀',
      risk: 80,
      target: 35
    }
  ];
  
  const strategyDescriptions = {
    quantum_momentum: {
      name: 'Quantum Momentum',
      description: 'Quantum-enhanced momentum strategies',
      complexity: 'Advanced',
      risk: 'Medium',
      returns: 'High'
    },
    ai_mean_reversion: {
      name: 'AI Mean Reversion',
      description: 'AI-driven mean reversion trading',
      complexity: 'Medium',
      risk: 'Low',
      returns: 'Medium'
    },
    cross_asset: {
      name: 'Cross-Asset Arbitrage',
      description: 'Multi-asset arbitrage opportunities',
      complexity: 'Medium',
      risk: 'Medium',
      returns: 'Medium'
    },
    volatility_targeting: {
      name: 'Volatility Targeting',
      description: 'Dynamic volatility-based allocation',
      complexity: 'Advanced',
      risk: 'Low',
      returns: 'Medium'
    },
    statistical_arbitrage: {
      name: 'Statistical Arbitrage',
      description: 'Statistical price relationship exploitation',
      complexity: 'Expert',
      risk: 'Medium',
      returns: 'High'
    },
    deep_rl: {
      name: 'Deep Reinforcement Learning',
      description: 'Advanced neural network strategies',
      complexity: 'Expert',
      risk: 'High',
      returns: 'Very High'
    }
  };
  
  const handleNext = () => {
    const newCompleted = new Set(completed);
    newCompleted.add(activeStep);
    setCompleted(newCompleted);
    setActiveStep((prevStep) => prevStep + 1);
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  const handleUserTypeChange = (newUserType) => {
    const userTypeConfig = userTypeOptions.find(opt => opt.value === newUserType);
    setConfig(prev => ({
      ...prev,
      userType: newUserType,
      automationLevel: userTypeConfig.defaults.automationLevel,
      riskTolerance: userTypeConfig.defaults.riskTolerance,
      profitTarget: userTypeConfig.defaults.profitTarget
    }));
  };
  
  const handleInvestmentGoalChange = (newGoal) => {
    const goalConfig = investmentGoals.find(goal => goal.value === newGoal);
    setConfig(prev => ({
      ...prev,
      investmentGoal: newGoal,
      riskTolerance: goalConfig.risk,
      profitTarget: goalConfig.target
    }));
  };
  
  const handleStrategyToggle = (strategy) => {
    setConfig(prev => ({
      ...prev,
      strategies: {
        ...prev.strategies,
        [strategy]: !prev.strategies[strategy]
      }
    }));
  };
  
  const handleComplete = async () => {
    setLoading(true);
    try {
      // Convert config to API format
      const apiConfig = {
        automation_level: config.automationLevel / 100,
        risk_tolerance: config.riskTolerance / 100,
        profit_target: config.profitTarget / 100,
        use_quantum: config.useQuantum,
        use_ai: config.useAI,
        enable_live_trading: !config.paperTrading
      };
      
      // Call completion handler
      await onComplete(apiConfig);
      onClose();
    } catch (error) {
      console.error('Setup completion failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              What's your trading experience level?
            </Typography>
            <Grid container spacing={3} sx={{ mt: 1 }}>
              {userTypeOptions.map((option) => (
                <Grid item xs={12} md={4} key={option.value}>
                  <Card 
                    sx={{ 
                      cursor: 'pointer',
                      border: config.userType === option.value ? 2 : 1,
                      borderColor: config.userType === option.value ? 'primary.main' : 'divider',
                      '&:hover': { borderColor: 'primary.main' }
                    }}
                    onClick={() => handleUserTypeChange(option.value)}
                  >
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h3" sx={{ mb: 1 }}>{option.icon}</Typography>
                      <Typography variant="h6" gutterBottom>{option.label}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {option.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            
            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Overall Experience (0-10 years)</Typography>
              <Slider
                value={config.experience}
                onChange={(e, value) => setConfig(prev => ({ ...prev, experience: value }))}
                min={0}
                max={100}
                marks={[
                  { value: 0, label: '0' },
                  { value: 25, label: '2.5' },
                  { value: 50, label: '5' },
                  { value: 75, label: '7.5' },
                  { value: 100, label: '10+' }
                ]}
                sx={{ mt: 2 }}
              />
            </Box>
          </Box>
        );
      
      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Define your investment profile
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 1 }}>
              {investmentGoals.map((goal) => (
                <Grid item xs={12} md={4} key={goal.value}>
                  <Card
                    sx={{
                      cursor: 'pointer',
                      border: config.investmentGoal === goal.value ? 2 : 1,
                      borderColor: config.investmentGoal === goal.value ? 'primary.main' : 'divider',
                      '&:hover': { borderColor: 'primary.main' }
                    }}
                    onClick={() => handleInvestmentGoalChange(goal.value)}
                  >
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h3" sx={{ mb: 1 }}>{goal.icon}</Typography>
                      <Typography variant="h6" gutterBottom>{goal.label}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {goal.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            
            <Grid container spacing={3} sx={{ mt: 3 }}>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Portfolio Size (USD)</Typography>
                <TextField
                  fullWidth
                  type="number"
                  value={config.portfolioSize}
                  onChange={(e) => setConfig(prev => ({ ...prev, portfolioSize: parseInt(e.target.value) }))}
                  InputProps={{
                    startAdornment: '$'
                  }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <FormLabel>Time Horizon</FormLabel>
                  <RadioGroup
                    value={config.timeHorizon}
                    onChange={(e) => setConfig(prev => ({ ...prev, timeHorizon: e.target.value }))}
                    row
                  >
                    <FormControlLabel value="short" control={<Radio />} label="Short (< 1 year)" />
                    <FormControlLabel value="medium" control={<Radio />} label="Medium (1-5 years)" />
                    <FormControlLabel value="long" control={<Radio />} label="Long (5+ years)" />
                  </RadioGroup>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        );
      
      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select your preferred trading strategies
            </Typography>
            <Alert severity="info" sx={{ mb: 3 }}>
              Choose strategies that match your risk tolerance and complexity preference. You can always adjust these later.
            </Alert>
            
            <Grid container spacing={2}>
              {Object.entries(strategyDescriptions).map(([key, strategy]) => (
                <Grid item xs={12} md={6} key={key}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box>
                          <Typography variant="h6" gutterBottom>{strategy.name}</Typography>
                          <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                            {strategy.description}
                          </Typography>
                        </Box>
                        <Switch
                          checked={config.strategies[key]}
                          onChange={() => handleStrategyToggle(key)}
                          color="primary"
                        />
                      </Box>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip label={`Complexity: ${strategy.complexity}`} size="small" variant="outlined" />
                        <Chip label={`Risk: ${strategy.risk}`} size="small" variant="outlined" />
                        <Chip label={`Returns: ${strategy.returns}`} size="small" variant="outlined" />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        );
      
      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure automation and risk parameters
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography gutterBottom>Automation Level: {config.automationLevel}%</Typography>
                  <Slider
                    value={config.automationLevel}
                    onChange={(e, value) => setConfig(prev => ({ ...prev, automationLevel: value }))}
                    min={0}
                    max={100}
                    marks={[
                      { value: 0, label: 'Manual' },
                      { value: 50, label: 'Hybrid' },
                      { value: 100, label: 'Full Auto' }
                    ]}
                  />
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    Higher automation requires less manual intervention
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography gutterBottom>Risk Tolerance: {config.riskTolerance}%</Typography>
                  <Slider
                    value={config.riskTolerance}
                    onChange={(e, value) => setConfig(prev => ({ ...prev, riskTolerance: value }))}
                    min={0}
                    max={100}
                    marks={[
                      { value: 0, label: 'Conservative' },
                      { value: 50, label: 'Balanced' },
                      { value: 100, label: 'Aggressive' }
                    ]}
                  />
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    Higher risk tolerance allows for more aggressive strategies
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography gutterBottom>Profit Target: {config.profitTarget}% annually</Typography>
                  <Slider
                    value={config.profitTarget}
                    onChange={(e, value) => setConfig(prev => ({ ...prev, profitTarget: value }))}
                    min={5}
                    max={50}
                    marks={[
                      { value: 5, label: '5%' },
                      { value: 20, label: '20%' },
                      { value: 50, label: '50%' }
                    ]}
                  />
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    Annual return target for the portfolio
                  </Typography>
                </Box>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Maximum Drawdown: {config.maxDrawdown}%</Typography>
              <Slider
                value={config.maxDrawdown}
                onChange={(e, value) => setConfig(prev => ({ ...prev, maxDrawdown: value }))}
                min={1}
                max={20}
                marks={[
                  { value: 1, label: '1%' },
                  { value: 5, label: '5%' },
                  { value: 10, label: '10%' },
                  { value: 20, label: '20%' }
                ]}
              />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Maximum acceptable portfolio decline from peak
              </Typography>
            </Box>
          </Box>
        );
      
      case 4:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Advanced system features
            </Typography>
            
            <List>
              <ListItem>
                <ListItemAvatar>
                  <Avatar><Bolt /></Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary="Quantum Computing"
                  secondary="Enable quantum-enhanced optimization algorithms"
                />
                <ListItemSecondaryAction>
                  <Switch
                    checked={config.useQuantum}
                    onChange={(e) => setConfig(prev => ({ ...prev, useQuantum: e.target.checked }))}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemAvatar>
                  <Avatar><SmartToy /></Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary="AI Enhancements"
                  secondary="Use advanced machine learning models"
                />
                <ListItemSecondaryAction>
                  <Switch
                    checked={config.useAI}
                    onChange={(e) => setConfig(prev => ({ ...prev, useAI: e.target.checked }))}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemAvatar>
                  <Avatar><Eco /></Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary="ESG Mode"
                  secondary="Focus on environmentally and socially responsible investments"
                />
                <ListItemSecondaryAction>
                  <Switch
                    checked={config.esgMode}
                    onChange={(e) => setConfig(prev => ({ ...prev, esgMode: e.target.checked }))}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemAvatar>
                  <Avatar><Speed /></Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary="Turbo Mode"
                  secondary="Maximize execution speed and frequency"
                />
                <ListItemSecondaryAction>
                  <Switch
                    checked={config.turboMode}
                    onChange={(e) => setConfig(prev => ({ ...prev, turboMode: e.target.checked }))}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <ListItem>
                <ListItemAvatar>
                  <Avatar><Security /></Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary="Paper Trading Mode"
                  secondary="Start with simulated trading (recommended for first setup)"
                />
                <ListItemSecondaryAction>
                  <Switch
                    checked={config.paperTrading}
                    onChange={(e) => setConfig(prev => ({ ...prev, paperTrading: e.target.checked }))}
                  />
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </Box>
        );
      
      case 5:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review your configuration
            </Typography>
            
            <Alert severity="success" sx={{ mb: 3 }}>
              <Typography variant="h6">Configuration Complete!</Typography>
              Your Ultimate Arbitrage System is ready to launch with these settings.
            </Alert>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Profile Summary</Typography>
                    <Typography>Experience: {config.userType}</Typography>
                    <Typography>Goal: {config.investmentGoal.replace('_', ' ')}</Typography>
                    <Typography>Portfolio: ${config.portfolioSize.toLocaleString()}</Typography>
                    <Typography>Time Horizon: {config.timeHorizon}</Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Risk & Returns</Typography>
                    <Typography>Automation: {config.automationLevel}%</Typography>
                    <Typography>Risk Tolerance: {config.riskTolerance}%</Typography>
                    <Typography>Profit Target: {config.profitTarget}% annually</Typography>
                    <Typography>Max Drawdown: {config.maxDrawdown}%</Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Active Strategies</Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {Object.entries(config.strategies)
                        .filter(([key, enabled]) => enabled)
                        .map(([key, enabled]) => (
                          <Chip
                            key={key}
                            label={strategyDescriptions[key].name}
                            color="primary"
                            variant="filled"
                          />
                        ))
                      }
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Advanced Features</Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {config.useQuantum && <Chip label="Quantum Computing" color="secondary" />}
                      {config.useAI && <Chip label="AI Enhanced" color="primary" />}
                      {config.esgMode && <Chip label="ESG Mode" color="success" />}
                      {config.turboMode && <Chip label="Turbo Mode" color="warning" />}
                      {config.paperTrading && <Chip label="Paper Trading" color="info" />}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        );
      
      default:
        return 'Unknown step';
    }
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { minHeight: '70vh' }
      }}
    >
      <DialogTitle>
        <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
          🚀 Ultimate Arbitrage System Setup
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Let's configure your quantum-powered trading system
        </Typography>
      </DialogTitle>
      
      <DialogContent sx={{ mt: 2 }}>
        <Stepper activeStep={activeStep} orientation="horizontal" sx={{ mb: 4 }}>
          {steps.map((step, index) => (
            <Step key={step.label} completed={completed.has(index)}>
              <StepLabel
                icon={completed.has(index) ? <CheckCircle color="success" /> : step.icon}
              >
                <Typography variant="subtitle2">{step.label}</Typography>
              </StepLabel>
            </Step>
          ))}
        </Stepper>
        
        {loading && <LinearProgress sx={{ mb: 2 }} />}
        
        <Box sx={{ minHeight: 400 }}>
          {getStepContent(activeStep)}
        </Box>
      </DialogContent>
      
      <DialogActions sx={{ p: 3, justifyContent: 'space-between' }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
          size="large"
        >
          Back
        </Button>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button onClick={onClose} size="large">
            Cancel
          </Button>
          
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleComplete}
              disabled={loading}
              size="large"
              startIcon={<Rocket />}
            >
              {loading ? 'Launching...' : 'Launch System'}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
              size="large"
            >
              Next
            </Button>
          )}
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default SetupWizard;

