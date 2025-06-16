import React from 'react';
import { Container, Paper, Box } from '@mui/material';
import SystemHealthMonitor from '../components/SystemHealthMonitor';
import QuickActionsPanel from '../components/QuickActionsPanel';
import PortfolioAnalytics from '../components/PortfolioAnalytics';
import RealTimeProfitGraph from '../components/RealTimeProfitGraph';
import AIStrategyAssistant from '../components/AIStrategyAssistant';
import WalletConfigPanel from '../components/WalletConfigPanel';

const WalletManagement = () => {
  const handleConfigChange = (newConfig) => {
    // Handle any changes to wallet configuration
    console.log('Wallet configuration changed:', newConfig);
  };

return (
    <Container maxWidth="xl">
<Box sx={{ py: 3 }}>
        <SystemHealthMonitor />
        <RealTimeProfitGraph />
        <QuickActionsPanel />
        <PortfolioAnalytics />
        <AIStrategyAssistant />
        <Paper elevation={0} sx={{ p: 0, bgcolor: 'transparent' }}>
          <WalletConfigPanel onConfigChange={handleConfigChange} />
        </Paper>
      </Box>
    </Container>
  );
};

export default WalletManagement;
