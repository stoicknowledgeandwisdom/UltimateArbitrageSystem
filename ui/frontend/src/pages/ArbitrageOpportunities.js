import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Container, 
  Grid, 
  Paper, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Button,
  Tabs,
  Tab,
  Box,
  CircularProgress,
  Chip,
  Divider
} from '@mui/material';
import { useMarketData } from '../contexts/MarketDataContext';

const ArbitrageOpportunities = () => {
  const { 
    marketData, 
    isLoading, 
    error,
    executeArbitrage,
    simulateArbitrage
  } = useMarketData();
  
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedOpportunity, setSelectedOpportunity] = useState(null);
  const [simulationResults, setSimulationResults] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  
  // Filter options
  const [filters, setFilters] = useState({
    minProfit: 0.1,
    exchanges: [],
    type: 'all' // 'all', 'triangular', 'cross-exchange'
  });
  
  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };
  
  const handleOpportunitySelect = (opportunity) => {
    setSelectedOpportunity(opportunity);
    setSimulationResults(null);
  };
  
  const handleSimulate = async () => {
    if (!selectedOpportunity) return;
    
    setIsSimulating(true);
    try {
      const results = await simulateArbitrage(selectedOpportunity.id);
      setSimulationResults(results);
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      setIsSimulating(false);
    }
  };
  
  const handleExecute = () => {
    if (!selectedOpportunity) return;
    executeArbitrage(selectedOpportunity.id);
  };
  
  // Get arbitrage opportunities from context
  const opportunities = marketData.arbitrageOpportunities || [];
  
  // Filter opportunities based on current filter settings
  const filteredOpportunities = opportunities.filter(opp => {
    if (filters.type !== 'all' && opp.type !== filters.type) return false;
    if (opp.profit < filters.minProfit) return false;
    if (filters.exchanges.length > 0) {
      const oppExchanges = [
        opp.exchange, 
        opp.buyExchange, 
        opp.sellExchange
      ].filter(Boolean);
      
      if (!oppExchanges.some(e => filters.exchanges.includes(e))) return false;
    }
    return true;
  });
  
  // Format profit percentage for display
  const formatProfit = (profit) => {
    return `${(profit * 100).toFixed(2)}%`;
  };
  
  // Get status color based on opportunity confidence
  const getStatusColor = (confidence) => {
    if (confidence >= 90) return 'success';
    if (confidence >= 75) return 'warning';
    return 'error';
  };
  
  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container justifyContent="center" alignItems="center" style={{ minHeight: '60vh' }}>
          <CircularProgress />
          <Typography variant="h6" sx={{ ml: 2 }}>
            Loading arbitrage opportunities...
          </Typography>
        </Grid>
      </Container>
    );
  }
  
  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h6" color="error">
            Error loading arbitrage opportunities: {error}
          </Typography>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom component="div">
        Arbitrage Opportunities
      </Typography>
      
      <Tabs value={selectedTab} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="All Opportunities" />
        <Tab label="Cross-Exchange" />
        <Tab label="Triangular" />
        <Tab label="Multi-Path" />
      </Tabs>
      
      <Grid container spacing={3}>
        {/* Opportunities List */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Available Opportunities ({filteredOpportunities.length})
            </Typography>
            
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell>Asset/Path</TableCell>
                    <TableCell>Exchange(s)</TableCell>
                    <TableCell>Profit</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredOpportunities.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        No arbitrage opportunities available
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredOpportunities.map((opp) => (
                      <TableRow 
                        key={opp.id} 
                        hover 
                        selected={selectedOpportunity?.id === opp.id}
                        onClick={() => handleOpportunitySelect(opp)}
                      >
                        <TableCell>{opp.type}</TableCell>
                        <TableCell>
                          {opp.type === 'triangular' 
                            ? opp.path.join(' → ') 
                            : opp.asset}
                        </TableCell>
                        <TableCell>
                          {opp.type === 'triangular' 
                            ? opp.exchange 
                            : `${opp.buyExchange} → ${opp.sellExchange}`}
                        </TableCell>
                        <TableCell>
                          {formatProfit(opp.profit)} (${opp.profitUsd.toFixed(2)})
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={`${opp.confidence.toFixed(1)}%`}
                            color={getStatusColor(opp.confidence)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Button 
                            size="small" 
                            variant="outlined"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleOpportunitySelect(opp);
                            }}
                          >
                            Details
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
        
        {/* Opportunity Details */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: '100%' }}>
            {selectedOpportunity ? (
              <>
                <Typography variant="h6" gutterBottom>
                  Opportunity Details
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle1">
                      ID: {selectedOpportunity.id}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2">Type:</Typography>
                    <Typography variant="subtitle1">
                      {selectedOpportunity.type.charAt(0).toUpperCase() + selectedOpportunity.type.slice(1)}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2">Profit:</Typography>
                    <Typography variant="subtitle1" color="primary">
                      {formatProfit(selectedOpportunity.profit)} (${selectedOpportunity.profitUsd.toFixed(2)})
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2">Confidence:</Typography>
                    <Typography variant="subtitle1">
                      {selectedOpportunity.confidence.toFixed(1)}%
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2">Estimated Fees:</Typography>
                    <Typography variant="subtitle1">
                      ${selectedOpportunity.estimatedFees.toFixed(2)}
                    </Typography>
                  </Grid>
                  
                  {selectedOpportunity.type === 'triangular' ? (
                    <>
                      <Grid item xs={12}>
                        <Typography variant="body2">Path:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.path.join(' → ')}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body2">Exchange:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.exchange}
                        </Typography>
                      </Grid>
                    </>
                  ) : (
                    <>
                      <Grid item xs={12}>
                        <Typography variant="body2">Asset:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.asset}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Buy Exchange:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.buyExchange}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Sell Exchange:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.sellExchange}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Buy Price:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.buyPrice.toFixed(6)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Sell Price:</Typography>
                        <Typography variant="subtitle1">
                          {selectedOpportunity.sellPrice.toFixed(6)}
                        </Typography>
                      </Grid>
                    </>
                  )}
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                {/* Simulation Results */}
                {simulationResults && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Simulation Results
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2">Expected Profit:</Typography>
                        <Typography variant="subtitle1" color="primary">
                          {formatProfit(simulationResults.expectedProfit)} 
                          (${simulationResults.expectedProfitUsd.toFixed(2)})
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Execution Time:</Typography>
                        <Typography variant="subtitle1">
                          {(simulationResults.executionTime / 1000).toFixed(2)}s
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Gas Estimate:</Typography>
                        <Typography variant="subtitle1">
                          ${simulationResults.gasEstimate.toFixed(4)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Success Probability:</Typography>
                        <Typography variant="subtitle1">
                          {(simulationResults.successProbability * 100).toFixed(1)}%
                        </Typography>
                      </Grid>
                    </Grid>
                    
                    <Divider sx={{ my: 2 }} />
                  </Box>
                )}
                
                {/* Action Buttons */}
                <Box sx={{ mt: 'auto', display: 'flex', justifyContent: 'space-between' }}>
                  <Button 
                    variant="outlined" 
                    onClick={handleSimulate}
                    disabled={isSimulating}
                  >
                    {isSimulating ? 'Simulating...' : 'Simulate Execution'}
                  </Button>
                  
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={handleExecute}
                    disabled={!simulationResults}
                  >
                    Execute Arbitrage
                  </Button>
                </Box>
              </>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Typography variant="h6" color="textSecondary">
                  Select an opportunity to view details
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ArbitrageOpportunities;

