import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { format } from 'date-fns';

// Styled components
const Container = styled.div`
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  background-color: #ffffff;
  overflow: hidden;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #e5e7eb;
`;

const Title = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: #111827;
  margin: 0;
`;

const RefreshButton = styled.button`
  background-color: transparent;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  padding: 6px 12px;
  color: #4b5563;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background-color: #f3f4f6;
    border-color: #9ca3af;
  }
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
`;

const TableHead = styled.thead`
  background-color: #f9fafb;
  
  th {
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    color: #374151;
    border-bottom: 1px solid #e5e7eb;
  }
`;

const TableBody = styled.tbody`
  tr {
    &:hover {
      background-color: #f9fafb;
    }
    
    &:not(:last-child) {
      border-bottom: 1px solid #e5e7eb;
    }
  }
  
  td {
    padding: 12px 16px;
    color: #4b5563;
  }
`;

const NoOpportunities = styled.div`
  padding: 32px;
  text-align: center;
  color: #6b7280;
  font-style: italic;
`;

const ProfitIndicator = styled.div`
  display: flex;
  align-items: center;
`;

const ProfitBadge = styled.span`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: 600;
  background-color: ${props => {
    if (props.profit >= 2.0) return '#10b981'; // Green for high profit
    if (props.profit >= 1.0) return '#22c55e'; // Light green for good profit
    if (props.profit >= 0.5) return '#f59e0b'; // Yellow for moderate profit
    return '#ef4444'; // Red for low profit
  }};
  color: white;
`;

const SparkIndicator = styled.div`
  display: inline-block;
  margin-left: 8px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: ${props => props.active ? '#ef4444' : 'transparent'};
  animation: ${props => props.active ? 'pulse 1.5s infinite' : 'none'};
  
  @keyframes pulse {
    0% {
      transform: scale(0.8);
      opacity: 0.8;
    }
    70% {
      transform: scale(1.1);
      opacity: 1;
    }
    100% {
      transform: scale(0.8);
      opacity: 0.8;
    }
  }
`;

const ExecuteButton = styled.button`
  background-color: #1e40af;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #1e3a8a;
  }
  
  &:disabled {
    background-color: #9ca3af;
    cursor: not-allowed;
  }
`;

const AutomateButton = styled.button`
  background-color: transparent;
  color: #4b5563;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  margin-left: 8px;
  transition: all 0.2s;
  
  &:hover {
    background-color: #f3f4f6;
    border-color: #9ca3af;
  }
`;

const StatusTag = styled.span`
  display: inline-block;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
  background-color: ${props => props.status === 'ready' ? '#dcfce7' : '#fee2e2'};
  color: ${props => props.status === 'ready' ? '#15803d' : '#b91c1c'};
  margin-left: 8px;
`;

const ArbitrageOpportunities = ({ opportunities = [], onRefresh }) => {
  const [executingId, setExecutingId] = useState(null);
  const [automatedIds, setAutomatedIds] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  // Update lastUpdated when opportunities change
  useEffect(() => {
    if (opportunities.length > 0) {
      setLastUpdated(new Date());
    }
  }, [opportunities]);
  
  // Function to execute an arbitrage opportunity
  const executeArbitrage = async (opportunityId) => {
    try {
      setExecutingId(opportunityId);
      // In a real implementation, this would call your arbitrage service
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API call
      
      // Handle success scenario
      console.log(`Executed arbitrage opportunity ${opportunityId}`);
      // You would typically update the UI or show a notification here
    } catch (error) {
      console.error('Failed to execute arbitrage:', error);
      // Handle error scenario
    } finally {
      setExecutingId(null);
    }
  };

  // Function to automate an arbitrage opportunity
  const automateArbitrage = (opportunityId) => {
    // In a real implementation, this would add the opportunity to automated trading strategy
    setAutomatedIds(prev => [...prev, opportunityId]);
    console.log(`Added arbitrage opportunity ${opportunityId} to automated trading`);
    // You would typically update the UI or show a notification here
  };

  // Function to handle refresh button click
  const handleRefresh = () => {
    if (onRefresh && typeof onRefresh === 'function') {
      onRefresh();
    }
    setLastUpdated(new Date());
  };

  return (
    <Container>
      <Header>
        <Title>Arbitrage Opportunities</Title>
        <div>
          <small>Last updated: {format(lastUpdated, 'HH:mm:ss')}</small>
          <RefreshButton onClick={handleRefresh}>
            Refresh
          </RefreshButton>
        </div>
      </Header>
      
      {(!opportunities || opportunities.length === 0) ? (
        <NoOpportunities>
          No arbitrage opportunities found at the moment. The system is continuously scanning markets.
        </NoOpportunities>
      ) : (
        <Table>
          <TableHead>
            <tr>
              <th>Exchanges</th>
              <th>Trading Pair</th>
              <th>Buy / Sell Price</th>
              <th>Profit %</th>
              <th>Est. Profit (USD)</th>
              <th>Updated</th>
              <th>Actions</th>
            </tr>
          </TableHead>
          <TableBody>
            {opportunities.map(opportunity => (
              <tr key={opportunity.id}>
                <td>
                  {opportunity.buyExchange} → {opportunity.sellExchange}
                </td>
                <td>{opportunity.tradingPair}</td>
                <td>
                  ${opportunity.buyPrice.toFixed(2)} / ${opportunity.sellPrice.toFixed(2)}
                </td>
                <td>
                  <ProfitIndicator>
                    <ProfitBadge profit={opportunity.profitPercentage}>
                      {opportunity.profitPercentage.toFixed(2)}%
                    </ProfitBadge>
                    <SparkIndicator active={opportunity.isHot} />
                  </ProfitIndicator>
                </td>
                <td>${opportunity.estimatedProfit.toFixed(2)}</td>
                <td>{format(new Date(opportunity.timestamp), 'HH:mm:ss')}</td>
                <td>
                  <ExecuteButton 
                    onClick={() => executeArbitrage(opportunity.id)}
                    disabled={executingId === opportunity.id || automatedIds.includes(opportunity.id)}
                  >
                    {executingId === opportunity.id ? 'Executing...' : 'Execute'}
                  </ExecuteButton>
                  {automatedIds.includes(opportunity.id) ? (
                    <StatusTag status="ready">Automated</StatusTag>
                  ) : (
                    <AutomateButton onClick={() => automateArbitrage(opportunity.id)}>
                      Automate
                    </AutomateButton>
                  )}
                </td>
              </tr>
            ))}
          </TableBody>
        </Table>
      )}
    </Container>
  );
};

export default ArbitrageOpportunities;

