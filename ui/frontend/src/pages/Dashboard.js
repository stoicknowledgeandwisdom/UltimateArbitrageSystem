import React, { useState, useEffect } from 'react';
import { useMarketData } from '../contexts/MarketDataContext';

// Components
import Card from '../components/ui/Card';
import BalanceSummary from '../components/dashboard/BalanceSummary';
import ProfitLossChart from '../components/dashboard/ProfitLossChart';
import ArbitrageOpportunities from '../components/dashboard/ArbitrageOpportunities';
import RecentTransactions from '../components/dashboard/RecentTransactions';
import SystemStatus from '../components/dashboard/SystemStatus';

const Dashboard = () => {
  const { marketData, isLoading } = useMarketData();
  const [totalBalance, setTotalBalance] = useState(0);
  const [profitLossData, setProfitLossData] = useState([]);

  useEffect(() => {
    // This would be replaced with actual API calls in a real implementation
    setTotalBalance(125432.65);
    
    // Sample data for profit/loss chart
    setProfitLossData([
      { date: '2023-05-01', profit: 1250 },
      { date: '2023-05-02', profit: 1820 },
      { date: '2023-05-03', profit: 1654 },
      { date: '2023-05-04', profit: 2100 },
      { date: '2023-05-05', profit: 1890 },
      { date: '2023-05-06', profit: 2340 },
      { date: '2023-05-07', profit: 2580 },
    ]);
  }, []);

  if (isLoading) {
    return <div>Loading dashboard data...</div>;
  }

  return (
    <div className="dashboard-container">
      <h1>Ultimate Arbitrage Dashboard</h1>
      
      <div className="dashboard-summary">
        <Card title="Portfolio Overview">
          <BalanceSummary totalBalance={totalBalance} />
        </Card>
      </div>
      
      <div className="dashboard-charts">
        <Card title="Profit/Loss Performance">
          <ProfitLossChart data={profitLossData} />
        </Card>
      </div>
      
      <div className="dashboard-opportunities">
        <Card title="Live Arbitrage Opportunities">
          <ArbitrageOpportunities opportunities={marketData.arbitrageOpportunities} />
        </Card>
      </div>
      
      <div className="dashboard-recent">
        <Card title="Recent Transactions">
          <RecentTransactions />
        </Card>
      </div>
      
      <div className="dashboard-status">
        <Card title="System Status">
          <SystemStatus />
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;

