import React, { useState } from 'react';
import { useMarketData } from '../contexts/MarketDataContext';

// Components
import ExchangeSelector from '../components/trading/ExchangeSelector';
import TradingPairSelector from '../components/trading/TradingPairSelector';
import PriceChart from '../components/trading/PriceChart';
import OrderBook from '../components/trading/OrderBook';
import TradeForm from '../components/trading/TradeForm';
import MarketDepth from '../components/trading/MarketDepth';

const TradingView = () => {
  const { 
    exchanges,
    selectedExchanges,
    setSelectedExchanges,
    tradingPairs,
    selectedPair,
    setSelectedPair,
    timeframe,
    setTimeframe,
    marketData,
    isLoading
  } = useMarketData();
  
  const [orderType, setOrderType] = useState('limit'); // 'limit', 'market', etc.
  
  const handleExchangeChange = (exchangeId) => {
    setSelectedExchanges([exchangeId]);
  };
  
  const handlePairChange = (pair) => {
    setSelectedPair(pair);
  };
  
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
  };
  
  if (isLoading) {
    return <div>Loading trading data...</div>;
  }

  return (
    <div className="trading-view-container">
      <div className="trading-view-header">
        <h1>Trading View</h1>
        <div className="trading-view-controls">
          <ExchangeSelector 
            exchanges={exchanges} 
            selectedExchange={selectedExchanges[0]} 
            onExchangeChange={handleExchangeChange}
          />
          <TradingPairSelector 
            pairs={tradingPairs} 
            selectedPair={selectedPair} 
            onPairChange={handlePairChange}
          />
          <div className="timeframe-selector">
            {['1m', '5m', '15m', '1h', '4h', '1d'].map(tf => (
              <button
                key={tf}
                className={timeframe === tf ? 'active' : ''}
                onClick={() => handleTimeframeChange(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
      </div>
      
      <div className="trading-view-main">
        <div className="trading-view-chart">
          <PriceChart 
            data={marketData.ohlcv[selectedExchanges[0]]?.[timeframe] || []} 
            pair={selectedPair}
            timeframe={timeframe}
          />
        </div>
        
        <div className="trading-view-sidebar">
          <OrderBook 
            orderbook={marketData.orderbooks[selectedExchanges[0]] || { bids: [], asks: [] }} 
            pair={selectedPair}
          />
          <TradeForm 
            pair={selectedPair} 
            exchange={selectedExchanges[0]}
            orderType={orderType}
            setOrderType={setOrderType}
            currentPrice={marketData.tickers[selectedExchanges[0]]?.last || 0}
          />
        </div>
      </div>
      
      <div className="trading-view-footer">
        <MarketDepth 
          orderbook={marketData.orderbooks[selectedExchanges[0]] || { bids: [], asks: [] }}
          pair={selectedPair}
        />
      </div>
    </div>
  );
};

export default TradingView;

