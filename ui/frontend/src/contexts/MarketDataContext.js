import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import * as marketDataService from '../services/marketDataService';

/**
 * Market Data Context
 * Provides market data and related functions to components throughout the application.
 */
const MarketDataContext = createContext();

/**
 * Market Data Provider Component
 * Fetches and manages market data, providing it to child components.
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Child components
 */
export const MarketDataProvider = ({ children }) => {
  // Exchange state
  const [exchanges, setExchanges] = useState([]);
  const [selectedExchanges, setSelectedExchanges] = useState([]);
  
  // Trading pair state
  const [tradingPairs, setTradingPairs] = useState({});
  const [selectedPair, setSelectedPair] = useState('');
  
  // Timeframe state (for charts)
  const [timeframe, setTimeframe] = useState('1h');
  
  // Market data state
  const [marketData, setMarketData] = useState({
    ohlcv: {}, // Structure: { exchangeId: { timeframe: data } }
    orderbooks: {}, // Structure: { exchangeId: orderbook }
    tickers: {}, // Structure: { exchangeId: ticker }
    arbitrageOpportunities: [] // List of arbitrage opportunities
  });
  
  // Loading state
  const [isLoading, setIsLoading] = useState(true);
  
  // Error state
  const [error, setError] = useState(null);
  
  // Data refresh intervals (in milliseconds)
  const refreshIntervals = {
    ohlcv: 60000, // 1 minute
    orderbooks: 5000, // 5 seconds
    tickers: 5000, // 5 seconds
    arbitrageOpportunities: 10000 // 10 seconds
  };

  /**
   * Fetch all exchanges
   */
  const fetchExchanges = useCallback(async () => {
    try {
      const exchangesData = await marketDataService.getExchanges();
      setExchanges(exchangesData);
      
      // If no exchange is selected, select the first one
      if (exchangesData.length > 0 && selectedExchanges.length === 0) {
        setSelectedExchanges([exchangesData[0].id]);
      }
    } catch (err) {
      console.error('Error fetching exchanges:', err);
      setError('Failed to fetch exchanges. Please try again later.');
    }
  }, [selectedExchanges]);

  /**
   * Fetch trading pairs for an exchange
   * @param {string} exchange - Exchange ID
   */
  const fetchTradingPairs = useCallback(async (exchange) => {
    try {
      const pairs = await marketDataService.getTradingPairs(exchange);
      setTradingPairs(prevPairs => ({
        ...prevPairs,
        [exchange]: pairs
      }));
      
      // If no pair is selected and pairs are available, select the first one
      if (!selectedPair && pairs.length > 0) {
        setSelectedPair(pairs[0].symbol);
      }
    } catch (err) {
      console.error(`Error fetching trading pairs for ${exchange}:`, err);
      setError(`Failed to fetch trading pairs for ${exchange}. Please try again later.`);
    }
  }, [selectedPair]);

  /**
   * Fetch OHLCV data for a trading pair on an exchange
   * @param {string} exchange - Exchange ID
   * @param {string} pair - Trading pair
   * @param {string} tf - Timeframe
   */
  const fetchOHLCV = useCallback(async (exchange, pair, tf) => {
    if (!exchange || !pair || !tf) return;
    
    try {
      const ohlcvData = await marketDataService.getOHLCV(exchange, pair, tf);
      
      setMarketData(prevData => ({
        ...prevData,
        ohlcv: {
          ...prevData.ohlcv,
          [exchange]: {
            ...(prevData.ohlcv[exchange] || {}),
            [tf]: ohlcvData
          }
        }
      }));
    } catch (err) {
      console.error(`Error fetching OHLCV data for ${pair} on ${exchange}:`, err);
      setError(`Failed to fetch chart data for ${pair}. Please try again later.`);
    }
  }, []);

  /**
   * Fetch order book for a trading pair on an exchange
   * @param {string} exchange - Exchange ID
   * @param {string} pair - Trading pair
   */
  const fetchOrderBook = useCallback(async (exchange, pair) => {
    if (!exchange || !pair) return;
    
    try {
      const orderBookData = await marketDataService.getOrderBook(exchange, pair);
      
      setMarketData(prevData => ({
        ...prevData,
        orderbooks: {
          ...prevData.orderbooks,
          [exchange]: orderBookData
        }
      }));
    } catch (err) {
      console.error(`Error fetching order book for ${pair} on ${exchange}:`, err);
      setError(`Failed to fetch order book for ${pair}. Please try again later.`);
    }
  }, []);

  /**
   * Fetch ticker for a trading pair on an exchange
   * @param {string} exchange - Exchange ID
   * @param {string} pair - Trading pair
   */
  const fetchTicker = useCallback(async (exchange, pair) => {
    if (!exchange || !pair) return;
    
    try {
      const tickerData = await marketDataService.getTicker(exchange, pair);
      
      setMarketData(prevData => ({
        ...prevData,
        tickers: {
          ...prevData.tickers,
          [exchange]: tickerData
        }
      }));
    } catch (err) {
      console.error(`Error fetching ticker for ${pair} on ${exchange}:`, err);
      setError(`Failed to fetch price data for ${pair}. Please try again later.`);
    }
  }, []);

  /**
   * Fetch arbitrage opportunities
   */
  const fetchArbitrageOpportunities = useCallback(async () => {
    try {
      const opportunities = await marketDataService.getArbitrageOpportunities();
      
      setMarketData(prevData => ({
        ...prevData,
        arbitrageOpportunities: opportunities
      }));
    } catch (err) {
      console.error('Error fetching arbitrage opportunities:', err);
      setError('Failed to fetch arbitrage opportunities. Please try again later.');
    }
  }, []);

  /**
   * Refresh all market data
   */
  const refreshData = useCallback(() => {
    if (selectedExchanges.length === 0 || !selectedPair) return;
    
    setIsLoading(true);
    
    // Fetch data for each selected exchange
    selectedExchanges.forEach(exchange => {
      fetchOHLCV(exchange, selectedPair, timeframe);
      fetchOrderBook(exchange, selectedPair);
      fetchTicker(exchange, selectedPair);
    });
    
    // Fetch arbitrage opportunities
    fetchArbitrageOpportunities();
    
    setIsLoading(false);
  }, [
    selectedExchanges, 
    selectedPair, 
    timeframe, 
    fetchOHLCV, 
    fetchOrderBook, 
    fetchTicker, 
    fetchArbitrageOpportunities
  ]);

  // Initial data load
  useEffect(() => {
    const loadInitialData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        await fetchExchanges();
        setIsLoading(false);
      } catch (err) {
        console.error('Error during initial data load:', err);
        setError('Failed to load initial data. Please refresh the page.');
        setIsLoading(false);
      }
    };
    
    loadInitialData();
  }, [fetchExchanges]);

  // Fetch trading pairs when selected exchanges change
  useEffect(() => {
    if (selectedExchanges.length === 0) return;
    
    selectedExchanges.forEach(exchange => {
      fetchTradingPairs(exchange);
    });
  }, [selectedExchanges, fetchTradingPairs]);

  // Refresh data when selected exchange, pair, or timeframe changes
  useEffect(() => {
    if (selectedExchanges.length === 0 || !selectedPair) return;
    
    refreshData();
    
    // Set up intervals for data refresh
    const ohlcvInterval = setInterval(() => {
      selectedExchanges.forEach(exchange => {
        fetchOHLCV(exchange, selectedPair, timeframe);
      });
    }, refreshIntervals.ohlcv);
    
    const orderbookInterval = setInterval(() => {
      selectedExchanges.forEach(exchange => {
        fetchOrderBook(exchange, selectedPair);
      });
    }, refreshIntervals.orderbooks);
    
    const tickerInterval = setInterval(() => {
      selectedExchanges.forEach(exchange => {
        fetchTicker(exchange, selectedPair);
      });
    }, refreshIntervals.tickers);
    
    const arbitrageInterval = setInterval(() => {
      fetchArbitrageOpportunities();
    }, refreshIntervals.arbitrageOpportunities);
    
    // Clean up intervals on unmount or when dependencies change
    return () => {
      clearInterval(ohlcvInterval);
      clearInterval(orderbookInterval);
      clearInterval(tickerInterval);
      clearInterval(arbitrageInterval);
    };
  }, [
    selectedExchanges, 
    selectedPair, 
    timeframe, 
    refreshData, 
    fetchOHLCV, 
    fetchOrderBook, 
    fetchTicker, 
    fetchArbitrageOpportunities
  ]);

  // Create value object for provider
  const contextValue = {
    // Exchange data
    exchanges,
    selectedExchanges,
    setSelectedExchanges,
    
    // Trading pair data
    tradingPairs,
    selectedPair,
    setSelectedPair,
    
    // Timeframe
    timeframe,
    setTimeframe,
    
    // Market data
    marketData,
    
    // Loading and error state
    isLoading,
    error,
    
    // Functions
    refreshData,
    
    // Advanced functions for arbitrage operations
    executeArbitrage: (opportunityId) => {
      console.log(`Executing arbitrage opportunity ${opportunityId}`);
      // Implementation would connect to the backend to execute the trade
      alert(`Arbitrage execution initiated for opportunity ${opportunityId}`);
      return true;
    },
    
    simulateArbitrage: (opportunityId) => {
      console.log(`Simulating arbitrage opportunity ${opportunityId}`);
      // Implementation would connect to the backend to simulate the trade
      return {
        success: true,
        estimatedProfit: 0.05,
        estimatedProfitUsd: 325.75,
        estimatedDuration: 12000, // ms
        estimatedGasCost: 0.01,
        estimatedSuccessRate: 0.95
      };
    }
  };

  return (
    <MarketDataContext.Provider value={contextValue}>
      {children}
    </MarketDataContext.Provider>
  );
};

/**
 * Custom hook to use the market data context
 * @returns {Object} Market data context
 */
export const useMarketData = () => {
  const context = useContext(MarketDataContext);
  
  if (!context) {
    throw new Error('useMarketData must be used within a MarketDataProvider');
  }
  
  return context;
};

