/**
 * Market Data Service
 * Provides functions for fetching various types of market data from cryptocurrency exchanges,
 * with support for both live API connections and a mock/demo mode for development.
 */

import axios from 'axios';

// Configuration for API endpoints and keys
const config = {
  // Base URL for the backend API
  apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api',
  
  // API keys for direct exchange connections (if applicable)
  apiKeys: {
    binance: process.env.REACT_APP_BINANCE_API_KEY,
    coinbase: process.env.REACT_APP_COINBASE_API_KEY,
    kraken: process.env.REACT_APP_KRAKEN_API_KEY,
    kucoin: process.env.REACT_APP_KUCOIN_API_KEY,
    // Add more exchanges as needed
  },
  
  // Enable mock data for development/testing
  useMockData: process.env.REACT_APP_USE_MOCK_DATA === 'true' || true,
  
  // Request timeout in milliseconds
  timeout: 10000,
  
  // Retry configuration
  maxRetries: 3,
  retryDelay: 1000,
};

// Mock data for development and testing
const mockData = {
  exchanges: [
    { id: 'binance', name: 'Binance', status: 'online', volume24h: 12453678901 },
    { id: 'coinbase', name: 'Coinbase', status: 'online', volume24h: 8765432109 },
    { id: 'kraken', name: 'Kraken', status: 'online', volume24h: 5678901234 },
    { id: 'kucoin', name: 'KuCoin', status: 'online', volume24h: 3456789012 }
  ],
  
  tradingPairs: {
    binance: [
      { symbol: 'BTC/USDT', base: 'BTC', quote: 'USDT', volume24h: 1234567890 },
      { symbol: 'ETH/USDT', base: 'ETH', quote: 'USDT', volume24h: 987654321 },
      { symbol: 'BNB/USDT', base: 'BNB', quote: 'USDT', volume24h: 456789012 },
      { symbol: 'SOL/USDT', base: 'SOL', quote: 'USDT', volume24h: 345678901 },
      { symbol: 'ETH/BTC', base: 'ETH', quote: 'BTC', volume24h: 234567890 }
    ],
    coinbase: [
      { symbol: 'BTC/USD', base: 'BTC', quote: 'USD', volume24h: 876543210 },
      { symbol: 'ETH/USD', base: 'ETH', quote: 'USD', volume24h: 765432109 },
      { symbol: 'SOL/USD', base: 'SOL', quote: 'USD', volume24h: 321098765 },
      { symbol: 'ETH/BTC', base: 'ETH', quote: 'BTC', volume24h: 210987654 }
    ],
    kraken: [
      { symbol: 'XBT/USD', base: 'XBT', quote: 'USD', volume24h: 654321098 },
      { symbol: 'ETH/USD', base: 'ETH', quote: 'USD', volume24h: 543210987 },
      { symbol: 'SOL/USD', base: 'SOL', quote: 'USD', volume24h: 432109876 },
      { symbol: 'ETH/XBT', base: 'ETH', quote: 'XBT', volume24h: 321098765 }
    ],
    kucoin: [
      { symbol: 'BTC/USDT', base: 'BTC', quote: 'USDT', volume24h: 210987654 },
      { symbol: 'ETH/USDT', base: 'ETH', quote: 'USDT', volume24h: 109876543 },
      { symbol: 'KCS/USDT', base: 'KCS', quote: 'USDT', volume24h: 98765432 },
      { symbol: 'ETH/BTC', base: 'ETH', quote: 'BTC', volume24h: 87654321 }
    ]
  },
  
  // Generate mock OHLCV data with a trend
  generateOHLCV: (pair, timeframe, count = 100) => {
    const now = new Date();
    const data = [];
    let lastClose = pair.includes('BTC') ? 50000 + Math.random() * 5000 : 2000 + Math.random() * 500;
    
    for (let i = count; i > 0; i--) {
      const timestamp = new Date(now);
      timestamp.setMinutes(now.getMinutes() - i * getTimeframeMinutes(timeframe));
      
      const open = lastClose;
      const high = open * (1 + Math.random() * 0.02);
      const low = open * (1 - Math.random() * 0.02);
      const close = (high + low) / 2 + (Math.random() - 0.5) * (high - low);
      const volume = Math.random() * 1000 * (pair.includes('BTC') ? 10 : 1);
      
      data.push({
        timestamp: timestamp.getTime(),
        open,
        high,
        low,
        close,
        volume
      });
      
      lastClose = close;
    }
    
    return data;
  },
  
  // Generate mock order book data
  generateOrderBook: (pair) => {
    const basePrice = pair.includes('BTC') ? 50000 + Math.random() * 5000 : 2000 + Math.random() * 500;
    const asks = [];
    const bids = [];
    
    // Generate asks (sell orders)
    for (let i = 0; i < 20; i++) {
      const price = basePrice * (1 + 0.0001 * i + Math.random() * 0.0001);
      const volume = Math.random() * 2 + 0.1;
      asks.push([price, volume]);
    }
    
    // Generate bids (buy orders)
    for (let i = 0; i < 20; i++) {
      const price = basePrice * (1 - 0.0001 * i - Math.random() * 0.0001);
      const volume = Math.random() * 2 + 0.1;
      bids.push([price, volume]);
    }
    
    return { asks, bids, timestamp: Date.now() };
  },
  
  // Generate mock ticker data
  generateTicker: (pair) => {
    const basePrice = pair.includes('BTC') ? 50000 + Math.random() * 5000 : 2000 + Math.random() * 500;
    
    return {
      symbol: pair,
      last: basePrice,
      bid: basePrice * 0.999,
      ask: basePrice * 1.001,
      high: basePrice * 1.02,
      low: basePrice * 0.98,
      volume: Math.random() * 1000,
      timestamp: Date.now()
    };
  },
  
  // Generate mock arbitrage opportunities
  generateArbitrageOpportunities: () => {
    const opportunities = [];
    
    // Simple cross-exchange opportunities
    opportunities.push({
      id: 'arb-1',
      type: 'cross-exchange',
      asset: 'ETH/BTC',
      buyExchange: 'binance',
      sellExchange: 'kraken',
      buyPrice: 0.0675 - Math.random() * 0.0005,
      sellPrice: 0.0675 + Math.random() * 0.0005,
      spread: 0.23 + Math.random() * 0.5,
      volume: 5 + Math.random() * 10,
      profit: 0.01 + Math.random() * 0.02,
      profitUsd: 250 + Math.random() * 500,
      timestamp: Date.now(),
      estimatedFees: 0.1 + Math.random() * 0.1,
      confidence: 85 + Math.random() * 10
    });
    
    opportunities.push({
      id: 'arb-2',
      type: 'cross-exchange',
      asset: 'SOL/USDT',
      buyExchange: 'kucoin',
      sellExchange: 'binance',
      buyPrice: 110 - Math.random() * 0.5,
      sellPrice: 110 + Math.random() * 0.5,
      spread: 0.45 + Math.random() * 0.3,
      volume: 100 + Math.random() * 200,
      profit: 0.25 + Math.random() * 0.5,
      profitUsd: 320 + Math.random() * 300,
      timestamp: Date.now(),
      estimatedFees: 0.15 + Math.random() * 0.1,
      confidence: 90 + Math.random() * 8
    });
    
    // Triangular arbitrage
    opportunities.push({
      id: 'arb-3',
      type: 'triangular',
      exchange: 'binance',
      path: ['BTC', 'ETH', 'USDT', 'BTC'],
      rates: [0.0675, 2800, 50000],
      spread: 0.42 + Math.random() * 0.3,
      profit: 0.12 + Math.random() * 0.2,
      profitUsd: 180 + Math.random() * 200,
      timestamp: Date.now(),
      estimatedFees: 0.08 + Math.random() * 0.1,
      confidence: 88 + Math.random() * 7
    });
    
    return opportunities;
  }
};

/**
 * Helper function to convert timeframe to minutes
 * @param {string} timeframe - Timeframe string (e.g., '1m', '1h', '1d')
 * @returns {number} - Minutes
 */
const getTimeframeMinutes = (timeframe) => {
  const unit = timeframe.slice(-1);
  const value = parseInt(timeframe.slice(0, -1));
  
  switch (unit) {
    case 'm': return value;
    case 'h': return value * 60;
    case 'd': return value * 24 * 60;
    default: return 1;
  }
};

/**
 * Creates an API client with retry logic
 * @returns {Object} - Axios instance
 */
const createApiClient = () => {
  const client = axios.create({
    baseURL: config.apiBaseUrl,
    timeout: config.timeout,
    headers: {
      'Content-Type': 'application/json'
    }
  });
  
  // Add response interceptor for retry logic
  client.interceptors.response.use(null, async (error) => {
    const { config } = error;
    
    // Skip if retry option not set or we've run out of retries
    if (!config || !config.retry || config.retryCount >= config.retry) {
      return Promise.reject(error);
    }
    
    // Set retryCount
    config.retryCount = config.retryCount || 0;
    config.retryCount += 1;
    
    // Calculate delay
    const delay = config.retryDelay || 1000;
    
    // Wait for the delay
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // Return the promise with the retry
    return client(config);
  });
  
  return client;
};

// Initialize API client
const apiClient = createApiClient();

/**
 * Get a list of available exchanges
 * @returns {Promise<Array>} Array of exchange objects
 */
export const getExchanges = async () => {
  try {
    if (config.useMockData) {
      return mockData.exchanges;
    }
    
    const response = await apiClient.get('/exchanges', {
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching exchanges:', error);
    throw new Error(`Failed to fetch exchanges: ${error.message}`);
  }
};

/**
 * Get trading pairs for a specific exchange
 * @param {string} exchange - Exchange ID
 * @returns {Promise<Array>} Array of trading pair objects
 */
export const getTradingPairs = async (exchange) => {
  try {
    if (config.useMockData) {
      return mockData.tradingPairs[exchange] || [];
    }
    
    const response = await apiClient.get(`/exchanges/${exchange}/pairs`, {
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error fetching trading pairs for ${exchange}:`, error);
    throw new Error(`Failed to fetch trading pairs: ${error.message}`);
  }
};

/**
 * Get OHLCV (candlestick) data for a specific exchange and trading pair
 * @param {string} exchange - Exchange ID
 * @param {string} pair - Trading pair symbol
 * @param {string} timeframe - Time frame for candlesticks (e.g., '1m', '5m', '1h', '1d')
 * @param {number} limit - Number of candles to fetch
 * @returns {Promise<Array>} Array of OHLCV data
 */
export const getOHLCV = async (exchange, pair, timeframe, limit = 100) => {
  try {
    if (config.useMockData) {
      return mockData.generateOHLCV(pair, timeframe, limit);
    }
    
    const response = await apiClient.get(`/exchanges/${exchange}/ohlcv`, {
      params: { pair, timeframe, limit },
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error fetching OHLCV data for ${pair} on ${exchange}:`, error);
    throw new Error(`Failed to fetch OHLCV data: ${error.message}`);
  }
};

/**
 * Get order book data for a specific exchange and trading pair
 * @param {string} exchange - Exchange ID
 * @param {string} pair - Trading pair symbol
 * @returns {Promise<Object>} Order book data with bids and asks
 */
export const getOrderBook = async (exchange, pair) => {
  try {
    if (config.useMockData) {
      return mockData.generateOrderBook(pair);
    }
    
    const response = await apiClient.get(`/exchanges/${exchange}/orderbook`, {
      params: { pair },
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error fetching order book for ${pair} on ${exchange}:`, error);
    throw new Error(`Failed to fetch order book data: ${error.message}`);
  }
};

/**
 * Get ticker data for a specific exchange and trading pair
 * @param {string} exchange - Exchange ID
 * @param {string} pair - Trading pair symbol
 * @returns {Promise<Object>} Ticker data
 */
export const getTicker = async (exchange, pair) => {
  try {
    if (config.useMockData) {
      return mockData.generateTicker(pair);
    }
    
    const response = await apiClient.get(`/exchanges/${exchange}/ticker`, {
      params: { pair },
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error fetching ticker for ${pair} on ${exchange}:`, error);
    throw new Error(`Failed to fetch ticker data: ${error.message}`);
  }
};

/**
 * Get arbitrage opportunities across exchanges
 * @param {Object} options - Filter options for arbitrage opportunities
 * @param {string[]} [options.exchanges] - Limit to specific exchanges
 * @param {string[]} [options.pairs] - Limit to specific trading pairs
 * @param {number} [options.minProfit] - Minimum profit percentage
 * @param {number} [options.minVolume] - Minimum volume
 * @returns {Promise<Array>} Array of arbitrage opportunities
 */
export const getArbitrageOpportunities = async (options = {}) => {
  try {
    if (config.useMockData) {
      let opportunities = mockData.generateArbitrageOpportunities();
      
      // Apply filters if provided in options
      if (options.exchanges && options.exchanges.length > 0) {
        opportunities = opportunities.filter(opp => 
          options.exchanges.includes(opp.exchange) || 
          (opp.buyExchange && options.exchanges.includes(opp.buyExchange)) ||
          (opp.sellExchange && options.exchanges.includes(opp.sellExchange))
        );
      }
      
      if (options.pairs && options.pairs.length > 0) {
        opportunities = opportunities.filter(opp => 
          options.pairs.some(pair => opp.asset && opp.asset.includes(pair))
        );
      }
      
      if (options.minProfit) {
        opportunities = opportunities.filter(opp => opp.profit >= options.minProfit);
      }
      
      if (options.minVolume) {
        opportunities = opportunities.filter(opp => opp.volume >= options.minVolume);
      }
      
      return opportunities;
    }
    
    const response = await apiClient.get('/arbitrage/opportunities', {
      params: options,
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching arbitrage opportunities:', error);
    throw new Error(`Failed to fetch arbitrage opportunities: ${error.message}`);
  }
};

/**
 * Get detailed information about a specific arbitrage opportunity
 * @param {string} opportunityId - Arbitrage opportunity ID
 * @returns {Promise<Object>} Detailed information about the opportunity
 */
export const getArbitrageOpportunityDetails = async (opportunityId) => {
  try {
    if (config.useMockData) {
      // For mock data, generate a more detailed version of the basic opportunity
      const opportunities = mockData.generateArbitrageOpportunities();
      const opportunity = opportunities.find(opp => opp.id === opportunityId);
      
      if (!opportunity) {
        throw new Error(`Opportunity with ID ${opportunityId} not found`);
      }
      
      // Add additional details to the basic opportunity
      return {
        ...opportunity,
        detailedPath: opportunity.type === 'triangular' 
          ? opportunity.path.map((currency, index, arr) => {
              if (index === arr.length - 1) return null;
              return {
                from: currency,
                to: arr[(index + 1) % arr.length],
                rate: opportunity.rates[index] || 0,
                exchange: opportunity.exchange
              };
            }).filter(Boolean)
          : [
              { 
                from: opportunity.asset.split('/')[0], 
                to: opportunity.asset.split('/')[1], 
                rate: opportunity.buyPrice, 
                exchange: opportunity.buyExchange 
              },
              { 
                from: opportunity.asset.split('/')[1], 
                to: opportunity.asset.split('/')[0], 
                rate: opportunity.sellPrice, 
                exchange: opportunity.sellExchange 
              }
            ],
        executionSteps: opportunity.type === 'triangular'
          ? opportunity.path.map((currency, index, arr) => {
              if (index === arr.length - 1) return null;
              return {
                step: index + 1,
                action: index === 0 ? 'BUY' : (index === arr.length - 2 ? 'SELL' : 'SWAP'),
                from: currency,
                to: arr[(index + 1) % arr.length],
                rate: opportunity.rates[index] || 0,
                exchange: opportunity.exchange
              };
            }).filter(Boolean)
          : [
              { 
                step: 1, 
                action: 'BUY', 
                from: opportunity.asset.split('/')[1], 
                to: opportunity.asset.split('/')[0], 
                rate: opportunity.buyPrice, 
                exchange: opportunity.buyExchange 
              },
              { 
                step: 2, 
                action: 'SELL', 
                from: opportunity.asset.split('/')[0], 
                to: opportunity.asset.split('/')[1], 
                rate: opportunity.sellPrice, 
                exchange: opportunity.sellExchange 
              }
            ],
        simulationResults: {
          expectedProfit: opportunity.profit,
          expectedProfitUsd: opportunity.profitUsd,
          executionTime: Math.floor(Math.random() * 5000) + 1000, // 1-6 seconds
          gasEstimate: Math.random() * 0.05,
          slippageEstimate: Math.random() * 0.01,
          successProbability: 0.8 + Math.random() * 0.15
        },
        historicalPerformance: {
          lastExecuted: new Date(Date.now() - Math.floor(Math.random() * 86400000)).toISOString(),
          successRate: 0.7 + Math.random() * 0.25,
          averageProfit: opportunity.profit * (0.8 + Math.random() * 0.4),
          totalExecutions: Math.floor(Math.random() * 50)
        }
      };
    }
    
    const response = await apiClient.get(`/arbitrage/opportunities/${opportunityId}`, {
      retry: config.maxRetries,
      retryDelay: config.retryDelay
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error fetching details for arbitrage opportunity ${opportunityId}:`, error);
    throw new Error(`Failed to fetch arbitrage opportunity details: ${error.message}`);
  }
};
