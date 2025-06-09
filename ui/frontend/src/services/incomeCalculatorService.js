/**
 * Income Calculator Service
 * 
 * This service provides API integration for the automated income calculator,
 * handling all communication with the backend for earnings calculations,
 * real-time data, and investment recommendations.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const INCOME_API_BASE = `${API_BASE_URL}/api/income`;

/**
 * Generic API request handler with error handling and retry logic
 */
const apiRequest = async (url, options = {}) => {
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, defaultOptions);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API Request Error:', error);
    throw error;
  }
};

/**
 * Calculate earnings projection for a given investment amount
 * 
 * @param {number} amount - Investment amount in USD
 * @returns {Promise<Object>} Earnings projection data
 */
export const calculateEarningsProjection = async (amount) => {
  if (!amount || amount < 100) {
    throw new Error('Investment amount must be at least $100');
  }
  
  const url = `${INCOME_API_BASE}/projection/${amount}`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Get real-time earnings data
 * 
 * @returns {Promise<Object>} Real-time earnings data
 */
export const getRealTimeEarnings = async () => {
  const url = `${INCOME_API_BASE}/realtime`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Get investment recommendations based on available capital
 * 
 * @param {number} capital - Available capital in USD
 * @returns {Promise<Array>} List of investment recommendations
 */
export const getInvestmentRecommendations = async (capital) => {
  if (!capital || capital < 100) {
    throw new Error('Capital amount must be at least $100');
  }
  
  const url = `${INCOME_API_BASE}/recommendations/${capital}`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Get competitor comparison data
 * 
 * @returns {Promise<Object>} Competitor analysis data
 */
export const getCompetitorComparison = async () => {
  const url = `${INCOME_API_BASE}/comparison`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Get current automation level
 * 
 * @returns {Promise<Object>} Automation level data
 */
export const getAutomationLevel = async () => {
  const url = `${INCOME_API_BASE}/automation-level`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Optimize investment allocation
 * 
 * @param {Object} params - Optimization parameters
 * @param {number} params.capital - Available capital
 * @param {string} params.riskTolerance - Risk tolerance ('low', 'medium', 'high')
 * @param {number} params.targetAutomation - Target automation percentage
 * @returns {Promise<Object>} Optimization results
 */
export const optimizeInvestment = async (params) => {
  const { capital, riskTolerance = 'medium', targetAutomation = 90 } = params;
  
  if (!capital || capital < 100) {
    throw new Error('Capital amount must be at least $100');
  }
  
  const url = `${INCOME_API_BASE}/optimize`;
  const response = await apiRequest(url, {
    method: 'POST',
    body: JSON.stringify({
      capital,
      risk_tolerance: riskTolerance,
      target_automation: targetAutomation,
    }),
  });
  
  return response.data;
};

/**
 * Get performance history
 * 
 * @param {number} days - Number of days of history to retrieve (default: 30)
 * @returns {Promise<Array>} Performance history data
 */
export const getPerformanceHistory = async (days = 30) => {
  const url = `${INCOME_API_BASE}/performance-history?days=${days}`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Get calculator system status
 * 
 * @returns {Promise<Object>} System status data
 */
export const getCalculatorStatus = async () => {
  const url = `${INCOME_API_BASE}/status`;
  const response = await apiRequest(url);
  
  return response.data;
};

/**
 * Health check for the income calculator service
 * 
 * @returns {Promise<Object>} Health status
 */
export const healthCheck = async () => {
  const url = `${INCOME_API_BASE}/health`;
  const response = await apiRequest(url);
  
  return response;
};

/**
 * Real-time data subscription manager
 */
class RealTimeDataManager {
  constructor() {
    this.subscribers = new Set();
    this.isRunning = false;
    this.interval = null;
    this.updateInterval = 2000; // 2 seconds
  }
  
  /**
   * Subscribe to real-time data updates
   * 
   * @param {Function} callback - Callback function to receive updates
   * @returns {Function} Unsubscribe function
   */
  subscribe(callback) {
    this.subscribers.add(callback);
    
    // Start polling if this is the first subscriber
    if (this.subscribers.size === 1 && !this.isRunning) {
      this.start();
    }
    
    // Return unsubscribe function
    return () => {
      this.subscribers.delete(callback);
      
      // Stop polling if no more subscribers
      if (this.subscribers.size === 0) {
        this.stop();
      }
    };
  }
  
  /**
   * Start real-time data polling
   */
  start() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.interval = setInterval(async () => {
      try {
        const data = await getRealTimeEarnings();
        this.notifySubscribers(data);
      } catch (error) {
        console.error('Real-time data update error:', error);
        this.notifySubscribers(null, error);
      }
    }, this.updateInterval);
    
    console.log('Real-time data manager started');
  }
  
  /**
   * Stop real-time data polling
   */
  stop() {
    if (!this.isRunning) return;
    
    this.isRunning = false;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    
    console.log('Real-time data manager stopped');
  }
  
  /**
   * Notify all subscribers of data updates
   */
  notifySubscribers(data, error = null) {
    this.subscribers.forEach(callback => {
      try {
        callback(data, error);
      } catch (err) {
        console.error('Subscriber callback error:', err);
      }
    });
  }
  
  /**
   * Set update interval
   * 
   * @param {number} interval - Update interval in milliseconds
   */
  setUpdateInterval(interval) {
    this.updateInterval = interval;
    
    if (this.isRunning) {
      this.stop();
      this.start();
    }
  }
}

// Create singleton instance for real-time data management
export const realTimeDataManager = new RealTimeDataManager();

/**
 * Hook for React components to subscribe to real-time data
 * 
 * @param {boolean} enabled - Whether to enable real-time updates
 * @returns {Object} { data, error, isLoading }
 */
export const useRealTimeEarnings = (enabled = true) => {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [isLoading, setIsLoading] = React.useState(true);
  
  React.useEffect(() => {
    if (!enabled) {
      setIsLoading(false);
      return;
    }
    
    const unsubscribe = realTimeDataManager.subscribe((newData, newError) => {
      setData(newData);
      setError(newError);
      setIsLoading(false);
    });
    
    // Initial data fetch
    getRealTimeEarnings()
      .then(initialData => {
        setData(initialData);
        setError(null);
      })
      .catch(initialError => {
        setError(initialError);
      })
      .finally(() => {
        setIsLoading(false);
      });
    
    return unsubscribe;
  }, [enabled]);
  
  return { data, error, isLoading };
};

/**
 * Utility functions for data formatting and validation
 */
export const utils = {
  /**
   * Format currency value
   * 
   * @param {number} amount - Amount to format
   * @param {string} currency - Currency code (default: 'USD')
   * @returns {string} Formatted currency string
   */
  formatCurrency: (amount, currency = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  },
  
  /**
   * Format percentage value
   * 
   * @param {number} value - Percentage value
   * @param {number} decimals - Number of decimal places
   * @returns {string} Formatted percentage string
   */
  formatPercentage: (value, decimals = 2) => {
    return `${value.toFixed(decimals)}%`;
  },
  
  /**
   * Validate investment amount
   * 
   * @param {number} amount - Amount to validate
   * @returns {Object} Validation result { isValid, error }
   */
  validateInvestmentAmount: (amount) => {
    if (!amount || isNaN(amount)) {
      return { isValid: false, error: 'Invalid amount' };
    }
    
    if (amount < 100) {
      return { isValid: false, error: 'Minimum investment is $100' };
    }
    
    if (amount > 10000000) {
      return { isValid: false, error: 'Maximum investment is $10,000,000' };
    }
    
    return { isValid: true, error: null };
  },
  
  /**
   * Calculate compound interest
   * 
   * @param {number} principal - Initial investment
   * @param {number} rate - Daily interest rate (as decimal)
   * @param {number} days - Number of days
   * @returns {number} Final amount after compound interest
   */
  calculateCompoundInterest: (principal, rate, days) => {
    return principal * Math.pow(1 + rate, days);
  },
  
  /**
   * Get investment tier based on amount
   * 
   * @param {number} amount - Investment amount
   * @returns {string} Investment tier
   */
  getInvestmentTier: (amount) => {
    if (amount >= 100000) return 'enterprise';
    if (amount >= 10000) return 'professional';
    if (amount >= 1000) return 'growth';
    return 'starter';
  },
  
  /**
   * Get tier color
   * 
   * @param {string} tier - Investment tier
   * @returns {string} Color code
   */
  getTierColor: (tier) => {
    const colors = {
      starter: '#4caf50',
      growth: '#2196f3',
      professional: '#ff9800',
      enterprise: '#9c27b0',
    };
    return colors[tier] || '#666';
  },
};

export default {
  calculateEarningsProjection,
  getRealTimeEarnings,
  getInvestmentRecommendations,
  getCompetitorComparison,
  getAutomationLevel,
  optimizeInvestment,
  getPerformanceHistory,
  getCalculatorStatus,
  healthCheck,
  realTimeDataManager,
  useRealTimeEarnings,
  utils,
};

