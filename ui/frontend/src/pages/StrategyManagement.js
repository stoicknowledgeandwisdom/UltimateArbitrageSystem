import React, { useState, useEffect } from 'react';

// Components
import StrategyList from '../components/strategies/StrategyList';
import StrategyForm from '../components/strategies/StrategyForm';
import StrategyPerformance from '../components/strategies/StrategyPerformance';
import StrategyScheduler from '../components/strategies/StrategyScheduler';

// Services
import { strategyService } from '../services/strategyService';

const StrategyManagement = () => {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        setIsLoading(true);
        const data = await strategyService.getStrategies();
        setStrategies(data);
        if (data.length > 0) {
          setSelectedStrategy(data[0]);
        }
      } catch (err) {
        setError('Failed to load strategies. Please try again later.');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStrategies();
  }, []);

  const handleStrategySelect = (strategy) => {
    setSelectedStrategy(strategy);
    setIsEditing(false);
  };

  const handleCreateNew = () => {
    setSelectedStrategy({
      id: null,
      name: '',
      type: 'triangular_arbitrage',
      active: false,
      parameters: {},
      exchanges: [],
      tradingPairs: [],
      riskParameters: {
        maxPositionSize: 0,
        stopLoss: 0,
        takeProfit: 0
      },
      schedule: {
        enabled: false,
        startTime: null,
        endTime: null,
        days: []
      }
    });
    setIsEditing(true);
  };

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleSave = async (strategy) => {
    try {
      setIsLoading(true);
      let savedStrategy;
      
      if (strategy.id) {
        // Update existing strategy
        savedStrategy = await strategyService.updateStrategy(strategy.id, strategy);
        
        // Update strategies list
        setStrategies(strategies.map(s => 
          s.id === savedStrategy.id ? savedStrategy : s
        ));
      } else {
        // Create new strategy
        savedStrategy = await strategyService.createStrategy(strategy);
        
        // Add to strategies list
        setStrategies([...strategies, savedStrategy]);
      }
      
      setSelectedStrategy(savedStrategy);
      setIsEditing(false);
    } catch (err) {
      setError('Failed to save strategy. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (strategyId) => {
    if (window.confirm('Are you sure you want to delete this strategy?')) {
      try {
        setIsLoading(true);
        await strategyService.deleteStrategy(strategyId);
        
        // Remove from strategies list
        const updatedStrategies = strategies.filter(s => s.id !== strategyId);
        setStrategies(updatedStrategies);
        
        // Select the first strategy or clear selection if none left
        if (updatedStrategies.length > 0) {
          setSelectedStrategy(updatedStrategies[0]);
        } else {
          setSelectedStrategy(null);
        }
        
        setIsEditing(false);
      } catch (err) {
        setError('Failed to delete strategy

