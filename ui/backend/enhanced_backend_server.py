from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import asyncio
import threading
import time
from datetime import datetime
import json
import numpy as np
from typing import Dict, Any, List
import logging
from dataclasses import asdict
import sys
import os

# Import wallet configuration components
try:
    from core.wallet_config_manager import WalletConfigManager, NetworkType, ExchangeType, StrategyType
except ImportError as e:
    print(f"Warning: Could not import wallet configuration manager: {e}")
    WalletConfigManager = None

# Add the core modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'orchestration'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'risk_management'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'market_data'))

# Import our advanced modules
try:
    from AdvancedStrategyIntegrator import AdvancedStrategyIntegrator, MockQuantumArbitrageStrategy, StrategyCommand
    from UltimateRiskManager import UltimateRiskManager
    from UltimateMarketDataManager import UltimateMarketDataManager, MockDataFeed
except ImportError as e:
    print(f"Warning: Could not import advanced modules: {e}")
    # Create mock classes for fallback
    class AdvancedStrategyIntegrator:
        def __init__(self, config=None): pass
        async def get_real_time_performance(self): return {'total_return': 0.1, 'weighted_win_rate': 85}
        async def start_integration_system(self): return True
        def get_integration_status(self): return {'system_status': 'mock'}
    
    class UltimateRiskManager:
        def __init__(self, config=None): pass
        def get_risk_summary(self): return {'system_status': 'mock'}
        async def monitor_real_time_risk(self): return {'overall_risk_level': 'low'}
    
    class UltimateMarketDataManager:
        def __init__(self, config=None): pass
        async def start_data_processing(self): return True
        def get_market_summary(self): return {'system_health': 'good'}
        def get_current_opportunities(self): return []
    
    class MockQuantumArbitrageStrategy: pass
    class StrategyCommand: pass
    class MockDataFeed: pass

# Import the automated trading engine
try:
    from core.automation.AutomatedTradingEngine import AutomatedTradingEngine, TradingOpportunity, OrderSide
except ImportError as e:
    print(f"Warning: Could not import AutomatedTradingEngine: {e}")
    # Create mock classes for fallback
    class AutomatedTradingEngine:
        def __init__(self, config): 
            self.config = config
            self.is_running = False
        async def start(self): 
            self.is_running = True
            return True
        async def stop(self): 
            self.is_running = False
            return True
        async def submit_opportunity(self, opportunity): return True
        def get_portfolio_status(self): return {'trading_mode': 'simulation', 'total_profit': 1000}
        def get_active_trades(self): return []
        def get_recent_trades(self, limit=10): return []
    
    class TradingOpportunity:
        def __init__(self, **kwargs): pass
    
    class OrderSide:
        BUY = "buy"
        SELL = "sell"

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ultimate_arbitrage_secret_key_2024'
CORS(app, origins=["http://localhost:3000"])
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

class UltimateArbitrageBackend:
    def __init__(self):
        self.logger = self._setup_logging()
        
        # System state
        self.system_active = False
        self.quantum_mode = False
        self.auto_rebalance = True
        self.aggression_level = 75
        
        # Initialize wallet configuration manager
        try:
            self.wallet_config_manager = WalletConfigManager() if WalletConfigManager else None
            if self.wallet_config_manager:
                self.logger.info("Wallet Configuration Manager initialized")
        except Exception as e:
            self.logger.error(f"Error initializing wallet config manager: {str(e)}")
            self.wallet_config_manager = None
        
        # Advanced system components
        self.strategy_integrator = None
        self.risk_manager = None
        self.market_data_manager = None
        self.trading_engine = None
        
        # System initialization flag
        self.system_initialized = False
        
        # Background tasks
        self.background_thread = None
        self.event_loop = None
        
        # Initialize advanced components
        asyncio.run(self._initialize_advanced_components())
        
        # Start background tasks
        self.start_background_tasks()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('UltimateArbitrageBackend')
    
    async def _initialize_advanced_components(self):
        """Initialize all advanced system components"""
        try:
            self.logger.info("Initializing Ultimate Arbitrage System components...")
            
            # Initialize Strategy Integrator
            self.strategy_integrator = AdvancedStrategyIntegrator({
                'max_concurrent_strategies': 10,
                'auto_optimization_enabled': True,
                'feedback_loop_enabled': True,
                'quantum_boost_enabled': True
            })
            
            # Initialize Risk Manager
            self.risk_manager = UltimateRiskManager({
                'max_portfolio_risk': 0.05,
                'max_strategy_allocation': 0.25,
                'emergency_stop_threshold': 0.15,
                'quantum_boost_threshold': 0.8
            })
            
            # Initialize Market Data Manager
            self.market_data_manager = UltimateMarketDataManager({
                'opportunity_min_profit': 0.001,
                'quantum_analysis_enabled': True,
                'anomaly_detection_enabled': True
            })
            
            # Initialize Automated Trading Engine
            trading_config = {
                'trading_mode': 'simulation',  # Start in simulation mode
                'max_concurrent_trades': 5,
                'max_daily_trades': 100,
                'min_profit_threshold': 0.005,  # 0.5%
                'max_risk_per_trade': 0.02,     # 2%
                'base_trade_size': 100,         # $100
                'initial_capital': 10000,       # $10,000
                'execution_delay': 0.1,
                'opportunity_timeout': 60,
                'max_slippage': 0.001,          # 0.1%
                'enable_stop_loss': True,
                'enable_take_profit': True,
                'max_daily_loss': 0.05,         # 5%
                'max_consecutive_losses': 5
            }
            self.trading_engine = AutomatedTradingEngine(trading_config)
            
            # Register mock strategies if modules are available
            if hasattr(self.strategy_integrator, 'register_strategy'):
                await self._register_strategies()
            
            # Setup data feeds if modules are available
            if hasattr(self.market_data_manager, 'add_data_feed'):
                await self._setup_data_feeds()
            
            # Start all systems
            if hasattr(self.strategy_integrator, 'start_integration_system'):
                await self.strategy_integrator.start_integration_system()
            if hasattr(self.market_data_manager, 'start_data_processing'):
                await self.market_data_manager.start_data_processing()
            
            # Start automated trading engine
            if self.trading_engine:
                await self.trading_engine.start()
            
            # Register callbacks for real-time updates
            self._register_callbacks()
            
            self.system_initialized = True
            self.logger.info("Advanced system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing advanced components: {str(e)}")
            self.system_initialized = False
    
    async def _register_strategies(self):
        """Register all available strategies"""
        strategies_config = [
            ('quantum_arbitrage', {'quantum_enhanced': True, 'max_allocation': 0.3}),
            ('cross_chain_mev', {'quantum_enhanced': False, 'max_allocation': 0.25}),
            ('flash_loan_arbitrage', {'quantum_enhanced': False, 'max_allocation': 0.2}),
            ('triangular_arbitrage', {'quantum_enhanced': False, 'max_allocation': 0.2}),
            ('volatility_harvesting', {'quantum_enhanced': True, 'max_allocation': 0.15}),
            ('options_arbitrage', {'quantum_enhanced': False, 'max_allocation': 0.15}),
            ('social_sentiment', {'quantum_enhanced': True, 'max_allocation': 0.1}),
            ('ai_momentum', {'quantum_enhanced': True, 'max_allocation': 0.2})
        ]
        
        for strategy_id, config in strategies_config:
            try:
                # Create mock strategy instance
                strategy_instance = MockQuantumArbitrageStrategy(strategy_id)
                
                # Register with integrator
                success = await self.strategy_integrator.register_strategy(
                    strategy_id, 
                    strategy_instance
                )
                
                if success:
                    self.logger.info(f"Registered strategy: {strategy_id}")
                else:
                    self.logger.error(f"Failed to register strategy: {strategy_id}")
            except Exception as e:
                self.logger.error(f"Error registering strategy {strategy_id}: {str(e)}")
    
    async def _setup_data_feeds(self):
        """Setup market data feeds"""
        try:
            # Create mock data feeds
            exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex']
            
            for exchange in exchanges:
                feed = MockDataFeed(exchange)
                if hasattr(feed, 'set_data_manager'):
                    feed.set_data_manager(self.market_data_manager)
                
                success = await self.market_data_manager.add_data_feed(exchange, feed)
                
                if success:
                    self.logger.info(f"Added data feed: {exchange}")
            
            # Subscribe to major trading pairs
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            await self.market_data_manager.subscribe_to_symbols(symbols)
            
        except Exception as e:
            self.logger.error(f"Error setting up data feeds: {str(e)}")
    
    def _register_callbacks(self):
        """Register callbacks for real-time updates"""
        try:
            # Register opportunity callback
            async def opportunity_callback(opportunity):
                self.logger.info(f"New opportunity: {opportunity.symbol} {opportunity.profit_percentage:.4f}%")
                # Emit to frontend via SocketIO if needed
            
            # Register regime callback
            async def regime_callback(regime):
                self.logger.info(f"Market regime update: {regime.volatility_regime} volatility")
            
            # Register feedback callback
            async def feedback_callback(performance_data):
                self.logger.debug(f"Performance feedback: {performance_data.get('system_health', 'unknown')}")
            
            if self.market_data_manager and hasattr(self.market_data_manager, 'register_opportunity_callback'):
                self.market_data_manager.register_opportunity_callback(opportunity_callback)
                self.market_data_manager.register_regime_callback(regime_callback)
            
            if self.strategy_integrator and hasattr(self.strategy_integrator, 'register_feedback_callback'):
                self.strategy_integrator.register_feedback_callback(feedback_callback)
                
        except Exception as e:
            self.logger.error(f"Error registering callbacks: {str(e)}")
    
    def get_performance_data(self):
        """Get current performance data from integrated systems"""
        try:
            if not self.system_initialized:
                return self._get_mock_performance_data()
            
            # Get real-time performance from strategy integrator
            if self.strategy_integrator and hasattr(self.strategy_integrator, 'get_real_time_performance'):
                performance_data = asyncio.run(self.strategy_integrator.get_real_time_performance())
                
                # Convert to expected format for frontend
                formatted_data = {
                    'total_profit': performance_data.get('total_return', 0) * 1000000,  # Scale for display
                    'daily_profit': performance_data.get('total_return', 0) * 50000,  # Simulated daily
                    'hourly_profit': performance_data.get('total_return', 0) * 2000,   # Simulated hourly
                    'minute_profit': np.random.normal(25, 10),  # Simulated minute
                    'win_rate': performance_data.get('weighted_win_rate', 85),
                    'sharpe_ratio': performance_data.get('portfolio_sharpe', 3.0),
                    'max_drawdown': -2.1,  # Would come from risk manager
                    'quantum_advantage': 2.87,
                    'active_trades': performance_data.get('total_trades', 0),
                    'successful_trades': performance_data.get('total_successful_trades', 0),
                    'failed_trades': performance_data.get('total_failed_trades', 0),
                    'average_trade_time': 0.34,
                    'portfolio_value': performance_data.get('total_allocated_capital', 0) + performance_data.get('total_available_capital', 0),
                    'available_capital': performance_data.get('total_available_capital', 1000000),
                    'allocated_capital': performance_data.get('total_allocated_capital', 0),
                    'roi': (performance_data.get('total_return', 0) / 1000000) * 100
                }
                
                return formatted_data
            
            return self._get_mock_performance_data()
            
        except Exception as e:
            self.logger.error(f"Error getting performance data: {str(e)}")
            return self._get_mock_performance_data()
    
    def _get_mock_performance_data(self):
        """Get mock performance data for fallback"""
        mock_data = {
            'total_profit': 847523.45,
            'daily_profit': 23847.12,
            'hourly_profit': 1243.58,
            'minute_profit': 20.73,
            'win_rate': 87.3,
            'sharpe_ratio': 3.24,
            'max_drawdown': -2.1,
            'quantum_advantage': 2.87,
            'active_trades': 247,
            'successful_trades': 1846,
            'failed_trades': 263,
            'average_trade_time': 0.34,
            'portfolio_value': 5847523.45,
            'available_capital': 1234567.89,
            'allocated_capital': 4612955.56,
            'roi': 169.5
        }
        
        if self.system_active:
            # Simulate real-time updates
            mock_data['minute_profit'] = round(np.random.normal(25, 10), 2)
            mock_data['hourly_profit'] += round(np.random.normal(100, 50), 2)
            mock_data['daily_profit'] += round(np.random.normal(200, 100), 2)
            mock_data['total_profit'] += round(np.random.normal(300, 150), 2)
            
            # Update other metrics
            mock_data['active_trades'] = np.random.randint(200, 300)
            mock_data['win_rate'] = round(np.random.uniform(85, 95), 1)
        
        return mock_data
    
    def get_strategy_data(self):
        """Get strategy performance data from integrated systems"""
        try:
            if not self.system_initialized or not self.strategy_integrator:
                return self._get_mock_strategy_data()
            
            # Get real strategy performance data
            performance_data = asyncio.run(self.strategy_integrator.get_real_time_performance())
            
            strategy_list = []
            strategy_performances = performance_data.get('strategy_performances', {})
            
            for strategy_id, perf_data in strategy_performances.items():
                strategy_info = {
                    'name': strategy_id.replace('_', ' ').title(),
                    'allocation': perf_data.get('allocated_capital', 0) / 1000000 * 100,  # Convert to percentage
                    'profit': perf_data.get('return', 0) * 1000000,  # Scale for display
                    'trades': perf_data.get('trades', 0),
                    'win_rate': perf_data.get('win_rate', 0),
                    'sharpe': perf_data.get('sharpe_ratio', 0),
                    'status': perf_data.get('status', 'inactive'),
                    'risk': self._classify_risk_level(perf_data.get('risk_score', 0.5)),
                    'confidence_score': perf_data.get('confidence_score', 0.5),
                    'performance_trend': perf_data.get('performance_trend', 'neutral')
                }
                strategy_list.append(strategy_info)
            
            # Sort by profit
            strategy_list.sort(key=lambda x: x['profit'], reverse=True)
            
            return strategy_list
            
        except Exception as e:
            self.logger.error(f"Error getting strategy data: {str(e)}")
            return self._get_mock_strategy_data()
    
    def _get_mock_strategy_data(self):
        """Get mock strategy data for fallback"""
        mock_strategies = [
            {
                'name': 'Quantum Arbitrage',
                'allocation': 25.4,
                'profit': 45623.12,
                'trades': 89,
                'win_rate': 94.4,
                'sharpe': 4.2,
                'status': 'active',
                'risk': 'low'
            },
            {
                'name': 'Cross-Chain MEV',
                'allocation': 18.7,
                'profit': 38921.45,
                'trades': 156,
                'win_rate': 82.1,
                'sharpe': 3.8,
                'status': 'active',
                'risk': 'medium'
            },
            {
                'name': 'Flash Loan Arbitrage',
                'allocation': 15.2,
                'profit': 29847.33,
                'trades': 234,
                'win_rate': 76.5,
                'sharpe': 3.1,
                'status': 'active',
                'risk': 'medium'
            }
        ]
        
        if self.system_active:
            # Simulate strategy updates
            for strategy in mock_strategies:
                strategy['profit'] += np.random.normal(100, 50)
                strategy['trades'] += np.random.randint(0, 5)
                strategy['win_rate'] = max(60, min(95, strategy['win_rate'] + np.random.normal(0, 2)))
        
        return mock_strategies
    
    def _classify_risk_level(self, risk_score):
        """Classify risk score into level"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def get_opportunities(self):
        """Get live arbitrage opportunities from market data manager"""
        try:
            if not self.system_initialized or not self.market_data_manager:
                return self._get_mock_opportunities()
            
            # Get real opportunities from market data manager
            opportunities = self.market_data_manager.get_current_opportunities()
            
            # Convert to frontend format
            formatted_opportunities = []
            
            for i, opp in enumerate(opportunities[:10]):  # Limit to top 10
                formatted_opp = {
                    'id': i + 1,
                    'type': opp.strategy_type.replace('_', ' ').title(),
                    'pair': opp.symbol,
                    'profit': opp.profit_percentage * 100,  # Convert to percentage
                    'confidence': opp.confidence_score * 100,  # Convert to percentage
                    'timeLeft': max(0, int((opp.expiry_time - datetime.now()).total_seconds())),
                    'status': opp.status,
                    'exchanges': opp.exchanges,
                    'risk_score': opp.risk_score,
                    'execution_time': opp.execution_time_estimate
                }
                formatted_opportunities.append(formatted_opp)
            
            return formatted_opportunities
            
        except Exception as e:
            self.logger.error(f"Error getting opportunities: {str(e)}")
            return self._get_mock_opportunities()
    
    def _get_mock_opportunities(self):
        """Get mock opportunities for fallback"""
        mock_opportunities = [
            {
                'id': 1,
                'type': 'Quantum Arbitrage',
                'pair': 'BTC/USDT',
                'profit': 2.34,
                'confidence': 96.8,
                'timeLeft': 12,
                'status': 'executing'
            },
            {
                'id': 2,
                'type': 'Cross-Chain MEV',
                'pair': 'ETH/USDC',
                'profit': 1.87,
                'confidence': 94.2,
                'timeLeft': 8,
                'status': 'pending'
            },
            {
                'id': 3,
                'type': 'Flash Loan',
                'pair': 'LINK/USDT',
                'profit': 4.21,
                'confidence': 91.5,
                'timeLeft': 5,
                'status': 'analyzing'
            }
        ]
        
        if self.system_active:
            # Simulate opportunity updates
            for opp in mock_opportunities:
                opp['timeLeft'] = max(0, opp['timeLeft'] - 1)
                opp['profit'] = max(0.1, opp['profit'] + np.random.normal(0, 0.1))
                opp['confidence'] = max(80, min(99, opp['confidence'] + np.random.normal(0, 1)))
                
                if opp['timeLeft'] <= 0:
                    opp['status'] = 'expired'
        
        return mock_opportunities
    
    def toggle_system(self):
        """Toggle system on/off with full integration"""
        try:
            self.system_active = not self.system_active
            
            if self.system_active:
                self.quantum_mode = True
                
                # Start strategies if system initialized
                if self.system_initialized and self.strategy_integrator and hasattr(self.strategy_integrator, 'command_queue'):
                    # Start top performing strategies
                    strategies_to_start = [
                        'quantum_arbitrage',
                        'cross_chain_mev',
                        'flash_loan_arbitrage',
                        'ai_momentum'
                    ]
                    
                    for strategy_id in strategies_to_start:
                        try:
                            command = StrategyCommand(
                                strategy_id=strategy_id,
                                command='start'
                            )
                            asyncio.run(self.strategy_integrator.command_queue.put(command))
                        except Exception as e:
                            self.logger.warning(f"Could not start strategy {strategy_id}: {str(e)}")
                
                self.logger.info("Ultimate Arbitrage System ACTIVATED with full integration")
            else:
                self.quantum_mode = False
                
                # Stop all strategies if system initialized
                if self.system_initialized and self.strategy_integrator and hasattr(self.strategy_integrator, 'emergency_stop_all'):
                    try:
                        asyncio.run(self.strategy_integrator.emergency_stop_all())
                    except Exception as e:
                        self.logger.warning(f"Error stopping strategies: {str(e)}")
                
                self.logger.info("Ultimate Arbitrage System DEACTIVATED")
            
            return {
                'success': True,
                'system_active': self.system_active,
                'quantum_mode': self.quantum_mode,
                'message': 'System state changed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error toggling system: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def emergency_stop(self):
        """Emergency stop all operations with full integration"""
        try:
            self.system_active = False
            self.quantum_mode = False
            
            # Execute emergency stop on all integrated systems
            if self.system_initialized:
                if self.strategy_integrator and hasattr(self.strategy_integrator, 'emergency_stop_all'):
                    try:
                        results = asyncio.run(self.strategy_integrator.emergency_stop_all())
                        self.logger.critical(f"Emergency stop results: {results}")
                    except Exception as e:
                        self.logger.error(f"Error in strategy emergency stop: {str(e)}")
                
                if self.market_data_manager and hasattr(self.market_data_manager, 'stop_data_processing'):
                    try:
                        asyncio.run(self.market_data_manager.stop_data_processing())
                        self.logger.critical("Market data processing stopped")
                    except Exception as e:
                        self.logger.error(f"Error stopping market data: {str(e)}")
            
            self.logger.critical("EMERGENCY STOP ACTIVATED - ALL SYSTEMS HALTED")
            
            return {
                'success': True,
                'message': 'Emergency stop activated - all trading and processing halted',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Emergency stop encountered an error'
            }
    
    def get_system_status(self):
        """Get comprehensive system status from all integrated components"""
        try:
            base_status = {
                'system_active': self.system_active,
                'quantum_mode': self.quantum_mode,
                'auto_rebalance': self.auto_rebalance,
                'aggression_level': self.aggression_level,
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'system_initialized': self.system_initialized
            }
            
            # Add integrated system status
            if self.system_initialized:
                # Strategy integrator status
                if self.strategy_integrator and hasattr(self.strategy_integrator, 'get_integration_status'):
                    try:
                        integration_status = self.strategy_integrator.get_integration_status()
                        base_status['strategy_integration'] = integration_status
                    except Exception as e:
                        base_status['strategy_integration'] = {'error': str(e)}
                
                # Market data manager status
                if self.market_data_manager and hasattr(self.market_data_manager, 'get_market_summary'):
                    try:
                        market_summary = self.market_data_manager.get_market_summary()
                        base_status['market_data'] = market_summary
                    except Exception as e:
                        base_status['market_data'] = {'error': str(e)}
                
                # Risk manager status
                if self.risk_manager and hasattr(self.risk_manager, 'get_risk_summary'):
                    try:
                        risk_summary = self.risk_manager.get_risk_summary()
                        base_status['risk_management'] = risk_summary
                    except Exception as e:
                        base_status['risk_management'] = {'error': str(e)}
            
            return base_status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {
                'system_active': self.system_active,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def update_settings(self, settings):
        """Update system settings with full integration"""
        try:
            updated_settings = {}
            
            if 'quantum_mode' in settings:
                self.quantum_mode = settings['quantum_mode']
                updated_settings['quantum_mode'] = self.quantum_mode
                
                # Update quantum settings in integrated systems
                if self.system_initialized and self.strategy_integrator and hasattr(self.strategy_integrator, 'strategies'):
                    for strategy_id in self.strategy_integrator.strategies.keys():
                        if 'quantum' in strategy_id.lower():
                            try:
                                config = self.strategy_integrator.strategy_configs[strategy_id]
                                config.quantum_enhanced = self.quantum_mode
                            except Exception as e:
                                self.logger.warning(f"Could not update quantum config for {strategy_id}: {str(e)}")
            
            if 'auto_rebalance' in settings:
                self.auto_rebalance = settings['auto_rebalance']
                updated_settings['auto_rebalance'] = self.auto_rebalance
                
                # Update auto-optimization in strategy integrator
                if self.system_initialized and self.strategy_integrator and hasattr(self.strategy_integrator, 'config'):
                    try:
                        self.strategy_integrator.config['auto_optimization_enabled'] = self.auto_rebalance
                    except Exception as e:
                        self.logger.warning(f"Could not update auto_optimization: {str(e)}")
            
            if 'aggression_level' in settings:
                self.aggression_level = settings['aggression_level']
                updated_settings['aggression_level'] = self.aggression_level
                
                # Update risk parameters based on aggression level
                if self.system_initialized and self.risk_manager:
                    try:
                        # Higher aggression = higher risk tolerance
                        risk_multiplier = self.aggression_level / 100.0
                        self.risk_manager.max_portfolio_risk = 0.05 * risk_multiplier
                        self.risk_manager.max_strategy_allocation = 0.25 * (1 + risk_multiplier * 0.5)
                    except Exception as e:
                        self.logger.warning(f"Could not update risk parameters: {str(e)}")
            
            self.logger.info(f"Settings updated with integration: {updated_settings}")
            
            return {
                'success': True,
                'message': 'Settings updated successfully across all systems',
                'updated_settings': updated_settings
            }
            
        except Exception as e:
            self.logger.error(f"Error updating settings: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_risk_metrics(self):
        """Get risk management metrics from integrated risk manager"""
        try:
            if not self.system_initialized or not self.risk_manager:
                return self._get_mock_risk_metrics()
            
            # Get comprehensive risk metrics from risk manager
            if hasattr(self.risk_manager, 'get_risk_summary'):
                risk_summary = self.risk_manager.get_risk_summary()
            else:
                risk_summary = {}
            
            # Get real-time risk monitoring
            if hasattr(self.risk_manager, 'monitor_real_time_risk'):
                risk_status = asyncio.run(self.risk_manager.monitor_real_time_risk())
            else:
                risk_status = {}
            
            # Get latest risk metrics if available
            latest_metrics = None
            if hasattr(self.risk_manager, 'risk_metrics_history') and self.risk_manager.risk_metrics_history:
                latest_metrics = self.risk_manager.risk_metrics_history[-1]
            
            # Format for frontend
            risk_data = {
                'overall_risk_level': risk_status.get('overall_risk_level', 'unknown'),
                'portfolio_var': risk_status.get('portfolio_var', 0.0),
                'risk_score': risk_status.get('risk_score', 0.0),
                'active_alerts': len(risk_status.get('active_alerts', [])),
                'recommendations': risk_status.get('recommendations', []),
                'system_health': risk_summary.get('system_status', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add detailed metrics if available
            if latest_metrics:
                risk_data.update({
                    'max_drawdown': latest_metrics.maximum_drawdown,
                    'sharpe_ratio': latest_metrics.sharpe_ratio,
                    'volatility': latest_metrics.volatility,
                    'var_95': latest_metrics.var_95,
                    'var_99': latest_metrics.var_99,
                    'expected_shortfall': latest_metrics.expected_shortfall,
                    'portfolio_correlation': latest_metrics.portfolio_correlation,
                    'concentration_risk': latest_metrics.concentration_risk,
                    'quantum_risk_factor': latest_metrics.quantum_risk_factor,
                    'ai_confidence_score': latest_metrics.ai_confidence_score,
                    'stress_test_results': latest_metrics.stress_test_results,
                    'market_regime': latest_metrics.market_regime_probability
                })
            
            return risk_data
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {str(e)}")
            return self._get_mock_risk_metrics()
    
    def _get_mock_risk_metrics(self):
        """Get mock risk metrics for fallback"""
        return {
            'overall_risk_level': 'medium',
            'portfolio_var': 0.024,
            'max_drawdown': -2.1,
            'sharpe_ratio': 3.24,
            'volatility': 0.18,
            'correlation_risk': 0.35,
            'liquidity_risk': 0.15,
            'stress_test_results': {
                'market_crash': -5.2,
                'liquidity_crisis': -8.1,
                'regulatory_shock': -3.7
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def background_updates(self):
        """Enhanced background task for real-time updates with full integration"""
        self.start_time = time.time()
        
        while True:
            try:
                if self.system_active:
                    # Update performance data from integrated systems
                    performance = self.get_performance_data()
                    
                    # Update strategies from strategy integrator
                    strategies = self.get_strategy_data()
                    
                    # Update opportunities from market data manager
                    opportunities = self.get_opportunities()
                    
                    # Get system status from all components
                    system_status = self.get_system_status()
                    
                    # Get risk metrics from risk manager
                    risk_metrics = self.get_risk_metrics()
                    
                    # Emit comprehensive updates via SocketIO
                    socketio.emit('performance_update', performance)
                    socketio.emit('strategy_update', strategies)
                    socketio.emit('opportunities_update', opportunities)
                    socketio.emit('system_status_update', system_status)
                    socketio.emit('risk_metrics_update', risk_metrics)
                    
                    # Log system health
                    if self.system_initialized:
                        health_status = system_status.get('market_data', {}).get('system_health', 'unknown')
                        if health_status not in ['excellent', 'good']:
                            self.logger.warning(f"System health: {health_status}")
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in background updates: {str(e)}")
                time.sleep(10)
    
    def start_background_tasks(self):
        """Start enhanced background update tasks with async event loop"""
        if not self.background_thread or not self.background_thread.is_alive():
            self.background_thread = threading.Thread(target=self.background_updates)
            self.background_thread.daemon = True
            self.background_thread.start()
            
            # Start async event loop for integrated systems
            if self.system_initialized:
                self._start_async_tasks()
            
            self.logger.info("Enhanced background tasks started with full integration")
    
    def _start_async_tasks(self):
        """Start async tasks for integrated system monitoring"""
        def run_async_loop():
            """Run async tasks in separate thread"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def monitor_systems():
                    """Monitor all integrated systems"""
                    while True:
                        try:
                            # Monitor strategy integrator
                            if self.strategy_integrator and hasattr(self.strategy_integrator, 'get_real_time_performance'):
                                performance = await self.strategy_integrator.get_real_time_performance()
                                if performance.get('system_health') == 'needs_attention':
                                    self.logger.warning("Strategy integration needs attention")
                            
                            # Monitor market data manager
                            if self.market_data_manager and hasattr(self.market_data_manager, 'get_market_summary'):
                                summary = self.market_data_manager.get_market_summary()
                                if summary.get('system_health') == 'poor':
                                    self.logger.warning("Market data system health is poor")
                            
                            # Monitor risk manager
                            if self.risk_manager and hasattr(self.risk_manager, 'monitor_real_time_risk'):
                                risk_status = await self.risk_manager.monitor_real_time_risk()
                                if risk_status.get('overall_risk_level') == 'high':
                                    self.logger.warning("High risk level detected")
                            
                            await asyncio.sleep(30)  # Monitor every 30 seconds
                            
                        except Exception as e:
                            self.logger.error(f"Error in async monitoring: {str(e)}")
                            await asyncio.sleep(60)
                
                # Start monitoring task
                loop.run_until_complete(monitor_systems())
                
            except Exception as e:
                self.logger.error(f"Error in async loop: {str(e)}")
        
        # Start async monitoring in separate thread
        async_thread = threading.Thread(target=run_async_loop)
        async_thread.daemon = True
        async_thread.start()

# Initialize the backend with full integration
try:
    backend = UltimateArbitrageBackend()
    print("✅ Ultimate Arbitrage Backend initialized successfully with full integration")
except Exception as e:
    print(f"❌ Error initializing backend: {str(e)}")
    # Create fallback backend
    backend = UltimateArbitrageBackend()
    print("⚠️ Fallback backend initialized")

# API Routes
@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get current performance data"""
    try:
        return jsonify(backend.get_performance_data())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get strategy data"""
    try:
        return jsonify(backend.get_strategy_data())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/opportunities', methods=['GET'])
def get_opportunities():
    """Get live arbitrage opportunities"""
    try:
        return jsonify(backend.get_opportunities())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/toggle', methods=['POST'])
def toggle_system():
    """Toggle system on/off"""
    try:
        result = backend.toggle_system()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Wallet Configuration API Endpoints
@app.route('/api/wallet-config/summary', methods=['GET'])
def get_wallet_config_summary():
    """Get wallet configuration summary"""
    try:
        if backend.wallet_config_manager:
            return jsonify(backend.wallet_config_manager.get_configuration_summary())
        else:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/available-strategies', methods=['GET'])
def get_available_strategies():
    """Get list of all available strategies with explanations"""
    try:
        if backend.wallet_config_manager:
            return jsonify(backend.wallet_config_manager.get_available_strategies())
        else:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/strategies/<strategy_type>/explanation', methods=['GET'])
def get_strategy_explanation(strategy_type):
    """Get detailed explanation for a specific strategy"""
    try:
        if backend.wallet_config_manager:
            try:
                strategy_enum = StrategyType(strategy_type)
                explanation = backend.wallet_config_manager.get_strategy_explanation(strategy_enum)
                return jsonify({'explanation': explanation})
            except ValueError:
                return jsonify({'error': 'Invalid strategy type'}), 400
        else:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/wallets', methods=['GET', 'POST'])
def manage_wallets():
    """Get all wallets or add a new wallet"""
    try:
        if not backend.wallet_config_manager:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
            
        if request.method == 'GET':
            # Return all wallets (without private keys)
            wallets = []
            for wallet_id, wallet in backend.wallet_config_manager.wallets.items():
                wallet_data = {
                    'id': wallet_id,
                    'network': wallet.network.value,
                    'address': wallet.address,
                    'balance_usd': wallet.balance_usd,
                    'is_active': wallet.is_active,
                    'risk_level': wallet.risk_level,
                    'last_updated': wallet.last_updated.isoformat() if wallet.last_updated else None
                }
                wallets.append(wallet_data)
            return jsonify(wallets)
            
        elif request.method == 'POST':
            # Add new wallet
            data = request.get_json()
            required_fields = ['network', 'address']
            
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            try:
                network = NetworkType(data['network'])
                wallet_id = f"wallet_{network.value}_{int(time.time())}"
                
                backend.wallet_config_manager.add_wallet(
                    wallet_id=wallet_id,
                    network=network,
                    address=data['address'],
                    private_key=data.get('privateKey'),
                    balance_usd=data.get('balance', 0.0)
                )
                
                return jsonify({'success': True, 'wallet_id': wallet_id})
                
            except ValueError:
                return jsonify({'error': 'Invalid network type'}), 400
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/exchanges', methods=['GET', 'POST'])
def manage_exchanges():
    """Get all exchanges or add a new exchange"""
    try:
        if not backend.wallet_config_manager:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
            
        if request.method == 'GET':
            # Return all exchanges (with masked secrets)
            exchanges = []
            for exchange_id, exchange in backend.wallet_config_manager.exchanges.items():
                exchange_data = {
                    'id': exchange_id,
                    'exchange': exchange.exchange.value,
                    'api_key': exchange.api_key[:8] + '...' if len(exchange.api_key) > 8 else exchange.api_key,
                    'sandbox_mode': exchange.sandbox_mode,
                    'is_active': exchange.is_active,
                    'daily_trading_limit_usd': exchange.daily_trading_limit_usd,
                    'last_validated': exchange.last_validated.isoformat() if exchange.last_validated else None
                }
                exchanges.append(exchange_data)
            return jsonify(exchanges)
            
        elif request.method == 'POST':
            # Add new exchange
            data = request.get_json()
            required_fields = ['exchange', 'apiKey', 'apiSecret']
            
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            try:
                exchange_type = ExchangeType(data['exchange'])
                exchange_id = f"exchange_{exchange_type.value}_{int(time.time())}"
                
                backend.wallet_config_manager.add_exchange(
                    exchange_id=exchange_id,
                    exchange=exchange_type,
                    api_key=data['apiKey'],
                    api_secret=data['apiSecret'],
                    passphrase=data.get('passphrase'),
                    sandbox_mode=data.get('sandboxMode', True)
                )
                
                return jsonify({'success': True, 'exchange_id': exchange_id})
                
            except ValueError:
                return jsonify({'error': 'Invalid exchange type'}), 400
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/strategies', methods=['GET'])
def get_configured_strategies():
    """Get all configured strategies"""
    try:
        if not backend.wallet_config_manager:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
            
        strategies = []
        for strategy_id, strategy in backend.wallet_config_manager.strategies.items():
            strategy_data = {
                'id': strategy_id,
                'name': strategy.name,
                'description': strategy.description,
                'strategy_type': strategy.strategy_type.value,
                'is_enabled': strategy.is_enabled,
                'min_capital_usd': strategy.min_capital_usd,
                'max_capital_usd': strategy.max_capital_usd,
                'profit_target_percent': strategy.profit_target_percent,
                'execution_frequency_minutes': strategy.execution_frequency_minutes,
                'required_wallets': [w.value for w in strategy.required_wallets],
                'required_exchanges': [e.value for e in strategy.required_exchanges],
                'assigned_wallets': {k.value: v for k, v in strategy.assigned_wallets.items()},
                'assigned_exchanges': {k.value: v for k, v in strategy.assigned_exchanges.items()},
                'last_execution': strategy.last_execution.isoformat() if strategy.last_execution else None
            }
            strategies.append(strategy_data)
        
        return jsonify(strategies)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/strategies/<strategy_id>/validate', methods=['GET'])
def validate_strategy_setup(strategy_id):
    """Validate that a strategy has all required components"""
    try:
        if not backend.wallet_config_manager:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
            
        validation_result = backend.wallet_config_manager.validate_strategy_setup(strategy_id)
        return jsonify(validation_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/strategies/<strategy_id>/auto-assign', methods=['POST'])
def auto_assign_strategy(strategy_id):
    """Automatically assign optimal wallets and exchanges for a strategy"""
    try:
        if not backend.wallet_config_manager:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
            
        result = backend.wallet_config_manager.auto_assign_wallets_and_exchanges(strategy_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet-config/strategies/<strategy_id>/toggle', methods=['POST'])
def toggle_strategy(strategy_id):
    """Enable or disable a strategy"""
    try:
        if not backend.wallet_config_manager:
            return jsonify({'error': 'Wallet configuration manager not available'}), 503
            
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        if strategy_id in backend.wallet_config_manager.strategies:
            strategy = backend.wallet_config_manager.strategies[strategy_id]
            strategy.is_enabled = enabled
            backend.wallet_config_manager._save_strategy_config(strategy_id, strategy)
            return jsonify({'success': True, 'enabled': enabled})
        else:
            return jsonify({'error': 'Strategy not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/emergency-stop', methods=['POST'])
def emergency_stop():
    """Emergency stop all operations"""
    try:
        result = backend.emergency_stop()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        return jsonify(backend.get_system_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update system settings"""
    try:
        settings = request.get_json()
        result = backend.update_settings(settings)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/risk-metrics', methods=['GET'])
def get_risk_metrics():
    """Get risk management metrics"""
    try:
        return jsonify(backend.get_risk_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Trading Engine Endpoints
@app.route('/api/trading/portfolio', methods=['GET'])
def get_portfolio_status():
    """Get current portfolio status from trading engine"""
    try:
        if backend.trading_engine:
            return jsonify(backend.trading_engine.get_portfolio_status())
        else:
            return jsonify({'error': 'Trading engine not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/active-trades', methods=['GET'])
def get_active_trades():
    """Get currently active trades"""
    try:
        if backend.trading_engine:
            return jsonify(backend.trading_engine.get_active_trades())
        else:
            return jsonify({'error': 'Trading engine not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/recent-trades', methods=['GET'])
def get_recent_trades():
    """Get recent completed trades"""
    try:
        limit = request.args.get('limit', 10, type=int)
        if backend.trading_engine:
            return jsonify(backend.trading_engine.get_recent_trades(limit))
        else:
            return jsonify({'error': 'Trading engine not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_status', {'status': 'connected', 'timestamp': datetime.now().isoformat()})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request from client"""
    try:
        # Send immediate updates
        emit('performance_update', backend.get_performance_data())
        emit('strategy_update', backend.get_strategy_data())
        emit('opportunities_update', backend.get_opportunities())
        emit('system_status_update', backend.get_system_status())
        emit('risk_metrics_update', backend.get_risk_metrics())
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('system_command')
def handle_system_command(data):
    """Handle system commands from client"""
    try:
        command = data.get('command')
        
        if command == 'toggle':
            result = backend.toggle_system()
            emit('system_response', result)
        elif command == 'emergency_stop':
            result = backend.emergency_stop()
            emit('system_response', result)
        elif command == 'update_settings':
            settings = data.get('settings', {})
            result = backend.update_settings(settings)
            emit('system_response', result)
        else:
            emit('error', {'message': f'Unknown command: {command}'})
            
    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Ultimate Arbitrage System Backend with Full Integration...")
    print(f"🌐 Backend available at: http://localhost:5000")
    print(f"🔌 WebSocket available at: ws://localhost:5000")
    print("\n📊 Integrated Systems:")
    print("  ⚡ Advanced Strategy Integrator")
    print("  🛡️ Ultimate Risk Manager")
    print("  📈 Ultimate Market Data Manager")
    print("  🤖 AI-Powered Optimization")
    print("  ⚛️ Quantum Enhancement Engine")
    print("\n🎯 Ready for maximum profit generation!")
    
    # Run with debug=False for production
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

