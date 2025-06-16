#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultra-Advanced Portfolio Manager
==============================

Main application that integrates quantum computing, advanced AI, and comprehensive
market analysis for automated portfolio optimization and trading.

This is the central orchestrator that brings together:
- Quantum Computing Engine
- Advanced AI Neural Networks
- Ultra-Advanced Integration Engine
- Real-time Market Data
- Risk Management
- Execution System
"""

import asyncio
import logging
import sys
import os
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
import json
from dataclasses import asdict
import argparse
from concurrent.futures import ThreadPoolExecutor
import schedule
import time
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our advanced components
try:
    from ai.integration.ultra_advanced_integration import (
        UltraAdvancedIntegration, IntegrationConfig, UltraOptimizationResult
    )
except ImportError as e:
    print(f"Failed to import ultra-advanced integration: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ultra_advanced_portfolio_manager.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("UltraAdvancedPortfolioManager")


class UltraAdvancedPortfolioManager:
    """
    Ultra-Advanced Portfolio Manager that orchestrates quantum-AI optimization
    for automated portfolio management and trading.
    """
    
    def __init__(self, config_path: str = "config/ultra_advanced_config.yaml"):
        """
        Initialize the Ultra-Advanced Portfolio Manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Core components
        self.integration_engine = None
        self.current_portfolio = {}
        self.optimization_history = []
        self.performance_metrics = {}
        
        # System state
        self.is_running = False
        self.is_trading_hours = False
        self.last_optimization = None
        self.last_rebalancing = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config['system']['max_workers'])
        
        # Initialize directories
        self._initialize_directories()
        
        logger.info(f"UltraAdvancedPortfolioManager initialized with config: {config_path}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load environment variables
            self._load_environment_variables(config)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_environment_variables(self, config: Dict[str, Any]):
        """Load environment variables referenced in config."""
        import re
        
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Replace ${VAR_NAME} with environment variable
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, obj)
                for match in matches:
                    env_value = os.getenv(match, '')
                    obj = obj.replace(f'${{{match}}}', env_value)
                return obj
            else:
                return obj
        
        return replace_env_vars(config)
    
    def _initialize_directories(self):
        """Initialize required directories."""
        directories = [
            'logs',
            'data',
            'reports',
            'models',
            'cache'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info("Directories initialized")
    
    async def initialize(self):
        """Initialize all components of the portfolio manager."""
        logger.info("Initializing Ultra-Advanced Portfolio Manager...")
        
        try:
            # Initialize the integration engine
            self.integration_engine = UltraAdvancedIntegration(self.config)
            
            # Initialize quantum and AI engines
            await self.integration_engine.initialize(
                quantum_config=self._prepare_quantum_config(),
                ai_config=self._prepare_ai_config()
            )
            
            # Start the integration engine
            await self.integration_engine.start()
            
            logger.info("Ultra-Advanced Portfolio Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize portfolio manager: {e}")
            raise
    
    def _prepare_quantum_config(self) -> Dict[str, Any]:
        """Prepare quantum engine configuration."""
        quantum_config = self.config.get('quantum', {})
        
        return {
            'quantum_hardware': {
                'ibm_token': quantum_config.get('providers', {}).get('ibm', {}).get('token'),
                'dwave_token': quantum_config.get('providers', {}).get('dwave', {}).get('token'),
                'use_real_hardware': quantum_config.get('providers', {}).get('ibm', {}).get('use_real_hardware', False),
                'max_qubits': quantum_config.get('providers', {}).get('ibm', {}).get('max_qubits', 20),
                'shots': quantum_config.get('providers', {}).get('ibm', {}).get('shots', 1024),
                'optimization_level': quantum_config.get('providers', {}).get('ibm', {}).get('optimization_level', 2)
            }
        }
    
    def _prepare_ai_config(self) -> Dict[str, Any]:
        """Prepare AI engine configuration."""
        ai_config = self.config.get('ai', {})
        
        return {
            'ai_model': {
                'model_type': 'ensemble',
                'learning_rate': ai_config.get('training', {}).get('learning_rate', 0.001),
                'batch_size': ai_config.get('training', {}).get('batch_size', 64),
                'hidden_dim': ai_config.get('models', {}).get('transformer', {}).get('hidden_dim', 256),
                'num_layers': ai_config.get('models', {}).get('transformer', {}).get('num_layers', 4),
                'attention_heads': ai_config.get('models', {}).get('transformer', {}).get('attention_heads', 8),
                'sequence_length': ai_config.get('models', {}).get('transformer', {}).get('sequence_length', 60),
                'embedding_dim': ai_config.get('models', {}).get('transformer', {}).get('embedding_dim', 128),
                'dropout_rate': ai_config.get('models', {}).get('transformer', {}).get('dropout_rate', 0.2),
                'training_epochs': ai_config.get('training', {}).get('epochs', 1000),
                'early_stopping_patience': ai_config.get('training', {}).get('early_stopping_patience', 50),
                'l2_regularization': ai_config.get('training', {}).get('l2_regularization', 0.001),
                'gradient_clip_norm': ai_config.get('training', {}).get('gradient_clip_norm', 1.0)
            }
        }
    
    async def start(self):
        """Start the portfolio manager."""
        logger.info("Starting Ultra-Advanced Portfolio Manager...")
        
        self.is_running = True
        
        # Schedule optimization tasks
        self._schedule_optimization_tasks()
        
        # Start main event loop
        await self._main_loop()
    
    def _schedule_optimization_tasks(self):
        """Schedule optimization and rebalancing tasks."""
        frequency = self.config['portfolio']['rebalancing_frequency']
        
        if frequency == 'hourly':
            schedule.every().hour.do(self._schedule_optimization)
        elif frequency == 'daily':
            schedule.every().day.at("09:30").do(self._schedule_optimization)  # Market open
        elif frequency == 'weekly':
            schedule.every().monday.at("09:30").do(self._schedule_optimization)
        
        logger.info(f"Optimization scheduled with frequency: {frequency}")
    
    def _schedule_optimization(self):
        """Schedule an optimization task."""
        if self.is_running and self._is_trading_hours():
            asyncio.create_task(self._run_optimization())
    
    def _is_trading_hours(self) -> bool:
        """Check if it's currently trading hours."""
        now = datetime.now()
        
        # Simple check for US market hours (9:30 AM - 4:00 PM ET)
        # In production, you'd want more sophisticated timezone handling
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def _main_loop(self):
        """Main event loop for the portfolio manager."""
        logger.info("Starting main event loop...")
        
        while self.is_running:
            try:
                # Check scheduled tasks
                schedule.run_pending()
                
                # Check trading hours
                self.is_trading_hours = self._is_trading_hours()
                
                # Real-time monitoring if enabled
                if self.config['integration']['real_time_data']:
                    await self._real_time_monitoring()
                
                # Sleep for a short interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Sleep longer on error
    
    async def _real_time_monitoring(self):
        """Perform real-time portfolio monitoring."""
        try:
            if not self.current_portfolio:
                return
            
            # Get current market data for portfolio assets
            symbols = list(self.current_portfolio.keys())
            
            # Simple risk check (placeholder)
            # In production, this would include comprehensive risk monitoring
            
        except Exception as e:
            logger.warning(f"Real-time monitoring error: {e}")
    
    async def _run_optimization(self):
        """Run portfolio optimization."""
        logger.info("Starting portfolio optimization...")
        
        try:
            # Get asset universe
            symbols = self._get_asset_universe()
            
            # Get risk tolerance from config
            risk_tolerance = self.config['portfolio']['default_risk_tolerance']
            
            # Get investment horizon
            investment_horizon = self.config['portfolio']['optimization_horizon']
            
            # Run ultra-advanced optimization
            result = await self.integration_engine.ultra_optimize_portfolio(
                symbols=symbols,
                risk_tolerance=risk_tolerance,
                investment_horizon=investment_horizon
            )
            
            # Store result
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'result': asdict(result),
                'symbols': symbols
            })
            
            self.last_optimization = datetime.now()
            
            # Check if rebalancing is needed
            if self._should_rebalance(result):
                await self._execute_rebalancing(result)
            
            # Update performance metrics
            await self._update_performance_metrics(result)
            
            # Generate report
            await self._generate_optimization_report(result)
            
            logger.info(f"Optimization completed successfully. Sharpe ratio: {result.sharpe_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
    
    def _get_asset_universe(self) -> List[str]:
        """Get the asset universe for optimization."""
        universe = self.config['portfolio']['asset_universe']
        
        symbols = []
        symbols.extend(universe.get('stocks', []))
        symbols.extend(universe.get('etfs', []))
        
        # Only include crypto if enabled
        if self.config.get('portfolio', {}).get('include_crypto', False):
            symbols.extend(universe.get('crypto', []))
        
        return symbols
    
    def _should_rebalance(self, result: UltraOptimizationResult) -> bool:
        """Determine if portfolio rebalancing is needed."""
        if not self.current_portfolio:
            return True  # First optimization
        
        # Check confidence threshold
        if result.confidence_score < self.config['integration']['confidence_threshold']:
            logger.info(f"Skipping rebalancing due to low confidence: {result.confidence_score:.3f}")
            return False
        
        # Check if weights have changed significantly
        threshold = 0.05  # 5% threshold
        
        for i, symbol in enumerate(result.asset_names):
            new_weight = result.optimal_weights[i]
            current_weight = self.current_portfolio.get(symbol, 0)
            
            if abs(new_weight - current_weight) > threshold:
                return True
        
        return False
    
    async def _execute_rebalancing(self, result: UltraOptimizationResult):
        """Execute portfolio rebalancing."""
        logger.info("Executing portfolio rebalancing...")
        
        try:
            # Update current portfolio
            new_portfolio = {}
            for i, symbol in enumerate(result.asset_names):
                weight = result.optimal_weights[i]
                if weight > 0.01:  # Only include meaningful positions
                    new_portfolio[symbol] = weight
            
            self.current_portfolio = new_portfolio
            self.last_rebalancing = datetime.now()
            
            # In production, this would execute actual trades
            logger.info(f"Portfolio rebalanced. New allocation: {new_portfolio}")
            
            # Log rebalancing trades
            for trade in result.rebalancing_trades:
                logger.info(f"Trade: {trade['action']} {trade['symbol']} to {trade['target_weight']:.3f}")
            
        except Exception as e:
            logger.error(f"Rebalancing execution failed: {e}")
    
    async def _update_performance_metrics(self, result: UltraOptimizationResult):
        """Update performance metrics."""
        self.performance_metrics = {
            'last_optimization': datetime.now().isoformat(),
            'sharpe_ratio': result.sharpe_ratio,
            'expected_return': result.expected_return,
            'expected_volatility': result.expected_volatility,
            'confidence_score': result.confidence_score,
            'quantum_advantage': result.quantum_advantage,
            'market_regime': result.market_regime.regime_type,
            'optimization_time': result.optimization_time
        }
    
    async def _generate_optimization_report(self, result: UltraOptimizationResult):
        """Generate optimization report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/optimization_report_{timestamp}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'optimization_result': asdict(result),
                'current_portfolio': self.current_portfolio,
                'performance_metrics': self.performance_metrics,
                'system_analytics': self.integration_engine.get_performance_analytics()
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Optimization report generated: {report_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")
    
    async def manual_optimization(self, symbols: Optional[List[str]] = None, 
                                risk_tolerance: Optional[float] = None) -> UltraOptimizationResult:
        """Manually trigger portfolio optimization."""
        logger.info("Manual optimization triggered")
        
        if symbols is None:
            symbols = self._get_asset_universe()
        
        if risk_tolerance is None:
            risk_tolerance = self.config['portfolio']['default_risk_tolerance']
        
        investment_horizon = self.config['portfolio']['optimization_horizon']
        
        result = await self.integration_engine.ultra_optimize_portfolio(
            symbols=symbols,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon
        )
        
        # Store result
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'result': asdict(result),
            'symbols': symbols,
            'manual': True
        })
        
        logger.info("Manual optimization completed")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the portfolio manager."""
        return {
            'is_running': self.is_running,
            'is_trading_hours': self.is_trading_hours,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'last_rebalancing': self.last_rebalancing.isoformat() if self.last_rebalancing else None,
            'current_portfolio': self.current_portfolio,
            'performance_metrics': self.performance_metrics,
            'total_optimizations': len(self.optimization_history),
            'system_analytics': self.integration_engine.get_performance_analytics() if self.integration_engine else {}
        }
    
    async def stop(self):
        """Stop the portfolio manager."""
        logger.info("Stopping Ultra-Advanced Portfolio Manager...")
        
        self.is_running = False
        
        if self.integration_engine:
            await self.integration_engine.stop()
        
        logger.info("Ultra-Advanced Portfolio Manager stopped")


async def main():
    """Main function to run the Ultra-Advanced Portfolio Manager."""
    parser = argparse.ArgumentParser(description="Ultra-Advanced Portfolio Manager")
    parser.add_argument('--config', '-c', default='config/ultra_advanced_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--manual', '-m', action='store_true',
                       help='Run manual optimization only')
    parser.add_argument('--symbols', nargs='+',
                       help='Symbols for manual optimization')
    parser.add_argument('--risk-tolerance', type=float, default=0.5,
                       help='Risk tolerance for manual optimization')
    
    args = parser.parse_args()
    
    # Create portfolio manager
    manager = UltraAdvancedPortfolioManager(args.config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(manager.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize the manager
        await manager.initialize()
        
        if args.manual:
            # Run manual optimization
            result = await manager.manual_optimization(
                symbols=args.symbols,
                risk_tolerance=args.risk_tolerance
            )
            
            print("\n" + "="*80)
            print("ULTRA-ADVANCED OPTIMIZATION RESULTS")
            print("="*80)
            print(f"Expected Return: {result.expected_return:.4f} ({result.expected_return*100:.2f}%)")
            print(f"Expected Volatility: {result.expected_volatility:.4f} ({result.expected_volatility*100:.2f}%)")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
            print(f"Sortino Ratio: {result.sortino_ratio:.4f}")
            print(f"Calmar Ratio: {result.calmar_ratio:.4f}")
            print(f"Max Drawdown: {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)")
            print(f"VaR (95%): {result.var_95:.4f}")
            print(f"CVaR (95%): {result.cvar_95:.4f}")
            print(f"Confidence Score: {result.confidence_score:.4f}")
            print(f"Quantum Advantage: {result.quantum_advantage:.2f}x")
            print(f"Market Regime: {result.market_regime.regime_type} (confidence: {result.market_regime.confidence:.2f})")
            print(f"Optimization Time: {result.optimization_time:.2f} seconds")
            
            print("\nOptimal Portfolio Allocation:")
            print("-" * 40)
            for i, symbol in enumerate(result.asset_names):
                weight = result.optimal_weights[i]
                if weight > 0.001:
                    print(f"{symbol:8} {weight:8.3f} ({weight*100:6.2f}%)")
            
            print("\nStrategy Recommendations:")
            print("-" * 40)
            for rec in result.strategy_recommendations:
                print(f"• {rec}")
            
            if result.risk_alerts:
                print("\nRisk Alerts:")
                print("-" * 40)
                for alert in result.risk_alerts:
                    print(f"⚠️  {alert}")
            
        else:
            # Start continuous operation
            await manager.start()
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error running portfolio manager: {e}")
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())

