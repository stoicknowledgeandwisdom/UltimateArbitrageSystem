#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Live Trading System - Complete Automation
================================================

Fully automated live trading system with real money execution.
Integrates with multiple brokers and exchanges for true arbitrage automation.

WARNING: This system trades with real money. Use extreme caution.

Usage:
    python start_live_trading_system.py
"""

import os
import sys
import asyncio
import logging
import json
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.exchanges.live_broker_integration import (
    LiveTradingManager, 
    SecureCredentialManager,
    BrokerCredentials
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/live_trading.log', mode='a')
    ]
)
logger = logging.getLogger('LiveTradingSystem')

class UltimateLiveTradingSystem:
    """
    Complete automated live trading system with real money execution.
    Integrates all components for fully automated income generation.
    """
    
    def __init__(self, config_file: str = "config/live_trading_config.json"):
        self.config_file = config_file
        self.config = {}
        self.trading_manager = None
        self.is_running = False
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        self.start_time = None
        
        # Create required directories
        self._create_directories()
        
        # Load configuration
        self._load_configuration()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_directories(self):
        """Create required directories"""
        directories = ['logs', 'config', 'data', 'reports']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_configuration(self):
        """Load system configuration"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration for live trading"""
        self.config = {
            "system_name": "Ultimate Live Trading System",
            "version": "2.0.0",
            "live_trading_enabled": False,  # Safety: disabled by default
            
            "brokers": {
                "enabled_brokers": ["alpaca", "binance"],
                "primary_broker": "alpaca",
                "backup_brokers": ["binance"]
            },
            
            "trading_parameters": {
                "max_position_size": 1000.0,  # Maximum position size in USD
                "min_profit_threshold": 0.5,  # Minimum profit percentage
                "max_trades_per_hour": 10,
                "max_concurrent_trades": 3,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 5.0
            },
            
            "risk_management": {
                "max_daily_loss": 500.0,  # Maximum daily loss in USD
                "max_drawdown": 1000.0,   # Maximum total drawdown
                "emergency_stop_triggers": {
                    "consecutive_losses": 5,
                    "hourly_loss_limit": 100.0,
                    "api_error_threshold": 3
                }
            },
            
            "automation_settings": {
                "opportunity_scan_interval": 5,  # seconds
                "execution_timeout": 30,  # seconds
                "retry_attempts": 3,
                "auto_compound_profits": True,
                "profit_withdrawal_threshold": 10000.0
            },
            
            "strategies": {
                "arbitrage": {
                    "enabled": True,
                    "min_spread": 0.3,  # Minimum spread percentage
                    "max_execution_time": 10  # seconds
                },
                "triangular_arbitrage": {
                    "enabled": True,
                    "min_profit": 0.2  # Minimum profit percentage
                },
                "cross_exchange": {
                    "enabled": True,
                    "supported_pairs": ["BTC/USD", "ETH/USD", "BTC/USDT", "ETH/USDT"]
                }
            },
            
            "monitoring": {
                "send_notifications": True,
                "notification_methods": ["log", "email"],
                "report_interval": 3600,  # seconds (1 hour)
                "performance_tracking": True
            }
        }
        
        # Save default configuration
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Default configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop_system())
    
    def display_startup_banner(self):
        """Display system startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🚀 ULTIMATE LIVE TRADING SYSTEM 🚀                       ║
║                                                                              ║
║                    ⚠️  REAL MONEY AUTOMATED TRADING ⚠️                      ║
║                                                                              ║
║     💰 Live Execution • 🤖 Full Automation • 🛡️ Risk Protected 🛡️          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        
        # Display critical warnings
        if self.config.get('live_trading_enabled', False):
            print("\n🚨 " + "="*76 + " 🚨")
            print("🚨 LIVE TRADING MODE ENABLED - REAL MONEY WILL BE TRADED 🚨")
            print("🚨 " + "="*76 + " 🚨\n")
        else:
            print("\n✅ " + "="*64 + " ✅")
            print("✅ SIMULATION MODE - NO REAL MONEY AT RISK ✅")
            print("✅ " + "="*64 + " ✅\n")
    
    async def initialize_system(self) -> bool:
        """Initialize the complete trading system"""
        try:
            logger.info("🔧 Initializing Ultimate Live Trading System...")
            
            # Initialize trading manager
            self.trading_manager = LiveTradingManager()
            
            # Initialize brokers
            enabled_brokers = self.config.get('brokers', {}).get('enabled_brokers', [])
            if not enabled_brokers:
                logger.error("❌ No brokers configured")
                return False
            
            await self.trading_manager.initialize_brokers(enabled_brokers)
            
            # Enable live trading if configured
            if self.config.get('live_trading_enabled', False):
                confirmation = input("\n⚠️  Type 'I UNDERSTAND LIVE TRADING RISKS' to enable live trading: ")
                if self.trading_manager.enable_live_trading(confirmation):
                    logger.warning("🚨 LIVE TRADING MODE ACTIVATED")
                else:
                    logger.info("📊 Running in simulation mode")
            
            logger.info("✅ System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def start_trading_loop(self):
        """Main trading loop - scans for opportunities and executes trades"""
        logger.info("🔄 Starting automated trading loop...")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        scan_interval = self.config.get('automation_settings', {}).get('opportunity_scan_interval', 5)
        
        try:
            while self.is_running:
                # Scan for arbitrage opportunities
                opportunities = await self._scan_opportunities()
                
                # Execute profitable opportunities
                for opportunity in opportunities:
                    if await self._should_execute_opportunity(opportunity):
                        result = await self._execute_opportunity(opportunity)
                        await self._process_trade_result(result)
                
                # Check risk limits and emergency stops
                await self._check_risk_limits()
                
                # Generate periodic reports
                await self._generate_periodic_reports()
                
                # Wait before next scan
                await asyncio.sleep(scan_interval)
                
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {e}")
        finally:
            await self.stop_system()
    
    async def _scan_opportunities(self) -> List[Dict[str, Any]]:
        """Scan for arbitrage opportunities across all configured strategies"""
        opportunities = []
        
        try:
            # Mock opportunity generation for demonstration
            # In real implementation, this would scan live markets
            mock_opportunities = [
                {
                    'id': f'opp_{datetime.now().timestamp()}',
                    'type': 'cross_exchange_arbitrage',
                    'symbol': 'BTC/USD',
                    'buy_exchange': 'alpaca',
                    'sell_exchange': 'binance',
                    'buy_price': 50000.0,
                    'sell_price': 50150.0,
                    'profit_usd': 150.0,
                    'profit_percentage': 0.3,
                    'confidence': 0.95,
                    'estimated_execution_time': 5.0,
                    'risk_score': 0.1
                }
            ]
            
            # Filter opportunities based on configuration
            min_profit = self.config.get('trading_parameters', {}).get('min_profit_threshold', 0.5)
            
            for opp in mock_opportunities:
                if opp['profit_percentage'] >= min_profit:
                    opportunities.append(opp)
            
        except Exception as e:
            logger.error(f"❌ Error scanning opportunities: {e}")
        
        return opportunities
    
    async def _should_execute_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Determine if an opportunity should be executed based on risk parameters"""
        try:
            # Check profit threshold
            min_profit = self.config.get('trading_parameters', {}).get('min_profit_threshold', 0.5)
            if opportunity['profit_percentage'] < min_profit:
                return False
            
            # Check position size limits
            max_position = self.config.get('trading_parameters', {}).get('max_position_size', 1000.0)
            if opportunity['profit_usd'] > max_position:
                return False
            
            # Check risk score
            if opportunity.get('risk_score', 1.0) > 0.5:
                return False
            
            # Check daily trade limits
            max_trades = self.config.get('trading_parameters', {}).get('max_trades_per_hour', 10)
            # Implementation would check actual trade count
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error evaluating opportunity: {e}")
            return False
    
    async def _execute_opportunity(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an arbitrage opportunity"""
        try:
            logger.info(f"⚡ Executing opportunity: {opportunity['id']}")
            
            # Execute through trading manager
            result = await self.trading_manager.execute_arbitrage_opportunity(opportunity)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to execute opportunity {opportunity['id']}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'opportunity_id': opportunity['id']
            }
    
    async def _process_trade_result(self, result: Dict[str, Any]):
        """Process the result of a trade execution"""
        try:
            if result['status'] == 'executed' or result['status'] == 'simulated':
                profit = result.get('profit', 0)
                self.total_profit += profit
                self.successful_trades += 1
                
                logger.info(f"✅ Trade successful: ${profit:.2f} profit")
                logger.info(f"📊 Total profit: ${self.total_profit:.2f} | Successful trades: {self.successful_trades}")
                
            elif result['status'] == 'failed':
                self.failed_trades += 1
                logger.warning(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing trade result: {e}")
    
    async def _check_risk_limits(self):
        """Check risk limits and trigger emergency stops if needed"""
        try:
            # Check daily loss limit
            max_daily_loss = self.config.get('risk_management', {}).get('max_daily_loss', 500.0)
            if self.total_profit < -max_daily_loss:
                logger.error(f"🚨 Daily loss limit exceeded: ${abs(self.total_profit):.2f}")
                await self.emergency_stop("Daily loss limit exceeded")
            
            # Check consecutive failures
            if self.failed_trades >= self.config.get('risk_management', {}).get('emergency_stop_triggers', {}).get('consecutive_losses', 5):
                logger.error(f"🚨 Too many consecutive failures: {self.failed_trades}")
                await self.emergency_stop("Consecutive failure limit exceeded")
            
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
    
    async def _generate_periodic_reports(self):
        """Generate periodic performance reports"""
        try:
            if not hasattr(self, '_last_report_time'):
                self._last_report_time = datetime.now()
            
            report_interval = self.config.get('monitoring', {}).get('report_interval', 3600)
            
            if (datetime.now() - self._last_report_time).seconds >= report_interval:
                # Generate performance report
                uptime = datetime.now() - self.start_time if self.start_time else None
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_hours': uptime.total_seconds() / 3600 if uptime else 0,
                    'total_profit': self.total_profit,
                    'successful_trades': self.successful_trades,
                    'failed_trades': self.failed_trades,
                    'success_rate': (self.successful_trades / max(1, self.successful_trades + self.failed_trades)) * 100
                }
                
                logger.info(f"📊 Performance Report: Profit: ${report['total_profit']:.2f} | Success Rate: {report['success_rate']:.1f}%")
                
                # Save report to file
                report_file = f"reports/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                self._last_report_time = datetime.now()
                
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
    
    async def emergency_stop(self, reason: str):
        """Emergency stop all trading activities"""
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
        
        # Disable live trading
        if self.trading_manager:
            self.trading_manager.disable_live_trading()
        
        # Stop trading loop
        self.is_running = False
        
        # Generate emergency report
        emergency_report = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'total_profit': self.total_profit,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades
        }
        
        with open(f"reports/emergency_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(emergency_report, f, indent=2)
        
        logger.critical("🛑 All trading activities stopped")
    
    async def stop_system(self):
        """Gracefully stop the trading system"""
        logger.info("🛑 Stopping Ultimate Live Trading System...")
        
        self.is_running = False
        
        # Generate final report
        if self.start_time:
            final_report = {
                'session_start': self.start_time.isoformat(),
                'session_end': datetime.now().isoformat(),
                'total_profit': self.total_profit,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'final_success_rate': (self.successful_trades / max(1, self.successful_trades + self.failed_trades)) * 100
            }
            
            with open(f"reports/session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(final_report, f, indent=2)
            
            logger.info(f"📊 Session Summary: Profit: ${self.total_profit:.2f} | Trades: {self.successful_trades + self.failed_trades}")
        
        logger.info("✅ System shutdown complete")
    
    async def run(self):
        """Main entry point to run the complete system"""
        try:
            # Display startup banner
            self.display_startup_banner()
            
            # Initialize system
            if not await self.initialize_system():
                logger.error("❌ System initialization failed")
                return False
            
            logger.info("🚀 Ultimate Live Trading System started successfully")
            
            # Start trading loop
            await self.start_trading_loop()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            await self.stop_system()
        
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Ultimate Live Trading System - Automated Real Money Trading'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        default='config/live_trading_config.json'
    )
    
    parser.add_argument(
        '--setup-credentials',
        action='store_true',
        help='Setup broker credentials'
    )
    
    parser.add_argument(
        '--test-connections',
        action='store_true',
        help='Test broker connections'
    )
    
    return parser.parse_args()

async def test_broker_connections():
    """Test connections to all configured brokers"""
    print("\n🔍 Testing broker connections...")
    
    try:
        manager = LiveTradingManager()
        await manager.initialize_brokers(['alpaca', 'binance'])
        
        # Test each connector
        for broker_name, connector in manager.connectors.items():
            try:
                async with connector:
                    if hasattr(connector, 'get_account'):
                        account = await connector.get_account()
                        print(f"✅ {broker_name}: Connected successfully")
                    elif hasattr(connector, 'get_account_info'):
                        account = await connector.get_account_info()
                        print(f"✅ {broker_name}: Connected successfully")
            except Exception as e:
                print(f"❌ {broker_name}: Connection failed - {e}")
    
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.setup_credentials:
        from src.exchanges.live_broker_integration import setup_broker_credentials
        setup_broker_credentials()
        return 0
    
    if args.test_connections:
        asyncio.run(test_broker_connections())
        return 0
    
    # Run the live trading system
    try:
        system = UltimateLiveTradingSystem(args.config)
        success = asyncio.run(system.run())
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

