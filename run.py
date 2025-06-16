#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Arbitrage System Runner
=============================

Main entry point for the automated trading system.
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from decimal import Decimal
import signal
import argparse

from core.orchestration.AutomatedOrchestrator import AutomatedOrchestrator, OrchestratorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log')
    ]
)

logger = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system runner"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.orchestrator = None
        self.running = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _load_config(self, config_path: str) -> OrchestratorConfig:
        """Load system configuration"""
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                # Use default configuration
                config_data = {
                    'max_concurrent_strategies': 5,
                    'optimization_interval': 300,
                    'risk_check_interval': 60,
                    'metrics_interval': 30,
                    'profit_threshold': '0.001',
                    'emergency_stop_loss': '0.05',
                    'auto_recovery': True,
                    'quantum_enabled': True,
                    'max_position_value': '1000000',
                    'test_mode': True
                }
            
            # Convert decimal strings to Decimal
            for key in ['profit_threshold', 'emergency_stop_loss', 'max_position_value']:
                if key in config_data:
                    config_data[key] = Decimal(str(config_data[key]))
            
            return OrchestratorConfig(**config_data)
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)
    
    def _handle_shutdown(self, signum, frame):
        """Handle system shutdown signals"""
        logger.info("Shutdown signal received...")
        if self.running:
            asyncio.create_task(self.stop())
    
    async def start(self) -> None:
        """Start the trading system"""
        try:
            logger.info("Starting trading system...")
            
            # Initialize orchestrator
            self.orchestrator = AutomatedOrchestrator(self.config)
            
            # Start system
            if await self.orchestrator.start():
                self.running = True
                logger.info("Trading system started successfully")
                
                # Keep system running
                while self.running:
                    await asyncio.sleep(1)
            else:
                logger.error("Failed to start trading system")
                await self.stop()
            
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the trading system"""
        try:
            logger.info("Stopping trading system...")
            self.running = False
            
            if self.orchestrator:
                await self.orchestrator.stop()
            
            logger.info("Trading system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {str(e)}")
            sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Ultimate Arbitrage System Runner'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--test-mode',
        help='Run system in test mode',
        action='store_true',
        default=True
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Initialize system
        system = TradingSystem(args.config)
        
        # Run system
        asyncio.run(system.start())
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

