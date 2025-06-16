#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Arbitrage System Launcher
=================================

Launches the complete Ultimate Arbitrage System with all components:
- Quantum Portfolio Optimization
- Autonomous Evolution Engine
- Event-Driven Risk Adjustment
- Real-time Market Intelligence
- Integrated Signal Processing

This is the ULTIMATE profit-generating machine that surpasses
any competitor through zero-investment creative thinking and
comprehensive gray-hat analysis.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the orchestrator
from src.core.ultimate_arbitrage_orchestrator import create_ultimate_arbitrage_system

# Configure logging with beautiful output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print the ultimate system banner"""
    banner = """
██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗
██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗  
██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  
╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
 ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
                                                               
                   ARBITRAGE SYSTEM                            
                                                               
🌟 Zero-Investment Mindset | 🧠 Creative Beyond Measure        
⚖️ Gray Hat Analysis | 🚀 Surpassing All Competitors         
💎 Quantum Enhanced | 🔄 Autonomous Evolution                 
"""
    print(banner)
    print(f"\n🕒 System initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Embodying unlimited creative potential...")
    print("🎯 Target: Maximum profit with zero investment restrictions")
    print("🛡️ Gray hat comprehensive scenario coverage engaged")
    print("\n" + "="*70 + "\n")

async def main():
    """Main launcher function"""
    try:
        # Print beautiful banner
        print_banner()
        
        logger.info("🚀 Launching Ultimate Arbitrage System...")
        
        # Create and initialize the system
        logger.info("⚡ Creating system components...")
        system = await create_ultimate_arbitrage_system()
        
        logger.info("🎊 Ultimate Arbitrage System fully initialized!")
        logger.info("🌟 All components active and ready for profit generation")
        logger.info("💎 Zero-investment creative mindset: ENGAGED")
        logger.info("🧠 Beyond-measure thinking: ACTIVATED")
        logger.info("⚖️ Gray hat comprehensive analysis: ONLINE")
        logger.info("🚀 Competitor-surpassing mode: INITIATED")
        
        # Start the system
        logger.info("\n🔥 STARTING ULTIMATE PROFIT GENERATION 🔥\n")
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("\n⚡ Graceful shutdown initiated by user...")
    except Exception as e:
        logger.error(f"❌ Critical error in Ultimate Arbitrage System: {e}")
        logger.error("🔄 System will attempt graceful recovery...")
        
        # Attempt recovery
        try:
            if 'system' in locals():
                await system.shutdown()
        except:
            logger.error("⚠️ Emergency shutdown completed")
    
    finally:
        logger.info("💎 Ultimate Arbitrage System session completed")
        logger.info("🙏 Thank you for using the most advanced trading system ever created")

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'numpy', 'pandas', 'asyncio', 'yfinance', 'scikit-learn',
        'tensorflow', 'cvxpy', 'scipy', 'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        logger.info("📦 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        # The asyncio event loop will handle the actual shutdown
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Setup
    setup_signal_handlers()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("🚫 Cannot start system due to missing dependencies")
        sys.exit(1)
    
    # Run the ultimate system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚡ Ultimate Arbitrage System shutdown complete")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

