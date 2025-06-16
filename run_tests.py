#!/usr/bin/env python3
"""Run comprehensive tests for the Portfolio Optimizer."""

import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    try:
        from test_portfolio_optimizer import run_comprehensive_tests
        
        print("Running Portfolio Optimizer Test Suite...")
        success = asyncio.run(run_comprehensive_tests())
        
        if success:
            print("\nAll tests passed! System is ready for use.")
            sys.exit(0)
        else:
            print("\nSome tests failed. Check the output above.")
            sys.exit(1)
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

