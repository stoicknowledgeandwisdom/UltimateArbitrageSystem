#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimizer Setup and Deployment Script
==============================================

Automated setup script for the Advanced AI Portfolio Optimizer system.
Installs dependencies, configures environment, and validates the installation.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_optimizer_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PortfolioOptimizerSetup:
    """Setup and deployment manager for the portfolio optimizer."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.ai_dir = self.base_dir / 'ai'
        self.ui_dir = self.base_dir / 'ui'
        self.backend_dir = self.ui_dir / 'backend'
        self.frontend_dir = self.ui_dir / 'frontend'
        
        self.required_packages = [
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scipy>=1.7.0',
            'scikit-learn>=1.0.0',
            'fastapi>=0.68.0',
            'uvicorn>=0.15.0',
            'websockets>=10.0',
            'pydantic>=1.8.0',
            'asyncio-extras>=1.3.0'
        ]
        
        self.optional_packages = [
            'matplotlib>=3.5.0',  # For visualization
            'plotly>=5.0.0',      # For interactive charts
            'dash>=2.0.0',        # For dashboard
            'redis>=4.0.0',       # For caching
            'celery>=5.2.0'       # For background tasks
        ]
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} detected ✓")
        return True
    
    def install_packages(self, packages, optional=False):
        """Install required Python packages."""
        package_type = "optional" if optional else "required"
        logger.info(f"Installing {package_type} packages...")
        
        success_count = 0
        failed_packages = []
        
        for package in packages:
            try:
                logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if result.returncode == 0:
                    logger.info(f"✓ Successfully installed {package}")
                    success_count += 1
                else:
                    logger.warning(f"✗ Failed to install {package}: {result.stderr}")
                    failed_packages.append(package)
                    
                    if not optional:
                        return False
                        
            except subprocess.TimeoutExpired:
                logger.error(f"✗ Timeout installing {package}")
                failed_packages.append(package)
                if not optional:
                    return False
            except Exception as e:
                logger.error(f"✗ Error installing {package}: {e}")
                failed_packages.append(package)
                if not optional:
                    return False
        
        if failed_packages and not optional:
            logger.error(f"Failed to install required packages: {failed_packages}")
            return False
        
        logger.info(f"Package installation complete: {success_count}/{len(packages)} succeeded")
        return True
    
    def create_directories(self):
        """Create necessary directories."""
        logger.info("Creating directory structure...")
        
        directories = [
            self.ai_dir,
            self.ui_dir,
            self.backend_dir,
            self.frontend_dir / 'src' / 'pages',
            self.base_dir / 'logs',
            self.base_dir / 'data',
            self.base_dir / 'tests',
            self.base_dir / 'config'
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created directory: {directory}")
            except Exception as e:
                logger.error(f"✗ Failed to create directory {directory}: {e}")
                return False
        
        return True
    
    def create_config_files(self):
        """Create default configuration files."""
        logger.info("Creating configuration files...")
        
        # Portfolio optimizer config
        optimizer_config = {
            "quantum_enabled": True,
            "ai_enhancement": True,
            "risk_tolerance": 0.15,
            "target_return": 0.12,
            "rebalance_frequency": "daily",
            "use_black_litterman": True,
            "use_regime_detection": True,
            "use_factor_models": True,
            "dynamic_constraints": True,
            "max_weight": 0.4,
            "min_weight": 0.01,
            "transaction_cost": 0.001,
            "optimization_method": "quantum_enhanced",
            "risk_model": "factor_based",
            "return_model": "ml_enhanced"
        }
        
        # API config
        api_config = {
            "host": "0.0.0.0",
            "port": 8001,
            "debug": False,
            "cors_origins": ["*"],
            "max_connections": 100,
            "websocket_timeout": 300
        }
        
        # Database config (if needed)
        db_config = {
            "type": "sqlite",
            "path": "data/portfolio_optimizer.db",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0
        }
        
        config_files = [
            (self.base_dir / 'config' / 'optimizer_config.json', optimizer_config),
            (self.base_dir / 'config' / 'api_config.json', api_config),
            (self.base_dir / 'config' / 'database_config.json', db_config)
        ]
        
        for config_path, config_data in config_files:
            try:
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"✓ Created config file: {config_path}")
            except Exception as e:
                logger.error(f"✗ Failed to create config file {config_path}: {e}")
                return False
        
        return True
    
    def validate_installation(self):
        """Validate the installation by running tests."""
        logger.info("Validating installation...")
        
        try:
            # Import core modules
            sys.path.append(str(self.ai_dir))
            from portfolio_quantum_optimizer import QuantumPortfolioOptimizer
            logger.info("✓ Core optimizer module imports successfully")
            
            # Test basic functionality
            import asyncio
            import numpy as np
            import pandas as pd
            
            async def validation_test():
                # Create a simple test
                config = {"quantum_enabled": True, "ai_enhancement": True}
                optimizer = QuantumPortfolioOptimizer(config)
                
                # Generate test data
                assets = ['TEST1', 'TEST2', 'TEST3']
                dates = pd.date_range('2023-01-01', periods=100, freq='D')
                returns_data = pd.DataFrame(
                    np.random.normal(0.001, 0.02, (100, 3)),
                    index=dates,
                    columns=assets
                )
                
                # Run optimization
                result = await optimizer.optimize_portfolio(
                    assets=assets,
                    returns_data=returns_data
                )
                
                # Validate result
                assert len(result.weights) == len(assets)
                assert abs(sum(result.weights.values()) - 1.0) < 1e-6
                
                return True
            
            # Run the validation test
            success = asyncio.run(validation_test())
            
            if success:
                logger.info("✓ Installation validation passed")
                return True
            else:
                logger.error("✗ Installation validation failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}")
            return False
    
    def create_startup_scripts(self):
        """Create startup scripts for easy deployment."""
        logger.info("Creating startup scripts...")
        
        # API server startup script
        api_script = '''#!/usr/bin/env python3
"""Start the Portfolio Optimizer API server."""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "ui" / "backend"
sys.path.append(str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    from portfolio_optimizer_api import app
    
    print("Starting Portfolio Optimizer API Server...")
    print("API will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
'''
        
        # Test runner script
        test_script = '''#!/usr/bin/env python3
"""Run comprehensive tests for the Portfolio Optimizer."""

import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    from test_portfolio_optimizer import run_comprehensive_tests
    
    print("Running Portfolio Optimizer Test Suite...")
    success = asyncio.run(run_comprehensive_tests())
    
    if success:
        print("\n🎉 All tests passed! System is ready for use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
'''
        
        scripts = [
            (self.base_dir / 'start_api_server.py', api_script),
            (self.base_dir / 'run_tests.py', test_script)
        ]
        
        for script_path, script_content in scripts:
            try:
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Make executable on Unix-like systems
                if os.name != 'nt':
                    os.chmod(script_path, 0o755)
                
                logger.info(f"✓ Created startup script: {script_path}")
            except Exception as e:
                logger.error(f"✗ Failed to create script {script_path}: {e}")
                return False
        
        return True
    
    def generate_documentation(self):
        """Generate deployment documentation."""
        logger.info("Generating documentation...")
        
        readme_content = '''# Advanced AI Portfolio Optimizer

Quantum-enhanced portfolio optimization system with AI-driven insights.

## Features

- 🚀 Quantum-enhanced optimization algorithms
- 🤖 AI-driven return forecasting and risk modeling
- 📊 Real-time portfolio analytics and monitoring
- 🔄 Automated rebalancing with transaction cost optimization
- 📈 Advanced performance metrics and risk analysis
- 🌐 RESTful API with WebSocket support
- 📱 Modern React-based frontend interface

## Quick Start

### 1. Installation

```bash
# Run the setup script
python setup_portfolio_optimizer.py
```

### 2. Run Tests

```bash
# Validate the installation
python run_tests.py
```

### 3. Start API Server

```bash
# Start the backend API
python start_api_server.py
```

### 4. Access the System

- API Documentation: http://localhost:8001/docs
- API Health Check: http://localhost:8001/health
- WebSocket: ws://localhost:8001/ws

## API Endpoints

### Portfolio Optimization
```
POST /optimize
```
Optimize portfolio allocation using quantum-enhanced AI algorithms.

### Portfolio Metrics
```
POST /metrics
```
Calculate comprehensive portfolio performance metrics.

### Risk Analysis
```
POST /risk-analysis
```
Perform advanced risk analysis including VaR and stress testing.

### Rebalancing
```
POST /rebalance
```
Calculate optimal rebalancing trades with transaction costs.

## Configuration

Configuration files are located in the `config/` directory:

- `optimizer_config.json`: Core optimizer settings
- `api_config.json`: API server configuration
- `database_config.json`: Database and caching settings

## Architecture

```
Portfolio Optimizer System
├── ai/
│   └── portfolio_quantum_optimizer.py  # Core optimization engine
├── ui/
│   ├── backend/
│   │   └── portfolio_optimizer_api.py  # FastAPI backend
│   └── frontend/
│       └── src/pages/
│           └── OneClickIncomeMaximizer.js  # React frontend
├── config/                             # Configuration files
├── logs/                              # Log files
├── data/                              # Data storage
└── tests/                             # Test files
```

## Advanced Features

### Quantum Enhancement
The system uses quantum-inspired algorithms to explore multiple optimization scenarios simultaneously, providing superior portfolio allocations.

### AI-Driven Forecasting
Machine learning models analyze market patterns and generate return forecasts with confidence intervals.

### Real-Time Risk Management
Continuous monitoring of portfolio risk with automatic alerts and rebalancing recommendations.

### Multi-Objective Optimization
Simultaneous optimization for return, risk, diversification, and transaction costs.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Connection Issues**: Check if the server is running on port 8001
3. **Optimization Failures**: Verify input data format and constraints
4. **Performance Issues**: Consider reducing the number of assets or optimization iterations

### Support

For technical support or questions:
1. Check the logs in the `logs/` directory
2. Run the test suite to identify issues
3. Review the API documentation at `/docs`

## License

ProprietaryPortfolio Optimization System - All Rights Reserved
'''
        
        try:
            with open(self.base_dir / 'README.md', 'w') as f:
                f.write(readme_content)
            logger.info("✓ Generated README.md")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to generate documentation: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process."""
        logger.info("Starting Portfolio Optimizer Setup...")
        logger.info("=" * 50)
        
        setup_steps = [
            ("Python Version Check", self.check_python_version),
            ("Directory Creation", self.create_directories),
            ("Required Packages", lambda: self.install_packages(self.required_packages, False)),
            ("Optional Packages", lambda: self.install_packages(self.optional_packages, True)),
            ("Configuration Files", self.create_config_files),
            ("Startup Scripts", self.create_startup_scripts),
            ("Documentation", self.generate_documentation),
            ("Installation Validation", self.validate_installation)
        ]
        
        failed_steps = []
        
        for step_name, step_function in setup_steps:
            logger.info(f"\nExecuting: {step_name}")
            try:
                if step_function():
                    logger.info(f"✓ {step_name} completed successfully")
                else:
                    logger.error(f"✗ {step_name} failed")
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"✗ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SETUP SUMMARY")
        logger.info("=" * 50)
        
        if failed_steps:
            logger.error(f"Setup completed with {len(failed_steps)} failed steps:")
            for step in failed_steps:
                logger.error(f"  - {step}")
            logger.error("\nPlease review the errors above and retry.")
            return False
        else:
            logger.info("✅ Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Run tests: python run_tests.py")
            logger.info("2. Start API: python start_api_server.py")
            logger.info("3. Access docs: http://localhost:8001/docs")
            return True

if __name__ == "__main__":
    setup = PortfolioOptimizerSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎉 Portfolio Optimizer setup completed successfully!")
        print("📚 Check README.md for usage instructions.")
    else:
        print("\n❌ Setup failed. Check the logs for details.")
        sys.exit(1)

