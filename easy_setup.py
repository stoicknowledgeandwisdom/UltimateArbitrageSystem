#!/usr/bin/env python3
"""
Ultimate Arbitrage System - One-Click Automation Enhancement

This script provides a simple, guided setup process for the Ultimate Arbitrage System,
automating all technical aspects while only requiring minimal user input for
critical configuration options like API keys. The script handles:

1. System environment preparation
2. Dependency installation
3. Database initialization
4. Cloud resource distribution
5. Smart contract deployment
6. Strategy optimization
7. System launch and monitoring

Usage:
    python easy_setup.py [--mode=<mode>] [--config=<config_file>]

Options:
    --mode=<mode>         Setup mode: 'full' (default), 'update', or 'minimal'
    --config=<config_file> Path to custom configuration file
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import platform
import subprocess
import webbrowser
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import getpass
import secrets
import socket
import threading
import concurrent.futures
import pkg_resources
import re
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EasySetup")
os.makedirs("logs", exist_ok=True)

# System constants
VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = "config/system_config.json"
DEFAULT_ENV_PATH = ".env"
MIN_PYTHON_VERSION = (3, 8)
DATABASE_INIT_TIMEOUT = 300  # 5 minutes
FLASH_LOAN_DEPLOYMENT_TIMEOUT = 600  # 10 minutes
CLOUD_SETUP_TIMEOUT = 900  # 15 minutes
SYSTEM_LAUNCH_TIMEOUT = 120  # 2 minutes


class EasySetup:
    """Main class for the Ultimate Arbitrage System one-click setup."""
    
    def __init__(self, mode: str = "full", config_path: str = DEFAULT_CONFIG_PATH):
        """Initialize the setup system.
        
        Args:
            mode: Setup mode ('full', 'update', or 'minimal')
            config_path: Path to configuration file
        """
        self.mode = mode
        self.config_path = config_path
        self.config = {}
        self.env_vars = {}
        self.system_info = self._detect_system_info()
        self.deploy_processes = []
        self.progress_thread = None
        self.stop_progress = False
        
        # Create necessary directories
        os.makedirs("config", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/historical", exist_ok=True)
        os.makedirs("data/metrics", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        logger.info(f"EasySetup initialized in {mode} mode")
        logger.info(f"System information: {self.system_info}")
        
    def _detect_system_info(self) -> Dict[str, Any]:
        """Detect and collect system information.
        
        Returns:
            Dictionary containing system information
        """
        info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or "Unknown",
            "hostname": socket.gethostname(),
            "cores": os.cpu_count() or 0,
            "memory_gb": 0,
            "network_speed": 0,
            "disk_space_gb": 0
        }
        
        # Try to get memory information
        try:
            if info["os"] == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    mem_info = f.read()
                    mem_total = re.search(r'MemTotal:\s+(\d+)', mem_info)
                    if mem_total:
                        info["memory_gb"] = round(int(mem_total.group(1)) / 1024 / 1024, 2)
            elif info["os"] == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]
                memory_status = MEMORYSTATUSEX()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
                info["memory_gb"] = round(memory_status.ullTotalPhys / (1024**3), 2)
            elif info["os"] == "Darwin":  # macOS
                memory_cmd = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                if memory_cmd.returncode == 0:
                    mem_bytes = int(memory_cmd.stdout.split(':')[1].strip())
                    info["memory_gb"] = round(mem_bytes / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get memory information: {e}")
        
        # Try to get disk space
        try:
            if info["os"] in ["Linux", "Darwin"]:
                disk_cmd = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
                if disk_cmd.returncode == 0:
                    lines = disk_cmd.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        if len(parts) >= 2:
                            size_str = parts[1]
                            if size_str.endswith('G'):
                                info["disk_space_gb"] = float(size_str[:-1])
            elif info["os"] == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p('.'), None, None, ctypes.pointer(free_bytes))
                info["disk_space_gb"] = round(free_bytes.value / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get disk space information: {e}")
        
        # Estimate network speed (simplified)
        try:
            start = time.time()
            urllib.request.urlopen('https://www.google.com', timeout=2)
            elapsed = time.time() - start
            info["network_speed"] = round(1 / elapsed * 100, 2)  # Rough estimate in Mbps
        except:
            info["network_speed"] = 0
            
        return info
        
    def _check_python_version(self) -> bool:
        """Check if the Python version meets requirements.
        
        Returns:
            True if Python version is sufficient, False otherwise
        """
        current = sys.version_info[:2]
        if current < MIN_PYTHON_VERSION:
            logger.error(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher is required")
            return False
        return True
    
    def _install_dependencies(self) -> bool:
        """Install all required dependencies.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._start_progress("Installing dependencies")
            
            # Define core dependencies
            core_deps = [
                "numpy>=1.20.0", "pandas>=1.2.0", "scipy>=1.6.0",
                "requests>=2.25.0", "aiohttp>=3.7.3", "websockets>=9.1",
                "web3>=5.20.0", "ccxt>=1.50.0", "cryptography>=3.4.0",
                "sqlalchemy>=1.4.0", "psycopg2-binary>=2.8.6", "redis>=3.5.3",
                "pyyaml>=5.4.0", "python-dotenv>=0.15.0", "tenacity>=7.0.0",
                "dash>=2.0.0", "plotly>=5.0.0", "networkx>=2.5",
                "scikit-learn>=0.24.0", "tensorflow>=2.4.0", "torch>=1.8.0",
            ]
            
            # Add dependencies for different operating systems
            if self.system_info["os"] == "Windows":
                core_deps.append("pywin32>=300")
            
            # Calculate the number of worker processes based on available cores
            num_workers = max(1, min(self.system_info["cores"] // 2, 4))
            
            # Install dependencies in batches with multiple processes
            logger.info(f"Installing {len(core_deps)} dependencies using {num_workers} workers")
            
            batches = [core_deps[i:i + 5] for i in range(0, len(core_deps), 5)]
            for batch in batches:
                install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + batch
                logger.info(f"Running: {' '.join(install_cmd)}")
                process = subprocess.run(install_cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    logger.error(f"Failed to install dependencies: {process.stderr}")
                    logger.error(f"Command: {' '.join(install_cmd)}")
                    raise Exception(f"Dependency installation failed: {process.stderr}")
                
                logger.info(f"Successfully installed batch: {batch}")
            
            # Install PyTorch with appropriate CUDA version if GPU is available
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA is available, installing appropriate PyTorch version")
                    cuda_version = torch.version.cuda
                    cuda_cmd = [sys.executable, "-m", "pip", "install", f"torch=={torch.__version__}+cu{cuda_version.replace('.', '')}"]
                    subprocess.run(cuda_cmd, capture_output=True, text=True)
            except ImportError:
                logger.info("PyTorch not found, skipping CUDA check")
            except Exception as e:
                logger.warning(f"Error checking CUDA availability: {e}")
            
            self._stop_progress("Dependencies installed successfully")
            
            # Save requirements to file
            with open("requirements.txt", "w") as f:
                f.write("\n".join(core_deps))
            
            return True
        except Exception as e:
            self._stop_progress(f"Failed to install dependencies: {e}")
            logger.error(f"Failed to install dependencies: {e}", exc_info=True)
            return False
    
    def _configure_environment(self) -> bool:
        """Set up environment variables and configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._start_progress("Configuring environment")
            
            # Load existing configuration if available
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded existing configuration from {self.config_path}")
            else:
                # Create default configuration
                self.config = {
                    "system_name": "Ultimate Arbitrage System",
                    "version": VERSION,
                    "log_level": "INFO",
                    "max_parallel_executions": min(self.system_info["cores"], 8),
                    "execution_delay": 0.5,
                    "monitoring_interval": 10,
                    "data_backup_interval": 3600,
                    "health_check_interval": 60,
                    "exchanges": [],
                    "strategies": [],
                    "risk_parameters": {
                        "max_exposure_per_trade": 0.02,
                        "max_daily_loss": 0.05,
                        "max_open_positions": 3,
                        "min_profit_threshold": 0.005,
                        "slippage_tolerance": 0.002,
                        "max_position_duration": 3600
                    },
                    "cloud_distribution": {
                        "enabled": True,
                        "providers": ["oracle", "aws", "gcp", "azure"],
                        "auto_scaling": True,
                        "resource_allocation": "auto"
                    },
                    "profit_optimization": {
                        "enabled": True,
                        "reallocation_interval": 3600,
                        "performance_metrics_weight": {
                            "profit": 0.4,
                            "win_rate": 0.2,
                            "sharpe_ratio": 0.3,
                            "execution_time": 0.1
                        }
                    },
                    "monitoring": {
                        "dashboard_port": 8050,
                        "enable_email_alerts": False,
                        "enable_telegram_alerts": False
                    },
                    "flash_loans": {
                        "enabled": True,
                        "providers": ["aave", "dydx", "compound"],
                        "max_loan_amount": "auto",
                        "safety_module": True
                    }
                }
                logger.info("Created default configuration")
            
            # Generate a secure secret key for the application if not present
            if "secret_key" not in self.config:
                self.config["secret_key"] = secrets.token_hex(32)
                logger.info("Generated new secret key")
                
            # Auto-configure based on system info
            self._auto_configure_based_on_system()
            
            # Configure exchanges (minimal interactive input required)
            if not self.config.get("exchanges"):
                print("\n===== Exchange Configuration =====")
                print("The system needs at least one exchange API connection to function.")
                print("You can set this up now, or skip and add it later.")
                
                setup_exchange = input("Do you want to set up an exchange now? (y/n) [y]: ").lower() != "n"
                
                if setup_exchange:
                    exchange_id = input("Enter exchange ID (e.g., binance, coinbase, kraken): ").lower()
                    print(f"Setting up {exchange_id.capitalize()}...")
                    print("Please create an API key with read and trade permissions.")
                    
                    api_key = getpass.getpass("Enter API Key: ")
                    api_secret = getpass.getpass("Enter API Secret: ")
                    
                    self.config["exchanges"].append({
                        "id": exchange_id,
                        "type": exchange_id,
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "enabled": True,
                        "test_mode": False,
                        "test_safe": True
                    })
                    logger.info(f"Added exchange configuration for {exchange_id}")
            
            # Save configuration
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
            
            # Create .env file with environment variables
            self.env_vars = {
                "SYSTEM_VERSION": VERSION,
                "CONFIG_PATH": self.config_path,
                "LOG_LEVEL": self.config.get("log_level", "INFO"),
                "SECRET_KEY": self.config.get("secret_key", ""),
                "DASHBOARD_PORT": str(self.config.get("monitoring", {}).get("dashboard_port", 8050)),
                "TEST_MODE": "0"  # Default to production mode
            }
            
            env_content = "\n".join([f"{key}={value}" for key, value in self.env_vars.items()])
            with open(DEFAULT_ENV_PATH, "w") as f:
                f.write(env_content)
            logger.info(f"Created environment file at {DEFAULT_ENV_PATH}")
            
            self._stop_progress("Environment configured successfully")
            return True
            
        except Exception as e:
            self._stop_progress(f"Failed to configure environment: {e}")
            logger.error(f"Failed to configure environment: {e}", exc_info=True)
            return False
    
    def _auto_configure_based_on_system(self):
        """Automatically adjust configuration based on system capabilities."""
        # Adjust parallel execution based on CPU cores
        if self.system_info["cores"] > 0:
            recommended_parallel = max(1, min(self.system_info["cores"] - 1, 8))
            self.config["max_parallel_executions"] = recommended_parallel
            logger.info(f"Set max_parallel_executions to {recommended_parallel} based on {self.system_info['cores']} CPU cores")
        
        # Adjust memory-intensive operations based on available RAM
        if self.system_info["memory_gb"] > 0:
            if self.system_info["memory_gb"] < 4:
                # Low memory mode
                self.config["memory_optimization"] = "low"
                logger.info("Enabling low memory optimization mode")
            elif self.system_info["memory_gb"] >= 16:
                # High memory mode - can use more aggressive caching
                self.config["memory_optimization"] = "high"
                logger.info("Enabling high memory optimization mode")
            else:
                # Standard memory mode
                self.config["memory_optimization"] = "standard"
                logger.info("Using standard memory optimization mode")
        
        # Adjust network-related parameters based on network speed
        if self.system_info["network_speed"] > 0:
            if self.system_info["network_speed"] < 10:
                # Slow network - increase timeouts, reduce concurrent connections
                self.config["network_timeouts"] = {
                    "connect": 10.0,
                    "read": 30.0,
                    "write": 30.0
                }
                self.config["max_connections_per_exchange"] = 3
                logger.info("Configuring for slow network connection")
            elif self.system_info["network_speed"] >= 100:
                # Fast network - optimize for performance
                self.config["network_timeouts"] = {
                    "connect": 3.0,
                    "read": 10.0,
                    "write": 10.0
                }
                self.config["max_connections_per_exchange"] = 10
                logger.info("Configuring for high-speed network connection")
            else:
                # Standard network
                self.config["network_timeouts"] = {
                    "connect": 5.0,
                    "read": 20.0,
                    "write": 20.0
                }
                self.config["max_connections_per_exchange"] = 5
                logger.info("Configuring for standard network connection")
    
    def _setup_database(self) -> bool:
        """Set up and initialize the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._start_progress("Setting up database")
            
            # Determine database configuration from system setup
            db_config = self.config.get("database", {})
            if not db_config:
                db_config = {
                    "type": "sqlite" if self.mode == "minimal" else "postgresql",
                    "host": "localhost",
                    "port": 5432,
                    "name": "ultimate_arbitrage",
                    "user": "postgres",
                    "password": secrets.token_urlsafe(12),
                    "backup_interval": 86400  # Daily backups
                }
                self.config["database"] = db_config
                logger.info(f"Created default database configuration for {db_config['type']}")
            
            # For SQLite (minimal mode), just create the database directory
            if db_config["type"] == "sqlite":
                os.makedirs("data/database", exist_ok=True)
                db_path = "data/database/arbitrage.db"
                db_config["path"] = db_path
                logger.info(f"Using SQLite database at {db_path}")
                
                # Create tables using SQLAlchemy
                conn_str = f"sqlite:///{db_path}"
                self._create_database_tables(conn_str)
                
            # For PostgreSQL (full mode), initialize the PostgreSQL server
            else:
                # Check if PostgreSQL is installed
                if self._is_command_available("psql"):
                    logger.info("PostgreSQL is installed")
                else:
                    logger.warning("PostgreSQL is not installed. Installing...")
                    if not self._install_postgresql():
                        raise Exception("Failed to install PostgreSQL")
                
                # Check if PostgreSQL server is running
                if not self._is_postgresql_running():
                    logger.warning("PostgreSQL server is not running. Starting...")
                    if not self._start_postgresql():
                        raise Exception("Failed to start PostgreSQL server")
                
                # Create PostgreSQL database and user if needed
                self._create_postgresql_database(
                    db_config["name"],
                    db_config["user"],
                    db_config["password"]
                )
                
                # Create tables using SQLAlchemy
                conn_str = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
                self._create_database_tables(conn_str)
            
            # Update configuration with database settings
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            self._stop_progress("Database setup completed successfully")
            return True
            
        except Exception as e:
            self._stop_progress(f"Failed to set up database: {e}")
            logger.error(f"Failed to set up database: {e}", exc_info=True)
            return False
    
    def _is_command_available(self, command: str) -> bool:
        """Check if a command is available in the system.
        
        Args:
            command: Command to check
            
        Returns:
            True if command exists, False otherwise
        """
        try:
            if self.system_info["os"] == "Windows":
                result = subprocess.run(f"where {command}", shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(f"which {command}", shell=True, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _install_postgresql(self) -> bool:
        """Install PostgreSQL database server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system_info["os"] == "Windows":
                # On Windows, guide user to manual installation
                print("\nPostgreSQL installation required.")
                print("Please download and install PostgreSQL from https://www.postgresql.org/download/windows/")
                print("Use the following settings during installation:")
                print("- Password: Use a strong password")
                print("- Port: 5432")
                print("- Locale: Default")
                
                proceed = input("Press Enter when PostgreSQL installation is complete, or type 'skip' to skip: ")
                return proceed.lower() != "skip"
                
            elif self.system_info["os"] == "Linux":
                # On Linux, use apt or yum depending on distribution
                if os.path.exists("/etc/debian_version"):
                    # Debian/Ubuntu
                    subprocess.run("sudo apt-get update", shell=True, check=True)
                    subprocess.run("sudo apt-get install -y postgresql postgresql-contrib", shell=True, check=True)
                elif os.path.exists("/etc/redhat-release"):
                    # CentOS/RHEL
                    subprocess.run("sudo yum install -y postgresql-server postgresql-contrib", shell=True, check=True)
                    subprocess.run("sudo postgresql-setup initdb", shell=True, check=True)
                    subprocess.run("sudo systemctl enable postgresql", shell=True, check=True)
                else:
                    logger.warning("Unsupported Linux distribution")
                    return False
                return True
                
            elif self.system_info["os"] == "Darwin":
                # macOS - use Homebrew
                if not self._is_command_available("brew"):
                    logger.warning("Homebrew not installed. Please install Homebrew first: https://brew.sh/")
                    return False
                subprocess.run("brew install postgresql", shell=True, check=True)
                return True
            
            logger.warning(f"Unsupported OS: {self.system_info['os']}")
            return False
            
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to install PostgreSQL: {e}")
            return False
    
    def _is_postgresql_running(self) -> bool:
        """Check if PostgreSQL server is running.
        
        Returns:
            True if running, False otherwise
        """
        try:
            if self.system_info["os"] == "Windows":
                result = subprocess.run(
                    'sc query postgresql | findstr "RUNNING"',
                    shell=True, capture_output=True, text=True
                )
                return "RUNNING" in result.stdout
            elif self.system_info["os"] in ["Linux", "Darwin"]:
                result = subprocess.run(
                    "pg_isready",
                    shell=True, capture_output=True, text=True
                )
                return result.returncode == 0
            return False
        except Exception:
            return False
    
    def _start_postgresql(self) -> bool:
        """Start PostgreSQL server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system_info["os"] == "Windows":
                # On Windows, use services control
                subprocess.run('net start postgresql', shell=True, check=True)
            elif self.system_info["os"] == "Linux":
                # On Linux, use systemctl
                subprocess.run('sudo systemctl start postgresql', shell=True, check=True)
            elif self.system_info["os"] == "Darwin":
                # On macOS, use brew services
                subprocess.run('brew services start postgresql', shell=True, check=True)
            return self._is_postgresql_running()
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to start PostgreSQL: {e}")
            return False
    
    def _create_postgresql_database(self, db_name: str, db_user: str, db_password: str) -> bool:
        """Create PostgreSQL database and user.
        
        Args:
            db_name: Database name
            db_user: Database user
            db_password: Database password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if database exists
            db_exists_cmd = f'psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname=\'{db_name}\'"'
            result = subprocess.run(db_exists_cmd, shell=True, capture_output=True, text=True)
            
            if "1" not in result.stdout:
                # Create database
                create_db_cmd = f'psql -U postgres -c "CREATE DATABASE {db_name}"'
                subprocess.run(create_db_cmd, shell=True, check=True)
                logger.info(f"Created database: {db_name}")
            
            # Check if user exists
            
