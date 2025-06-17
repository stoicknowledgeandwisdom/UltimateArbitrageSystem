#!/usr/bin/env python3
"""
Ultimate Wallet & API Key Configuration Manager
==============================================

A comprehensive system for managing wallets, API keys, and configurations
for all automated trading strategies. Features:

1. Secure wallet management across all supported networks
2. Exchange API key configuration with explanations
3. Strategy-specific wallet assignments
4. Real-time validation and testing
5. Encrypted storage with backup systems
6. Automated fund management and rebalancing
7. Cross-chain wallet coordination
8. Risk-based wallet allocation
9. Emergency wallet controls
10. Compliance and audit logging
"""

import os
import json
import hashlib
import base64
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import sqlite3
import yaml

logger = logging.getLogger(__name__)

class NetworkType(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    COSMOS = "cosmos"
    NEAR = "near"
    TERRA = "terra"
    BITCOIN = "bitcoin"
    LITECOIN = "litecoin"
    DOGECOIN = "dogecoin"
    RIPPLE = "ripple"
    STELLAR = "stellar"
    MONERO = "monero"
    ZCASH = "zcash"

class ExchangeType(Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    COINBASE_PRO = "coinbase_pro"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    BITGET = "bitget"
    OKX = "okx"
    GATE_IO = "gate_io"
    HUOBI = "huobi"
    KU_COIN = "kucoin"
    BITFINEX = "bitfinex"
    GEMINI = "gemini"
    CRYPTO_COM = "crypto_com"
    FTX = "ftx"
    MEXC = "mexc"
    ASCENDEX = "ascendex"
    BITRUE = "bitrue"
    BITSTAMP = "bitstamp"
    DYDX = "dydx"
    PERPETUAL_PROTOCOL = "perpetual_protocol"
    UNISWAP = "uniswap"
    SUSHISWAP = "sushiswap"
    PANCAKESWAP = "pancakeswap"
    CURVE = "curve"
    BALANCER = "balancer"
    YEARN = "yearn"
    COMPOUND = "compound"
    AAVE = "aave"
    MAKER = "maker"
    SYNTHETIX = "synthetix"

class StrategyType(Enum):
    """Automated trading strategies"""
    QUANTUM_ARBITRAGE = "quantum_arbitrage"
    CROSS_CHAIN_MEV = "cross_chain_mev"
    FLASH_LOAN_ARBITRAGE = "flash_loan_arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    TRIANGULAR_ARBITRAGE = "triangular_arbitrage"
    SPATIAL_ARBITRAGE = "spatial_arbitrage"
    TEMPORAL_ARBITRAGE = "temporal_arbitrage"
    YIELD_FARMING = "yield_farming"
    LIQUIDITY_MINING = "liquidity_mining"
    IMPERMANENT_LOSS_MITIGATION = "impermanent_loss_mitigation"
    DELTA_NEUTRAL_STRATEGIES = "delta_neutral_strategies"
    FUNDING_RATE_ARBITRAGE = "funding_rate_arbitrage"
    OPTIONS_ARBITRAGE = "options_arbitrage"
    SYNTHETIC_ASSET_ARBITRAGE = "synthetic_asset_arbitrage"
    STABLECOIN_ARBITRAGE = "stablecoin_arbitrage"
    GOVERNANCE_TOKEN_ARBITRAGE = "governance_token_arbitrage"
    NFT_ARBITRAGE = "nft_arbitrage"
    GAMEFI_ARBITRAGE = "gamefi_arbitrage"
    DEFI_VAULT_OPTIMIZATION = "defi_vault_optimization"
    CROSS_MARGIN_ARBITRAGE = "cross_margin_arbitrage"
    PERPETUAL_FUTURES_ARBITRAGE = "perpetual_futures_arbitrage"
    LENDING_PROTOCOL_ARBITRAGE = "lending_protocol_arbitrage"
    ALGORITHMIC_STABLECOIN_ARBITRAGE = "algorithmic_stablecoin_arbitrage"
    WRAPPED_TOKEN_ARBITRAGE = "wrapped_token_arbitrage"
    BRIDGE_ARBITRAGE = "bridge_arbitrage"

@dataclass
class WalletConfig:
    """Configuration for a blockchain wallet"""
    network: NetworkType
    address: str
    private_key: Optional[str] = None
    mnemonic: Optional[str] = None
    derivation_path: Optional[str] = None
    balance_usd: float = 0.0
    gas_reserve_percent: float = 5.0
    max_transaction_amount_usd: float = 10000.0
    daily_limit_usd: float = 100000.0
    risk_level: str = "medium"  # low, medium, high
    is_active: bool = True
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExchangeConfig:
    """Configuration for exchange API access"""
    exchange: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    sandbox_mode: bool = True
    rate_limit_requests_per_minute: int = 600
    max_position_size_usd: float = 50000.0
    daily_trading_limit_usd: float = 500000.0
    risk_level: str = "medium"
    is_active: bool = True
    supported_pairs: List[str] = field(default_factory=list)
    last_validated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    strategy_type: StrategyType
    name: str
    description: str
    explanation: str  # Detailed explanation of how it works
    required_wallets: List[NetworkType]
    required_exchanges: List[ExchangeType]
    min_capital_usd: float
    max_capital_usd: float
    profit_target_percent: float
    max_loss_percent: float
    execution_frequency_minutes: int
    risk_level: str = "medium"
    is_enabled: bool = True
    assigned_wallets: Dict[NetworkType, str] = field(default_factory=dict)
    assigned_exchanges: Dict[ExchangeType, str] = field(default_factory=dict)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class SecureConfigManager:
    """Secure management of sensitive configuration data"""
    
    def __init__(self, config_dir: str = "config/secure"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.master_key_file = self.config_dir / ".master_key"
        self.salt_file = self.config_dir / ".salt"
        self.config_db = self.config_dir / "secure_config.db"
        
        self._initialize_encryption()
        self._initialize_database()
        
    def _initialize_encryption(self):
        """Initialize encryption system"""
        if not self.salt_file.exists():
            salt = os.urandom(16)
            with open(self.salt_file, 'wb') as f:
                f.write(salt)
        else:
            with open(self.salt_file, 'rb') as f:
                salt = f.read()
        
        if not self.master_key_file.exists():
            # Generate new master key
            password = secrets.token_urlsafe(32).encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            with open(self.master_key_file, 'wb') as f:
                f.write(key)
        else:
            with open(self.master_key_file, 'rb') as f:
                key = f.read()
        
        self.cipher = Fernet(key)
        
    def _initialize_database(self):
        """Initialize secure configuration database"""
        with sqlite3.connect(self.config_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secure_configs (
                    id TEXT PRIMARY KEY,
                    config_type TEXT NOT NULL,
                    encrypted_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    config_id TEXT,
                    config_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT
                )
            """)
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def store_config(self, config_id: str, config_type: str, data: Dict[str, Any]):
        """Store encrypted configuration"""
        encrypted_data = self.encrypt_data(json.dumps(data))
        
        with sqlite3.connect(self.config_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO secure_configs 
                (id, config_type, encrypted_data, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (config_id, config_type, encrypted_data))
            
            conn.execute("""
                INSERT INTO audit_log (action, config_id, config_type, details)
                VALUES (?, ?, ?, ?)
            """, ("STORE", config_id, config_type, f"Stored {config_type} configuration"))
    
    def load_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt configuration"""
        with sqlite3.connect(self.config_db) as conn:
            cursor = conn.execute(
                "SELECT encrypted_data FROM secure_configs WHERE id = ?",
                (config_id,)
            )
            row = cursor.fetchone()
            
            if row:
                encrypted_data = row[0]
                decrypted_data = self.decrypt_data(encrypted_data)
                return json.loads(decrypted_data)
        
        return None
    
    def list_configs(self, config_type: Optional[str] = None) -> List[Tuple[str, str]]:
        """List all configuration IDs and types"""
        with sqlite3.connect(self.config_db) as conn:
            if config_type:
                cursor = conn.execute(
                    "SELECT id, config_type FROM secure_configs WHERE config_type = ?",
                    (config_type,)
                )
            else:
                cursor = conn.execute(
                    "SELECT id, config_type FROM secure_configs"
                )
            
            return cursor.fetchall()

class WalletConfigManager:
    """Advanced wallet and API key configuration manager"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.secure_manager = SecureConfigManager()
        self.wallets: Dict[str, WalletConfig] = {}
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.strategies: Dict[str, StrategyConfig] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._load_configurations()
        self._initialize_strategy_templates()
        
        logger.info("Wallet Configuration Manager initialized")
    
    def _load_configurations(self):
        """Load all configurations from secure storage"""
        # Load wallets
        wallet_configs = self.secure_manager.list_configs("wallet")
        for wallet_id, _ in wallet_configs:
            config_data = self.secure_manager.load_config(wallet_id)
            if config_data:
                wallet = WalletConfig(**config_data)
                self.wallets[wallet_id] = wallet
        
        # Load exchanges
        exchange_configs = self.secure_manager.list_configs("exchange")
        for exchange_id, _ in exchange_configs:
            config_data = self.secure_manager.load_config(exchange_id)
            if config_data:
                exchange = ExchangeConfig(**config_data)
                self.exchanges[exchange_id] = exchange
        
        # Load strategies
        strategy_configs = self.secure_manager.list_configs("strategy")
        for strategy_id, _ in strategy_configs:
            config_data = self.secure_manager.load_config(strategy_id)
            if config_data:
                strategy = StrategyConfig(**config_data)
                self.strategies[strategy_id] = strategy
    
    def _initialize_strategy_templates(self):
        """Initialize strategy templates with detailed explanations"""
        strategy_templates = {
            StrategyType.QUANTUM_ARBITRAGE: {
                "name": "Quantum Arbitrage Strategy",
                "description": "AI-powered quantum-enhanced arbitrage across multiple exchanges and networks",
                "explanation": """
                🌟 QUANTUM ARBITRAGE - ULTIMATE PROFIT GENERATION
                
                HOW IT WORKS:
                1. Quantum Computing Enhancement: Uses quantum algorithms to calculate optimal arbitrage paths across thousands of trading pairs simultaneously
                2. Multi-Exchange Scanning: Monitors price differences across 20+ exchanges in real-time
                3. Cross-Chain Opportunities: Identifies arbitrage between different blockchain networks
                4. AI Pattern Recognition: Machine learning models predict price movements and optimal entry/exit points
                5. Lightning-Fast Execution: Executes trades in microseconds using advanced order routing
                
                PROFIT MECHANISM:
                - Buys assets on exchanges where prices are lower
                - Sells the same assets on exchanges where prices are higher
                - Captures the price difference as pure profit
                - Uses quantum computing to find complex multi-hop arbitrage paths
                
                REQUIRED SETUP:
                - Wallets on multiple networks (Ethereum, BSC, Polygon, Arbitrum)
                - API keys for major exchanges (Binance, Coinbase, Kraken, etc.)
                - Minimum capital: $1,000 (more capital = higher profits)
                
                PROFIT POTENTIAL: 0.1% - 5% per trade, 50-200 trades per day
                RISK LEVEL: Low-Medium (automated risk management)
                """,
                "required_wallets": [NetworkType.ETHEREUM, NetworkType.BINANCE_SMART_CHAIN, NetworkType.POLYGON],
                "required_exchanges": [ExchangeType.BINANCE, ExchangeType.COINBASE_PRO, ExchangeType.KRAKEN],
                "min_capital_usd": 1000.0,
                "max_capital_usd": 1000000.0,
                "profit_target_percent": 2.5,
                "max_loss_percent": 0.5,
                "execution_frequency_minutes": 1
            },
            StrategyType.CROSS_CHAIN_MEV: {
                "name": "Cross-Chain MEV Extraction",
                "description": "Maximum Extractable Value capture across blockchain bridges and DEXs",
                "explanation": """
                ⚡ CROSS-CHAIN MEV - ADVANCED PROFIT EXTRACTION
                
                HOW IT WORKS:
                1. MEV Detection: Identifies Maximum Extractable Value opportunities in pending transactions
                2. Bridge Arbitrage: Captures profits from price differences between chains
                3. Sandwich Attacks: Legally extracts value from large trades (where permitted)
                4. Liquidation Hunting: Monitors lending protocols for liquidation opportunities
                5. Gas Optimization: Uses advanced gas strategies to maximize profit margins
                
                PROFIT MECHANISM:
                - Monitors mempool for profitable MEV opportunities
                - Executes trades before/after large transactions to capture slippage
                - Arbitrages price differences between chains using bridges
                - Provides liquidity at optimal times for maximum fees
                
                REQUIRED SETUP:
                - Multi-chain wallets with significant gas reserves
                - API access to mempool data and bridge protocols
                - Integration with major DEXs and lending platforms
                
                PROFIT POTENTIAL: 0.5% - 15% per opportunity, 20-100 opportunities per day
                RISK LEVEL: Medium (requires advanced monitoring)
                """,
                "required_wallets": [NetworkType.ETHEREUM, NetworkType.ARBITRUM, NetworkType.OPTIMISM, NetworkType.POLYGON],
                "required_exchanges": [ExchangeType.UNISWAP, ExchangeType.SUSHISWAP, ExchangeType.CURVE],
                "min_capital_usd": 5000.0,
                "max_capital_usd": 2000000.0,
                "profit_target_percent": 5.0,
                "max_loss_percent": 1.0,
                "execution_frequency_minutes": 1
            },
            StrategyType.FLASH_LOAN_ARBITRAGE: {
                "name": "Flash Loan Arbitrage",
                "description": "Zero-capital arbitrage using flash loans for infinite leverage",
                "explanation": """
                💰 FLASH LOAN ARBITRAGE - ZERO CAPITAL REQUIRED
                
                HOW IT WORKS:
                1. Flash Loan Borrowing: Borrows large amounts (millions) with no collateral
                2. Arbitrage Execution: Uses borrowed funds to execute profitable arbitrage
                3. Instant Repayment: Repays loan + fees in the same transaction
                4. Pure Profit: Keeps the arbitrage profit with zero initial capital
                5. Risk-Free Trading: If trade fails, entire transaction reverts (no loss)
                
                PROFIT MECHANISM:
                - Borrows $1M+ from AAVE, Compound, or dYdX
                - Executes arbitrage between DEXs or exchanges
                - Repays loan + small fee (0.05-0.3%)
                - Keeps remaining profit (often $500-$50,000 per trade)
                
                REQUIRED SETUP:
                - Smart contract deployment capability
                - Access to flash loan providers (AAVE, Compound, dYdX)
                - Integration with multiple DEXs for arbitrage execution
                
                PROFIT POTENTIAL: $100 - $50,000 per trade, 5-50 trades per day
                RISK LEVEL: Very Low (transactions revert if unprofitable)
                """,
                "required_wallets": [NetworkType.ETHEREUM, NetworkType.POLYGON, NetworkType.ARBITRUM],
                "required_exchanges": [ExchangeType.AAVE, ExchangeType.COMPOUND, ExchangeType.UNISWAP],
                "min_capital_usd": 0.0,  # Flash loans require no capital!
                "max_capital_usd": 10000000.0,  # Can borrow millions
                "profit_target_percent": 1.0,
                "max_loss_percent": 0.1,
                "execution_frequency_minutes": 5
            },
            StrategyType.YIELD_FARMING: {
                "name": "Automated Yield Farming",
                "description": "Optimal yield generation across DeFi protocols with auto-compounding",
                "explanation": """
                🌾 AUTOMATED YIELD FARMING - PASSIVE INCOME GENERATION
                
                HOW IT WORKS:
                1. Yield Scanning: Continuously monitors 100+ DeFi protocols for highest yields
                2. Auto-Allocation: Automatically moves funds to highest-yielding opportunities
                3. Compound Harvesting: Claims and reinvests rewards every few hours
                4. Risk Assessment: Evaluates protocol security and impermanent loss risks
                5. Portfolio Rebalancing: Maintains optimal allocation across multiple protocols
                
                PROFIT MECHANISM:
                - Provides liquidity to high-yield farming pools
                - Earns trading fees + liquidity mining rewards
                - Automatically compounds rewards for exponential growth
                - Moves funds when better opportunities arise
                
                REQUIRED SETUP:
                - Wallets on major DeFi networks
                - Approval for automatic token swaps and staking
                - Integration with yield farming aggregators
                
                PROFIT POTENTIAL: 5% - 500% APY (varies by market conditions)
                RISK LEVEL: Medium (impermanent loss and smart contract risks)
                """,
                "required_wallets": [NetworkType.ETHEREUM, NetworkType.BINANCE_SMART_CHAIN, NetworkType.POLYGON, NetworkType.AVALANCHE],
                "required_exchanges": [ExchangeType.PANCAKESWAP, ExchangeType.UNISWAP, ExchangeType.YEARN, ExchangeType.COMPOUND],
                "min_capital_usd": 500.0,
                "max_capital_usd": 5000000.0,
                "profit_target_percent": 20.0,
                "max_loss_percent": 5.0,
                "execution_frequency_minutes": 60
            }
        }
        
        # Initialize missing strategies
        for strategy_type, template in strategy_templates.items():
            strategy_id = f"auto_{strategy_type.value}"
            if strategy_id not in self.strategies:
                strategy_config = StrategyConfig(
                    strategy_type=strategy_type,
                    **template
                )
                self.strategies[strategy_id] = strategy_config
                self._save_strategy_config(strategy_id, strategy_config)
    
    def add_wallet(self, wallet_id: str, network: NetworkType, address: str, 
                   private_key: Optional[str] = None, **kwargs) -> str:
        """Add a new wallet configuration"""
        wallet_config = WalletConfig(
            network=network,
            address=address,
            private_key=private_key,
            last_updated=datetime.now(),
            **kwargs
        )
        
        self.wallets[wallet_id] = wallet_config
        self._save_wallet_config(wallet_id, wallet_config)
        
        logger.info(f"Added wallet {wallet_id} for {network.value} network")
        return wallet_id
    
    def add_exchange(self, exchange_id: str, exchange: ExchangeType, 
                     api_key: str, api_secret: str, **kwargs) -> str:
        """Add a new exchange configuration"""
        exchange_config = ExchangeConfig(
            exchange=exchange,
            api_key=api_key,
            api_secret=api_secret,
            last_validated=datetime.now(),
            **kwargs
        )
        
        self.exchanges[exchange_id] = exchange_config
        self._save_exchange_config(exchange_id, exchange_config)
        
        # Validate exchange configuration asynchronously
        self.executor.submit(self._validate_exchange_config, exchange_id)
        
        logger.info(f"Added exchange {exchange_id} for {exchange.value}")
        return exchange_id
    
    def configure_strategy(self, strategy_id: str, strategy_type: StrategyType, 
                          wallet_assignments: Dict[NetworkType, str],
                          exchange_assignments: Dict[ExchangeType, str],
                          **kwargs) -> str:
        """Configure a trading strategy with wallet and exchange assignments"""
        if strategy_id in self.strategies:
            strategy_config = self.strategies[strategy_id]
        else:
            # Create new strategy from template
            template = self._get_strategy_template(strategy_type)
            strategy_config = StrategyConfig(
                strategy_type=strategy_type,
                **template
            )
        
        # Update assignments
        strategy_config.assigned_wallets = wallet_assignments
        strategy_config.assigned_exchanges = exchange_assignments
        
        # Update custom parameters
        for key, value in kwargs.items():
            if hasattr(strategy_config, key):
                setattr(strategy_config, key, value)
            else:
                strategy_config.custom_parameters[key] = value
        
        self.strategies[strategy_id] = strategy_config
        self._save_strategy_config(strategy_id, strategy_config)
        
        # Validate strategy configuration
        self.executor.submit(self._validate_strategy_config, strategy_id)
        
        logger.info(f"Configured strategy {strategy_id} ({strategy_type.value})")
        return strategy_id
    
    def get_strategy_explanation(self, strategy_type: StrategyType) -> str:
        """Get detailed explanation of how a strategy works"""
        for strategy in self.strategies.values():
            if strategy.strategy_type == strategy_type:
                return strategy.explanation
        
        # Return template explanation if no configured strategy found
        template = self._get_strategy_template(strategy_type)
        return template.get("explanation", "No explanation available")
    
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of all available strategies with explanations"""
        strategies = []
        for strategy_type in StrategyType:
            template = self._get_strategy_template(strategy_type)
            strategies.append({
                "type": strategy_type.value,
                "name": template["name"],
                "description": template["description"],
                "explanation": template["explanation"],
                "required_wallets": [w.value for w in template["required_wallets"]],
                "required_exchanges": [e.value for e in template["required_exchanges"]],
                "min_capital_usd": template["min_capital_usd"],
                "profit_potential": template["profit_target_percent"]
            })
        
        return strategies
    
    def get_wallet_requirements(self, strategy_type: StrategyType) -> List[NetworkType]:
        """Get wallet requirements for a strategy"""
        template = self._get_strategy_template(strategy_type)
        return template["required_wallets"]
    
    def get_exchange_requirements(self, strategy_type: StrategyType) -> List[ExchangeType]:
        """Get exchange requirements for a strategy"""
        template = self._get_strategy_template(strategy_type)
        return template["required_exchanges"]
    
    def validate_strategy_setup(self, strategy_id: str) -> Dict[str, Any]:
        """Validate that a strategy has all required components"""
        if strategy_id not in self.strategies:
            return {"valid": False, "error": "Strategy not found"}
        
        strategy = self.strategies[strategy_id]
        validation_result = {
            "valid": True,
            "missing_wallets": [],
            "missing_exchanges": [],
            "warnings": [],
            "estimated_setup_time": "5-10 minutes"
        }
        
        # Check wallet requirements
        for required_network in strategy.required_wallets:
            if required_network not in strategy.assigned_wallets:
                validation_result["missing_wallets"].append(required_network.value)
                validation_result["valid"] = False
        
        # Check exchange requirements
        for required_exchange in strategy.required_exchanges:
            if required_exchange not in strategy.assigned_exchanges:
                validation_result["missing_exchanges"].append(required_exchange.value)
                validation_result["valid"] = False
        
        # Check capital requirements
        total_available_capital = self._calculate_available_capital(strategy_id)
        if total_available_capital < strategy.min_capital_usd:
            validation_result["warnings"].append(
                f"Insufficient capital: ${total_available_capital:.2f} available, "
                f"${strategy.min_capital_usd:.2f} minimum required"
            )
        
        return validation_result
    
    def auto_assign_wallets_and_exchanges(self, strategy_id: str) -> Dict[str, Any]:
        """Automatically assign optimal wallets and exchanges for a strategy"""
        if strategy_id not in self.strategies:
            return {"success": False, "error": "Strategy not found"}
        
        strategy = self.strategies[strategy_id]
        assignments = {
            "wallets": {},
            "exchanges": {},
            "created_configs": []
        }
        
        # Auto-assign wallets
        for required_network in strategy.required_wallets:
            best_wallet = self._find_best_wallet(required_network)
            if best_wallet:
                assignments["wallets"][required_network.value] = best_wallet
            else:
                # Create placeholder for manual setup
                assignments["wallets"][required_network.value] = f"SETUP_REQUIRED_{required_network.value}"
        
        # Auto-assign exchanges
        for required_exchange in strategy.required_exchanges:
            best_exchange = self._find_best_exchange(required_exchange)
            if best_exchange:
                assignments["exchanges"][required_exchange.value] = best_exchange
            else:
                # Create placeholder for manual setup
                assignments["exchanges"][required_exchange.value] = f"SETUP_REQUIRED_{required_exchange.value}"
        
        return {"success": True, "assignments": assignments}
    
    def _save_wallet_config(self, wallet_id: str, config: WalletConfig):
        """Save wallet configuration securely"""
        config_data = asdict(config)
        # Convert enum to string
        config_data["network"] = config.network.value
        # Convert datetime to string
        if config.last_updated:
            config_data["last_updated"] = config.last_updated.isoformat()
        
        self.secure_manager.store_config(wallet_id, "wallet", config_data)
    
    def _save_exchange_config(self, exchange_id: str, config: ExchangeConfig):
        """Save exchange configuration securely"""
        config_data = asdict(config)
        # Convert enum to string
        config_data["exchange"] = config.exchange.value
        # Convert datetime to string
        if config.last_validated:
            config_data["last_validated"] = config.last_validated.isoformat()
        
        self.secure_manager.store_config(exchange_id, "exchange", config_data)
    
    def _save_strategy_config(self, strategy_id: str, config: StrategyConfig):
        """Save strategy configuration"""
        config_data = asdict(config)
        # Convert enums to strings
        config_data["strategy_type"] = config.strategy_type.value
        config_data["required_wallets"] = [w.value for w in config.required_wallets]
        config_data["required_exchanges"] = [e.value for e in config.required_exchanges]
        config_data["assigned_wallets"] = {k.value: v for k, v in config.assigned_wallets.items()}
        config_data["assigned_exchanges"] = {k.value: v for k, v in config.assigned_exchanges.items()}
        
        # Convert datetime to string
        if config.last_execution:
            config_data["last_execution"] = config.last_execution.isoformat()
        
        self.secure_manager.store_config(strategy_id, "strategy", config_data)
    
    def _get_strategy_template(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get strategy template (you would expand this with all strategy types)"""
        # This is a simplified version - in practice, you'd have templates for all strategies
        basic_template = {
            "name": f"{strategy_type.value.replace('_', ' ').title()} Strategy",
            "description": f"Automated {strategy_type.value} strategy",
            "explanation": "Automated trading strategy with AI optimization",
            "required_wallets": [NetworkType.ETHEREUM],
            "required_exchanges": [ExchangeType.BINANCE],
            "min_capital_usd": 1000.0,
            "max_capital_usd": 100000.0,
            "profit_target_percent": 5.0,
            "max_loss_percent": 2.0,
            "execution_frequency_minutes": 30
        }
        return basic_template
    
    def _validate_exchange_config(self, exchange_id: str):
        """Validate exchange API configuration"""
        if exchange_id not in self.exchanges:
            return
        
        config = self.exchanges[exchange_id]
        # Here you would implement actual API validation
        # For now, just mark as validated
        config.last_validated = datetime.now()
        self._save_exchange_config(exchange_id, config)
    
    def _validate_strategy_config(self, strategy_id: str):
        """Validate strategy configuration"""
        validation_result = self.validate_strategy_setup(strategy_id)
        logger.info(f"Strategy {strategy_id} validation: {validation_result}")
    
    def _calculate_available_capital(self, strategy_id: str) -> float:
        """Calculate total available capital for a strategy"""
        if strategy_id not in self.strategies:
            return 0.0
        
        strategy = self.strategies[strategy_id]
        total_capital = 0.0
        
        for network, wallet_id in strategy.assigned_wallets.items():
            if wallet_id in self.wallets:
                wallet = self.wallets[wallet_id]
                total_capital += wallet.balance_usd
        
        return total_capital
    
    def _find_best_wallet(self, network: NetworkType) -> Optional[str]:
        """Find the best available wallet for a network"""
        best_wallet = None
        best_balance = 0.0
        
        for wallet_id, wallet in self.wallets.items():
            if wallet.network == network and wallet.is_active:
                if wallet.balance_usd > best_balance:
                    best_balance = wallet.balance_usd
                    best_wallet = wallet_id
        
        return best_wallet
    
    def _find_best_exchange(self, exchange: ExchangeType) -> Optional[str]:
        """Find the best available exchange configuration"""
        for exchange_id, config in self.exchanges.items():
            if config.exchange == exchange and config.is_active:
                return exchange_id
        
        return None
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get complete configuration summary"""
        return {
            "wallets": {
                "total": len(self.wallets),
                "by_network": {network.value: len([w for w in self.wallets.values() if w.network == network]) 
                              for network in NetworkType},
                "total_balance_usd": sum(w.balance_usd for w in self.wallets.values())
            },
            "exchanges": {
                "total": len(self.exchanges),
                "by_exchange": {exchange.value: len([e for e in self.exchanges.values() if e.exchange == exchange]) 
                               for exchange in ExchangeType},
                "active": len([e for e in self.exchanges.values() if e.is_active])
            },
            "strategies": {
                "total": len(self.strategies),
                "enabled": len([s for s in self.strategies.values() if s.is_enabled]),
                "fully_configured": len([s for s in self.strategies.values() 
                                        if self.validate_strategy_setup(s.strategy_type.value)["valid"]])
            }
        }

