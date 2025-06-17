#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-DEX Arbitrage Engine - Intra-Chain Profit Maximizer
=========================================================

High-frequency arbitrage engine that scans multiple DEXs on the same blockchain
to identify and execute profitable price differences with lightning speed.

Features:
- ⚡ Ultra-Fast DEX Scanning (Uniswap V2/V3, SushiSwap, Curve, Balancer)
- 🔄 Flash Loan Integration for Capital-Free Arbitrage
- 📊 Real-time Liquidity Analysis
- 🎯 MEV Protection & Frontrunning Defense
- 💰 Automated Profit Optimization
- 🛡️ Smart Slippage Management
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DEXOpportunity:
    """DEX arbitrage opportunity"""
    opportunity_id: str
    blockchain: str
    asset_pair: str
    dex_from: str
    dex_to: str
    price_from: float
    price_to: float
    price_difference: float
    price_difference_percent: float
    liquidity_from: float
    liquidity_to: float
    gas_cost: float
    flash_loan_fee: float
    estimated_profit: float
    estimated_profit_percent: float
    max_trade_size: float
    execution_time_ms: int
    confidence_score: float
    mev_risk_score: float
    timestamp: datetime

@dataclass
class DEXConfig:
    """DEX configuration"""
    dex_name: str
    blockchain: str
    router_address: str
    factory_address: str
    fee_percent: float
    supports_flash_loans: bool
    avg_gas_cost: float
    liquidity_threshold: float
    active: bool

@dataclass
class FlashLoanProvider:
    """Flash loan provider configuration"""
    provider_name: str
    blockchain: str
    supported_assets: List[str]
    fee_percent: float
    max_loan_amount: float
    min_loan_amount: float
    gas_cost: float
    reliability_score: float
    active: bool

class MultiDEXArbitrageEngine:
    """Advanced multi-DEX arbitrage engine for intra-chain opportunities"""
    
    def __init__(self, blockchain: str = "ethereum"):
        self.blockchain = blockchain
        self.active_opportunities: List[DEXOpportunity] = []
        self.executed_arbitrages: List[Dict] = []
        self.dexs: List[DEXConfig] = []
        self.flash_loan_providers: List[FlashLoanProvider] = []
        self.is_running = False
        
        # Performance metrics
        self.total_dex_profit = 0.0
        self.successful_dex_arbitrages = 0
        self.failed_dex_arbitrages = 0
        self.mev_attacks_detected = 0
        
        # Market data cache
        self.price_data: Dict[str, Dict[str, float]] = {}
        self.liquidity_data: Dict[str, Dict[str, float]] = {}
        self.gas_tracker = {"current_gas_price": 20.0, "recommended_gas": 25.0}
        
        # Initialize configurations
        self._initialize_dex_configs()
        self._initialize_flash_loan_providers()
        
        logger.info(f"⚡ Multi-DEX Arbitrage Engine initialized for {blockchain}")
    
    def _initialize_dex_configs(self):
        """Initialize DEX configurations"""
        if self.blockchain == "ethereum":
            self.dexs = [
                DEXConfig(
                    dex_name="Uniswap V3",
                    blockchain="ethereum",
                    router_address="0xE592427A0AEce92De3Edee1F18E0157C05861564",
                    factory_address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
                    fee_percent=0.05,  # 0.05% average fee
                    supports_flash_loans=True,
                    avg_gas_cost=150000,
                    liquidity_threshold=100000,
                    active=True
                ),
                DEXConfig(
                    dex_name="Uniswap V2",
                    blockchain="ethereum",
                    router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                    factory_address="0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                    fee_percent=0.30,  # 0.30% fee
                    supports_flash_loans=False,
                    avg_gas_cost=120000,
                    liquidity_threshold=50000,
                    active=True
                ),
                DEXConfig(
                    dex_name="SushiSwap",
                    blockchain="ethereum",
                    router_address="0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                    factory_address="0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
                    fee_percent=0.30,  # 0.30% fee
                    supports_flash_loans=True,
                    avg_gas_cost=130000,
                    liquidity_threshold=30000,
                    active=True
                ),
                DEXConfig(
                    dex_name="Curve",
                    blockchain="ethereum",
                    router_address="0x8301AE4fc9c624d1D396cbDAa1ed877821D7C511",
                    factory_address="0xF18056Bbd320E96A48e3Fbf8bC061322531aac99",
                    fee_percent=0.04,  # 0.04% fee for stables
                    supports_flash_loans=False,
                    avg_gas_cost=100000,
                    liquidity_threshold=200000,
                    active=True
                ),
                DEXConfig(
                    dex_name="Balancer V2",
                    blockchain="ethereum",
                    router_address="0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                    factory_address="0x8E9aa87E45f92bad84D5F8DD1bff34Fb92637dE9",
                    fee_percent=0.10,  # Variable fees, 0.1% average
                    supports_flash_loans=True,
                    avg_gas_cost=140000,
                    liquidity_threshold=80000,
                    active=True
                )
            ]
    
    def _initialize_flash_loan_providers(self):
        """Initialize flash loan provider configurations"""
        self.flash_loan_providers = [
            FlashLoanProvider(
                provider_name="Aave V3",
                blockchain=self.blockchain,
                supported_assets=["USDC", "USDT", "DAI", "WETH", "WBTC"],
                fee_percent=0.05,  # 0.05% flash loan fee
                max_loan_amount=10000000.0,
                min_loan_amount=1000.0,
                gas_cost=200000,
                reliability_score=0.99,
                active=True
            ),
            FlashLoanProvider(
                provider_name="Balancer Flash Loans",
                blockchain=self.blockchain,
                supported_assets=["USDC", "USDT", "DAI", "WETH", "WBTC", "BAL"],
                fee_percent=0.00,  # No fee for Balancer flash loans
                max_loan_amount=5000000.0,
                min_loan_amount=100.0,
                gas_cost=180000,
                reliability_score=0.97,
                active=True
            ),
            FlashLoanProvider(
                provider_name="dYdX",
                blockchain=self.blockchain,
                supported_assets=["USDC", "DAI", "WETH"],
                fee_percent=0.00,  # No fee but requires 2 wei profit
                max_loan_amount=2000000.0,
                min_loan_amount=1.0,
                gas_cost=250000,
                reliability_score=0.95,
                active=True
            )
        ]
    
    async def start_dex_hunting(self):
        """Start multi-DEX arbitrage hunting"""
        if self.is_running:
            logger.warning("DEX hunting already running")
            return
        
        self.is_running = True
        logger.info(f"⚡ Starting multi-DEX arbitrage hunting on {self.blockchain}...")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._price_monitoring_loop()),
            asyncio.create_task(self._opportunity_detection_loop()),
            asyncio.create_task(self._execution_engine_loop()),
            asyncio.create_task(self._mev_protection_loop()),
            asyncio.create_task(self._gas_tracker_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in DEX hunting: {e}")
        finally:
            self.is_running = False
    
    async def _price_monitoring_loop(self):
        """Monitor prices across all DEXs"""
        while self.is_running:
            try:
                # Update price data for all DEXs
                for dex in self.dexs:
                    if dex.active:
                        dex_prices = await self._fetch_dex_prices(dex.dex_name)
                        self.price_data[dex.dex_name] = dex_prices
                        
                        dex_liquidity = await self._fetch_dex_liquidity(dex.dex_name)
                        self.liquidity_data[dex.dex_name] = dex_liquidity
                
                await asyncio.sleep(1)  # Update every second for high frequency
                
            except Exception as e:
                logger.error(f"Error in price monitoring: {e}")
                await asyncio.sleep(2)
    
    async def _opportunity_detection_loop(self):
        """Detect DEX arbitrage opportunities"""
        while self.is_running:
            try:
                if len(self.price_data) >= 2:
                    new_opportunities = await self._detect_dex_opportunities()
                    
                    for opportunity in new_opportunities:
                        if self._is_profitable_dex_opportunity(opportunity):
                            self.active_opportunities.append(opportunity)
                            logger.info(
                                f"⚡ DEX opportunity: {opportunity.asset_pair} "
                                f"{opportunity.dex_from} -> {opportunity.dex_to} "
                                f"Profit: ${opportunity.estimated_profit:.2f} "
                                f"({opportunity.estimated_profit_percent:.3f}%)"
                            )
                    
                    # Remove stale opportunities (very short window for DEX arb)
                    current_time = datetime.now()
                    self.active_opportunities = [
                        opp for opp in self.active_opportunities
                        if (current_time - opp.timestamp).total_seconds() < 5  # 5 second window
                    ]
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"Error in opportunity detection: {e}")
                await asyncio.sleep(1)
    
    async def _execution_engine_loop(self):
        """Execute profitable DEX arbitrages"""
        while self.is_running:
            try:
                if self.active_opportunities:
                    # Sort by profit potential adjusted for MEV risk
                    self.active_opportunities.sort(
                        key=lambda x: x.estimated_profit * x.confidence_score * (1 - x.mev_risk_score),
                        reverse=True
                    )
                    
                    best_opportunity = self.active_opportunities[0]
                    
                    if await self._should_execute_dex_opportunity(best_opportunity):
                        execution_result = await self._execute_dex_arbitrage(best_opportunity)
                        
                        if execution_result['success']:
                            self.successful_dex_arbitrages += 1
                            self.total_dex_profit += execution_result['actual_profit']
                            logger.info(f"⚡ DEX arbitrage executed: ${execution_result['actual_profit']:.2f}")
                        else:
                            self.failed_dex_arbitrages += 1
                            logger.warning(f"❌ DEX arbitrage failed: {execution_result['error']}")
                        
                        # Remove executed opportunity
                        self.active_opportunities.remove(best_opportunity)
                        self.executed_arbitrages.append(execution_result)
                
                await asyncio.sleep(0.1)  # Ultra high-frequency execution (100ms)
                
            except Exception as e:
                logger.error(f"Error in execution engine: {e}")
                await asyncio.sleep(0.5)
    
    async def _mev_protection_loop(self):
        """Monitor for MEV attacks and adjust strategies"""
        while self.is_running:
            try:
                # Simulate MEV detection
                mev_activity = await self._detect_mev_activity()
                
                if mev_activity['high_risk']:
                    self.mev_attacks_detected += 1
                    logger.warning(f"🛡️ MEV attack detected! Adjusting strategy...")
                    
                    # Increase gas prices for faster execution
                    self.gas_tracker['recommended_gas'] *= 1.2
                    
                    # Filter out high MEV risk opportunities
                    self.active_opportunities = [
                        opp for opp in self.active_opportunities
                        if opp.mev_risk_score < 0.7
                    ]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in MEV protection: {e}")
                await asyncio.sleep(10)
    
    async def _gas_tracker_loop(self):
        """Track and optimize gas prices"""
        while self.is_running:
            try:
                # Simulate gas price monitoring
                gas_data = await self._fetch_gas_prices()
                self.gas_tracker.update(gas_data)
                
                await asyncio.sleep(15)  # Update gas every 15 seconds
                
            except Exception as e:
                logger.error(f"Error in gas tracking: {e}")
                await asyncio.sleep(15)
    
    async def _fetch_dex_prices(self, dex_name: str) -> Dict[str, float]:
        """Fetch current prices for trading pairs on a specific DEX"""
        # Simulate price data with DEX-specific variations
        base_pairs = {
            "USDC/USDT": 1.0001,
            "WETH/USDC": 3000.0,
            "WBTC/USDC": 45000.0,
            "DAI/USDC": 1.0002,
            "USDT/DAI": 0.9999
        }
        
        # DEX-specific price variations (different fees and liquidity cause price differences)
        dex_variations = {
            "Uniswap V3": np.random.uniform(0.9998, 1.0002),
            "Uniswap V2": np.random.uniform(0.9996, 1.0004),
            "SushiSwap": np.random.uniform(0.9995, 1.0005),
            "Curve": np.random.uniform(0.9999, 1.0001),  # Very tight for stables
            "Balancer V2": np.random.uniform(0.9997, 1.0003)
        }
        
        variation = dex_variations.get(dex_name, 1.0)
        
        return {
            pair: price * variation * np.random.uniform(0.99995, 1.00005)
            for pair, price in base_pairs.items()
        }
    
    async def _fetch_dex_liquidity(self, dex_name: str) -> Dict[str, float]:
        """Fetch liquidity data for trading pairs on a specific DEX"""
        # Simulate liquidity data
        base_liquidity = {
            "USDC/USDT": np.random.uniform(1000000, 10000000),
            "WETH/USDC": np.random.uniform(5000000, 50000000),
            "WBTC/USDC": np.random.uniform(1000000, 20000000),
            "DAI/USDC": np.random.uniform(2000000, 15000000),
            "USDT/DAI": np.random.uniform(1000000, 8000000)
        }
        
        # DEX-specific liquidity multipliers
        multipliers = {
            "Uniswap V3": 2.5,
            "Uniswap V2": 2.0,
            "SushiSwap": 1.2,
            "Curve": 3.0,  # High liquidity for stables
            "Balancer V2": 1.0
        }
        
        multiplier = multipliers.get(dex_name, 1.0)
        
        return {
            pair: liquidity * multiplier
            for pair, liquidity in base_liquidity.items()
        }
    
    async def _detect_dex_opportunities(self) -> List[DEXOpportunity]:
        """Detect DEX arbitrage opportunities"""
        opportunities = []
        
        try:
            dex_names = list(self.price_data.keys())
            
            for i, dex_from in enumerate(dex_names):
                for dex_to in dex_names[i+1:]:
                    from_prices = self.price_data[dex_from]
                    to_prices = self.price_data[dex_to]
                    
                    # Find common trading pairs
                    common_pairs = set(from_prices.keys()) & set(to_prices.keys())
                    
                    for pair in common_pairs:
                        price_from = from_prices[pair]
                        price_to = to_prices[pair]
                        
                        # Check both directions
                        opportunities.extend([
                            await self._create_dex_opportunity(
                                dex_from, dex_to, pair, price_from, price_to
                            ),
                            await self._create_dex_opportunity(
                                dex_to, dex_from, pair, price_to, price_from
                            )
                        ])
        
        except Exception as e:
            logger.error(f"Error detecting DEX opportunities: {e}")
        
        return [opp for opp in opportunities if opp is not None]
    
    async def _create_dex_opportunity(self, dex_from: str, dex_to: str, 
                                    asset_pair: str, price_from: float, price_to: float) -> Optional[DEXOpportunity]:
        """Create a DEX arbitrage opportunity"""
        try:
            if price_to <= price_from:
                return None  # No profit potential
            
            price_difference = price_to - price_from
            price_difference_percent = (price_difference / price_from) * 100
            
            # Get liquidity data
            liquidity_from = self.liquidity_data.get(dex_from, {}).get(asset_pair, 0)
            liquidity_to = self.liquidity_data.get(dex_to, {}).get(asset_pair, 0)
            
            # Calculate costs
            gas_cost = await self._calculate_dex_gas_cost(dex_from, dex_to)
            
            # Find best flash loan provider
            flash_loan_provider = await self._find_best_flash_loan_provider(asset_pair.split('/')[0])
            flash_loan_fee = 0.0
            if flash_loan_provider:
                flash_loan_fee = flash_loan_provider.fee_percent / 100 * price_from
            
            # Calculate maximum trade size based on liquidity
            max_trade_size = min(liquidity_from, liquidity_to) * 0.1  # 10% of pool liquidity
            
            # Calculate estimated profit
            estimated_profit = (price_difference * max_trade_size) - gas_cost - flash_loan_fee
            estimated_profit_percent = (estimated_profit / (price_from * max_trade_size)) * 100
            
            if estimated_profit <= 5.0:  # Minimum $5 profit for DEX arb
                return None
            
            # Calculate scores
            confidence_score = self._calculate_dex_confidence_score(
                dex_from, dex_to, price_difference_percent, liquidity_from, liquidity_to
            )
            mev_risk_score = self._calculate_mev_risk_score(
                asset_pair, price_difference_percent, max_trade_size
            )
            
            return DEXOpportunity(
                opportunity_id=f"dex_{dex_from}_{dex_to}_{asset_pair.replace('/', '_')}_{int(time.time() * 1000)}",
                blockchain=self.blockchain,
                asset_pair=asset_pair,
                dex_from=dex_from,
                dex_to=dex_to,
                price_from=price_from,
                price_to=price_to,
                price_difference=price_difference,
                price_difference_percent=price_difference_percent,
                liquidity_from=liquidity_from,
                liquidity_to=liquidity_to,
                gas_cost=gas_cost,
                flash_loan_fee=flash_loan_fee,
                estimated_profit=estimated_profit,
                estimated_profit_percent=estimated_profit_percent,
                max_trade_size=max_trade_size,
                execution_time_ms=500,  # Fast DEX execution
                confidence_score=confidence_score,
                mev_risk_score=mev_risk_score,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error creating DEX opportunity: {e}")
            return None
    
    async def _calculate_dex_gas_cost(self, dex_from: str, dex_to: str) -> float:
        """Calculate gas cost for DEX arbitrage"""
        # Get DEX configs
        dex_from_config = next((d for d in self.dexs if d.dex_name == dex_from), None)
        dex_to_config = next((d for d in self.dexs if d.dex_name == dex_to), None)
        
        if not dex_from_config or not dex_to_config:
            return 100.0  # Default gas cost
        
        # Total gas = flash loan setup + swap on dex_from + swap on dex_to + flash loan repay
        total_gas = 200000 + dex_from_config.avg_gas_cost + dex_to_config.avg_gas_cost + 100000
        
        # Convert to USD (simplified)
        gas_price_gwei = self.gas_tracker['recommended_gas']
        gas_cost_eth = (total_gas * gas_price_gwei * 1e9) / 1e18
        gas_cost_usd = gas_cost_eth * 3000  # ETH price assumption
        
        return gas_cost_usd
    
    async def _find_best_flash_loan_provider(self, asset: str) -> Optional[FlashLoanProvider]:
        """Find the best flash loan provider for an asset"""
        suitable_providers = [
            provider for provider in self.flash_loan_providers
            if provider.active and asset in provider.supported_assets
        ]
        
        if not suitable_providers:
            return None
        
        # Sort by fee (lower is better) and reliability
        suitable_providers.sort(key=lambda p: p.fee_percent + (1 - p.reliability_score))
        
        return suitable_providers[0]
    
    def _calculate_dex_confidence_score(self, dex_from: str, dex_to: str, 
                                      price_diff_percent: float, liquidity_from: float, liquidity_to: float) -> float:
        """Calculate confidence score for DEX opportunity"""
        # Price difference confidence (higher difference = higher confidence, but watch for manipulation)
        price_confidence = min(1.0, price_diff_percent / 0.5)  # Max confidence at 0.5% price diff
        
        # Liquidity confidence
        min_liquidity = min(liquidity_from, liquidity_to)
        liquidity_confidence = min(1.0, min_liquidity / 1000000)  # Max confidence at $1M liquidity
        
        # DEX reliability
        dex_reliability = {
            "Uniswap V3": 0.95,
            "Uniswap V2": 0.90,
            "SushiSwap": 0.85,
            "Curve": 0.92,
            "Balancer V2": 0.88
        }
        
        from_reliability = dex_reliability.get(dex_from, 0.8)
        to_reliability = dex_reliability.get(dex_to, 0.8)
        
        overall_confidence = (price_confidence * 0.4 + 
                            liquidity_confidence * 0.3 + 
                            from_reliability * 0.15 + 
                            to_reliability * 0.15)
        
        return min(1.0, overall_confidence)
    
    def _calculate_mev_risk_score(self, asset_pair: str, price_diff_percent: float, trade_size: float) -> float:
        """Calculate MEV (Maximum Extractable Value) risk score"""
        # Higher price differences attract more MEV bots
        price_risk = min(1.0, price_diff_percent / 1.0)  # Max risk at 1% price diff
        
        # Larger trades are more attractive to MEV bots
        size_risk = min(1.0, trade_size / 100000)  # Max risk at $100k trade
        
        # Asset-specific MEV risk
        asset_risks = {
            "WETH/USDC": 0.8,  # High MEV activity
            "WBTC/USDC": 0.7,
            "USDC/USDT": 0.3,  # Lower MEV for stables
            "DAI/USDC": 0.3,
            "USDT/DAI": 0.2
        }
        
        asset_risk = asset_risks.get(asset_pair, 0.5)
        
        overall_mev_risk = (price_risk * 0.4 + size_risk * 0.3 + asset_risk * 0.3)
        
        return min(1.0, overall_mev_risk)
    
    def _is_profitable_dex_opportunity(self, opportunity: DEXOpportunity) -> bool:
        """Check if DEX opportunity is profitable"""
        return (opportunity.estimated_profit > 5.0 and  # Minimum $5 profit
                opportunity.estimated_profit_percent > 0.05 and  # Minimum 0.05% profit
                opportunity.confidence_score > 0.6 and  # Good confidence
                opportunity.liquidity_from > 10000 and  # Minimum liquidity
                opportunity.liquidity_to > 10000 and
                opportunity.mev_risk_score < 0.8)  # Acceptable MEV risk
    
    async def _should_execute_dex_opportunity(self, opportunity: DEXOpportunity) -> bool:
        """Determine if DEX opportunity should be executed"""
        # Check if opportunity is still fresh (very short window for DEX)
        time_since_detection = (datetime.now() - opportunity.timestamp).total_seconds()
        if time_since_detection > 3:  # 3 second window
            return False
        
        # Check if prices haven't moved significantly
        current_from_price = self.price_data.get(opportunity.dex_from, {}).get(opportunity.asset_pair, 0)
        current_to_price = self.price_data.get(opportunity.dex_to, {}).get(opportunity.asset_pair, 0)
        
        if abs(current_from_price - opportunity.price_from) / opportunity.price_from > 0.002:  # 0.2% movement
            return False
        
        if abs(current_to_price - opportunity.price_to) / opportunity.price_to > 0.002:
            return False
        
        return True
    
    async def _execute_dex_arbitrage(self, opportunity: DEXOpportunity) -> Dict:
        """Execute DEX arbitrage opportunity"""
        try:
            execution_start = time.time()
            
            # Simulate DEX arbitrage execution
            logger.info(f"🔄 Executing flash loan arbitrage: {opportunity.asset_pair} "
                       f"{opportunity.dex_from} -> {opportunity.dex_to}")
            
            # Simulate execution time
            await asyncio.sleep(np.random.uniform(0.5, 1.5))
            
            # Simulate success/failure based on confidence and MEV risk
            success_probability = opportunity.confidence_score * (1 - opportunity.mev_risk_score)
            is_successful = np.random.random() < success_probability
            
            execution_time = (time.time() - execution_start) * 1000
            
            if is_successful:
                # Calculate actual profit with some variance
                actual_profit = opportunity.estimated_profit * np.random.uniform(0.8, 1.2)
                
                return {
                    'success': True,
                    'opportunity_id': opportunity.opportunity_id,
                    'blockchain': opportunity.blockchain,
                    'asset_pair': opportunity.asset_pair,
                    'dex_from': opportunity.dex_from,
                    'dex_to': opportunity.dex_to,
                    'trade_size': opportunity.max_trade_size,
                    'estimated_profit': opportunity.estimated_profit,
                    'actual_profit': actual_profit,
                    'execution_time_ms': execution_time,
                    'gas_cost': opportunity.gas_cost,
                    'flash_loan_fee': opportunity.flash_loan_fee,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'opportunity_id': opportunity.opportunity_id,
                    'blockchain': opportunity.blockchain,
                    'asset_pair': opportunity.asset_pair,
                    'estimated_profit': opportunity.estimated_profit,
                    'actual_profit': 0,
                    'execution_time_ms': execution_time,
                    'error': 'MEV attack or price movement',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error executing DEX arbitrage: {e}")
            return {
                'success': False,
                'opportunity_id': opportunity.opportunity_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _detect_mev_activity(self) -> Dict:
        """Detect MEV (Maximum Extractable Value) activity"""
        # Simulate MEV detection
        mev_score = np.random.uniform(0, 1)
        
        return {
            'high_risk': mev_score > 0.7,
            'mev_score': mev_score,
            'recommended_action': 'increase_gas' if mev_score > 0.7 else 'normal'
        }
    
    async def _fetch_gas_prices(self) -> Dict:
        """Fetch current gas prices"""
        # Simulate gas price data
        base_gas = 20.0
        network_congestion = np.random.uniform(0.8, 2.0)
        
        return {
            'current_gas_price': base_gas * network_congestion,
            'recommended_gas': base_gas * network_congestion * 1.1,
            'fast_gas': base_gas * network_congestion * 1.3
        }
    
    def get_dex_performance_metrics(self) -> Dict[str, Any]:
        """Get DEX arbitrage performance metrics"""
        total_executions = self.successful_dex_arbitrages + self.failed_dex_arbitrages
        success_rate = self.successful_dex_arbitrages / max(1, total_executions)
        
        return {
            'total_dex_profit_usd': self.total_dex_profit,
            'successful_dex_arbitrages': self.successful_dex_arbitrages,
            'failed_dex_arbitrages': self.failed_dex_arbitrages,
            'success_rate': success_rate,
            'active_opportunities': len(self.active_opportunities),
            'mev_attacks_detected': self.mev_attacks_detected,
            'current_gas_price': self.gas_tracker['current_gas_price'],
            'active_dexs': len([d for d in self.dexs if d.active]),
            'timestamp': datetime.now().isoformat()
        }
    
    async def stop_dex_hunting(self):
        """Stop DEX arbitrage hunting"""
        self.is_running = False
        logger.info("🛑 DEX hunting stopped")

# Example usage
async def main():
    """Demonstrate multi-DEX arbitrage engine"""
    dex_engine = MultiDEXArbitrageEngine("ethereum")
    
    # Start DEX hunting for a demo
    try:
        # Run for 30 seconds
        await asyncio.wait_for(dex_engine.start_dex_hunting(), timeout=30.0)
    except asyncio.TimeoutError:
        await dex_engine.stop_dex_hunting()
    
    # Display performance metrics
    metrics = dex_engine.get_dex_performance_metrics()
    print("\n⚡ MULTI-DEX ARBITRAGE PERFORMANCE METRICS ⚡")
    print("=" * 50)
    print(f"Total DEX Profit: ${metrics['total_dex_profit_usd']:.2f}")
    print(f"Successful Arbitrages: {metrics['successful_dex_arbitrages']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Active Opportunities: {metrics['active_opportunities']}")
    print(f"MEV Attacks Detected: {metrics['mev_attacks_detected']}")
    print(f"Current Gas Price: {metrics['current_gas_price']:.1f} Gwei")

if __name__ == "__main__":
    asyncio.run(main())

