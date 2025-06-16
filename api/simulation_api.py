from fastapi import FastAPI, HTTPException, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections import defaultdict

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Ultimate Investment Simulation API",
    description="Real-money simulation with quantum portfolio optimization and AI strategies",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class SimulationConfig(BaseModel):
    name: str
    initial_capital: float
    risk_tolerance: float = 0.5
    quantum_enabled: bool = True
    ai_strategies: List[str] = ["momentum", "mean_reversion", "volatility_breakout"]
    assets: List[str] = ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ"]
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    max_position_size: float = 0.2  # Max 20% per position
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.15  # 15% take profit
    leverage: float = 1.0
    commission_rate: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    duration_days: int = 30

class SimulationStep(BaseModel):
    simulation_id: str
    force_rebalance: bool = False
    market_event: Optional[str] = None

class SimulationResult(BaseModel):
    simulation_id: str
    step: int
    timestamp: datetime
    portfolio_value: float
    cash: float
    positions: Dict[str, Any]
    daily_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    strategy_signals: Dict[str, Any]
    quantum_allocation: Dict[str, float]
    ai_confidence: float
    risk_metrics: Dict[str, float]
    trade_history: List[Dict[str, Any]]
    performance_attribution: Dict[str, float]

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    value_history: List[float]
    trade_history: List[Dict[str, Any]]
    daily_returns: List[float]
    timestamp: datetime
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(qty * prices.get(symbol, 0) for symbol, qty in self.positions.items())
        return self.cash + position_value
    
    def get_allocation(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current allocation percentages"""
        total_value = self.get_total_value(prices)
        if total_value == 0:
            return {}
        
        allocation = {}
        for symbol, qty in self.positions.items():
            allocation[symbol] = (qty * prices.get(symbol, 0)) / total_value
        allocation['cash'] = self.cash / total_value
        return allocation

class QuantumPortfolioOptimizer:
    """Simplified quantum portfolio optimizer for simulation"""
    
    def __init__(self, quantum_enabled: bool = True):
        self.quantum_enabled = quantum_enabled
        self.risk_models = {}
        
    async def optimize_portfolio(
        self, 
        assets: List[str], 
        market_data: Dict[str, pd.DataFrame],
        risk_tolerance: float,
        current_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """Generate optimal portfolio allocation"""
        try:
            # Simulate quantum-enhanced optimization
            np.random.seed(int(time.time()) % 1000)
            
            if self.quantum_enabled:
                # Quantum-inspired optimization with momentum and mean reversion
                weights = self._quantum_optimize(assets, market_data, risk_tolerance)
            else:
                # Classical mean-variance optimization
                weights = self._classical_optimize(assets, market_data, risk_tolerance)
            
            # Apply constraints
            weights = self._apply_constraints(weights, current_allocation)
            
            return weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            # Fallback to equal weight
            return {asset: 1.0/len(assets) for asset in assets}
    
    def _quantum_optimize(self, assets: List[str], market_data: Dict[str, pd.DataFrame], risk_tolerance: float) -> Dict[str, float]:
        """Quantum-inspired optimization algorithm"""
        num_assets = len(assets)
        
        # Quantum-inspired random walk with momentum
        quantum_weights = np.random.dirichlet(np.ones(num_assets) * (1 + risk_tolerance))
        
        # Apply momentum and volatility adjustments
        adjustments = []
        for i, asset in enumerate(assets):
            if asset in market_data and len(market_data[asset]) > 20:
                returns = market_data[asset]['close'].pct_change().dropna()
                momentum = returns.tail(10).mean()
                volatility = returns.tail(20).std()
                
                # Quantum enhancement factor
                quantum_factor = 1 + (momentum * risk_tolerance) - (volatility * (1 - risk_tolerance))
                adjustments.append(max(0.1, quantum_factor))
            else:
                adjustments.append(1.0)
        
        # Apply adjustments and renormalize
        adjusted_weights = quantum_weights * np.array(adjustments)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        return {asset: float(weight) for asset, weight in zip(assets, adjusted_weights)}
    
    def _classical_optimize(self, assets: List[str], market_data: Dict[str, pd.DataFrame], risk_tolerance: float) -> Dict[str, float]:
        """Classical mean-variance optimization"""
        # Simplified mean-variance with risk tolerance
        weights = np.random.dirichlet(np.ones(len(assets)) * risk_tolerance * 10)
        return {asset: float(weight) for asset, weight in zip(assets, weights)}
    
    def _apply_constraints(self, weights: Dict[str, float], current_allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply position size and turnover constraints"""
        max_weight = 0.25  # Max 25% per position
        
        # Cap individual positions
        total_weight = sum(weights.values())
        constrained_weights = {}
        
        for asset, weight in weights.items():
            normalized_weight = weight / total_weight if total_weight > 0 else 0
            constrained_weights[asset] = min(normalized_weight, max_weight)
        
        # Renormalize
        total_constrained = sum(constrained_weights.values())
        if total_constrained > 0:
            constrained_weights = {asset: weight / total_constrained for asset, weight in constrained_weights.items()}
        
        return constrained_weights

class AIStrategyEngine:
    """AI-powered trading strategy engine"""
    
    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        self.signals_history = defaultdict(list)
        
    async def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """Generate trading signals from multiple AI strategies"""
        signals = {}
        confidence_scores = {}
        
        try:
            for strategy in self.strategies:
                if strategy == "momentum":
                    signals[strategy] = self._momentum_strategy(market_data)
                elif strategy == "mean_reversion":
                    signals[strategy] = self._mean_reversion_strategy(market_data)
                elif strategy == "volatility_breakout":
                    signals[strategy] = self._volatility_breakout_strategy(market_data)
                elif strategy == "ml_ensemble":
                    signals[strategy] = self._ml_ensemble_strategy(market_data)
                
                # Calculate confidence based on signal strength
                confidence_scores[strategy] = self._calculate_confidence(signals.get(strategy, {}))
            
            # Meta-learning ensemble
            ensemble_signals = self._ensemble_signals(signals, confidence_scores, risk_tolerance)
            
            return {
                "individual_signals": signals,
                "confidence_scores": confidence_scores,
                "ensemble_signals": ensemble_signals,
                "overall_confidence": np.mean(list(confidence_scores.values()))
            }
            
        except Exception as e:
            logger.error(f"AI strategy error: {e}")
            return {"individual_signals": {}, "confidence_scores": {}, "ensemble_signals": {}, "overall_confidence": 0.5}
    
    def _momentum_strategy(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Momentum-based trading signals"""
        signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                signals[symbol] = 0.0
                continue
                
            # Calculate momentum indicators
            returns = data['close'].pct_change().dropna()
            short_momentum = returns.tail(5).mean()
            long_momentum = returns.tail(20).mean()
            
            # Generate signal (-1 to 1)
            if short_momentum > long_momentum * 1.02:  # 2% threshold
                signals[symbol] = min(1.0, short_momentum * 10)
            elif short_momentum < long_momentum * 0.98:
                signals[symbol] = max(-1.0, short_momentum * 10)
            else:
                signals[symbol] = 0.0
                
        return signals
    
    def _mean_reversion_strategy(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Mean reversion trading signals"""
        signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                signals[symbol] = 0.0
                continue
                
            # Bollinger Bands mean reversion
            prices = data['close']
            ma = prices.rolling(20).mean()
            std = prices.rolling(20).std()
            
            current_price = prices.iloc[-1]
            upper_band = ma.iloc[-1] + 2 * std.iloc[-1]
            lower_band = ma.iloc[-1] - 2 * std.iloc[-1]
            
            if current_price > upper_band:
                signals[symbol] = -0.8  # Sell signal
            elif current_price < lower_band:
                signals[symbol] = 0.8   # Buy signal
            else:
                # Gradual mean reversion
                deviation = (current_price - ma.iloc[-1]) / std.iloc[-1]
                signals[symbol] = -deviation * 0.5
                
        return signals
    
    def _volatility_breakout_strategy(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Volatility breakout trading signals"""
        signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                signals[symbol] = 0.0
                continue
                
            # ATR-based volatility breakout
            high = data['high'] if 'high' in data.columns else data['close']
            low = data['low'] if 'low' in data.columns else data['close']
            close = data['close']
            
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            atr = tr.rolling(14).mean()
            
            # Breakout detection
            current_price = close.iloc[-1]
            prev_high = high.rolling(20).max().iloc[-2]
            prev_low = low.rolling(20).min().iloc[-2]
            
            if current_price > prev_high + atr.iloc[-1] * 0.5:
                signals[symbol] = 0.9  # Strong buy
            elif current_price < prev_low - atr.iloc[-1] * 0.5:
                signals[symbol] = -0.9  # Strong sell
            else:
                signals[symbol] = 0.0
                
        return signals
    
    def _ml_ensemble_strategy(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Machine learning ensemble strategy"""
        signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < 50:
                signals[symbol] = 0.0
                continue
                
            try:
                # Feature engineering
                features = self._extract_features(data)
                
                # Simulated ML predictions (replace with actual ML models)
                prediction = np.tanh(np.random.normal(0, 0.3))  # Simulate ML output
                
                # Ensemble with technical indicators
                tech_signal = self._technical_ensemble(data)
                
                # Combine ML and technical signals
                signals[symbol] = 0.7 * prediction + 0.3 * tech_signal
                
            except Exception as e:
                logger.warning(f"ML strategy error for {symbol}: {e}")
                signals[symbol] = 0.0
                
        return signals
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract ML features from market data"""
        # Price-based features
        returns = data['close'].pct_change().dropna()
        
        features = [
            returns.tail(5).mean(),   # Short-term momentum
            returns.tail(20).mean(),  # Long-term momentum
            returns.tail(5).std(),    # Short-term volatility
            returns.tail(20).std(),   # Long-term volatility
            len(returns[returns > 0]) / len(returns),  # Win rate
        ]
        
        return np.array(features)
    
    def _technical_ensemble(self, data: pd.DataFrame) -> float:
        """Technical indicator ensemble"""
        close = data['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = (50 - rsi.iloc[-1]) / 50  # Normalize to [-1, 1]
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_signal = np.tanh((macd.iloc[-1] - signal_line.iloc[-1]) / close.iloc[-1] * 100)
        
        # Combine signals
        return (rsi_signal + macd_signal) / 2
    
    def _calculate_confidence(self, signals: Dict[str, float]) -> float:
        """Calculate confidence score for strategy signals"""
        if not signals:
            return 0.0
        
        signal_values = list(signals.values())
        avg_strength = np.mean(np.abs(signal_values))
        consistency = 1 - np.std(signal_values) if len(signal_values) > 1 else 1.0
        
        return min(1.0, avg_strength * consistency)
    
    def _ensemble_signals(self, signals: Dict[str, Dict[str, float]], confidence_scores: Dict[str, float], risk_tolerance: float) -> Dict[str, float]:
        """Combine signals from multiple strategies"""
        ensemble = {}
        all_symbols = set()
        
        # Collect all symbols
        for strategy_signals in signals.values():
            all_symbols.update(strategy_signals.keys())
        
        # Weighted ensemble
        for symbol in all_symbols:
            weighted_signal = 0.0
            total_weight = 0.0
            
            for strategy, strategy_signals in signals.items():
                if symbol in strategy_signals:
                    weight = confidence_scores.get(strategy, 0.5)
                    weighted_signal += strategy_signals[symbol] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_signal = weighted_signal / total_weight
                # Apply risk tolerance
                ensemble[symbol] = ensemble_signal * risk_tolerance
            else:
                ensemble[symbol] = 0.0
        
        return ensemble

class MarketDataProvider:
    """Simulated market data provider"""
    
    def __init__(self):
        self.data_cache = {}
        self.last_update = {}
        
    async def get_market_data(self, symbols: List[str], days: int = 100) -> Dict[str, pd.DataFrame]:
        """Get historical market data for symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # Generate realistic market data simulation
                data = self._generate_market_data(symbol, days)
                market_data[symbol] = data
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                # Fallback data
                market_data[symbol] = self._generate_fallback_data(symbol, days)
        
        return market_data
    
    def _generate_market_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate realistic market data simulation"""
        np.random.seed(hash(symbol) % 1000)
        
        # Base parameters
        base_price = np.random.uniform(50, 500)
        volatility = np.random.uniform(0.15, 0.35)
        drift = np.random.uniform(-0.0005, 0.0005)
        
        # Generate price series using geometric Brownian motion
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        
        prices = [base_price]
        for i in range(1, days):
            dt = 1/252  # Daily step
            dW = np.random.normal(0, np.sqrt(dt))
            price_change = prices[-1] * (drift * dt + volatility * dW)
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [price * np.random.uniform(1.0, 1.02) for price in prices],
            'low': [price * np.random.uniform(0.98, 1.0) for price in prices],
            'close': prices,
            'volume': [np.random.randint(100000, 10000000) for _ in range(days)]
        })
        
        # Add some market microstructure noise
        df['close'] = df['close'] * (1 + np.random.normal(0, 0.001, len(df)))
        df['high'] = np.maximum(df['high'], df['close'])
        df['low'] = np.minimum(df['low'], df['close'])
        
        return df
    
    def _generate_fallback_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate simple fallback data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        base_price = 100
        
        return pd.DataFrame({
            'date': dates,
            'open': [base_price] * days,
            'high': [base_price * 1.01] * days,
            'low': [base_price * 0.99] * days,
            'close': [base_price] * days,
            'volume': [1000000] * days
        })
    
    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current market prices"""
        prices = {}
        
        for symbol in symbols:
            if symbol in self.data_cache:
                # Simulate real-time price movement
                last_price = self.data_cache[symbol]['close'].iloc[-1]
                change = np.random.normal(0, 0.01)  # 1% daily volatility
                prices[symbol] = last_price * (1 + change)
            else:
                # Default price
                prices[symbol] = 100.0
        
        return prices

class SimulationEngine:
    """Core simulation engine orchestrating all components"""
    
    def __init__(self):
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self.market_data_provider = MarketDataProvider()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def create_simulation(self, config: SimulationConfig) -> str:
        """Create a new simulation instance"""
        simulation_id = str(uuid.uuid4())
        
        try:
            # Initialize components
            quantum_optimizer = QuantumPortfolioOptimizer(config.quantum_enabled)
            ai_engine = AIStrategyEngine(config.ai_strategies)
            
            # Initialize portfolio
            portfolio = Portfolio(
                cash=config.initial_capital,
                positions={},
                value_history=[config.initial_capital],
                trade_history=[],
                daily_returns=[],
                timestamp=datetime.now()
            )
            
            # Get initial market data
            market_data = await self.market_data_provider.get_market_data(config.assets)
            
            # Store simulation state
            self.active_simulations[simulation_id] = {
                "config": config,
                "portfolio": portfolio,
                "quantum_optimizer": quantum_optimizer,
                "ai_engine": ai_engine,
                "market_data": market_data,
                "step": 0,
                "start_time": datetime.now(),
                "last_update": datetime.now(),
                "status": "active",
                "performance_metrics": {},
                "risk_metrics": {}
            }
            
            logger.info(f"Created simulation {simulation_id} with ${config.initial_capital:,.2f} initial capital")
            return simulation_id
            
        except Exception as e:
            logger.error(f"Error creating simulation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create simulation: {str(e)}")
    
    async def step_simulation(self, simulation_id: str, force_rebalance: bool = False) -> SimulationResult:
        """Execute one simulation step"""
        if simulation_id not in self.active_simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        sim_data = self.active_simulations[simulation_id]
        
        try:
            # Update market data
            current_prices = await self.market_data_provider.get_real_time_prices(sim_data["config"].assets)
            
            # Generate AI signals
            ai_signals = await sim_data["ai_engine"].generate_signals(
                sim_data["market_data"],
                sim_data["portfolio"],
                sim_data["config"].risk_tolerance
            )
            
            # Portfolio optimization
            current_allocation = sim_data["portfolio"].get_allocation(current_prices)
            optimal_allocation = await sim_data["quantum_optimizer"].optimize_portfolio(
                sim_data["config"].assets,
                sim_data["market_data"],
                sim_data["config"].risk_tolerance,
                current_allocation
            )
            
            # Execute trades based on signals and optimization
            trades = self._generate_trades(
                sim_data,
                current_prices,
                optimal_allocation,
                ai_signals["ensemble_signals"],
                force_rebalance
            )
            
            # Apply trades to portfolio
            self._execute_trades(sim_data["portfolio"], trades, current_prices, sim_data["config"])
            
            # Update portfolio value and metrics
            portfolio_value = sim_data["portfolio"].get_total_value(current_prices)
            self._update_performance_metrics(sim_data, portfolio_value)
            
            # Increment step
            sim_data["step"] += 1
            sim_data["last_update"] = datetime.now()
            
            # Create result
            result = SimulationResult(
                simulation_id=simulation_id,
                step=sim_data["step"],
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                cash=sim_data["portfolio"].cash,
                positions={symbol: qty for symbol, qty in sim_data["portfolio"].positions.items() if qty != 0},
                daily_return=sim_data["portfolio"].daily_returns[-1] if sim_data["portfolio"].daily_returns else 0.0,
                total_return=(portfolio_value / sim_data["config"].initial_capital - 1) * 100,
                sharpe_ratio=sim_data["performance_metrics"].get("sharpe_ratio", 0.0),
                max_drawdown=sim_data["performance_metrics"].get("max_drawdown", 0.0),
                strategy_signals=ai_signals["individual_signals"],
                quantum_allocation=optimal_allocation,
                ai_confidence=ai_signals["overall_confidence"],
                risk_metrics=sim_data["risk_metrics"],
                trade_history=trades,
                performance_attribution=self._calculate_performance_attribution(sim_data, current_prices)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Simulation step failed: {str(e)}")
    
    def _generate_trades(
        self,
        sim_data: Dict[str, Any],
        current_prices: Dict[str, float],
        optimal_allocation: Dict[str, float],
        ai_signals: Dict[str, float],
        force_rebalance: bool
    ) -> List[Dict[str, Any]]:
        """Generate trades based on signals and optimization"""
        trades = []
        portfolio = sim_data["portfolio"]
        config = sim_data["config"]
        
        current_allocation = portfolio.get_allocation(current_prices)
        portfolio_value = portfolio.get_total_value(current_prices)
        
        for symbol in config.assets:
            current_weight = current_allocation.get(symbol, 0.0)
            target_weight = optimal_allocation.get(symbol, 0.0)
            
            # Apply AI signal modulation
            ai_signal = ai_signals.get(symbol, 0.0)
            target_weight *= (1 + ai_signal * config.risk_tolerance * 0.2)  # Max 20% signal impact
            target_weight = max(0.0, min(config.max_position_size, target_weight))
            
            # Check if rebalancing is needed
            weight_diff = abs(target_weight - current_weight)
            
            if force_rebalance or weight_diff > 0.05:  # 5% threshold
                target_value = target_weight * portfolio_value
                current_value = current_weight * portfolio_value
                
                if target_value > current_value:  # Buy
                    trade_value = target_value - current_value
                    quantity = trade_value / current_prices[symbol]
                    
                    if trade_value >= portfolio.cash * 0.01:  # Min 1% of cash
                        trades.append({
                            "symbol": symbol,
                            "action": "buy",
                            "quantity": quantity,
                            "price": current_prices[symbol],
                            "value": trade_value,
                            "timestamp": datetime.now(),
                            "reason": "rebalance"
                        })
                        
                elif target_value < current_value:  # Sell
                    trade_value = current_value - target_value
                    quantity = trade_value / current_prices[symbol]
                    current_position = portfolio.positions.get(symbol, 0.0)
                    
                    if quantity <= current_position and trade_value >= 100:  # Min $100 trade
                        trades.append({
                            "symbol": symbol,
                            "action": "sell",
                            "quantity": quantity,
                            "price": current_prices[symbol],
                            "value": trade_value,
                            "timestamp": datetime.now(),
                            "reason": "rebalance"
                        })
        
        # Stop loss and take profit checks
        trades.extend(self._check_stop_loss_take_profit(sim_data, current_prices))
        
        return trades
    
    def _check_stop_loss_take_profit(
        self,
        sim_data: Dict[str, Any],
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check for stop loss and take profit triggers"""
        trades = []
        portfolio = sim_data["portfolio"]
        config = sim_data["config"]
        
        for symbol, quantity in portfolio.positions.items():
            if quantity <= 0:
                continue
            
            # Find average purchase price from trade history
            avg_price = self._get_average_price(portfolio.trade_history, symbol)
            if avg_price is None:
                continue
            
            current_price = current_prices.get(symbol, avg_price)
            pnl_percent = (current_price - avg_price) / avg_price
            
            # Stop loss check
            if pnl_percent <= -config.stop_loss:
                trades.append({
                    "symbol": symbol,
                    "action": "sell",
                    "quantity": quantity,
                    "price": current_price,
                    "value": quantity * current_price,
                    "timestamp": datetime.now(),
                    "reason": "stop_loss"
                })
            
            # Take profit check
            elif pnl_percent >= config.take_profit:
                trades.append({
                    "symbol": symbol,
                    "action": "sell",
                    "quantity": quantity * 0.5,  # Partial profit taking
                    "price": current_price,
                    "value": quantity * current_price * 0.5,
                    "timestamp": datetime.now(),
                    "reason": "take_profit"
                })
        
        return trades
    
    def _get_average_price(self, trade_history: List[Dict[str, Any]], symbol: str) -> Optional[float]:
        """Calculate average purchase price for a symbol"""
        total_cost = 0.0
        total_quantity = 0.0
        
        for trade in trade_history:
            if trade["symbol"] == symbol:
                if trade["action"] == "buy":
                    total_cost += trade["value"]
                    total_quantity += trade["quantity"]
                elif trade["action"] == "sell":
                    # Adjust for FIFO accounting
                    if total_quantity > 0:
                        avg_price = total_cost / total_quantity
                        sold_cost = trade["quantity"] * avg_price
                        total_cost -= sold_cost
                        total_quantity -= trade["quantity"]
        
        return total_cost / total_quantity if total_quantity > 0 else None
    
    def _execute_trades(
        self,
        portfolio: Portfolio,
        trades: List[Dict[str, Any]],
        current_prices: Dict[str, float],
        config: SimulationConfig
    ):
        """Execute trades and update portfolio"""
        for trade in trades:
            symbol = trade["symbol"]
            action = trade["action"]
            quantity = trade["quantity"]
            price = trade["price"]
            
            # Apply slippage and commission
            if action == "buy":
                executed_price = price * (1 + config.slippage)
                commission = trade["value"] * config.commission_rate
                total_cost = quantity * executed_price + commission
                
                if total_cost <= portfolio.cash:
                    portfolio.cash -= total_cost
                    portfolio.positions[symbol] = portfolio.positions.get(symbol, 0.0) + quantity
                    
                    # Record trade
                    trade["executed_price"] = executed_price
                    trade["commission"] = commission
                    trade["status"] = "executed"
                    portfolio.trade_history.append(trade)
                else:
                    trade["status"] = "rejected_insufficient_funds"
            
            elif action == "sell":
                current_position = portfolio.positions.get(symbol, 0.0)
                
                if quantity <= current_position:
                    executed_price = price * (1 - config.slippage)
                    proceeds = quantity * executed_price
                    commission = proceeds * config.commission_rate
                    net_proceeds = proceeds - commission
                    
                    portfolio.cash += net_proceeds
                    portfolio.positions[symbol] -= quantity
                    
                    # Clean up zero positions
                    if portfolio.positions[symbol] <= 1e-8:
                        del portfolio.positions[symbol]
                    
                    # Record trade
                    trade["executed_price"] = executed_price
                    trade["commission"] = commission
                    trade["net_proceeds"] = net_proceeds
                    trade["status"] = "executed"
                    portfolio.trade_history.append(trade)
                else:
                    trade["status"] = "rejected_insufficient_position"
    
    def _update_performance_metrics(self, sim_data: Dict[str, Any], portfolio_value: float):
        """Update performance and risk metrics"""
        portfolio = sim_data["portfolio"]
        config = sim_data["config"]
        
        # Update value history
        portfolio.value_history.append(portfolio_value)
        
        # Calculate daily return
        if len(portfolio.value_history) > 1:
            daily_return = (portfolio_value / portfolio.value_history[-2] - 1) * 100
            portfolio.daily_returns.append(daily_return)
        
        # Performance metrics
        if len(portfolio.value_history) > 2:
            returns = np.array(portfolio.daily_returns) / 100
            
            # Sharpe ratio (annualized)
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio.value_history)
            drawdowns = (np.array(portfolio.value_history) - peak) / peak
            max_drawdown = np.min(drawdowns) * 100
            
            # Win rate
            positive_returns = len([r for r in returns if r > 0])
            win_rate = positive_returns / len(returns) if len(returns) > 0 else 0.0
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252) * 100
            
            sim_data["performance_metrics"] = {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "volatility": volatility,
                "total_return": (portfolio_value / config.initial_capital - 1) * 100
            }
            
            # Risk metrics
            sim_data["risk_metrics"] = {
                "var_95": np.percentile(returns, 5) * 100,  # 5% VaR
                "current_drawdown": drawdowns[-1] * 100,
                "beta": self._calculate_beta(returns),
                "sortino_ratio": self._calculate_sortino_ratio(returns)
            }
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate beta relative to market (simplified)"""
        # Simplified beta calculation (assumes market return of 0.01% daily)
        market_returns = np.full_like(returns, 0.0001)
        if len(returns) > 1 and np.std(market_returns) > 0:
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 0 else 1.0
        return 1.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            return np.mean(returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0
        return np.inf if np.mean(returns) > 0 else 0.0
    
    def _calculate_performance_attribution(self, sim_data: Dict[str, Any], current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance attribution by asset"""
        portfolio = sim_data["portfolio"]
        config = sim_data["config"]
        attribution = {}
        
        total_value = portfolio.get_total_value(current_prices)
        initial_value = config.initial_capital
        
        for symbol in config.assets:
            if symbol in portfolio.positions and portfolio.positions[symbol] > 0:
                position_value = portfolio.positions[symbol] * current_prices.get(symbol, 0)
                weight = position_value / total_value if total_value > 0 else 0
                
                # Simplified attribution (position return contribution)
                avg_price = self._get_average_price(portfolio.trade_history, symbol)
                if avg_price:
                    asset_return = (current_prices.get(symbol, avg_price) / avg_price - 1)
                    attribution[symbol] = asset_return * weight * 100
                else:
                    attribution[symbol] = 0.0
            else:
                attribution[symbol] = 0.0
        
        return attribution
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get current simulation status and metrics"""
        if simulation_id not in self.active_simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        sim_data = self.active_simulations[simulation_id]
        current_prices = await self.market_data_provider.get_real_time_prices(sim_data["config"].assets)
        portfolio_value = sim_data["portfolio"].get_total_value(current_prices)
        
        return {
            "simulation_id": simulation_id,
            "status": sim_data["status"],
            "step": sim_data["step"],
            "start_time": sim_data["start_time"],
            "last_update": sim_data["last_update"],
            "config": sim_data["config"].dict(),
            "portfolio_value": portfolio_value,
            "cash": sim_data["portfolio"].cash,
            "positions": dict(sim_data["portfolio"].positions),
            "performance_metrics": sim_data["performance_metrics"],
            "risk_metrics": sim_data["risk_metrics"],
            "total_trades": len(sim_data["portfolio"].trade_history)
        }
    
    async def stop_simulation(self, simulation_id: str) -> Dict[str, str]:
        """Stop a simulation"""
        if simulation_id not in self.active_simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        self.active_simulations[simulation_id]["status"] = "stopped"
        logger.info(f"Stopped simulation {simulation_id}")
        
        return {"message": f"Simulation {simulation_id} stopped successfully"}
    
    async def list_simulations(self) -> List[Dict[str, Any]]:
        """List all active simulations"""
        simulations = []
        
        for sim_id, sim_data in self.active_simulations.items():
            simulations.append({
                "simulation_id": sim_id,
                "name": sim_data["config"].name,
                "status": sim_data["status"],
                "step": sim_data["step"],
                "start_time": sim_data["start_time"],
                "initial_capital": sim_data["config"].initial_capital,
                "current_value": sim_data["portfolio"].value_history[-1] if sim_data["portfolio"].value_history else 0
            })
        
        return simulations

# Global simulation engine instance
simulation_engine = SimulationEngine()

# WebSocket connections for real-time updates
active_connections: Dict[str, WebSocket] = {}

# API Endpoints
@app.post("/simulations", response_model=dict)
async def create_simulation(config: SimulationConfig):
    """Create a new investment simulation"""
    simulation_id = await simulation_engine.create_simulation(config)
    return {"simulation_id": simulation_id, "message": "Simulation created successfully"}

@app.post("/simulations/{simulation_id}/step", response_model=SimulationResult)
async def step_simulation(simulation_id: str, step_config: SimulationStep = None):
    """Execute one simulation step"""
    force_rebalance = step_config.force_rebalance if step_config else False
    result = await simulation_engine.step_simulation(simulation_id, force_rebalance)
    
    # Broadcast to WebSocket clients
    await broadcast_update(simulation_id, result.dict())
    
    return result

@app.get("/simulations/{simulation_id}/status")
async def get_simulation_status(simulation_id: str):
    """Get simulation status and metrics"""
    return await simulation_engine.get_simulation_status(simulation_id)

@app.post("/simulations/{simulation_id}/stop")
async def stop_simulation(simulation_id: str):
    """Stop a simulation"""
    return await simulation_engine.stop_simulation(simulation_id)

@app.get("/simulations")
async def list_simulations():
    """List all simulations"""
    return await simulation_engine.list_simulations()

@app.post("/simulations/{simulation_id}/run_steps/{num_steps}")
async def run_multiple_steps(simulation_id: str, num_steps: int):
    """Run multiple simulation steps"""
    results = []
    
    for i in range(num_steps):
        try:
            result = await simulation_engine.step_simulation(simulation_id)
            results.append(result.dict())
            
            # Broadcast each step
            await broadcast_update(simulation_id, result.dict())
            
            # Small delay between steps
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in step {i+1}: {e}")
            break
    
    return {"steps_completed": len(results), "results": results}

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{simulation_id}")
async def websocket_endpoint(websocket: WebSocket, simulation_id: str):
    await websocket.accept()
    active_connections[simulation_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
            elif data == "get_status":
                if simulation_id in simulation_engine.active_simulations:
                    status = await simulation_engine.get_simulation_status(simulation_id)
                    await websocket.send_text(json.dumps(status))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if simulation_id in active_connections:
            del active_connections[simulation_id]

async def broadcast_update(simulation_id: str, data: dict):
    """Broadcast updates to WebSocket clients"""
    if simulation_id in active_connections:
        try:
            await active_connections[simulation_id].send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error broadcasting to {simulation_id}: {e}")
            # Remove dead connection
            if simulation_id in active_connections:
                del active_connections[simulation_id]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_simulations": len(simulation_engine.active_simulations),
        "websocket_connections": len(active_connections),
        "timestamp": datetime.now()
    }

# Advanced endpoints for analytics
@app.get("/simulations/{simulation_id}/analytics")
async def get_simulation_analytics(simulation_id: str):
    """Get detailed analytics for a simulation"""
    if simulation_id not in simulation_engine.active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim_data = simulation_engine.active_simulations[simulation_id]
    portfolio = sim_data["portfolio"]
    
    # Prepare analytics data
    analytics = {
        "value_history": portfolio.value_history,
        "daily_returns": portfolio.daily_returns,
        "trade_history": portfolio.trade_history,
        "performance_metrics": sim_data["performance_metrics"],
        "risk_metrics": sim_data["risk_metrics"],
        "drawdown_series": [],
        "rolling_sharpe": [],
        "trade_statistics": {}
    }
    
    # Calculate additional analytics
    if len(portfolio.value_history) > 1:
        # Drawdown series
        values = np.array(portfolio.value_history)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak * 100
        analytics["drawdown_series"] = drawdowns.tolist()
        
        # Rolling Sharpe ratio (30-day)
        if len(portfolio.daily_returns) >= 30:
            returns = np.array(portfolio.daily_returns) / 100
            rolling_sharpe = []
            for i in range(30, len(returns)):
                window_returns = returns[i-30:i]
                if np.std(window_returns) > 0:
                    sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
            analytics["rolling_sharpe"] = rolling_sharpe
        
        # Trade statistics
        trades = portfolio.trade_history
        if trades:
            buy_trades = [t for t in trades if t["action"] == "buy"]
            sell_trades = [t for t in trades if t["action"] == "sell"]
            
            analytics["trade_statistics"] = {
                "total_trades": len(trades),
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "avg_trade_size": np.mean([t["value"] for t in trades]),
                "total_commission": sum(t.get("commission", 0) for t in trades),
                "avg_holding_period": 5.0  # Simplified calculation
            }
    
    return analytics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

