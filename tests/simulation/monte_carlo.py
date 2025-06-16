"""Monte Carlo Market Simulation Framework

Generate synthetic volatility and volume scenarios for comprehensive testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from scipy import stats
from scipy.stats import norm, lognorm, gamma, beta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


class MarketRegime(Enum):
    """Market regime types"""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    CRASH = "crash"
    BUBBLE = "bubble"


@dataclass
class MarketParameters:
    """Parameters for market simulation"""
    # Price dynamics
    initial_price: float = 50000.0
    drift: float = 0.0  # Annual drift
    volatility: float = 0.20  # Annual volatility
    
    # Jump diffusion parameters
    jump_intensity: float = 0.1  # Jumps per year
    jump_mean: float = 0.0  # Mean jump size
    jump_std: float = 0.05  # Jump volatility
    
    # Volume parameters
    base_volume: float = 1000.0
    volume_volatility: float = 0.5
    volume_correlation: float = 0.3  # Correlation with price changes
    
    # Microstructure parameters
    bid_ask_spread: float = 0.001  # 10 bps spread
    market_impact: float = 0.0001  # Price impact coefficient
    orderbook_depth: int = 20  # Number of levels
    
    # Regime switching
    regime_switch_probability: float = 0.05  # Daily probability
    regime_persistence: float = 0.95  # Probability of staying in regime
    
    # Intraday patterns
    intraday_volatility_pattern: List[float] = field(default_factory=lambda: 
        [1.2, 1.0, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,  # Morning
         1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8])  # Afternoon/Evening
    
    intraday_volume_pattern: List[float] = field(default_factory=lambda:
        [1.5, 1.2, 0.8, 0.6, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6,
         1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.7, 0.8, 1.0, 1.3, 1.8, 2.0])


@dataclass 
class SimulationResult:
    """Result of Monte Carlo simulation"""
    prices: np.ndarray
    volumes: np.ndarray
    returns: np.ndarray
    timestamps: np.ndarray
    regimes: np.ndarray
    orderbooks: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    parameters: MarketParameters


class MonteCarloSimulator:
    """Advanced Monte Carlo market simulator"""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Precompute normal random numbers for performance
        self._random_cache_size = 100000
        self._refresh_random_cache()
        self._random_index = 0
    
    def _refresh_random_cache(self):
        """Refresh cache of random numbers"""
        self._normal_cache = np.random.normal(0, 1, self._random_cache_size)
        self._uniform_cache = np.random.uniform(0, 1, self._random_cache_size)
        self._random_index = 0
    
    def _get_random_normal(self) -> float:
        """Get cached random normal"""
        if self._random_index >= self._random_cache_size:
            self._refresh_random_cache()
        
        value = self._normal_cache[self._random_index]
        self._random_index += 1
        return value
    
    def _get_random_uniform(self) -> float:
        """Get cached random uniform"""
        if self._random_index >= self._random_cache_size:
            self._refresh_random_cache()
        
        value = self._uniform_cache[self._random_index]
        self._random_index += 1
        return value
    
    def simulate_price_path(self,
                           params: MarketParameters,
                           duration_days: int = 30,
                           timestep_minutes: int = 1) -> SimulationResult:
        """Simulate complete price path with multiple factors"""
        
        # Calculate simulation parameters
        total_minutes = duration_days * 24 * 60
        n_steps = total_minutes // timestep_minutes
        dt = timestep_minutes / (24 * 60 * 365)  # Convert to years
        
        self.logger.info(f"Simulating {n_steps} steps over {duration_days} days")
        
        # Initialize arrays
        prices = np.zeros(n_steps + 1)
        volumes = np.zeros(n_steps + 1)
        returns = np.zeros(n_steps)
        regimes = np.zeros(n_steps + 1, dtype=int)
        timestamps = np.array([datetime.now() + timedelta(minutes=i * timestep_minutes) 
                              for i in range(n_steps + 1)])
        
        # Set initial values
        prices[0] = params.initial_price
        volumes[0] = params.base_volume
        current_regime = MarketRegime.NORMAL
        regimes[0] = list(MarketRegime).index(current_regime)
        
        # Storage for detailed data
        orderbooks = []
        trades = []
        
        # Simulation loop
        for i in range(n_steps):
            # Update regime if needed
            if self._get_random_uniform() < params.regime_switch_probability * dt * 365:
                current_regime = self._switch_regime(current_regime, params)
            regimes[i + 1] = list(MarketRegime).index(current_regime)
            
            # Get regime-adjusted parameters
            regime_params = self._get_regime_parameters(params, current_regime)
            
            # Generate price movement
            price_change = self._generate_price_change(
                prices[i], regime_params, dt, timestamps[i]
            )
            prices[i + 1] = prices[i] * (1 + price_change)
            returns[i] = price_change
            
            # Generate volume
            volumes[i + 1] = self._generate_volume(
                params, price_change, timestamps[i]
            )
            
            # Generate orderbook if requested (every 10 steps to save memory)
            if i % 10 == 0:
                orderbook = self._generate_orderbook(
                    prices[i + 1], params, timestamps[i + 1]
                )
                orderbooks.append(orderbook)
                
                # Generate some trades
                if i > 0:
                    trade_count = np.random.poisson(volumes[i + 1] / 100)
                    for _ in range(min(trade_count, 10)):
                        trade = self._generate_trade(
                            prices[i + 1], volumes[i + 1], timestamps[i + 1]
                        )
                        trades.append(trade)
        
        # Calculate statistics
        statistics = self._calculate_statistics(prices, returns, volumes)
        
        return SimulationResult(
            prices=prices,
            volumes=volumes,
            returns=returns,
            timestamps=timestamps,
            regimes=regimes,
            orderbooks=orderbooks,
            trades=trades,
            statistics=statistics,
            parameters=params
        )
    
    def _switch_regime(self, current_regime: MarketRegime, 
                      params: MarketParameters) -> MarketRegime:
        """Switch market regime based on probabilities"""
        # Stay in current regime with high probability
        if self._get_random_uniform() < params.regime_persistence:
            return current_regime
        
        # Switch to new regime
        regimes = list(MarketRegime)
        regimes.remove(current_regime)  # Don't switch to same regime
        
        # Weight probabilities based on current regime
        if current_regime == MarketRegime.CRASH:
            # After crash, likely to go to mean reverting or normal
            weights = [0.4, 0.1, 0.1, 0.1, 0.1, 0.2, 0.0]  # Exclude BUBBLE
        elif current_regime == MarketRegime.BUBBLE:
            # After bubble, likely to crash or high volatility
            weights = [0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.2]  # Include CRASH
        else:
            # Equal probabilities for other transitions
            weights = [1.0 / len(regimes)] * len(regimes)
        
        return np.random.choice(regimes, p=weights)
    
    def _get_regime_parameters(self, base_params: MarketParameters,
                              regime: MarketRegime) -> MarketParameters:
        """Adjust parameters based on market regime"""
        params = MarketParameters(**base_params.__dict__)  # Copy
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            params.volatility *= 2.0
            params.jump_intensity *= 3.0
            params.volume_volatility *= 1.5
        
        elif regime == MarketRegime.LOW_VOLATILITY:
            params.volatility *= 0.5
            params.jump_intensity *= 0.3
            params.volume_volatility *= 0.7
        
        elif regime == MarketRegime.TRENDING_UP:
            params.drift = 0.15  # 15% annual drift
            params.volatility *= 0.8
        
        elif regime == MarketRegime.TRENDING_DOWN:
            params.drift = -0.10  # -10% annual drift
            params.volatility *= 1.2
        
        elif regime == MarketRegime.MEAN_REVERTING:
            params.drift *= -0.5  # Mean reversion
            params.volatility *= 0.6
        
        elif regime == MarketRegime.CRASH:
            params.drift = -0.50  # -50% annual drift
            params.volatility *= 3.0
            params.jump_intensity *= 10.0
            params.jump_mean = -0.05  # Negative jumps
        
        elif regime == MarketRegime.BUBBLE:
            params.drift = 0.30  # 30% annual drift
            params.volatility *= 1.5
            params.jump_mean = 0.02  # Positive jumps
        
        return params
    
    def _generate_price_change(self, current_price: float,
                              params: MarketParameters,
                              dt: float,
                              timestamp: datetime) -> float:
        """Generate price change using jump diffusion model"""
        
        # Intraday pattern adjustment
        hour = timestamp.hour
        intraday_vol_factor = params.intraday_volatility_pattern[hour]
        
        # Geometric Brownian Motion component
        gbm_change = (params.drift - 0.5 * params.volatility**2) * dt + \
                     params.volatility * np.sqrt(dt) * self._get_random_normal() * intraday_vol_factor
        
        # Jump component (Poisson jumps)
        jump_change = 0.0
        if self._get_random_uniform() < params.jump_intensity * dt:
            jump_size = params.jump_mean + params.jump_std * self._get_random_normal()
            jump_change = jump_size
        
        return gbm_change + jump_change
    
    def _generate_volume(self, params: MarketParameters,
                        price_change: float,
                        timestamp: datetime) -> float:
        """Generate trading volume correlated with price movements"""
        
        # Base volume with intraday pattern
        hour = timestamp.hour
        intraday_vol_factor = params.intraday_volume_pattern[hour]
        base_vol = params.base_volume * intraday_vol_factor
        
        # Volume correlation with price changes
        correlation_component = params.volume_correlation * abs(price_change) * base_vol
        
        # Random component
        random_component = params.volume_volatility * base_vol * abs(self._get_random_normal())
        
        # Ensure positive volume
        volume = max(base_vol * 0.1, base_vol + correlation_component + random_component)
        
        return volume
    
    def _generate_orderbook(self, price: float,
                           params: MarketParameters,
                           timestamp: datetime) -> Dict[str, Any]:
        """Generate realistic orderbook"""
        
        spread = params.bid_ask_spread * price
        mid_price = price
        
        bids = []
        asks = []
        
        for i in range(params.orderbook_depth):
            # Exponential distribution of levels
            level_spacing = spread * (1 + i * 0.1)
            
            bid_price = mid_price - spread/2 - i * level_spacing
            ask_price = mid_price + spread/2 + i * level_spacing
            
            # Volume decreases with distance from mid
            volume_factor = np.exp(-i * 0.2)
            bid_volume = params.base_volume * volume_factor * (0.5 + 0.5 * self._get_random_uniform())
            ask_volume = params.base_volume * volume_factor * (0.5 + 0.5 * self._get_random_uniform())
            
            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])
        
        return {
            'timestamp': timestamp.isoformat(),
            'timestamp_ns': int(timestamp.timestamp() * 1_000_000_000),
            'symbol': 'BTC/USDT',
            'bids': bids,
            'asks': asks,
            'mid_price': mid_price,
            'spread': spread
        }
    
    def _generate_trade(self, price: float, volume: float,
                       timestamp: datetime) -> Dict[str, Any]:
        """Generate individual trade"""
        
        # Random trade size (exponential distribution)
        trade_size = np.random.exponential(volume / 100)
        
        # Random side
        side = 'buy' if self._get_random_uniform() > 0.5 else 'sell'
        
        # Small price variation around mid
        price_variation = price * 0.0001 * self._get_random_normal()
        trade_price = price + price_variation
        
        return {
            'timestamp': timestamp.isoformat(),
            'timestamp_ns': int(timestamp.timestamp() * 1_000_000_000),
            'symbol': 'BTC/USDT',
            'side': side,
            'amount': trade_size,
            'price': trade_price,
            'id': f"trade_{int(timestamp.timestamp() * 1000)}_{np.random.randint(1000, 9999)}"
        }
    
    def _calculate_statistics(self, prices: np.ndarray,
                             returns: np.ndarray,
                             volumes: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        
        return {
            'price_statistics': {
                'initial_price': float(prices[0]),
                'final_price': float(prices[-1]),
                'min_price': float(np.min(prices)),
                'max_price': float(np.max(prices)),
                'total_return': float((prices[-1] / prices[0]) - 1),
                'max_drawdown': float(self._calculate_max_drawdown(prices))
            },
            'return_statistics': {
                'mean_return': float(np.mean(returns)),
                'volatility': float(np.std(returns) * np.sqrt(365 * 24 * 60)),  # Annualized
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'sharpe_ratio': float(np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 60)),
                'var_95': float(np.percentile(returns, 5)),
                'var_99': float(np.percentile(returns, 1))
            },
            'volume_statistics': {
                'mean_volume': float(np.mean(volumes)),
                'total_volume': float(np.sum(volumes)),
                'volume_volatility': float(np.std(volumes) / np.mean(volumes)),
                'max_volume': float(np.max(volumes)),
                'min_volume': float(np.min(volumes))
            }
        }
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)
    
    def run_monte_carlo(self,
                       params: MarketParameters,
                       n_simulations: int = 1000,
                       duration_days: int = 30,
                       parallel: bool = True) -> List[SimulationResult]:
        """Run multiple Monte Carlo simulations"""
        
        self.logger.info(f"Running {n_simulations} Monte Carlo simulations")
        
        if parallel and n_simulations > 1:
            # Use multiprocessing for parallel execution
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = []
                for i in range(n_simulations):
                    # Create new simulator for each process
                    future = executor.submit(
                        self._run_single_simulation,
                        params, duration_days, i
                    )
                    futures.append(future)
                
                results = [future.result() for future in futures]
        else:
            # Sequential execution
            results = []
            for i in range(n_simulations):
                result = self.simulate_price_path(params, duration_days)
                results.append(result)
        
        self.logger.info(f"Completed {n_simulations} simulations")
        return results
    
    @staticmethod
    def _run_single_simulation(params: MarketParameters,
                              duration_days: int,
                              seed: int) -> SimulationResult:
        """Run single simulation (for multiprocessing)"""
        simulator = MonteCarloSimulator(random_seed=seed)
        return simulator.simulate_price_path(params, duration_days)
    
    def analyze_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze Monte Carlo results"""
        
        final_prices = [r.prices[-1] for r in results]
        total_returns = [(r.prices[-1] / r.prices[0]) - 1 for r in results]
        max_drawdowns = [r.statistics['price_statistics']['max_drawdown'] for r in results]
        volatilities = [r.statistics['return_statistics']['volatility'] for r in results]
        
        return {
            'final_price_stats': {
                'mean': float(np.mean(final_prices)),
                'std': float(np.std(final_prices)),
                'min': float(np.min(final_prices)),
                'max': float(np.max(final_prices)),
                'percentiles': {
                    '5%': float(np.percentile(final_prices, 5)),
                    '25%': float(np.percentile(final_prices, 25)),
                    '50%': float(np.percentile(final_prices, 50)),
                    '75%': float(np.percentile(final_prices, 75)),
                    '95%': float(np.percentile(final_prices, 95))
                }
            },
            'return_stats': {
                'mean': float(np.mean(total_returns)),
                'std': float(np.std(total_returns)),
                'min': float(np.min(total_returns)),
                'max': float(np.max(total_returns)),
                'probability_positive': float(np.mean(np.array(total_returns) > 0)),
                'percentiles': {
                    '5%': float(np.percentile(total_returns, 5)),
                    '25%': float(np.percentile(total_returns, 25)),
                    '50%': float(np.percentile(total_returns, 50)),
                    '75%': float(np.percentile(total_returns, 75)),
                    '95%': float(np.percentile(total_returns, 95))
                }
            },
            'risk_stats': {
                'mean_max_drawdown': float(np.mean(max_drawdowns)),
                'worst_drawdown': float(np.min(max_drawdowns)),
                'mean_volatility': float(np.mean(volatilities)),
                'max_volatility': float(np.max(volatilities))
            },
            'simulation_count': len(results)
        }
    
    def save_results(self, results: List[SimulationResult],
                    output_path: Path) -> None:
        """Save simulation results to files"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary analysis
        analysis = self.analyze_results(results)
        with open(output_path / 'monte_carlo_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save individual results (compressed)
        for i, result in enumerate(results):
            result_data = {
                'prices': result.prices.tolist(),
                'volumes': result.volumes.tolist(),
                'returns': result.returns.tolist(),
                'timestamps': [t.isoformat() for t in result.timestamps],
                'regimes': result.regimes.tolist(),
                'statistics': result.statistics,
                'parameters': result.parameters.__dict__
            }
            
            with open(output_path / f'simulation_{i:04d}.json', 'w') as f:
                json.dump(result_data, f)
        
        self.logger.info(f"Saved {len(results)} simulation results to {output_path}")

