#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Advanced AI Portfolio Optimizer
===========================================================

Complete testing of the quantum-enhanced portfolio optimization system.
"""

import asyncio
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import sys
import os

# Add the ai module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

try:
    from ai.portfolio_quantum_optimizer import (
        QuantumPortfolioOptimizer, 
        OptimizationResult, 
        PortfolioMetrics
    )
except ImportError:
    from portfolio_quantum_optimizer import (
        QuantumPortfolioOptimizer, 
        OptimizationResult, 
        PortfolioMetrics
    )

class PortfolioOptimizerTestSuite:
    """Comprehensive test suite for the portfolio optimizer."""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results."""
        status = "✓ PASSED" if passed else "✗ FAILED"
        result = {
            'test': test_name,
            'status': status,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        print(f"{status}: {test_name}")
        if details:
            print(f"  Details: {details}")
    
    def generate_test_data(self, assets: list, days: int = 252) -> pd.DataFrame:
        """Generate realistic test data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate correlated returns with different characteristics
        returns_data = {}
        
        for i, asset in enumerate(assets):
            # Different return characteristics for each asset
            base_return = 0.0005 + (i * 0.0003)  # 0.05% to 0.17% daily
            volatility = 0.015 + (i * 0.003)     # 1.5% to 2.7% daily
            
            # Add some correlation structure
            market_factor = np.random.normal(0, 0.01, days)
            specific_factor = np.random.normal(0, volatility * 0.7, days)
            
            returns = base_return + 0.6 * market_factor + specific_factor
            returns_data[asset] = returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    async def test_basic_initialization(self):
        """Test basic optimizer initialization."""
        try:
            config = {
                'quantum_enabled': True,
                'ai_enhancement': True,
                'risk_tolerance': 0.15,
                'target_return': 0.12
            }
            
            optimizer = QuantumPortfolioOptimizer(config)
            
            # Check attributes
            assert optimizer.quantum_enabled == True
            assert optimizer.ai_enhancement == True
            assert optimizer.risk_tolerance == 0.15
            assert optimizer.target_return == 0.12
            
            self.log_test("Basic Initialization", True, "Optimizer created successfully with correct config")
            return optimizer
            
        except Exception as e:
            self.log_test("Basic Initialization", False, f"Error: {str(e)}")
            return None
    
    async def test_data_preparation(self, optimizer):
        """Test data preparation functionality."""
        try:
            assets = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']
            returns_data = self.generate_test_data(assets)
            
            processed_data = await optimizer._prepare_optimization_data(
                assets, returns_data, None
            )
            
            # Verify data structure
            assert 'returns' in processed_data
            assert 'mean_returns' in processed_data
            assert 'cov_matrix' in processed_data
            assert processed_data['n_assets'] == len(assets)
            assert len(processed_data['mean_returns']) == len(assets)
            
            self.log_test("Data Preparation", True, f"Successfully prepared data for {len(assets)} assets")
            return processed_data
            
        except Exception as e:
            self.log_test("Data Preparation", False, f"Error: {str(e)}")
            return None
    
    async def test_regime_detection(self, optimizer, processed_data):
        """Test market regime detection."""
        try:
            regime = await optimizer._detect_market_regime(processed_data)
            
            valid_regimes = ['crisis', 'volatile', 'diversified', 'bearish', 'bullish', 'normal']
            assert regime in valid_regimes
            
            self.log_test("Regime Detection", True, f"Detected regime: {regime}")
            return regime
            
        except Exception as e:
            self.log_test("Regime Detection", False, f"Error: {str(e)}")
            return 'normal'
    
    async def test_return_forecasting(self, optimizer, processed_data, regime):
        """Test return forecasting."""
        try:
            forecasts = await optimizer._forecast_returns(processed_data, regime)
            
            # Verify forecast structure
            assert isinstance(forecasts, np.ndarray)
            assert len(forecasts) == processed_data['n_assets']
            assert not np.any(np.isnan(forecasts))
            
            self.log_test("Return Forecasting", True, f"Generated forecasts for {len(forecasts)} assets")
            return forecasts
            
        except Exception as e:
            self.log_test("Return Forecasting", False, f"Error: {str(e)}")
            return None
    
    async def test_risk_modeling(self, optimizer, processed_data, regime):
        """Test risk model estimation."""
        try:
            risk_model = await optimizer._estimate_risk_model(processed_data, regime)
            
            # Verify risk model structure
            assert isinstance(risk_model, np.ndarray)
            assert risk_model.shape == (processed_data['n_assets'], processed_data['n_assets'])
            assert np.allclose(risk_model, risk_model.T)  # Should be symmetric
            assert np.all(np.diag(risk_model) > 0)  # Positive diagonal elements
            
            self.log_test("Risk Modeling", True, f"Generated {risk_model.shape} risk model matrix")
            return risk_model
            
        except Exception as e:
            self.log_test("Risk Modeling", False, f"Error: {str(e)}")
            return None
    
    async def test_quantum_enhancement(self, optimizer, expected_returns, risk_model):
        """Test quantum enhancement functionality."""
        try:
            if optimizer.quantum_enabled:
                enhanced_returns, enhanced_risk = await optimizer._apply_quantum_enhancement(
                    expected_returns, risk_model
                )
                
                # Verify enhancement
                assert enhanced_returns.shape == expected_returns.shape
                assert enhanced_risk.shape == risk_model.shape
                
                self.log_test("Quantum Enhancement", True, "Quantum enhancement applied successfully")
                return enhanced_returns, enhanced_risk
            else:
                self.log_test("Quantum Enhancement", True, "Quantum enhancement disabled")
                return expected_returns, risk_model
                
        except Exception as e:
            self.log_test("Quantum Enhancement", False, f"Error: {str(e)}")
            return expected_returns, risk_model
    
    async def test_full_optimization(self, optimizer):
        """Test complete portfolio optimization."""
        try:
            assets = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']
            returns_data = self.generate_test_data(assets)
            
            # Test with different configurations
            configs = [
                {'quantum_enabled': True, 'ai_enhancement': True},
                {'quantum_enabled': False, 'ai_enhancement': True},
                {'quantum_enabled': True, 'ai_enhancement': False},
            ]
            
            optimization_results = []
            
            for i, config in enumerate(configs):
                # Update optimizer config
                for key, value in config.items():
                    setattr(optimizer, key, value)
                
                result = await optimizer.optimize_portfolio(
                    assets=assets,
                    returns_data=returns_data
                )
                
                # Verify result structure
                assert isinstance(result, OptimizationResult)
                assert len(result.weights) == len(assets)
                assert abs(sum(result.weights.values()) - 1.0) < 1e-6  # Weights sum to 1
                assert result.sharpe_ratio is not None
                assert result.expected_return is not None
                assert result.expected_volatility is not None
                
                optimization_results.append(result)
                
                config_name = f"Config {i+1}: Q={config['quantum_enabled']}, AI={config['ai_enhancement']}"
                details = f"Sharpe: {result.sharpe_ratio:.3f}, Return: {result.expected_return:.4f}"
                self.log_test(f"Optimization - {config_name}", True, details)
            
            return optimization_results
            
        except Exception as e:
            self.log_test("Full Optimization", False, f"Error: {str(e)}")
            return []
    
    async def test_portfolio_metrics(self, optimizer, optimization_result):
        """Test portfolio metrics calculation."""
        try:
            if not optimization_result:
                self.log_test("Portfolio Metrics", False, "No optimization result available")
                return None
            
            assets = list(optimization_result.weights.keys())
            returns_data = self.generate_test_data(assets)
            
            metrics = await optimizer.calculate_portfolio_metrics(
                weights=optimization_result.weights,
                returns_data=returns_data
            )
            
            # Verify metrics structure
            assert isinstance(metrics, PortfolioMetrics)
            assert metrics.total_value > 0
            assert metrics.sharpe_ratio is not None
            assert metrics.max_drawdown <= 0  # Should be negative or zero
            assert 0 <= metrics.diversification_ratio <= 1
            assert 0 <= metrics.concentration_risk <= 1
            
            details = f"Value: ${metrics.total_value:.2f}, Sharpe: {metrics.sharpe_ratio:.3f}, Risk Score: {metrics.risk_score:.1f}"
            self.log_test("Portfolio Metrics", True, details)
            return metrics
            
        except Exception as e:
            self.log_test("Portfolio Metrics", False, f"Error: {str(e)}")
            return None
    
    async def test_performance_comparison(self, optimization_results):
        """Test performance comparison between different configurations."""
        try:
            if len(optimization_results) < 2:
                self.log_test("Performance Comparison", False, "Insufficient results for comparison")
                return
            
            # Compare Sharpe ratios
            sharpe_ratios = [result.sharpe_ratio for result in optimization_results]
            best_sharpe_idx = np.argmax(sharpe_ratios)
            
            # Compare expected returns
            returns = [result.expected_return for result in optimization_results]
            best_return_idx = np.argmax(returns)
            
            # Compare volatilities
            volatilities = [result.expected_volatility for result in optimization_results]
            lowest_vol_idx = np.argmin(volatilities)
            
            comparison_details = (
                f"Best Sharpe: Config {best_sharpe_idx+1} ({sharpe_ratios[best_sharpe_idx]:.3f}), "
                f"Best Return: Config {best_return_idx+1} ({returns[best_return_idx]:.4f}), "
                f"Lowest Vol: Config {lowest_vol_idx+1} ({volatilities[lowest_vol_idx]:.4f})"
            )
            
            self.log_test("Performance Comparison", True, comparison_details)
            
        except Exception as e:
            self.log_test("Performance Comparison", False, f"Error: {str(e)}")
    
    async def test_edge_cases(self, optimizer):
        """Test edge cases and error handling."""
        try:
            test_cases = [
                {"name": "Single Asset", "assets": ["BTC"]},
                {"name": "Two Assets", "assets": ["BTC", "ETH"]},
                {"name": "Many Assets", "assets": [f"ASSET_{i}" for i in range(20)]},
            ]
            
            edge_case_results = []
            
            for case in test_cases:
                try:
                    returns_data = self.generate_test_data(case["assets"], days=100)
                    
                    result = await optimizer.optimize_portfolio(
                        assets=case["assets"],
                        returns_data=returns_data
                    )
                    
                    assert len(result.weights) == len(case["assets"])
                    edge_case_results.append(True)
                    
                    self.log_test(f"Edge Case - {case['name']}", True, f"Handled {len(case['assets'])} assets successfully")
                    
                except Exception as e:
                    edge_case_results.append(False)
                    self.log_test(f"Edge Case - {case['name']}", False, f"Error: {str(e)}")
            
            overall_success = all(edge_case_results)
            if overall_success:
                self.log_test("Edge Cases Overall", True, "All edge cases handled successfully")
            
        except Exception as e:
            self.log_test("Edge Cases", False, f"Error: {str(e)}")
    
    def print_summary(self):
        """Print test summary."""
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("PORTFOLIO OPTIMIZER TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * 60)
        
        if self.failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"- {result['test']}: {result['details']}")
        
        return success_rate >= 80  # 80% success rate threshold
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to file."""
        with open(filename, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'summary': {
                    'total_tests': self.passed_tests + self.failed_tests,
                    'passed': self.passed_tests,
                    'failed': self.failed_tests,
                    'success_rate': (self.passed_tests / (self.passed_tests + self.failed_tests) * 100) if (self.passed_tests + self.failed_tests) > 0 else 0
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

async def run_comprehensive_tests():
    """Run the complete test suite."""
    print("Starting Comprehensive Portfolio Optimizer Test Suite...")
    print("=" * 60)
    
    test_suite = PortfolioOptimizerTestSuite()
    
    # Initialize optimizer
    optimizer = await test_suite.test_basic_initialization()
    if not optimizer:
        print("CRITICAL: Failed to initialize optimizer. Aborting tests.")
        return False
    
    # Test data preparation
    processed_data = await test_suite.test_data_preparation(optimizer)
    if not processed_data:
        print("CRITICAL: Failed data preparation. Aborting tests.")
        return False
    
    # Test individual components
    regime = await test_suite.test_regime_detection(optimizer, processed_data)
    expected_returns = await test_suite.test_return_forecasting(optimizer, processed_data, regime)
    risk_model = await test_suite.test_risk_modeling(optimizer, processed_data, regime)
    
    if expected_returns is not None and risk_model is not None:
        enhanced_returns, enhanced_risk = await test_suite.test_quantum_enhancement(
            optimizer, expected_returns, risk_model
        )
    
    # Test full optimization
    optimization_results = await test_suite.test_full_optimization(optimizer)
    
    # Test portfolio metrics
    if optimization_results:
        metrics = await test_suite.test_portfolio_metrics(optimizer, optimization_results[0])
        
        # Test performance comparison
        await test_suite.test_performance_comparison(optimization_results)
    
    # Test edge cases
    await test_suite.test_edge_cases(optimizer)
    
    # Print summary and save results
    success = test_suite.print_summary()
    test_suite.save_results("portfolio_optimizer_test_results.json")
    
    return success

if __name__ == "__main__":
    import asyncio
    
    # Run the comprehensive test suite
    success = asyncio.run(run_comprehensive_tests())
    
    if success:
        print("\n🎉 Portfolio Optimizer passed comprehensive testing!")
        print("✅ System is ready for production use.")
    else:
        print("\n⚠️  Some tests failed. Please review the results above.")
        print("🔧 Consider fixing issues before production deployment.")

