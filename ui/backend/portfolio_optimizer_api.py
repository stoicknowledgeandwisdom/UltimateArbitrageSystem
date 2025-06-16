#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimizer API Backend
==============================

Fast API backend service for the Advanced AI Portfolio Optimizer.
Provides RESTful endpoints for portfolio optimization, risk analysis,
and real-time performance monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import numpy as np

# Import our portfolio optimizer
try:
    from ..ai.portfolio_quantum_optimizer import QuantumPortfolioOptimizer, OptimizationResult, PortfolioMetrics
except ImportError:
    # Fallback import for testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ai.portfolio_quantum_optimizer import QuantumPortfolioOptimizer, OptimizationResult, PortfolioMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class OptimizationRequest(BaseModel):
    """Portfolio optimization request."""
    assets: List[str] = Field(..., description="List of asset symbols")
    returns_data: Optional[Dict[str, List[float]]] = Field(None, description="Historical returns data")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Additional market data")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Portfolio constraints")
    config: Optional[Dict[str, Any]] = Field(None, description="Optimizer configuration")

class PortfolioWeights(BaseModel):
    """Portfolio weights for metrics calculation."""
    weights: Dict[str, float] = Field(..., description="Asset weights")
    returns_data: Optional[Dict[str, List[float]]] = Field(None, description="Historical returns data")
    benchmark_returns: Optional[List[float]] = Field(None, description="Benchmark returns")

class RebalanceRequest(BaseModel):
    """Portfolio rebalancing request."""
    current_weights: Dict[str, float] = Field(..., description="Current portfolio weights")
    target_weights: Dict[str, float] = Field(..., description="Target portfolio weights")
    transaction_costs: Optional[float] = Field(0.001, description="Transaction costs")
    max_turnover: Optional[float] = Field(0.5, description="Maximum turnover")

class RiskAnalysisRequest(BaseModel):
    """Risk analysis request."""
    weights: Dict[str, float] = Field(..., description="Portfolio weights")
    returns_data: Dict[str, List[float]] = Field(..., description="Historical returns data")
    confidence_levels: Optional[List[float]] = Field([0.95, 0.99], description="VaR confidence levels")
    stress_scenarios: Optional[List[Dict[str, Any]]] = Field(None, description="Stress test scenarios")

# FastAPI app
app = FastAPI(
    title="Advanced AI Portfolio Optimizer API",
    description="Quantum-enhanced portfolio optimization with AI-driven insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global optimizer instance
optimizer = None
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the optimizer on startup."""
    global optimizer
    
    # Default configuration
    config = {
        'quantum_enabled': True,
        'ai_enhancement': True,
        'risk_tolerance': 0.15,
        'target_return': 0.12,
        'use_black_litterman': True,
        'use_regime_detection': True,
        'use_factor_models': True,
        'dynamic_constraints': True
    }
    
    optimizer = QuantumPortfolioOptimizer(config)
    logger.info("Portfolio Optimizer API started successfully")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced AI Portfolio Optimizer API",
        "version": "1.0.0",
        "features": [
            "Quantum-enhanced optimization",
            "AI-driven return forecasting",
            "Real-time risk analysis",
            "Multi-objective optimization",
            "Advanced performance metrics"
        ],
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "optimizer_ready": optimizer is not None,
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections)
    }

@app.post("/optimize", response_model=Dict[str, Any])
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio allocation using quantum-enhanced AI algorithms.
    
    Returns optimal weights, expected returns, risk metrics, and confidence scores.
    """
    try:
        if optimizer is None:
            raise HTTPException(status_code=500, detail="Optimizer not initialized")
        
        logger.info(f"Starting portfolio optimization for {len(request.assets)} assets")
        
        # Prepare returns data
        if request.returns_data:
            returns_df = pd.DataFrame(request.returns_data)
        else:
            # Generate sample data for demo
            returns_df = generate_sample_returns(request.assets)
        
        # Update optimizer config if provided
        if request.config:
            for key, value in request.config.items():
                setattr(optimizer, key, value)
        
        # Perform optimization
        result = await optimizer.optimize_portfolio(
            assets=request.assets,
            returns_data=returns_df,
            market_data=request.market_data,
            constraints=request.constraints
        )
        
        # Convert result to dict for JSON serialization
        response = {
            "optimization_result": {
                "weights": result.weights,
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "optimization_method": result.optimization_method,
                "convergence_status": result.convergence_status,
                "iterations": result.iterations,
                "quantum_enhancement": result.quantum_enhancement,
                "ai_enhancement": result.ai_enhancement,
                "confidence_score": result.confidence_score,
                "risk_metrics": result.risk_metrics,
                "timestamp": result.timestamp.isoformat()
            },
            "recommendations": generate_recommendations(result),
            "next_rebalance": (datetime.now() + timedelta(days=1)).isoformat(),
            "status": "success"
        }
        
        # Broadcast to connected websockets
        await broadcast_update("optimization_complete", response)
        
        logger.info(f"Portfolio optimization completed successfully. Sharpe ratio: {result.sharpe_ratio:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics", response_model=Dict[str, Any])
async def calculate_metrics(request: PortfolioWeights):
    """
    Calculate comprehensive portfolio performance metrics.
    
    Returns detailed risk-return metrics, drawdown analysis, and performance ratios.
    """
    try:
        if optimizer is None:
            raise HTTPException(status_code=500, detail="Optimizer not initialized")
        
        logger.info("Calculating portfolio metrics")
        
        # Prepare data
        if request.returns_data:
            returns_df = pd.DataFrame(request.returns_data)
        else:
            # Generate sample data
            assets = list(request.weights.keys())
            returns_df = generate_sample_returns(assets)
        
        benchmark_series = None
        if request.benchmark_returns:
            benchmark_series = pd.Series(request.benchmark_returns)
        
        # Calculate metrics
        metrics = await optimizer.calculate_portfolio_metrics(
            weights=request.weights,
            returns_data=returns_df,
            benchmark_returns=benchmark_series
        )
        
        # Convert to dict
        response = {
            "metrics": {
                "total_value": metrics.total_value,
                "daily_return": metrics.daily_return,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
                "expected_return": metrics.expected_return,
                "beta": metrics.beta,
                "alpha": metrics.alpha,
                "information_ratio": metrics.information_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "treynor_ratio": metrics.treynor_ratio,
                "quantum_advantage": metrics.quantum_advantage,
                "ai_confidence": metrics.ai_confidence,
                "risk_score": metrics.risk_score,
                "diversification_ratio": metrics.diversification_ratio,
                "concentration_risk": metrics.concentration_risk
            },
            "risk_assessment": assess_risk_level(metrics),
            "performance_grade": grade_performance(metrics),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-analysis", response_model=Dict[str, Any])
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Perform comprehensive risk analysis including VaR, stress testing, and scenario analysis.
    """
    try:
        logger.info("Performing risk analysis")
        
        # Prepare data
        returns_df = pd.DataFrame(request.returns_data)
        weights_array = np.array([request.weights[asset] for asset in returns_df.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights_array).sum(axis=1)
        
        # Value at Risk calculation
        var_results = {}
        for conf_level in request.confidence_levels:
            var_value = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            var_results[f"var_{int(conf_level*100)}"] = float(var_value)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = portfolio_returns[portfolio_returns <= var_results["var_95"]].mean()
        
        # Risk decomposition
        risk_contributions = calculate_risk_contributions(weights_array, returns_df)
        
        # Stress testing
        stress_results = {}
        if request.stress_scenarios:
            stress_results = perform_stress_tests(request.weights, request.stress_scenarios)
        else:
            # Default stress scenarios
            stress_results = default_stress_tests(portfolio_returns)
        
        # Monte Carlo simulation
        mc_results = monte_carlo_simulation(portfolio_returns, 1000)
        
        response = {
            "var_analysis": var_results,
            "expected_shortfall_95": float(es_95),
            "risk_contributions": risk_contributions,
            "stress_test_results": stress_results,
            "monte_carlo": mc_results,
            "risk_summary": {
                "overall_risk_level": classify_risk_level(var_results["var_95"]),
                "diversification_benefit": calculate_diversification_benefit(weights_array),
                "concentration_warning": check_concentration_risk(weights_array),
                "volatility_regime": classify_volatility_regime(portfolio_returns)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebalance", response_model=Dict[str, Any])
async def calculate_rebalancing(request: RebalanceRequest):
    """
    Calculate optimal rebalancing trades considering transaction costs and constraints.
    """
    try:
        logger.info("Calculating portfolio rebalancing")
        
        current_weights = np.array(list(request.current_weights.values()))
        target_weights = np.array(list(request.target_weights.values()))
        
        # Calculate required trades
        trades = target_weights - current_weights
        
        # Calculate turnover
        turnover = np.sum(np.abs(trades))
        
        # Check if rebalancing is needed
        if turnover < 0.05:  # 5% threshold
            return {
                "rebalancing_needed": False,
                "turnover": float(turnover),
                "recommendation": "No rebalancing needed - portfolio is close to target",
                "timestamp": datetime.now().isoformat()
            }
        
        # Apply turnover constraint
        if turnover > request.max_turnover:
            # Scale down trades
            scaling_factor = request.max_turnover / turnover
            trades = trades * scaling_factor
            adjusted_target = current_weights + trades
        else:
            adjusted_target = target_weights
        
        # Calculate transaction costs
        transaction_cost = turnover * request.transaction_costs
        
        # Calculate trade details
        assets = list(request.current_weights.keys())
        trade_details = []
        
        for i, asset in enumerate(assets):
            if abs(trades[i]) > 0.001:  # Minimum trade threshold
                trade_details.append({
                    "asset": asset,
                    "current_weight": float(current_weights[i]),
                    "target_weight": float(adjusted_target[i]),
                    "trade_amount": float(trades[i]),
                    "trade_direction": "buy" if trades[i] > 0 else "sell"
                })
        
        response = {
            "rebalancing_needed": True,
            "turnover": float(turnover),
            "transaction_cost": float(transaction_cost),
            "trade_details": trade_details,
            "adjusted_targets": {
                asset: float(adjusted_target[i]) 
                for i, asset in enumerate(assets)
            },
            "rebalancing_efficiency": calculate_rebalancing_efficiency(
                current_weights, adjusted_target, transaction_cost
            ),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Rebalancing calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to Portfolio Optimizer API",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for messages or send periodic updates
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            
            if websocket in active_connections:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "optimizer_status": "active" if optimizer else "inactive"
                })
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# Utility functions

def generate_sample_returns(assets: List[str], days: int = 252) -> pd.DataFrame:
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Different return characteristics for different assets
    returns_data = {}
    for i, asset in enumerate(assets):
        # Vary mean and volatility by asset
        mean_return = 0.0008 + (i * 0.0002)  # 0.08% to 0.16% daily
        volatility = 0.015 + (i * 0.005)     # 1.5% to 3.0% daily
        
        returns_data[asset] = np.random.normal(mean_return, volatility, days)
    
    return pd.DataFrame(returns_data, index=dates)

def generate_recommendations(result: OptimizationResult) -> List[Dict[str, Any]]:
    """Generate investment recommendations based on optimization result."""
    recommendations = []
    
    # Sharpe ratio recommendation
    if result.sharpe_ratio > 2.0:
        recommendations.append({
            "type": "performance",
            "priority": "high",
            "message": f"Excellent risk-adjusted returns (Sharpe: {result.sharpe_ratio:.2f})",
            "action": "Consider increasing allocation to this strategy"
        })
    elif result.sharpe_ratio < 1.0:
        recommendations.append({
            "type": "performance",
            "priority": "medium",
            "message": f"Low risk-adjusted returns (Sharpe: {result.sharpe_ratio:.2f})",
            "action": "Review strategy or reduce allocation"
        })
    
    # Concentration recommendation
    max_weight = max(result.weights.values())
    if max_weight > 0.4:
        recommendations.append({
            "type": "diversification",
            "priority": "high",
            "message": f"High concentration risk (max weight: {max_weight:.1%})",
            "action": "Consider reducing largest position and diversifying"
        })
    
    # Quantum enhancement recommendation
    if result.quantum_enhancement and result.confidence_score > 0.8:
        recommendations.append({
            "type": "technology",
            "priority": "low",
            "message": "Quantum enhancement is providing significant benefits",
            "action": "Continue leveraging quantum optimization features"
        })
    
    return recommendations

def assess_risk_level(metrics: PortfolioMetrics) -> Dict[str, Any]:
    """Assess overall portfolio risk level."""
    
    # Risk factors
    factors = {
        "volatility": "high" if metrics.volatility > 0.2 else "medium" if metrics.volatility > 0.1 else "low",
        "max_drawdown": "high" if abs(metrics.max_drawdown) > 0.2 else "medium" if abs(metrics.max_drawdown) > 0.1 else "low",
        "concentration": "high" if metrics.concentration_risk > 0.3 else "medium" if metrics.concentration_risk > 0.2 else "low",
        "diversification": "low" if metrics.diversification_ratio > 0.8 else "medium" if metrics.diversification_ratio > 0.6 else "high"
    }
    
    # Overall assessment
    high_risk_count = sum(1 for level in factors.values() if level == "high")
    
    if high_risk_count >= 2:
        overall = "high"
    elif high_risk_count == 1:
        overall = "medium"
    else:
        overall = "low"
    
    return {
        "overall_risk": overall,
        "risk_factors": factors,
        "risk_score": metrics.risk_score,
        "recommendations": generate_risk_recommendations(factors)
    }

def grade_performance(metrics: PortfolioMetrics) -> Dict[str, Any]:
    """Grade portfolio performance."""
    
    # Grading criteria
    sharpe_grade = "A" if metrics.sharpe_ratio > 2 else "B" if metrics.sharpe_ratio > 1.5 else "C" if metrics.sharpe_ratio > 1 else "D"
    return_grade = "A" if metrics.expected_return > 0.15 else "B" if metrics.expected_return > 0.1 else "C" if metrics.expected_return > 0.05 else "D"
    risk_grade = "A" if metrics.volatility < 0.1 else "B" if metrics.volatility < 0.15 else "C" if metrics.volatility < 0.2 else "D"
    
    # Overall grade
    grades = [sharpe_grade, return_grade, risk_grade]
    grade_values = {"A": 4, "B": 3, "C": 2, "D": 1}
    avg_grade = sum(grade_values[g] for g in grades) / len(grades)
    
    overall_grade = "A" if avg_grade >= 3.5 else "B" if avg_grade >= 2.5 else "C" if avg_grade >= 1.5 else "D"
    
    return {
        "overall_grade": overall_grade,
        "component_grades": {
            "risk_adjusted_return": sharpe_grade,
            "expected_return": return_grade,
            "risk_management": risk_grade
        },
        "score": round(avg_grade, 2)
    }

def calculate_risk_contributions(weights: np.ndarray, returns_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate risk contribution of each asset."""
    
    # Portfolio variance
    cov_matrix = returns_df.cov().values
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    
    # Marginal risk contributions
    marginal_contrib = 2 * np.dot(cov_matrix, weights)
    
    # Risk contributions
    risk_contrib = weights * marginal_contrib / portfolio_var
    
    return {
        asset: float(risk_contrib[i]) 
        for i, asset in enumerate(returns_df.columns)
    }

def perform_stress_tests(weights: Dict[str, float], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform custom stress tests."""
    # Placeholder for custom stress testing
    return {"message": "Custom stress testing not implemented yet"}

def default_stress_tests(portfolio_returns: pd.Series) -> Dict[str, float]:
    """Perform default stress tests."""
    
    # Historical stress scenarios
    worst_day = portfolio_returns.min()
    worst_week = portfolio_returns.rolling(5).sum().min()
    worst_month = portfolio_returns.rolling(20).sum().min()
    
    return {
        "worst_single_day": float(worst_day),
        "worst_week": float(worst_week),
        "worst_month": float(worst_month),
        "market_crash_simulation": float(worst_day * 2),  # 2x worst day
        "volatility_spike": float(portfolio_returns.std() * 3)  # 3 sigma event
    }

def monte_carlo_simulation(portfolio_returns: pd.Series, n_simulations: int) -> Dict[str, Any]:
    """Perform Monte Carlo simulation."""
    
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    
    # Simulate future returns
    simulations = np.random.normal(mean_return, volatility, (n_simulations, 252))  # 1 year
    
    # Calculate cumulative returns
    cumulative_returns = (1 + simulations).cumprod(axis=1)
    final_values = cumulative_returns[:, -1]
    
    return {
        "expected_value_1y": float(np.mean(final_values)),
        "value_at_risk_5_1y": float(np.percentile(final_values, 5)),
        "value_at_risk_1_1y": float(np.percentile(final_values, 1)),
        "probability_of_loss": float(np.mean(final_values < 1.0)),
        "probability_of_gain_10": float(np.mean(final_values > 1.10))
    }

def classify_risk_level(var_95: float) -> str:
    """Classify risk level based on VaR."""
    if var_95 < -0.05:
        return "high"
    elif var_95 < -0.03:
        return "medium"
    else:
        return "low"

def calculate_diversification_benefit(weights: np.ndarray) -> float:
    """Calculate diversification benefit."""
    # Herfindahl-Hirschman Index
    hhi = np.sum(weights ** 2)
    max_diversification = 1.0 / len(weights)  # Equal weights
    
    benefit = (1 - hhi) / (1 - max_diversification)
    return float(np.clip(benefit, 0, 1))

def check_concentration_risk(weights: np.ndarray) -> bool:
    """Check if portfolio has concentration risk."""
    max_weight = np.max(weights)
    top_3_weight = np.sum(np.sort(weights)[-3:])
    
    return max_weight > 0.4 or top_3_weight > 0.7

def classify_volatility_regime(portfolio_returns: pd.Series) -> str:
    """Classify current volatility regime."""
    current_vol = portfolio_returns.tail(30).std()
    historical_vol = portfolio_returns.std()
    
    vol_ratio = current_vol / historical_vol
    
    if vol_ratio > 1.5:
        return "high"
    elif vol_ratio > 1.2:
        return "elevated"
    elif vol_ratio < 0.8:
        return "low"
    else:
        return "normal"

def calculate_rebalancing_efficiency(current: np.ndarray, target: np.ndarray, cost: float) -> float:
    """Calculate rebalancing efficiency score."""
    # Distance to target
    distance = np.sum(np.abs(target - current))
    
    # Efficiency (inverse of cost per unit distance)
    if distance > 0:
        efficiency = 1 / (cost / distance + 1)
    else:
        efficiency = 1.0
    
    return float(efficiency)

def generate_risk_recommendations(risk_factors: Dict[str, str]) -> List[str]:
    """Generate risk management recommendations."""
    recommendations = []
    
    if risk_factors["volatility"] == "high":
        recommendations.append("Consider reducing portfolio volatility through diversification")
    
    if risk_factors["concentration"] == "high":
        recommendations.append("Reduce position sizes in largest holdings")
    
    if risk_factors["diversification"] == "high":
        recommendations.append("Improve diversification across asset classes")
    
    return recommendations

async def broadcast_update(message_type: str, data: Any):
    """Broadcast update to all connected WebSocket clients."""
    if active_connections:
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connections
        for connection in active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                active_connections.remove(connection)

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "portfolio_optimizer_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

