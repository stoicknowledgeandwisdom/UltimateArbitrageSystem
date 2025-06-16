"""Main orchestrator for the Real-Time Optimization & ML Module."""

import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd
from enum import Enum

# Import all components
from .feature_store import FeastFeatureStore, FeatureConfig
from .streaming_etl import FlinkETLPipeline, FlinkConfig
from .reinforcement_learning import DDPGAgent, DDPGConfig
from .meta_controller import ContextualBanditController, ContextualBanditConfig, MarketRegime, TradingStrategy
from .ops import DeviceManager, DeviceManagerConfig
from .safety import ExplainabilityDashboard, ExplanationConfig


class SystemStatus(Enum):
    """System status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class MLOptimizationConfig:
    """Master configuration for ML optimization system."""
    # Component configurations
    feature_store_config: FeatureConfig = field(default_factory=FeatureConfig)
    etl_config: FlinkConfig = field(default_factory=FlinkConfig)
    ddpg_config: DDPGConfig = field(default_factory=DDPGConfig)
    bandit_config: ContextualBanditConfig = field(default_factory=ContextualBanditConfig)
    device_config: DeviceManagerConfig = field(default_factory=DeviceManagerConfig)
    explanation_config: ExplanationConfig = field(default_factory=ExplanationConfig)
    
    # System-wide settings
    enable_feature_store: bool = True
    enable_streaming_etl: bool = True
    enable_reinforcement_learning: bool = True
    enable_meta_controller: bool = True
    enable_explainability: bool = True
    enable_safety_checks: bool = True
    
    # Performance settings
    max_concurrent_models: int = 5
    model_update_interval: float = 300.0  # 5 minutes
    feature_update_interval: float = 1.0  # 1 second
    health_check_interval: float = 30.0  # 30 seconds
    
    # Risk management
    max_position_size: float = 1.0
    max_daily_loss: float = 0.1  # 10%
    emergency_stop_loss: float = 0.05  # 5%
    risk_budget_per_strategy: float = 0.02  # 2%


@dataclass
class TradingDecision:
    """Trading decision output from the ML system."""
    timestamp: datetime
    symbol: str
    exchange: str
    strategy: str
    action: str  # 'buy', 'sell', 'hold'
    position_size: float
    confidence: float
    expected_return: float
    risk_score: float
    features_used: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None
    vetoed: bool = False
    veto_reason: Optional[str] = None


class MLOptimizationOrchestrator:
    """Main orchestrator for the Real-Time Optimization & ML Module."""
    
    def __init__(self, config: MLOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # System status
        self.status = SystemStatus.INITIALIZING
        self.error_message: Optional[str] = None
        self.start_time: Optional[datetime] = None
        
        # Component instances
        self.device_manager: Optional[DeviceManager] = None
        self.feature_store: Optional[FeastFeatureStore] = None
        self.etl_pipeline: Optional[FlinkETLPipeline] = None
        self.rl_agents: Dict[str, DDPGAgent] = {}
        self.meta_controller: Optional[ContextualBanditController] = None
        self.explainability_dashboard: Optional[ExplainabilityDashboard] = None
        
        # Model registry
        self.active_models: Dict[str, Any] = {}
        self.shadow_models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Decision tracking
        self.trading_decisions: deque = deque(maxlen=10000)
        self.execution_results: deque = deque(maxlen=10000)
        
        # Risk management
        self.current_positions: Dict[str, float] = defaultdict(float)
        self.daily_pnl: float = 0.0
        self.risk_budget_used: Dict[str, float] = defaultdict(float)
        
        # Performance monitoring
        self.system_metrics: Dict[str, Any] = {
            'decisions_per_second': 0.0,
            'latency_ms': 0.0,
            'feature_freshness': 0.0,
            'model_accuracy': 0.0,
            'system_uptime': 0.0
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread management
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing ML Optimization System...")
            
            # Initialize device manager first (needed for GPU/TPU detection)
            if self.config.device_config:
                self.device_manager = DeviceManager(self.config.device_config)
                self.logger.info(f"Device manager initialized with primary device: {self.device_manager.primary_device}")
            
            # Initialize feature store
            if self.config.enable_feature_store:
                self.feature_store = FeastFeatureStore(self.config.feature_store_config)
                self.logger.info("Feature store initialized")
            
            # Initialize streaming ETL
            if self.config.enable_streaming_etl:
                self.etl_pipeline = FlinkETLPipeline(self.config.etl_config)
                self.logger.info("Streaming ETL pipeline initialized")
            
            # Initialize meta-controller
            if self.config.enable_meta_controller:
                self.meta_controller = ContextualBanditController(self.config.bandit_config)
                self.logger.info("Meta-controller initialized")
            
            # Initialize explainability dashboard
            if self.config.enable_explainability:
                self.explainability_dashboard = ExplainabilityDashboard(self.config.explanation_config)
                self.logger.info("Explainability dashboard initialized")
            
            # Initialize RL agents (will be created on-demand)
            if self.config.enable_reinforcement_learning:
                self.logger.info("Reinforcement learning enabled")
            
            self.status = SystemStatus.RUNNING
            self.logger.info("ML Optimization System initialization complete")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.error_message = str(e)
            self.logger.error(f"Failed to initialize ML Optimization System: {e}")
            raise
    
    async def start(self):
        """Start the ML optimization system."""
        try:
            self.start_time = datetime.now()
            self.logger.info("Starting ML Optimization System...")
            
            # Start streaming ETL
            if self.etl_pipeline:
                await self.etl_pipeline.start()
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._feature_update_loop()),
                asyncio.create_task(self._model_update_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._risk_monitoring_loop())
            ]
            
            self.status = SystemStatus.RUNNING
            self.logger.info("ML Optimization System started successfully")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.error_message = str(e)
            self.logger.error(f"Failed to start ML Optimization System: {e}")
            raise
    
    async def stop(self):
        """Stop the ML optimization system."""
        try:
            self.logger.info("Stopping ML Optimization System...")
            self.shutdown_event.set()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Stop streaming ETL
            if self.etl_pipeline:
                await self.etl_pipeline.stop()
            
            # Stop explainability dashboard
            if self.explainability_dashboard:
                self.explainability_dashboard.stop_dashboard()
            
            # Stop device manager monitoring
            if self.device_manager:
                self.device_manager.stop_monitoring()
            
            self.status = SystemStatus.STOPPED
            self.logger.info("ML Optimization System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping ML Optimization System: {e}")
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        """Process incoming market data and generate trading decisions."""
        try:
            start_time = datetime.now()
            
            # Extract basic info
            symbol = market_data.get('symbol', '')
            exchange = market_data.get('exchange', '')
            
            if not symbol or not exchange:
                return None
            
            # Process through ETL pipeline
            if self.etl_pipeline:
                await self.etl_pipeline.process_message({
                    'type': 'market_data',
                    'symbol': symbol,
                    'exchange': exchange,
                    'data': market_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Extract features
            features = await self._extract_features(symbol, exchange, market_data)
            
            if not features:
                return None
            
            # Select strategy using meta-controller
            strategy_info = None
            if self.meta_controller:
                strategy, confidence, strategy_info = await self.meta_controller.select_strategy(features)
                if not strategy:
                    return None
            else:
                strategy = "default"
                confidence = 0.5
            
            # Get RL agent for strategy
            rl_agent = await self._get_or_create_rl_agent(strategy)
            
            # Generate action using RL agent
            feature_array = self._features_to_array(features)
            action = rl_agent.select_action(feature_array, add_noise=True)
            action_info = rl_agent.get_action_info(action)
            
            # Calculate expected return and risk
            expected_return = self._calculate_expected_return(action_info, features)
            risk_score = self._calculate_risk_score(action_info, features, symbol, exchange)
            
            # Check risk limits
            if not self._check_risk_limits(symbol, action_info, risk_score):
                return None
            
            # Create trading decision
            decision = TradingDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                exchange=exchange,
                strategy=strategy,
                action=self._action_to_string(action_info),
                position_size=action_info['position_size'],
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                features_used=features
            )
            
            # Generate explanation if enabled
            if self.explainability_dashboard and self.config.enable_explainability:
                try:
                    explanation = await self.explainability_dashboard.explain_prediction(
                        f"rl_agent_{strategy}",
                        feature_array,
                        expected_return,
                        {'strategy': strategy, 'action_info': action_info}
                    )
                    decision.explanation = {
                        'method': explanation.explanation_method,
                        'confidence': explanation.confidence,
                        'top_features': self._get_top_explanation_features(explanation)
                    }
                except Exception as e:
                    self.logger.error(f"Error generating explanation: {e}")
            
            # Apply safety checks
            if self.config.enable_safety_checks:
                veto_result = await self._apply_safety_checks(decision)
                if veto_result:
                    decision.vetoed = True
                    decision.veto_reason = veto_result
                    self.logger.warning(f"Decision vetoed: {veto_result}")
            
            # Store decision
            with self._lock:
                self.trading_decisions.append(decision)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.system_metrics['latency_ms'] = processing_time
            
            # Trigger event handlers
            await self._trigger_event('decision_generated', decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
    
    async def _extract_features(self, symbol: str, exchange: str, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML models."""
        try:
            features = {}
            
            # Get features from feature store if available
            if self.feature_store:
                entity_keys = {'symbol': symbol, 'exchange': exchange}
                feature_names = self.feature_store.get_feature_names()
                
                online_features = await self.feature_store.get_online_features(
                    entity_keys, feature_names
                )
                features.update(online_features)
            
            # Add raw market data features
            numeric_fields = ['price', 'volume', 'bid_price', 'ask_price', 'spread']
            for field in numeric_fields:
                if field in market_data:
                    features[f'raw_{field}'] = float(market_data[field])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        # Use consistent feature ordering
        feature_names = sorted(features.keys())
        return np.array([features.get(name, 0.0) for name in feature_names])
    
    async def _get_or_create_rl_agent(self, strategy: str) -> DDPGAgent:
        """Get or create RL agent for strategy."""
        if strategy not in self.rl_agents:
            # Create new RL agent
            config = DDPGConfig()
            if self.device_manager:
                config.device = self.device_manager.get_torch_device().type
            
            agent = DDPGAgent(config)
            self.rl_agents[strategy] = agent
            
            self.logger.info(f"Created new RL agent for strategy: {strategy}")
        
        return self.rl_agents[strategy]
    
    def _calculate_expected_return(self, action_info: Dict[str, Any], features: Dict[str, float]) -> float:
        """Calculate expected return for action."""
        # Simple expected return calculation
        # In practice, this would use more sophisticated models
        position_size = abs(action_info.get('position_size', 0.0))
        price_momentum = features.get('price_change_1m', 0.0)
        volatility = features.get('price_volatility_1m', 0.01)
        
        # Expected return based on momentum and position size
        expected_return = position_size * price_momentum / (1.0 + volatility)
        return float(expected_return)
    
    def _calculate_risk_score(self, action_info: Dict[str, Any], features: Dict[str, float], 
                             symbol: str, exchange: str) -> float:
        """Calculate risk score for action."""
        # Risk score calculation
        position_size = abs(action_info.get('position_size', 0.0))
        volatility = features.get('price_volatility_5m', 0.01)
        spread = features.get('spread_pct', 0.001)
        
        # Base risk from position size and volatility
        base_risk = position_size * volatility
        
        # Liquidity risk from spread
        liquidity_risk = spread * position_size
        
        # Concentration risk from existing positions
        current_position = abs(self.current_positions.get(f"{symbol}:{exchange}", 0.0))
        concentration_risk = current_position * 0.1
        
        total_risk = base_risk + liquidity_risk + concentration_risk
        return float(min(1.0, total_risk))
    
    def _check_risk_limits(self, symbol: str, action_info: Dict[str, Any], risk_score: float) -> bool:
        """Check if action is within risk limits."""
        try:
            position_size = abs(action_info.get('position_size', 0.0))
            
            # Check maximum position size
            if position_size > self.config.max_position_size:
                return False
            
            # Check daily loss limit
            if self.daily_pnl < -self.config.max_daily_loss:
                return False
            
            # Check emergency stop loss
            if self.daily_pnl < -self.config.emergency_stop_loss:
                return False
            
            # Check risk score
            if risk_score > 0.8:  # High risk threshold
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _action_to_string(self, action_info: Dict[str, Any]) -> str:
        """Convert action info to string."""
        position_size = action_info.get('position_size', 0.0)
        
        if position_size > 0.1:
            return 'buy'
        elif position_size < -0.1:
            return 'sell'
        else:
            return 'hold'
    
    def _get_top_explanation_features(self, explanation) -> List[Dict[str, Any]]:
        """Get top features from explanation."""
        if not explanation.shap_values:
            return []
        
        # Sort by absolute importance
        sorted_features = sorted(
            explanation.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        return [{'feature': f, 'importance': v} for f, v in sorted_features]
    
    async def _apply_safety_checks(self, decision: TradingDecision) -> Optional[str]:
        """Apply safety checks to trading decision."""
        try:
            # Check for extreme position sizes
            if abs(decision.position_size) > 0.9:
                return "Extreme position size detected"
            
            # Check for low confidence
            if decision.confidence < 0.3:
                return "Low confidence decision"
            
            # Check for high risk
            if decision.risk_score > 0.7:
                return "High risk score"
            
            # Check for recent similar decisions (avoid over-trading)
            recent_decisions = [d for d in list(self.trading_decisions)[-10:] 
                             if d.symbol == decision.symbol and d.exchange == decision.exchange]
            
            if len(recent_decisions) > 3:
                return "Too many recent decisions for this symbol"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in safety checks: {e}")
            return "Safety check error"
    
    async def update_execution_result(self, decision_id: str, result: Dict[str, Any]):
        """Update system with execution results."""
        try:
            # Find the corresponding decision
            decision = None
            for d in reversed(self.trading_decisions):
                if str(id(d)) == decision_id:  # Simple ID matching
                    decision = d
                    break
            
            if not decision:
                return
            
            # Extract result information
            executed = result.get('executed', False)
            pnl = result.get('pnl', 0.0)
            execution_price = result.get('execution_price', 0.0)
            quantity = result.get('quantity', 0.0)
            
            # Update positions
            position_key = f"{decision.symbol}:{decision.exchange}"
            if executed:
                if decision.action == 'buy':
                    self.current_positions[position_key] += quantity
                elif decision.action == 'sell':
                    self.current_positions[position_key] -= quantity
            
            # Update daily PnL
            self.daily_pnl += pnl
            
            # Update RL agent
            if decision.strategy in self.rl_agents:
                agent = self.rl_agents[decision.strategy]
                
                # Create next state (simplified)
                next_features = decision.features_used
                next_state = self._features_to_array(next_features)
                
                # Calculate reward
                reward = self._calculate_reward(pnl, executed, decision)
                
                # Store transition (simplified - in practice, you'd need the actual state)
                feature_array = self._features_to_array(decision.features_used)
                action_array = np.array([decision.position_size, 0.5, 0.5])  # Simplified
                
                agent.store_transition(
                    feature_array, action_array, reward, next_state, not executed
                )
                
                # Update agent
                if len(agent.replay_buffer) > agent.config.batch_size:
                    update_info = agent.update()
                    self.logger.debug(f"Agent update: {update_info}")
            
            # Update meta-controller
            if self.meta_controller:
                trade_result = {
                    'pnl': pnl,
                    'duration_minutes': 5.0,  # Simplified
                    'success': executed and pnl > 0,
                    'market_data': decision.features_used,
                    'regime': MarketRegime.UNKNOWN  # Would detect from current state
                }
                
                await self.meta_controller.update_strategy_performance(
                    decision.strategy, trade_result
                )
            
            # Store execution result
            execution_result = {
                'timestamp': datetime.now(),
                'decision': decision,
                'result': result,
                'pnl': pnl,
                'executed': executed
            }
            
            with self._lock:
                self.execution_results.append(execution_result)
            
            # Trigger event
            await self._trigger_event('execution_result', execution_result)
            
        except Exception as e:
            self.logger.error(f"Error updating execution result: {e}")
    
    def _calculate_reward(self, pnl: float, executed: bool, decision: TradingDecision) -> float:
        """Calculate reward for RL agent."""
        if not executed:
            return -0.1  # Small penalty for not executing
        
        # Normalize PnL
        normalized_pnl = np.tanh(pnl * 100)
        
        # Confidence bonus
        confidence_bonus = (decision.confidence - 0.5) * 0.2
        
        # Risk penalty
        risk_penalty = decision.risk_score * 0.1
        
        reward = normalized_pnl + confidence_bonus - risk_penalty
        return float(np.clip(reward, -1.0, 1.0))
    
    async def _feature_update_loop(self):
        """Background loop for feature updates."""
        while not self.shutdown_event.is_set():
            try:
                # Update feature freshness metric
                if self.feature_store:
                    # Calculate feature freshness (simplified)
                    self.system_metrics['feature_freshness'] = 0.95
                
                await asyncio.sleep(self.config.feature_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in feature update loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _model_update_loop(self):
        """Background loop for model updates."""
        while not self.shutdown_event.is_set():
            try:
                # Update model performance metrics
                for strategy, agent in self.rl_agents.items():
                    stats = agent.get_agent_stats()
                    self.model_performance[strategy] = {
                        'total_steps': stats['total_steps'],
                        'training_mode': stats['training_mode'],
                        'buffer_size': stats['buffer_size']
                    }
                
                # Calculate overall model accuracy (simplified)
                if self.execution_results:
                    recent_results = list(self.execution_results)[-100:]
                    successful = sum(1 for r in recent_results if r['pnl'] > 0)
                    self.system_metrics['model_accuracy'] = successful / len(recent_results)
                
                await asyncio.sleep(self.config.model_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in model update loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _health_check_loop(self):
        """Background loop for health checks."""
        while not self.shutdown_event.is_set():
            try:
                # Update system uptime
                if self.start_time:
                    uptime = (datetime.now() - self.start_time).total_seconds()
                    self.system_metrics['system_uptime'] = uptime
                
                # Check component health
                component_health = await self._check_component_health()
                
                if not all(component_health.values()):
                    self.logger.warning(f"Component health issues: {component_health}")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _risk_monitoring_loop(self):
        """Background loop for risk monitoring."""
        while not self.shutdown_event.is_set():
            try:
                # Check daily PnL against limits
                if self.daily_pnl < -self.config.emergency_stop_loss:
                    self.logger.critical(f"Emergency stop loss triggered: {self.daily_pnl}")
                    await self._trigger_event('emergency_stop', {'pnl': self.daily_pnl})
                
                # Check position concentrations
                total_exposure = sum(abs(pos) for pos in self.current_positions.values())
                if total_exposure > self.config.max_position_size * 5:
                    self.logger.warning(f"High total exposure: {total_exposure}")
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _check_component_health(self) -> Dict[str, bool]:
        """Check health of all components."""
        health = {}
        
        try:
            # Check ETL pipeline
            if self.etl_pipeline:
                etl_health = await self.etl_pipeline.health_check()
                health['etl_pipeline'] = etl_health.get('status') == 'healthy'
            
            # Check feature store
            if self.feature_store:
                health['feature_store'] = True  # Simplified check
            
            # Check device manager
            if self.device_manager:
                device_stats = self.device_manager.get_all_device_stats()
                health['device_manager'] = len(device_stats['devices']) > 0
            
            # Check RL agents
            health['rl_agents'] = len(self.rl_agents) > 0
            
            # Check meta-controller
            if self.meta_controller:
                controller_stats = self.meta_controller.get_controller_stats()
                health['meta_controller'] = controller_stats['bandit_arms'] > 0
            
        except Exception as e:
            self.logger.error(f"Error checking component health: {e}")
        
        return health
    
    async def _trigger_event(self, event_type: str, data: Any):
        """Trigger event handlers."""
        try:
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Error triggering event {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        self.event_handlers[event_type].append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            return {
                'status': self.status.value,
                'error_message': self.error_message,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'metrics': self.system_metrics.copy(),
                'components': {
                    'feature_store': self.feature_store is not None,
                    'etl_pipeline': self.etl_pipeline is not None,
                    'meta_controller': self.meta_controller is not None,
                    'explainability': self.explainability_dashboard is not None,
                    'device_manager': self.device_manager is not None
                },
                'active_models': len(self.rl_agents),
                'decisions_generated': len(self.trading_decisions),
                'execution_results': len(self.execution_results),
                'current_positions': dict(self.current_positions),
                'daily_pnl': self.daily_pnl
            }
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading decisions."""
        with self._lock:
            recent = list(self.trading_decisions)[-count:]
            return [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'symbol': d.symbol,
                    'exchange': d.exchange,
                    'strategy': d.strategy,
                    'action': d.action,
                    'position_size': d.position_size,
                    'confidence': d.confidence,
                    'expected_return': d.expected_return,
                    'risk_score': d.risk_score,
                    'vetoed': d.vetoed,
                    'veto_reason': d.veto_reason
                }
                for d in recent
            ]
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

