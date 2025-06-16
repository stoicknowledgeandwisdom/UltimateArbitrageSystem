"""
StrategyManagerExtension - Extends the StrategyManager with weight optimization capabilities.

This module provides advanced functionality for managing strategy weights, including:
- Gradual weight adjustments over time
- Historical weight tracking
- Weight change notifications
- Weight snapshot creation for analysis and rollback
- Performance metrics correlation with weight changes
- Automated circuit breakers for strategy weight adjustments
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
import pickle

logger = logging.getLogger(__name__)

class StrategyManagerExtension:
    """
    Extension for the StrategyManager class that provides advanced weight management 
    capabilities. This extension is designed to work with the StrategyOptimizer for
    AI-driven strategy allocation.
    
    Features:
    - Gradual weight adjustments to minimize market impact
    - Historical weight tracking for analysis and backtesting
    - Weight change notifications for system components
    - Weight snapshots for rollback and recovery
    - Performance correlation analysis between weight changes and results
    - Automated circuit breakers for weight adjustments
    """

    def __init__(self, strategy_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StrategyManagerExtension.

        Args:
            strategy_manager: The StrategyManager instance to extend
            config: Configuration dictionary for the extension
        """
        self.strategy_manager = strategy_manager
        self.config = config or {}
        
        # Initialize configuration settings with defaults
        self.weight_history_dir = self.config.get('weight_history_dir', 'data/weight_history')
        self.snapshot_dir = self.config.get('snapshot_dir', 'data/weight_snapshots')
        self.snapshot_interval = self.config.get('snapshot_interval', 3600)  # 1 hour
        self.gradual_change_enabled = self.config.get('gradual_change_enabled', True)
        self.max_immediate_change = self.config.get('max_immediate_change', 0.10)  # 10% max immediate change
        self.notify_changes = self.config.get('notify_changes', True)
        self.max_snapshots = self.config.get('max_snapshots', 50)  # Maximum number of snapshots to keep
        self.history_retention_days = self.config.get('history_retention_days', 30)
        
        # Risk management settings
        self.emergency_fallback_weights = self.config.get('emergency_fallback_weights', {})
        self.circuit_breaker_enabled = self.config.get('circuit_breaker_enabled', True)
        self.min_strategy_count = self.config.get('min_strategy_count', 3)  # Minimum active strategies
        
        # Performance tracking
        self.track_performance = self.config.get('track_performance', True)
        self.performance_correlation_window = self.config.get('performance_correlation_window', 7)  # days
        
        # Internal state
        self.weight_history = []
        self.last_snapshot_time = time.time()
        self.weight_change_lock = threading.Lock()
        self.active_gradual_changes = {}
        self.last_notification_time = {}  # To prevent notification spam
        self.notification_min_interval = 60  # seconds between notifications for same strategy
        
        # Create necessary directories
        os.makedirs(self.weight_history_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Load any existing history
        self._load_weight_history()
        
        # Register with event system if available
        self._register_with_event_system()
            
        logger.info(f"StrategyManagerExtension initialized with snapshot interval: {self.snapshot_interval}s")

    def update_strategy_weights(self, new_weights: Dict[str, float], 
                                reason: str = "optimizer_update", 
                                immediate: bool = False) -> Dict[str, float]:
        """
        Update the weights of strategies based on the new weights provided.

        Args:
            new_weights: Dictionary mapping strategy IDs to new weight values
            reason: The reason for the weight update
            immediate: Whether to apply changes immediately or gradually

        Returns:
            The actual weights applied after constraints
        """
        with self.weight_change_lock:
            logger.info(f"Updating strategy weights. Reason: {reason}")
            
            # Get current weights
            current_weights = self._get_current_weights()
            
            # Check for emergency circuit breaker conditions
            if self.circuit_breaker_enabled and self._check_circuit_breaker(new_weights):
                logger.warning("Circuit breaker triggered - using emergency fallback weights")
                if self.emergency_fallback_weights:
                    new_weights = self.emergency_fallback_weights
                else:
                    logger.warning("No emergency fallback weights defined - keeping current weights")
                    return current_weights
            
            # Normalize the new weights
            normalized_weights = self._normalize_weights(new_weights)
            
            # Apply category constraints to ensure balance across strategy types
            constrained_weights = self._apply_category_constraints(normalized_weights)
            
            # Validate with risk controller
            approved_weights = self._validate_with_risk_controller(constrained_weights)
            
            # If gradual changes are enabled and not immediate, schedule the changes
            if self.gradual_change_enabled and not immediate:
                self._schedule_gradual_changes(current_weights, approved_weights)
                actual_weights = current_weights  # Return current weights, as changes will be gradual
            else:
                # Apply the changes immediately
                self._apply_weight_changes(approved_weights)
                actual_weights = approved_weights
            
            # Save the weight history
            self._save_weight_history(actual_weights, reason)
            
            # Create snapshot if needed
            current_time = time.time()
            if current_time - self.last_snapshot_time >= self.snapshot_interval:
                self._create_weight_snapshot(actual_weights)
                self.last_snapshot_time = current_time
            
            # Notify about weight changes
            if self.notify_changes:
                self._notify_weight_changes(current_weights, actual_weights, reason)
            
            # If performance tracking is enabled, analyze correlations
            if self.track_performance:
                self._analyze_performance_correlation()
            
            return actual_weights

    def _check_circuit_breaker(self, new_weights: Dict[str, float]) -> bool:
        """
        Check if circuit breaker conditions are met to prevent dangerous weight allocations.
        
        Args:
            new_weights: The proposed new weights
            
        Returns:
            True if circuit breaker should be triggered, False otherwise
        """
        # Check if too few strategies would be active
        active_count = sum(1 for w in new_weights.values() if w > 0.01)
        if active_count < self.min_strategy_count:
            logger.warning(f"Circuit breaker: only {active_count} strategies would be active (min: {self.min_strategy_count})")
            return True
            
        # Check if any single strategy has too much weight
        max_weight = max(new_weights.values()) if new_weights else 0
        max_single_strategy_weight = self.config.get('max_single_strategy_weight', 0.5)
        if max_weight > max_single_strategy_weight:
            logger.warning(f"Circuit breaker: strategy weight {max_weight:.2f} exceeds maximum {max_single_strategy_weight:.2f}")
            return True
            
        # Check if weights sum is valid
        total_weight = sum(new_weights.values())
        if total_weight < 0.9 or total_weight > 1.1:
            logger.warning(f"Circuit breaker: weight sum {total_weight:.2f} is outside valid range")
            return True
            
        return False

    def _get_current_weights(self) -> Dict[str, float]:
        """
        Get the current weights of all strategies.
        
        Returns:
            Dictionary mapping strategy IDs to current weight values
        """
        weights = {}
        for strategy_id, strategy in self.strategy_manager.strategies.items():
            weights[strategy_id] = getattr(strategy, 'weight', 1.0)
        
        # Normalize weights if they don't sum to 1.0
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            for strategy_id in weights:
                weights[strategy_id] /= total
                
        return weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to ensure they sum to 1.0.
        
        Args:
            weights: Dictionary mapping strategy IDs to weight values
            
        Returns:
            Normalized weights dictionary
        """
        total = sum(weights.values())
        if total <= 0:
            # If total is zero or negative, assign equal weights
            equal_weight = 1.0 / len(weights) if weights else 0.0
            return {strategy_id: equal_weight for strategy_id in weights}
        
        # Normalize
        return {strategy_id: weight / total for strategy_id, weight in weights.items()}

    def _apply_category_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply constraints to ensure balance across strategy categories.
        
        Args:
            weights: Dictionary mapping strategy IDs to weight values
            
        Returns:
            Dictionary with weights adjusted according to category constraints
        """
        # Get category constraints from config
        category_constraints = self.config.get('category_constraints', {})
        if not category_constraints:
            return weights  # No constraints to apply
        
        # Group strategies by category
        categories = {}
        for strategy_id, strategy in self.strategy_manager.strategies.items():
            category = getattr(strategy, 'category', 'default')
            if category not in categories:
                categories[category] = []
            categories[category].append(strategy_id)
        
        # Calculate current weight per category
        category_weights = {}
        for category, strategy_ids in categories.items():
            category_weights[category] = sum(weights.get(sid, 0.0) for sid in strategy_ids)
        
        # Adjust weights to meet constraints
        adjusted_weights = copy.deepcopy(weights)
        
        for category, min_max in category_constraints.items():
            if category not in categories:
                continue  # Skip if no strategies in this category
                
            min_weight = min_max.get('min', 0.0)
            max_weight = min_max.get('max', 1.0)
            
            # If current weight for this category is outside the constraints
            current = category_weights.get(category, 0.0)
            strategy_ids = categories[category]
            
            if current < min_weight and strategy_ids:
                # Increase weights to meet minimum
                scale_factor = min_weight / max(current, 0.0001)
                for sid in strategy_ids:
                    adjusted_weights[sid] = weights.get(sid, 0.0) * scale_factor
                    
            elif current > max_weight and strategy_ids:
                # Decrease weights to meet maximum
                scale_factor = max_weight / current
                for sid in strategy_ids:
                    adjusted_weights[sid] = weights.get(sid, 0.0) * scale_factor
        
        # Re-normalize
        return self._normalize_weights(adjusted_weights)

    def _validate_with_risk_controller(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate the proposed weights with the risk controller.
        
        Args:
            weights: Dictionary mapping strategy IDs to weight values
            
        Returns:
            Dictionary with weights approved by the risk controller
        """
        risk_controller = getattr(self.strategy_manager, 'risk_controller', None)
        if not risk_controller:
            return weights  # No risk controller available
        
        try:
            # Check if the risk controller has a validate_weights method
            if hasattr(risk_controller, 'validate_strategy_weights'):
                return risk_controller.validate_strategy_weights(weights)
            
            # Check if risk controller has a simpler validate method
            if hasattr(risk_controller, 'validate'):
                validation = risk_controller.validate({
                    'action': 'update_strategy_weights',
                    'weights': weights
                })
                if validation.get('approved', True):
                    return validation.get('modified_weights', weights)
                else:
                    logger.warning("Risk controller rejected weight changes")
                    return self._get_current_weights()  # Revert to current weights
                    
            return weights  # No validation method found
            
        except Exception as e:
            logger.error(f"Error validating weights with risk controller: {e}")
            return self._get_current_weights()  # On error, revert to current weights

    def _apply_weight_changes(self, weights: Dict[str, float]) -> None:
        """
        Apply the weight changes to the strategies.
        
        Args:
            weights: Dictionary mapping strategy IDs to weight values
        """
        for strategy_id, weight in weights.items():
            if strategy_id in self.strategy_manager.strategies:
                strategy = self.strategy_manager.strategies[strategy_id]
                
                # Check if the strategy has a set_weight method
                if hasattr(strategy, 'set_weight'):
                    strategy.set_weight(weight)
                else:
                    # Otherwise, set the weight attribute directly
                    setattr(strategy, 'weight', weight)
                
                logger.debug(f"Applied weight {weight:.4f} to strategy {strategy_id}")
            else:
                logger.warning(f"Strategy {strategy_id} not found when applying weights")

    def _schedule_gradual_changes(self, current_weights: Dict[str, float], 
                               target_weights: Dict[str, float]) -> None:
        """
        Schedule gradual changes to strategy weights over time to minimize market impact
        and ensure smooth transitions between allocations.
        
        Args:
            current_weights: Dictionary mapping strategy IDs to current weight values
            target_weights: Dictionary mapping strategy IDs to target weight values
        """
        # Stop any existing gradual change threads
        for thread_id in list(self.active_gradual_changes.keys()):
            self.active_gradual_changes[thread_id]['active'] = False
        
        # Calculate absolute weight changes
        changes = {}
        for strategy_id, target in target_weights.items():
            current = current_weights.get(strategy_id, 0.0)
            changes[strategy_id] = abs(target - current)
        
        # If no significant changes, return early
        if max(changes.values(), default=0.0) < 0.01:
            logger.debug("No significant weight changes to schedule")
            return
        
        # Determine duration based on the largest change
        max_change = max(changes.values())
        max_duration = self.config.get('max_gradual_change_duration', 3600)  # 1 hour default
        min_duration = self.config.get('min_gradual_change_duration', 300)   # 5 min

import logging
import json
import time
import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from risk_management.risk_controller import RiskController
from utils.notification_service import NotificationService
from utils.config_loader import ConfigLoader
from utils.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class StrategyManagerExtension:
    """
    Extension class for StrategyManager to handle dynamic strategy weight adjustments.
    This class provides methods to update strategy weights based on optimizer recommendations
    while ensuring risk management constraints are maintained.
    """
    
    def __init__(
        self, 
        risk_controller: RiskController,
        db_manager: Optional[DatabaseManager] = None,
        config_path: str = "config/strategy_weights.json",
        notification_service: Optional[NotificationService] = None
    ):
        """
        Initialize the StrategyManagerExtension.
        
        Args:
            risk_controller: Risk controller instance for validation
            db_manager: Database manager for persistence (optional)
            config_path: Path to the strategy weights configuration
            notification_service: Service for sending notifications about weight changes
        """
        self.risk_controller = risk_controller
        self.db_manager = db_manager
        self.config_path = config_path
        self.notification_service = notification_service
        
        # Load configuration
        self.config = ConfigLoader.load_config(config_path)
        self.weight_history = []
        self.pending_changes = {}
        self.last_update_time = None
        
        # Configuration options
        self.gradual_change_threshold = self.config.get("gradual_change_threshold", 0.15)
        self.max_immediate_change = self.config.get("max_immediate_change", 0.10)
        self.min_strategy_weight = self.config.get("min_strategy_weight", 0.01)
        self.min_category_weight = self.config.get("min_category_weight", 0.05)
        self.category_constraints = self.config.get("category_constraints", {})

    def update_strategy_weights(
        self, 
        strategy_manager: Any, 
        new_weights: Dict[str, float], 
        reason: str = "optimizer_recommendation",
        force: bool = False,
        gradual: bool = True
    ) -> bool:
        """
        Update strategy weights based on optimizer recommendations.
        
        Args:
            strategy_manager: The StrategyManager instance to update
            new_weights: Dictionary mapping strategy IDs to their new weights
            reason: Reason for the weight update
            force: Whether to bypass validation checks
            gradual: Whether to apply changes gradually for large adjustments
            
        Returns:
            bool: True if weights were updated successfully, False otherwise
        """
        try:
            logger.info(f"Updating strategy weights: {len(new_weights)} strategies affected")
            
            # Get current weights for comparison
            current_weights = self._get_current_weights(strategy_manager)
            
            # Check if the provided weights are valid
            if not self._validate_weights(new_weights):
                logger.error("Invalid weights provided - must sum to 1.0")
                return False
                
            # Apply category constraints (e.g., min/max allocation per strategy type)
            constrained_weights = self._apply_category_constraints(
                strategy_manager, new_weights
            )
            
            # Validate with risk controller
            if not force and not self._validate_with_risk_controller(
                strategy_manager, constrained_weights, current_weights
            ):
                logger.warning("Weight changes rejected by risk controller")
                return False
                
            # For large changes, schedule gradual adjustment if enabled
            if gradual:
                significant_changes = {
                    sid: weight for sid, weight in constrained_weights.items()
                    if sid in current_weights and 
                    abs(weight - current_weights.get(sid, 0)) > self.gradual_change_threshold
                }
                
                if significant_changes:
                    logger.info(f"Scheduling gradual changes for {len(significant_changes)} strategies")
                    self._schedule_gradual_changes(
                        strategy_manager, significant_changes, current_weights
                    )
                    # Remove the gradual changes from immediate application
                    for sid in significant_changes:
                        # Apply only a portion of the change immediately
                        max_change = self.max_immediate_change
                        current = current_weights.get(sid, 0)
                        target = constrained_weights[sid]
                        direction = 1 if target > current else -1
                        immediate_change = min(abs(target - current), max_change) * direction
                        constrained_weights[sid] = current + immediate_change
            
            # Apply the weight changes to the strategy manager
            success = self._apply_weight_changes(strategy_manager, constrained_weights)
            
            if success:
                # Record the change in history
                self._save_weight_history(current_weights, constrained_weights, reason)
                # Create a weight snapshot for recovery if needed
                self._create_weight_snapshot(constrained_weights, reason)
                # Notify about the weight changes
                self._notify_weight_changes(current_weights, constrained_weights, reason)
                
                self.last_update_time = time.time()
                logger.info("Strategy weights updated successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {str(e)}", exc_info=True)
            return False
    
    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        """
        Validate that weights are properly formatted and sum to approximately 1.0.
        
        Args:
            weights: Dictionary of strategy weights
            
        Returns:
            bool: True if weights are valid, False otherwise
        """
        if not weights:
            return False
            
        weight_sum = sum(weights.values())
        # Allow small floating point errors
        return 0.99 <= weight_sum <= 1.01 and all(w >= 0 for w in weights.values())
    
    def _get_current_weights(self, strategy_manager: Any) -> Dict[str, float]:
        """
        Get the current weights from the strategy manager.
        
        Args:
            strategy_manager: The StrategyManager instance
            
        Returns:
            Dict[str, float]: Current strategy weights
        """
        try:
            if hasattr(strategy_manager, "get_strategy_weights"):
                return strategy_manager.get_strategy_weights()
            
            # Fallback if direct method not available
            weights = {}
            total_allocation = 0
            
            for strategy_id, strategy in strategy_manager.strategies.items():
                if hasattr(strategy, "allocation"):
                    weights[strategy_id] = strategy.allocation
                    total_allocation += strategy.allocation
            
            # Normalize if needed
            if total_allocation > 0:
                return {sid: w/total_allocation for sid, w in weights.items()}
            return weights
        
        except Exception as e:
            logger.error(f"Error getting current weights: {str(e)}")
            return {}
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to ensure they sum to 1.0.
        
        Args:
            weights: Dictionary of strategy weights
            
        Returns:
            Dict[str, float]: Normalized weights
        """
        total = sum(weights.values())
        if total == 0:
            # If all weights are zero, distribute equally
            equal_weight = 1.0 / len(weights) if weights else 0
            return {sid: equal_weight for sid in weights}
        
        return {sid: weight/total for sid, weight in weights.items()}
    
    def _apply_category_constraints(
        self, 
        strategy_manager: Any, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply category-based constraints to strategy weights.
        
        Args:
            strategy_manager: The StrategyManager instance
            weights: Dictionary of proposed strategy weights
            
        Returns:
            Dict[str, float]: Adjusted weights that meet category constraints
        """
        # Group strategies by category
        categories: Dict[str, Dict[str, float]] = {}
        
        # Get strategy categories
        for strategy_id, weight in weights.items():
            strategy = strategy_manager.strategies.get(strategy_id)
            if not strategy:
                logger.warning(f"Strategy {strategy_id} not found, removing from weights")
                continue
                
            category = getattr(strategy, "category", "default")
            if category not in categories:
                categories[category] = {}
            categories[category][strategy_id] = weight
        
        # Apply min/max constraints per category
        adjusted_weights = weights.copy()
        total_adjustment = 0
        
        for category, strats in categories.items():
            category_total = sum(strats.values())
            constraints = self.category_constraints.get(category, {})
            
            min_allocation = constraints.get("min_allocation", self.min_category_weight)
            max_allocation = constraints.get("max_allocation", 1.0)
            
            # Ensure category meets minimum allocation
            if category_total < min_allocation and category_total > 0:
                scale_factor = min_allocation / category_total
                for sid in strats:
                    original = adjusted_weights[sid]
                    adjusted_weights[sid] = original * scale_factor
                    total_adjustment += adjusted_weights[sid] - original
            
            # Ensure category doesn't exceed maximum allocation
            elif category_total > max_allocation:
                scale_factor = max_allocation / category_total
                for sid in strats:
                    original = adjusted_weights[sid]
                    adjusted_weights[sid] = original * scale_factor
                    total_adjustment += adjusted_weights[sid] - original
        
        # Compensate for adjustments by scaling other categories
        if abs(total_adjustment) > 0.001:
            # Find strategies not in adjusted categories
            unadjusted_strategies = set(weights.keys()) - set(
                sid for cat_strats in categories.values() for sid in cat_strats
            )
            
            if unadjusted_strategies:
                unadjusted_total = sum(weights[sid] for sid in unadjusted_strategies)
                if unadjusted_total > 0:
                    # Distribute the adjustment among unadjusted strategies
                    adjustment_per_unit = total_adjustment / unadjusted_total
                    for sid in unadjusted_strategies:
                        adjusted_weights[sid] = weights[sid] * (1 - adjustment_per_unit)
        
        # Final normalization to ensure sum is 1.0
        return self._normalize_weights(adjusted_weights)
    
    def _validate_with_risk_controller(
        self, 
        strategy_manager: Any,
        new_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> bool:
        """
        Validate proposed weight changes with the risk controller.
        
        Args:
            strategy_manager: The StrategyManager instance
            new_weights: Dictionary of proposed strategy weights
            current_weights: Dictionary of current strategy weights
            
        Returns:
            bool: True if weight changes are approved by risk controller
        """
        # Skip if no risk controller available
        if not self.risk_controller:
            return True
            
        # Calculate the weight changes
        changes = {}
        for sid, new_weight in new_weights.items():
            current = current_weights.get(sid, 0)
            changes[sid] = new_weight - current
        
        # Check with risk controller if changes are acceptable
        try:
            risk_assessment = self.risk_controller.assess_allocation_change(
                strategy_manager.strategies, changes
            )
            
            # If risk score exceeds threshold, reject changes
            if risk_assessment.get("risk_score", 0) > risk_assessment.get("threshold", 100):
                logger.warning(
                    f"Risk assessment failed: score {risk_assessment.get('risk_score')} "
                    f"exceeds threshold {risk_assessment.get('threshold')}"
                )
                return False
                
            # Check individual strategy risk limits
            for sid, change in changes.items():
                if abs(change) > risk_assessment.get("max_strategy_change", 0.25):
                    # Apply clamping rather than rejecting
                    max_change = risk_assessment.get("max_strategy_change", 0.25)
                    direction = 1 if change > 0 else -1
                    new_weights[sid] = current_weights.get(sid, 0) + (max_change * direction)
                    logger.info(f"Clamped change for {sid} to {max_change * direction}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk validation: {str(e)}")
            # Default to cautious approach on error
            return False
    
    def _apply_weight_changes(
        self, 
        strategy_manager: Any,
        new_weights: Dict[str, float]
    ) -> bool:
        """
        Apply weight changes to the strategy manager.
        
        Args:
            strategy_manager: The StrategyManager instance
            new_weights: Dictionary of new strategy weights
            
        Returns:
            bool: True if changes were applied successfully
        """
        try:
            # If the manager has a dedicated method for updating weights, use it
            if hasattr(strategy_manager, "set_strategy_weights"):
                return strategy_manager.set_strategy_weights(new_weights)
            
            # Otherwise update allocation attribute directly
            for strategy_id, weight in new_weights.items():
                if strategy_id in strategy_manager.strategies:
                    strategy = strategy_manager.strategies[strategy_id]
                    if hasattr(strategy, "allocation"):
                        strategy.allocation = weight
                    else:
                        logger.warning(f"Strategy {strategy_id} has no allocation attribute")
            
            # Update any internal tracking the manager might have
            if hasattr(strategy_manager, "update_allocation_tracking"):
                strategy_manager.update_allocation_tracking()
                
            return True
            
        except Exception as e:
            logger.error(f"Error applying weight changes: {str(e)}")
            return False
    
    def _schedule_gradual_changes(
        self,
        strategy_manager: Any,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> None:
        """
        Schedule gradual changes for strategies with large weight adjustments.
        
        Args:
            strategy_manager: The StrategyManager instance
            target_weights: Dictionary of target strategy weights
            current_weights: Dictionary of current strategy weights
        """
        now = time.time()
        adjustment_period = self.config.get("gradual_adjustment_period_days", 7) * 86400
        
        # Clear old pending changes that have expired
        self.pending_changes = {
            sid: data for sid, data in self.pending_changes.items()
            if now < data["end_time"]
        }
        
        # Schedule new gradual changes
        for sid, target_weight in target_weights.items():
            current = current_weights.get(sid, 0)
            

"""
Strategy Manager Extension Module

This module provides an extension to the StrategyManager class that adds dynamic
strategy weight adjustment capabilities based on optimizer recommendations with
advanced risk management, performance analytics, and gradual transition features.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import threading
import numpy as np
import pandas as pd
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class StrategyManagerExtension:
    """
    Extension to the StrategyManager class that provides methods for dynamically
    updating strategy weights based on optimizer recommendations.
    
    This class works alongside the StrategyOptimizer to apply weight changes
    while respecting risk constraints and category allocation rules. It supports:
    
    - Gradual weight transitions
    - Category-based constraints
    - Risk controller integration
    - Performance analytics
    - Weight change history tracking
    - Automated notifications
    - A/B testing of weighting schemes
    """
    
    def __init__(self, strategy_manager, risk_controller=None, config_path=None, 
                 database_manager=None, notification_service=None):
        """
        Initialize the StrategyManagerExtension.
        
        Args:
            strategy_manager: The base StrategyManager instance to extend
            risk_controller: Optional risk controller for validating weight changes
            config_path: Path to the extension configuration file
            database_manager: Optional database manager for persistent storage
            notification_service: Optional notification service for alerts
        """
        self.strategy_manager = strategy_manager
        self.risk_controller = risk_controller
        self.database_manager = database_manager
        self.notification_service = notification_service
        
        # Default configuration
        self.config = {
            "min_weight_threshold": 0.01,
            "max_weight_change": 0.20,  # Max 20% change at once
            "category_constraints": {
                "zero_capital": {"min": 0.2, "max": 0.5, "priority": 1},
                "minimal_capital": {"min": 0.1, "max": 0.4, "priority": 2},
                "standard": {"min": 0.2, "max": 0.6, "priority": 3},
                "high_frequency": {"min": 0.0, "max": 0.3, "priority": 4}
            },
            "market_condition_weights": {
                "volatile": {
                    "zero_capital": 0.35,
                    "minimal_capital": 0.3,
                    "standard": 0.25,
                    "high_frequency": 0.1
                },
                "trending": {
                    "zero_capital": 0.2,
                    "minimal_capital": 0.2,
                    "standard": 0.3,
                    "high_frequency": 0.3
                },
                "sideways": {
                    "zero_capital": 0.25,
                    "minimal_capital": 0.25,
                    "standard": 0.4,
                    "high_frequency": 0.1
                },
                "crisis": {
                    "zero_capital": 0.5,
                    "minimal_capital": 0.3,
                    "standard": 0.15,
                    "high_frequency": 0.05
                }
            },
            "risk_limits": {
                "max_total_capital_at_risk": 0.4,  # Max 40% of capital at risk
                "max_category_capital_at_risk": 0.25,  # Max 25% per category
                "max_strategy_capital_at_risk": 0.1  # Max 10% per strategy
            },
            "gradual_change": {
                "enabled": True,
                "steps": 3,
                "interval_hours": 2,
                "adaptive_interval": True,  # Adjust interval based on change magnitude
                "max_change_per_step": 0.08  # Max 8% change per step
            },
            "history": {
                "enabled": True,
                "snapshot_interval_hours": 24,
                "max_snapshots": 30,
                "path": "data/weight_history",
                "analytics_enabled": True,
                "correlation_analysis": True,
                "regression_analysis": True
            },
            "notifications": {
                "enabled": True,
                "threshold": 0.1,  # Notify on weight changes > 10%
                "channels": ["log", "email", "dashboard"],
                "include_analytics": True,
                "priority_levels": {
                    "critical": 0.25,  # Changes > 25%
                    "high": 0.15,      # Changes > 15%
                    "medium": 0.1,     # Changes > 10%
                    "low": 0.05        # Changes > 5%
                }
            },
            "optimization": {
                "auto_tune": True,     # Auto-tune constraints based on performance
                "performance_window": 30,  # Days to consider for performance
                "min_data_points": 10,
                "rebalance_frequency": 24  # Hours
            }
        }
        
        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    # Deep merge of configuration
                    self._deep_update(self.config, custom_config)
                logger.info(f"Loaded custom weight manager configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_path}: {e}")
        
        # Initialize weight history storage
        self.weight_history = []
        self.pending_changes = {}
        self.change_lock = threading.Lock()
        self.last_optimization_time = datetime.now() - timedelta(days=1)
        
        # Runtime statistics
        self.stats = {
            "updates": 0,
            "gradual_changes": 0,
            "constraint_violations": 0,
            "risk_validations": 0,
            "weight_snapshots": 0
        }
        
        # Ensure history directory exists
        if self.config["history"]["enabled"]:
            os.makedirs(self.config["history"]["path"], exist_ok=True)
            
        # Start scheduler for gradual changes if needed
        if self.config["gradual_change"]["enabled"]:
            self._start_change_scheduler()
    
    def _deep_update(self, original, update):
        """
        Recursively update a nested dictionary structure.
        
        Args:
            original: Original dictionary to update
            update: Dictionary with updates to apply
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def update_strategy_weights(self, new_weights: Dict[str, float], 
                                reason: str = "optimizer_recommendation",
                                market_condition: str = None,
                                gradual: bool = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Update the weights of strategies based on optimizer recommendations.
        
        Args:
            new_weights: Dictionary mapping strategy IDs to their new weights
            reason: Reason for the weight update (for logging/history)
            market_condition: Current market condition (volatile, trending, etc.)
            gradual: Override the gradual change setting from config
            metadata: Additional metadata to store with weight change history
            
        Returns:
            Dictionary of actually applied weights after constraints
        """
        start_time = time.time()
        logger.info(f"Updating strategy weights. Reason: {reason}, Market condition: {market_condition}")
        
        try:
            with self.change_lock:
                # Get current weights for comparison
                current_weights = self._get_current_weights()
                
                # Normalize weights to ensure they sum to 1.0
                normalized_weights = self._normalize_weights(new_weights)
                
                # Apply category-based constraints (min/max allocations per category)
                constrained_weights = self._apply_category_constraints(
                    normalized_weights, 
                    market_condition=market_condition
                )
                
                # Validate changes with risk controller (if available)
                validated_weights = self._validate_with_risk_controller(
                    current_weights, 
                    constrained_weights
                )
                
                # Track constraint violations
                if not np.isclose(sum(validated_weights.values()), 1.0, rtol=1e-5):
                    logger.warning(f"Weight sum after validation: {sum(validated_weights.values())}")
                    validated_weights = self._normalize_weights(validated_weights)
                    self.stats["constraint_violations"] += 1
                
                # Determine if changes should be applied gradually
                use_gradual = self.config["gradual_change"]["enabled"]
                if gradual is not None:  # Override if specifically requested
                    use_gradual = gradual
                    
                if use_gradual and self._should_use_gradual_changes(current_weights, validated_weights):
                    # Schedule gradual changes over time
                    self._schedule_gradual_changes(current_weights, validated_weights)
                    logger.info("Gradual weight changes scheduled")
                    self.stats["gradual_changes"] += 1
                    
                    # Only apply first step of changes now
                    final_weights = self._apply_weight_changes(
                        current_weights, 
                        validated_weights, 
                        step=1, 
                        total_steps=self.config["gradual_change"]["steps"]
                    )
                else:
                    # Apply changes immediately
                    final_weights = self._apply_weight_changes(current_weights, validated_weights)
                    
                # Save weight change history
                self._save_weight_history(current_weights, final_weights, reason, metadata)
                
                # Periodic snapshot of weights (if enabled)
                self._create_weight_snapshot(final_weights)
                
                # Notify of significant weight changes
                self._notify_weight_changes(current_weights, final_weights)
                
                # Update stats
                self.stats["updates"] += 1
                
                execution_time = time.time() - start_time
                logger.debug(f"Weight update completed in {execution_time:.2f} seconds")
                
                return final_weights
                
        except Exception as e:
            logger.error(f"Error updating strategy weights: {e}", exc_info=True)
            return self._get_current_weights()
    
    def _get_current_weights(self) -> Dict[str, float]:
        """
        Get the current weights of all strategies from the strategy manager.
        
        Returns:
            Dictionary mapping strategy IDs to their current weights
        """
        try:
            # This implementation depends on how weights are stored in the base StrategyManager
            current_weights = {}
            
            # First check if strategy_manager has a get_weights method
            if hasattr(self.strategy_manager, "get_strategy_weights"):
                return self.strategy_manager.get_strategy_weights()
            
            # Otherwise, get all strategies and extract weights
            strategies = self.strategy_manager.get_all_strategies()
            
            # Get current weight for each strategy
            for strategy_id, strategy in strategies.items():
                # Get weight attribute, default to equal weight if not set
                weight = getattr(strategy, "weight", None)
                if weight is None:
                    # Equal weighting if weights aren't set
                    weight = 1.0 / len(strategies)
                current_weights[strategy_id] = weight
                
            # Normalize to ensure they sum to 1.0
            return self._normalize_weights(current_weights)
            
        except Exception as e:
            logger.error(f"Error getting current weights: {e}", exc_info=True)
            return {}
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to ensure they sum to 1.0 and handle edge cases.
        
        Args:
            weights: Dictionary mapping strategy IDs to their weights
            
        Returns:
            Dictionary of normalized weights
        """
        # Handle empty dictionary case
        if not weights:
            logger.warning("Empty weights dictionary provided")
            return {}
            
        # Filter out non-positive weights
        filtered_weights = {k: max(0, v) for k, v in weights.items()}
        
        total_weight = sum(filtered_weights.values())
        
        if total_weight <= 0:
            logger.warning("All weights are zero or negative. Falling back to equal weighting.")
            equal_weight = 1.0 / len(weights)
            return {strategy_id: equal_weight for strategy_id in weights}
            
        normalized = {}
        for strategy_id, weight in filtered_weights.items():
            # Normalize to sum to 1.0
            normalized[strategy_id] = weight / total_weight
            
            # Apply minimum threshold if configured
            if normalized[strategy_id] < self.config["min_weight_threshold"]:
                normalized[strategy_id] = 0.0
                
        # Re-normalize after applying thresholds
        active_weights = {k: v for k, v in normalized.items() if v > 0}
        
        if not active_weights:
            logger.warning("All weights below threshold. Falling back to equal weighting.")
            equal_weight = 1.0 / len(weights)
            return {strategy_id: equal_weight for strategy_id in weights}
            
        # Final normalization pass
        total_active = sum(active_weights.values())
        return {k: (v / total_active if k in active_weights else 0.0) for k, v in normalized.items()}
    
    def _apply_category_constraints(self, weights: Dict[str, float], 
                                   market_condition: str = None) -> Dict[str, float]:
        """
        Apply constraints based on strategy categories to ensure proper diversification.
        
        Args:
            weights: Dictionary mapping strategy IDs to their weights
            market_condition: Current market condition (volatile, trending, etc.)
            
        Returns:
            Dictionary of weights after applying category constraints
        """
        # Group strategies by category
        categories = defaultdict(dict)
        strategies = self.strategy_manager.get_all_strategies()
        
        # Available categories
        category_weights = {}
        total_allocated = 0.0
        
        # First pass: group strategies by category and calculate per-category allocations
        for strategy_id, weight in weights.items():
            if strategy_id not in strategies:
                logger.warning(f"Strategy {strategy_id} not found in strategy manager")
                continue
                
            strategy = strategies[strategy_id]
            category = getattr(strategy, "category", "standard")
            
            categories[category][strategy_id] = weight
            
            if category not in category_weights:
                category_weights[category] = 0.0
                
            category_weights[category] += weight
            total_allocated += weight
        
        # Check if we have market condition-specific category weights
        target_category_weights = {}
        
        if market_condition and market_condition in self.config["market_condition_weights"]:
            logger.info(f"Using {market_condition} market condition weights")
            target_category_

"""
Strategy Manager Extension Module

This module extends the functionality of the StrategyManager class with advanced methods
for dynamic strategy weight management based on AI optimizer recommendations. It provides
comprehensive weight validation, normalization, transition management, persistence, and
monitoring capabilities to ensure optimal capital allocation across trading strategies.

Features:
- Multiple weighting modes (absolute, relative, tiered)
- Gradual weight transition to prevent abrupt allocation changes
- Performance impact analysis before and after weight updates
- Automatic weight normalization and validation
- Weight change constraints with min/max boundaries
- Comprehensive logging and audit trail
- Persistence with recovery mechanisms
- Integration with risk management systems
"""

import logging
import numpy as np
import json
import os
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import copy
import math
import hashlib
import traceback

# Setup logging with more detailed format
logger = logging.getLogger(__name__)

class WeightingMode(Enum):
    """Enumeration of supported weighting modes"""
    ABSOLUTE = "absolute"  # Weights are used exactly as provided
    RELATIVE = "relative"  # Weights are normalized relative to their sum
    PROPORTIONAL = "proportional"  # Weights are adjusted proportionally from current values
    TIERED = "tiered"  # Strategies are grouped into tiers with weights distributed within tiers
    ZERO_SUM = "zero_sum"  # Increases in some weights are balanced by decreases in others

class WeightChangePolicy(Enum):
    """Enumeration of policies for handling weight changes"""
    IMMEDIATE = "immediate"  # Apply changes immediately
    GRADUAL = "gradual"  # Apply changes gradually over time
    THRESHOLD = "threshold"  # Only apply changes above a certain threshold
    SCHEDULED = "scheduled"  # Apply changes at scheduled intervals

class StrategyManagerExtension:
    """
    Extension class for the StrategyManager to add advanced weight management capabilities.
    
    This class provides enhanced methods for dynamic strategy weight updating based on 
    AI optimizer recommendations, with sophisticated validation, transition management,
    and performance monitoring.
    """
    
    # Constants for weight management
    DEFAULT_MIN_WEIGHT = 0.01  # 1% minimum allocation to any strategy
    DEFAULT_MAX_WEIGHT = 0.50  # 50% maximum allocation to any strategy
    DEFAULT_MAX_CHANGE_PERCENT = 0.20  # 20% maximum change in a single update
    WEIGHT_PRECISION = 6  # Decimal places for weight precision
    WEIGHT_SUM_TOLERANCE = 0.0001  # Tolerance for weight sum validation
    
    # Persistence settings
    HISTORY_BASE_DIR = os.path.join("logs", "weight_history")
    SNAPSHOT_BASE_DIR = os.path.join("data", "strategy_weights")
    DEFAULT_BACKUP_COUNT = 10  # Number of backup snapshots to keep
    
    @classmethod
    def update_strategy_weights(cls, 
                               strategy_manager, 
                               weights: Dict[str, float],
                               weight_mode: Union[WeightingMode, str] = WeightingMode.ABSOLUTE,
                               change_policy: Union[WeightChangePolicy, str] = WeightChangePolicy.IMMEDIATE,
                               min_weight: float = DEFAULT_MIN_WEIGHT,
                               max_weight: float = DEFAULT_MAX_WEIGHT,
                               max_change_percent: float = DEFAULT_MAX_CHANGE_PERCENT,
                               transition_period: int = 0,
                               category_constraints: Optional[Dict[str, Dict[str, float]]] = None,
                               risk_controller = None,
                               save_history: bool = True,
                               create_snapshot: bool = True,
                               notify_changes: bool = True,
                               dry_run: bool = False) -> Dict[str, Any]:
        """
        Updates the weights of strategies in the StrategyManager based on 
        values provided by the StrategyOptimizer with advanced features.
        
        Args:
            strategy_manager: The StrategyManager instance to update
            weights: Dictionary mapping strategy IDs to their new weights
            weight_mode: Mode for interpreting weights (absolute, relative, etc.)
            change_policy: Policy for how weight changes are applied
            min_weight: Minimum allowed weight value
            max_weight: Maximum allowed weight value
            max_change_percent: Maximum percent change allowed in a single update
            transition_period: Number of seconds over which to transition weights
            category_constraints: Dict of category constraints {category: {min_weight, max_weight, max_strategies}}
            risk_controller: Risk controller instance for validating weight changes
            save_history: Whether to save weight update history
            create_snapshot: Whether to create a snapshot for recovery
            notify_changes: Whether to log notification messages about changes
            dry_run: If True, calculate changes but don't apply them
            
        Returns:
            Dict containing the update results with keys:
                - success: Boolean indicating if update was successful
                - updated: List of strategy IDs that were successfully updated
                - skipped: List of strategy IDs that were skipped (not found or unchanged)
                - errors: Dict mapping strategy IDs to error messages
                - metrics: Dict of metrics about the weight changes
                - timestamp: ISO formatted timestamp of the update
                - transition_id: ID of the transition schedule if gradual changes are used
                
        Raises:
            ValueError: If weights configuration is invalid
            TypeError: If weights or other parameters have incorrect types
            RuntimeError: If the update operation encounters a critical error
        """
        # Start time for performance tracking
        start_time = time.time()
        
        # Convert enum string values to proper enums if needed
        if isinstance(weight_mode, str):
            try:
                weight_mode = WeightingMode(weight_mode)
            except ValueError:
                weight_mode = WeightingMode.ABSOLUTE
                
        if isinstance(change_policy, str):
            try:
                change_policy = WeightChangePolicy(change_policy)
            except ValueError:
                change_policy = WeightChangePolicy.IMMEDIATE
        
        # Input validation
        if not isinstance(weights, dict):
            raise TypeError("Weights must be provided as a dictionary mapping strategy IDs to weight values")
            
        # Initialize detailed results object
        results = {
            "success": False,
            "updated": [],
            "skipped": [],
            "errors": {},
            "warnings": [],
            "metrics": {
                "total_weight_before": 0.0,
                "total_weight_after": 0.0,
                "largest_increase": {"strategy_id": None, "amount": 0.0, "percent": 0.0},
                "largest_decrease": {"strategy_id": None, "amount": 0.0, "percent": 0.0},
                "strategies_increased": 0,
                "strategies_decreased": 0,
                "strategies_unchanged": 0,
                "processing_time_ms": 0
            },
            "timestamp": datetime.now().isoformat(),
            "transition_id": None,
            "weight_mode": weight_mode.value,
            "change_policy": change_policy.value,
            "dry_run": dry_run
        }
        
        # Get current weights from strategy manager
        try:
            current_weights, strategy_dict = cls._get_current_weights(strategy_manager)
            results["metrics"]["total_weight_before"] = sum(current_weights.values())
        except Exception as e:
            error_msg = f"Failed to retrieve current strategy weights: {str(e)}"
            logger.error(error_msg)
            results["errors"]["initialization"] = error_msg
            results["metrics"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return results

        # Create a copy of the weights to work with
        target_weights = copy.deepcopy(weights)
        
        # Validate and normalize weights based on the weight_mode
        try:
            normalized_weights = cls._normalize_weights(
                current_weights=current_weights,
                target_weights=target_weights,
                weight_mode=weight_mode,
                min_weight=min_weight,
                max_weight=max_weight,
                strategy_dict=strategy_dict
            )
            
            # Validate category constraints if provided
            if category_constraints:
                normalized_weights = cls._apply_category_constraints(
                    weights=normalized_weights,
                    strategy_dict=strategy_dict,
                    category_constraints=category_constraints
                )
        except Exception as e:
            error_msg = f"Failed to normalize and validate weights: {str(e)}"
            logger.error(error_msg)
            results["errors"]["validation"] = error_msg
            results["metrics"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return results
        
        # Calculate weight changes and apply constraints
        weight_changes = {}
        try:
            for strategy_id, new_weight in normalized_weights.items():
                if strategy_id in current_weights:
                    current_weight = current_weights[strategy_id]
                    
                    # Calculate absolute and percentage change
                    abs_change = new_weight - current_weight
                    pct_change = abs_change / current_weight if current_weight > 0 else float('inf')
                    
                    # Apply max change percent constraint
                    if abs(pct_change) > max_change_percent and max_change_percent > 0:
                        constrained_change = current_weight * max_change_percent * (1 if abs_change > 0 else -1)
                        new_weight = current_weight + constrained_change
                        results["warnings"].append(
                            f"Weight change for {strategy_id} exceeds maximum allowed change "
                            f"({abs(pct_change):.2%} > {max_change_percent:.2%}). "
                            f"Limiting change to {max_change_percent:.2%}."
                        )
                    
                    # Record the change
                    weight_changes[strategy_id] = {
                        "current": current_weight,
                        "target": new_weight,
                        "absolute_change": new_weight - current_weight,
                        "percent_change": (new_weight - current_weight) / max(current_weight, 0.0001) * 100 if current_weight else 0
                    }
                    
                    # Update metrics for largest increases/decreases
                    if new_weight > current_weight:
                        results["metrics"]["strategies_increased"] += 1
                        if new_weight - current_weight > results["metrics"]["largest_increase"]["amount"]:
                            results["metrics"]["largest_increase"] = {
                                "strategy_id": strategy_id,
                                "amount": new_weight - current_weight,
                                "percent": (new_weight - current_weight) / max(current_weight, 0.0001) * 100
                            }
                    elif new_weight < current_weight:
                        results["metrics"]["strategies_decreased"] += 1
                        if current_weight - new_weight > results["metrics"]["largest_decrease"]["amount"]:
                            results["metrics"]["largest_decrease"] = {
                                "strategy_id": strategy_id,
                                "amount": current_weight - new_weight,
                                "percent": (current_weight - new_weight) / current_weight * 100
                            }
                    else:
                        results["metrics"]["strategies_unchanged"] += 1
                else:
                    # New strategy
                    weight_changes[strategy_id] = {
                        "current": 0.0,
                        "target": new_weight,
                        "absolute_change": new_weight,
                        "percent_change": 100.0
                    }
                    results["metrics"]["strategies_increased"] += 1
        except Exception as e:
            error_msg = f"Failed to calculate weight changes: {str(e)}"
            logger.error(error_msg)
            results["errors"]["calculation"] = error_msg
            results["metrics"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return results
        
        # Check with risk controller if available
        if risk_controller is not None:
            try:
                risk_approved = cls._validate_with_risk_controller(
                    risk_controller=risk_controller,
                    current_weights=current_weights,
                    new_weights=normalized_weights,
                    weight_changes=weight_changes
                )
                
                if not risk_approved:
                    results["errors"]["risk_control"] = "Weight changes rejected by risk controller"
                    results["metrics"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
                    return results
            except Exception as e:
                error_msg = f"Error during risk validation: {str(e)}"
                logger.error(error_msg)
                results["warnings"].append(error_msg)
                # Continue with the update despite risk validation error
        
        # If this is a dry run, return the calculated changes without applying them
        if dry_run:
            results["metrics"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
            results["applied"] = False
            results["normalized_weights"] = normalized_weights
            results["weight_changes"] = weight_changes
            return results
        
        # Determine how to apply the changes based on the change policy
        if change_policy == WeightChangePolicy.IMMEDIATE:
            # Apply changes immediately
            update_results = cls._apply_weight_changes(
                strategy_manager=strategy_manager,
                strategy_dict=strategy_dict,
                normalized_weights=normalized_weights
            )
            results.update(update_results)
            
        elif change_policy == WeightChangePolicy.GRADUAL and transition_period > 0:
            # Schedule gradual changes
            transition_id = cls._schedule_gradual_changes(
                strategy_manager=strategy_manager,
                strategy_dict=strategy_dict,
                current_weights=current_weights,
                target_weights=normalized_weights,
                transition_period=transition_period
            )
            results["transition_id"] = transition_id
            results["success"] = True
            results["updated"] = list(normalized_weights.keys())
            
        elif change_policy == WeightChangePolicy.THRESHOLD:
            # Apply only changes above threshold
            significant_changes = {
                strategy_id: weight 
                for strategy_id, weight in normalized_weights.items()
                if strategy_id in weight_changes and 
                abs(weight_changes[strategy_id]["percent_change"]) >= max_change_percent * 100
            }
            
            if significant_changes:
                update_results = cls._apply_weight_changes(
                    strategy_manager=strategy_manager,
                    strategy_dict=strategy_dict,
                    normalized_weights=significant_changes
                )
                results.update(update_results)
            else:
                results["success"] = True
                results["skipped"] = list(normalized_weights.keys())
                results["warnings"].append(
                    f"No weight changes exceed the threshold of {max_change_percent:.2%}. No updates were made."
                )
        
        # Calculate final metrics
        try:
            final_weights, _ = cls._get_current_weights(strategy_manager)
            results["metrics"]["total_weight_after"] = sum(final_weights.values())
        except Exception as e:
            results["warnings"].append(f"Failed to calculate final weight metrics: {str(e)}")
        
        # Save history if requested
        if save_history:
            try:
                history_path = cls._save_weight_history(
                    current_weights=current_weights,
                    new_weights=normalized_weights,
                    weight_changes=weight_changes,
                    results=results
                )
                results["history_path"] = history_path
                logger.info(f"Weight history saved to {history_path}")
            except Exception as e:
                error_msg = f"Failed to save weight history: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                results["warnings"].append(error_msg)
        
        # Create snapshot if requested
        if create_snapshot:
            try:
                snapshot_path = cls._create_weight_snapshot(
                    strategy_manager=strategy_manager,
                    weights=normalized_weights,
                    results=results
                )
                results["snapshot_path"] = snapshot_path
                logger.info(f"Weight snapshot created at {snapshot_path}")
            except Exception as e:
                error_msg = f"Failed to create weight snapshot: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                results["warnings"].append(error_msg)
        
        # Send notifications about significant changes if requested
        if notify_changes and results["success"]:
            cls._notify_weight_changes(
                weight_changes=weight_changes,
                results=results
            )
            
        # Calculate total processing time
        results["metrics"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return results

    @classmethod
    def _get_current_weights(cls, strategy_manager) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Retrieves the current weights and strategy objects from the strategy manager.
        
        Args:
            strategy_manager: The StrategyManager instance
            
        Returns:
            Tuple containing:
                - Dictionary mapping strategy IDs to their current weights
                - Dictionary mapping strategy IDs to their strategy objects
                
        Raises:
            AttributeError: If strategy_manager doesn't have required attributes
            RuntimeError: If retrieval fails for any other reason
        """
        try:
            # Initialize dictionaries
            current_weights = {}
            strategy_dict = {}
            
            # Get strategies from the strategy manager
            strategies = strategy_manager.get_all_strategies()
            
            if not strategies:
                logger.warning("No strategies found in strategy manager")
                return {}, {}
                
            # Extract weights and strategy objects
            for strategy_id, strategy in strategies.items():
                # Try multiple attribute names for backward compatibility
                for weight_attr in ["weight", "allocation", "capital_allocation", "allocation_weight"]:
                    if hasattr(strategy, weight_attr):
                        current_weight = getattr(strategy, weight_attr, 0.0)
                        break
                else:
                    # Default if no weight attribute is found
                    current_weight = 0.0
                    logger.debug(f"No weight attribute found for strategy {strategy_id}, defaulting to 0.0")
                
                # Validate weight is numeric
                if not isinstance(current_weight, (int, float)):
                    logger.warning(f"Non-numeric weight for strategy {strategy_id}, defaulting to 0.0")
                    current_weight = 0.0
                    
                # Ensure weight is within valid range
                current_weight = max(0.0, min(1.0, current_weight))
                
                # Store with proper precision
                current_weights[strategy_id] = round(current_weight, cls.WEIGHT_PRECISION)
                strategy_dict[strategy_id] = strategy
                
            # Log summary
            logger.debug(f"Retrieved weights for {len(current_weights)} strategies. Total weight: {sum(current_weights.values()):.4f}")
            return current_weights, strategy_dict
            
        except AttributeError as e:
            logger.error(f"Strategy manager lacks required attributes: {str(e)}\n{traceback.format_exc()}")
            raise AttributeError(f"Could not retrieve strategy weights: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to retrieve current strategy weights: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Error retrieving current weights: {str(e)}")

    @classmethod
    def _normalize_weights(cls, 
                          current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          weight_mode: WeightingMode,
                          min_weight: float,
                          max_weight: float,
                          strategy_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalizes strategy weights according to the specified weight mode.
        
        Args:
            current_weights: Dictionary mapping strategy IDs to their current weights
            target_weights: Dictionary mapping strategy IDs to their target weights
            weight_mode: The weighting mode to use for normalization
            min_weight: Minimum allowed weight for any strategy
            max_weight: Maximum allowed weight for any strategy
            strategy_dict: Dictionary mapping strategy IDs to their strategy objects
            
        Returns:
            Dictionary of normalized strategy weights
            
        Raises:
            ValueError: If weight normalization fails due to invalid inputs
        """
        # Input validation
        if not current_weights:
            logger.warning("Current weights dictionary is empty")
        if not target_weights:
            raise ValueError("Target weights dictionary cannot be empty")
        if min_weight < 0 or min_weight > 1:
            raise ValueError(f"Minimum weight must be between 0 and 1, got {min_weight}")
        if max_weight < min_weight or max_weight > 1:
            raise ValueError(f"Maximum weight must be between {min_weight} and 1, got {max_weight}")
            
        # Start with empty normalized weights
        normalized = {}
        
        # Log the normalization process
        logger.info(f"Normalizing weights using mode: {weight_mode.value}")
        logger.debug(f"Input weights sum: {sum(target_weights.values()):.4f}, " 
                     f"Current weights sum: {sum(current_weights.values()):.4f}")
        
        # Handle different weighting modes
        if weight_mode == WeightingMode.ABSOLUTE:
            # Use weights exactly as provided, just ensure they're within bounds
            for strategy_id, weight in target_weights.items():
                if strategy_id in strategy_dict:
                    normalized[strategy_id] = max(min_weight, min(max_weight, weight))
                else:
                    logger.warning(f"Strategy {strategy_id} not found in strategy_dict, skipping")
                    
        elif weight_mode == WeightingMode.RELATIVE:
            # Normalize weights to sum to 1.0
            total_weight = sum(weight for sid, weight in target_weights.items() if sid in strategy_dict)
            if total_weight <= 0:
                raise ValueError("Sum of target weights must be positive for RELATIVE mode")
                
            # First pass: calculate initial normalized weights
            for strategy_id, weight in target_weights.items():
                if strategy_id in strategy_dict:
                    if weight < 0:
                        logger.warning(f"Negative weight for strategy {strategy_id}, setting to 0")
                        weight = 0
                    normalized[strategy_id] = weight / total_weight if total_weight > 0 else 0
                    
            # Second pass: apply min/max constraints
            constrained_strategies = []
            for strategy_id in list(normalized.keys()):
                if normalized[strategy_id] < min_weight and normalized[strategy_id] > 0:
                    normalized[strategy_id] = min_weight
                    constrained_strategies.append(strategy_id)
                elif normalized[strategy_id] > max_weight:
                    normalized[strategy_id] = max_weight
                    constrained_strategies.append(strategy_id)
            
            # If any constraints were applied, redistribute excess/deficit
            if constrained_strategies:
                logger.debug(f"Applied constraints to {len(constrained_strategies)} strategies")
                
                # Calculate how much weight is available for redistribution
                constrained_total = sum(normalized[sid] for sid in constrained_strategies)
                unconstrained_strategies = [sid for sid in normalized if sid not in constrained_strategies]
                
                if unconstrained_strategies:
                    # Determine target total (should still be 1.0)
                    target_total = 1.0
                    
                    # Calculate how much weight to distribute among unconstrained strategies
                    remaining_weight = target_total - constrained_total
                    
                    # Get original weights for unconstrained strategies
                    unconstrained_original = {
                        sid: target_weights.get(sid, 0.0) for sid in unconstrained_strategies
                    }
                    unconstrained_sum = sum(unconstrained_original.values())
                    
                    # Redistribute proportionally
                    if unconstrained_sum > 0:
                        for sid in unconstrained_strategies:
                            proportion = unconstrained_original[sid] / unconstrained_sum
                            normalized[sid] = remaining_weight * proportion
                    else:
                        # If all were 0, distribute evenly
                        even_weight = remaining_weight / len(unconstrained_strategies)
                        for sid in unconstrained_strategies:
                            normalized[sid] = even_weight
                else:
                    # If all strategies are constrained, scale to sum to 1
                    scale = 1.0 / constrained_total if constrained_total > 0 else 0
                    for sid in normalized:
                        normalized[sid] *= scale
                        
        elif weight_mode == WeightingMode.PROPORTIONAL:
            # Adjust weights proportionally from current values
            weight_changes = {}
            
            # Calculate proportional changes
            for strategy_id, target_weight in target_weights.items():
                if strategy_id in strategy_dict:
                    current = current_weights.get(strategy_id, 0.0)
                    if current <= 0 and target_weight > 0:
                        # New strategy or zero-weight strategy
                        normalized[strategy_id] = target_weight
                        weight_changes[strategy_id] = target_weight
                    elif current > 0:
                        # Calculate multiplicative factor
                        factor = target_weight / current if current > 0 else 0
                        new_weight = current * factor
                        normalized[strategy_id] = new_weight
                        weight_changes[strategy_id] = new_weight - current
            
            # Apply min/max constraints
            total_adjusted = 0
            for strategy_id in list(normalized.keys()):
                if normalized[strategy_id] < min_weight and normalized[strategy_id] > 0:
                    normalized[strategy_id] = min_weight
                elif normalized[strategy_id] > max_weight:
                    normalized[strategy_id] = max_weight
                total_adjusted += normalized[strategy_id]
                    
            # Re-normalize if total is significantly different
            if abs(total_adjusted - sum(current_weights.values())) > cls.WEIGHT_SUM_TOLERANCE:
                scale = sum(current_weights.values()) / total_adjusted if total_adjusted > 0 else 0
                for sid in normalized:
                    normalized[sid] *= scale
                    
        elif weight_mode == WeightingMode.TIERED:
            # Advanced tiered allocation with performance-based weighting
            tiers = {}
            tier_stats = {}
            
            # Extract tier information and performance metrics
            for strategy_id, weight in target_weights.items():
                if strategy_id in strategy_dict:
                    strategy = strategy_dict[strategy_id]
                    
                    # Get tier from integer part of weight
                    tier = max(0, int(weight))
                    
                    # Extract performance metrics if available
                    performance = getattr(strategy, "performance_score", None)
                    if performance is None:
                        # Try to calculate from other metrics
                        sharpe = getattr(strategy, "sharpe_ratio", 0.0)
                        profit = getattr(strategy, "profit_factor", 1.0)
                        win_rate = getattr(strategy, "win_rate", 0.5)
                        performance = (sharpe * 0.4) + (profit * 0.4) + (win_rate * 0.2)
                    
                    if tier not in tiers:
                        tiers[tier] = []
                        tier_stats[tier] = {"count": 0, "total_performance": 0.0}
                    
                    tiers[tier].append({
                        "id": strategy_id,
                        "performance": performance
                    })
                    
                    tier_stats[tier]["count"] += 1
                    tier_stats[tier]["total_performance"] += performance
            
            # If no valid tiers, default to equal weighting
            if not tiers:
                logger.warning("No valid tiers found, defaulting to equal weighting")
                strategies = list(strategy_dict.keys())
                equal_weight = 1.0 / len(strategies) if strategies else 0.0
                for strategy_id in strategies:
                    normalized[strategy_id] = equal_weight
            else:
                # Allocate weights to each tier based on tier level and count
                total_tier_weight = sum(tier * stats["count"] for tier, stats in tier_stats.items())
                
                # Distribute weight across tiers
                remaining_weight = 1.0
                available_tiers = sorted(tiers.keys(), reverse=True)  # Higher tiers first
                
                for tier in available_tiers:
                    # Calculate tier allocation
                    if total_tier_weight > 0:
                        tier_allocation = (tier * tier_stats[tier]["count"] / total_tier_weight) * remaining_weight
                    else:
                        # Fallback if tier weight calculation fails
                        tier_allocation = remaining_weight / len(available_tiers)
                    
                    strategies_in_tier = tiers[tier]
                    
                    # Sort strategies within tier by performance
                    strategies_in_tier.sort(key=lambda x: x["performance"], reverse=True)
                    
                    # Distribute tier allocation based on performance
                    tier_performance = tier_stats[tier]["total_performance"]
                    if tier_performance > 0:
                        for strategy in strategies_in_tier:
                            perf_weight = strategy["performance"] / tier_performance
                            normalized[strategy["id"]] = tier_allocation * perf_weight
                    else:
                        # Equal distribution within tier
                        equal_weight = tier_allocation / len(strategies_in_tier)
                        for strategy in strategies_in_tier:
                            normalized[strategy["id"]] = equal_weight
                    
                    remaining_weight -= tier_allocation
                        
        elif weight_mode == WeightingMode.ZERO_SUM:
            # Advanced zero-sum rebalancing with performance-based priorities
            increases = {}
            decreases = {}
            neutral = {}
            
            # Calculate potential changes and categorize them
            for strategy_id, target in target_weights.items():
                if strategy_id in strategy_dict:
                    current = current_weights.get(strategy_id, 0.0)
                    change = target - current
                    
                    # Get strategy performance for prioritization
                    strategy = strategy_dict[strategy_id]
                    priority = getattr(strategy, "reallocation_priority", 1.0)
                    
                    if abs(change) < cls.WEIGHT_S
