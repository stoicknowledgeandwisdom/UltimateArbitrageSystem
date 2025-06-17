#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core System Components for Ultimate Arbitrage System
=================================================

This module provides the foundational components for the Ultimate Arbitrage System,
including the Master Orchestrator and all core coordination engines.
"""

from .ultimate_master_orchestrator import (
    UltimateMasterOrchestrator,
    get_ultimate_orchestrator,
    TradingSignal,
    ComponentHealth,
    SystemMetrics,
    SignalType,
    ExecutionPriority,
    ComponentStatus,
    SignalFusionEngine,
    PerformanceOptimizer,
    AdvancedHealthMonitor,
    ExecutionCoordinator
)

__all__ = [
    'UltimateMasterOrchestrator',
    'get_ultimate_orchestrator',
    'TradingSignal',
    'ComponentHealth',
    'SystemMetrics',
    'SignalType',
    'ExecutionPriority',
    'ComponentStatus',
    'SignalFusionEngine',
    'PerformanceOptimizer',
    'AdvancedHealthMonitor',
    'ExecutionCoordinator'
]

__version__ = '1.0.0'
__author__ = 'Ultimate Arbitrage System'
__description__ = 'Core orchestration components for maximum profit generation'

