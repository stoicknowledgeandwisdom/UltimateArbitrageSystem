#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Order Management System
==============================

Comprehensive order management with:
- Multi-broker support
- Advanced order types
- Risk management integration
- Real-time execution monitoring
- Automated position management
- Performance analytics
"""

from .base_broker import BaseBroker, Order, OrderStatus, OrderType, Trade, Position
from .alpaca_broker import AlpacaBroker
from .order_manager import AdvancedOrderManager, OrderRequest, ExecutionStrategy
from .risk_controller import RiskController, RiskCheck, RiskViolation
from .execution_algorithms import ExecutionAlgorithm, TWAPAlgorithm, VWAPAlgorithm

__all__ = [
    'BaseBroker',
    'Order',
    'OrderStatus', 
    'OrderType',
    'Trade',
    'Position',
    'AlpacaBroker',
    'AdvancedOrderManager',
    'OrderRequest',
    'ExecutionStrategy',
    'RiskController',
    'RiskCheck',
    'RiskViolation',
    'ExecutionAlgorithm',
    'TWAPAlgorithm',
    'VWAPAlgorithm'
]

