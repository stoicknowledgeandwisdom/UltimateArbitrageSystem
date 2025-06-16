"""
SAGA Pattern Operators for Resilient Task Execution
Implements idempotent tasks with retry & compensation actions
"""

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException
from typing import Any, Dict, List, Callable, Optional
import logging
import json
from datetime import datetime

class SAGAOperator(BaseOperator):
    """
    Base SAGA operator that implements the SAGA pattern for distributed transactions
    """
    
    template_fields = ['saga_context']
    
    @apply_defaults
    def __init__(
        self,
        saga_id: str,
        transaction_func: Callable,
        compensation_func: Optional[Callable] = None,
        saga_context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.saga_id = saga_id
        self.transaction_func = transaction_func
        self.compensation_func = compensation_func
        self.saga_context = saga_context or {}
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context):
        """
        Execute the SAGA transaction with compensation handling
        """
        try:
            # Check if this transaction has already been executed (idempotency)
            if self._is_already_executed(context):
                self.logger.info(f"Transaction {self.saga_id} already executed, skipping")
                return self._get_cached_result(context)
            
            # Execute the main transaction
            result = self._execute_transaction(context)
            
            # Mark as executed for idempotency
            self._mark_as_executed(context, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transaction {self.saga_id} failed: {e}")
            
            # Execute compensation if available
            if self.compensation_func:
                try:
                    self.logger.info(f"Executing compensation for {self.saga_id}")
                    self.compensation_func(context, self.saga_context)
                except Exception as comp_e:
                    self.logger.error(f"Compensation failed for {self.saga_id}: {comp_e}")
            
            raise AirflowException(f"SAGA transaction {self.saga_id} failed: {e}")
    
    def _execute_transaction(self, context):
        """
        Execute the main transaction with retries
        """
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                return self.transaction_func(context, self.saga_context)
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise e
                self.logger.warning(f"Transaction {self.saga_id} failed, retry {retry_count}/{self.max_retries}")
    
    def _is_already_executed(self, context) -> bool:
        """
        Check if this transaction has already been executed (idempotency check)
        """
        try:
            execution_key = f"saga_{self.saga_id}_executed"
            return context['task_instance'].xcom_pull(key=execution_key) is not None
        except:
            return False
    
    def _mark_as_executed(self, context, result):
        """
        Mark this transaction as executed for idempotency
        """
        execution_key = f"saga_{self.saga_id}_executed"
        context['task_instance'].xcom_push(
            key=execution_key,
            value={
                'executed_at': datetime.now().isoformat(),
                'result': result
            }
        )
    
    def _get_cached_result(self, context):
        """
        Get the cached result from a previous execution
        """
        execution_key = f"saga_{self.saga_id}_executed"
        cached_data = context['task_instance'].xcom_pull(key=execution_key)
        return cached_data.get('result') if cached_data else None

class FundRebalanceOperator(SAGAOperator):
    """
    Specialized operator for fund rebalancing with SAGA pattern
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        def rebalance_transaction(context, saga_context):
            """
            Main rebalancing transaction
            """
            event_data = context['task_instance'].xcom_pull(key='kafka_event_data')
            
            # Placeholder for actual rebalancing logic
            # This would integrate with Transferwise API to find cheapest route
            rebalance_plan = {
                'source_exchange': 'binance',
                'target_exchange': 'coinbase',
                'amount': 10000,
                'currency': 'USD',
                'route': 'transferwise_optimal'
            }
            
            # Simulate API calls
            result = {
                'transaction_id': f"rebalance_{datetime.now().timestamp()}",
                'status': 'completed',
                'plan': rebalance_plan,
                'fees': 25.50
            }
            
            return result
        
        def rebalance_compensation(context, saga_context):
            """
            Compensation transaction for failed rebalancing
            """
            # Reverse the rebalancing operation
            # This would typically involve moving funds back or canceling pending transfers
            pass
        
        super().__init__(
            saga_id='fund_rebalance',
            transaction_func=rebalance_transaction,
            compensation_func=rebalance_compensation,
            *args,
            **kwargs
        )

class FundingRateHedgeOperator(SAGAOperator):
    """
    Specialized operator for funding rate hedge updates with SAGA pattern
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        def hedge_transaction(context, saga_context):
            """
            Main hedge update transaction
            """
            event_data = context['task_instance'].xcom_pull(key='kafka_event_data')
            
            # Placeholder for actual hedge update logic
            hedge_update = {
                'symbol': 'BTC-PERP',
                'old_hedge_ratio': 0.75,
                'new_hedge_ratio': 0.80,
                'deviation_bps': event_data.get('deviation_bps', 25) if event_data else 25
            }
            
            result = {
                'transaction_id': f"hedge_{datetime.now().timestamp()}",
                'status': 'completed',
                'update': hedge_update
            }
            
            return result
        
        def hedge_compensation(context, saga_context):
            """
            Compensation transaction for failed hedge updates
            """
            # Restore previous hedge ratios
            pass
        
        super().__init__(
            saga_id='funding_rate_hedge',
            transaction_func=hedge_transaction,
            compensation_func=hedge_compensation,
            *args,
            **kwargs
        )

class EmergencyLiquidationOperator(SAGAOperator):
    """
    Specialized operator for emergency liquidation with SAGA pattern
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        def liquidation_transaction(context, saga_context):
            """
            Main liquidation transaction
            """
            event_data = context['task_instance'].xcom_pull(key='kafka_event_data')
            
            # Placeholder for actual liquidation logic
            liquidation_plan = {
                'positions_to_liquidate': [
                    {'exchange': 'binance', 'symbol': 'BTC-PERP', 'size': -2.5},
                    {'exchange': 'coinbase', 'symbol': 'ETH-USD', 'size': 10.0}
                ],
                'execution_method': 'market_order',
                'priority': 'emergency'
            }
            
            result = {
                'transaction_id': f"liquidation_{datetime.now().timestamp()}",
                'status': 'completed',
                'plan': liquidation_plan,
                'executed_at': datetime.now().isoformat()
            }
            
            return result
        
        def liquidation_compensation(context, saga_context):
            """
            Compensation transaction for emergency liquidation
            Note: Emergency liquidations typically cannot be reversed,
            but we can log and alert for manual intervention
            """
            # Log for manual review and potential position rebuilding
            pass
        
        super().__init__(
            saga_id='emergency_liquidation',
            transaction_func=liquidation_transaction,
            compensation_func=liquidation_compensation,
            *args,
            **kwargs
        )

