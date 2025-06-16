"""
Auto-Rebalance DAG with SAGA Pattern
Event-driven fund rebalancing using Transferwise API with resilience
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from sensors.kafka_event_sensor import MarketShockSensor, CapitalThresholdSensor
from operators.saga_operators import FundRebalanceOperator
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}

# Define the DAG - Event-driven
with DAG(
    'auto_rebalance_saga',
    default_args=default_args,
    description='Auto-rebalance funds across exchanges using SAGA pattern',
    schedule_interval=None,  # Event-driven only
    start_date=days_ago(1),
    catchup=False,
    tags=['rebalance', 'saga', 'event-driven'],
    max_active_runs=1,  # Prevent concurrent rebalancing
) as dag:

    # Listen for market shock events
    market_shock_sensor = MarketShockSensor(
        task_id='market_shock_sensor',
        poke_interval=30,
        timeout=300
    )
    
    # Listen for capital threshold events
    capital_threshold_sensor = CapitalThresholdSensor(
        task_id='capital_threshold_sensor',
        threshold_type='rebalance',
        poke_interval=30,
        timeout=300
    )
    
    # Execute fund rebalancing with SAGA pattern
    execute_rebalance = FundRebalanceOperator(
        task_id='execute_fund_rebalance',
        trigger_rule='one_success'  # Trigger on any sensor success
    )
    
    # Set task dependencies
    [market_shock_sensor, capital_threshold_sensor] >> execute_rebalance

