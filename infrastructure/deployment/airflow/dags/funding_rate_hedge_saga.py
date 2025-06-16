"""
Funding Rate Hedge Update DAG with SAGA Pattern
Hybrid cron + event-driven hedge updates with resilience
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.dummy import DummyOperator
from sensors.kafka_event_sensor import FundingRateDeviationSensor
from operators.saga_operators import FundingRateHedgeOperator
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
    'retry_delay': timedelta(minutes=2),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Cron-based DAG for regular hedge updates
with DAG(
    'funding_rate_hedge_cron',
    default_args=default_args,
    description='Regular funding rate hedge updates every 15 minutes',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['funding', 'hedge', 'cron'],
    max_active_runs=1,
) as cron_dag:

    # Dummy task to mark cron execution
    cron_trigger = DummyOperator(
        task_id='cron_trigger'
    )
    
    # Execute hedge update with SAGA pattern
    execute_cron_hedge_update = FundingRateHedgeOperator(
        task_id='execute_cron_hedge_update'
    )
    
    cron_trigger >> execute_cron_hedge_update

# Event-driven DAG for deviation-based hedge updates
with DAG(
    'funding_rate_hedge_event',
    default_args=default_args,
    description='Event-driven funding rate hedge updates on >20 bps deviation',
    schedule_interval=None,  # Event-driven only
    start_date=days_ago(1),
    catchup=False,
    tags=['funding', 'hedge', 'event-driven'],
    max_active_runs=1,
) as event_dag:

    # Listen for funding rate deviation events
    funding_deviation_sensor = FundingRateDeviationSensor(
        task_id='funding_deviation_sensor',
        min_deviation_bps=20,
        poke_interval=10,
        timeout=120
    )
    
    # Execute hedge update with SAGA pattern
    execute_event_hedge_update = FundingRateHedgeOperator(
        task_id='execute_event_hedge_update'
    )
    
    funding_deviation_sensor >> execute_event_hedge_update

