"""
Emergency Stop-Loss DAG with SAGA Pattern
Critical event-driven position liquidation with maximum resilience
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.email import EmailOperator
from sensors.kafka_event_sensor import CapitalThresholdSensor, MarketShockSensor
from operators.saga_operators import EmergencyLiquidationOperator
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'email_on_success': True,  # Critical to notify on success
    'retries': 5,  # More retries for critical operations
    'retry_delay': timedelta(seconds=30),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=10),
}

# Define the DAG - High priority event-driven
with DAG(
    'emergency_stop_loss_saga',
    default_args=default_args,
    description='Emergency stop loss DAG with SAGA pattern',
    schedule_interval=None,  # Event-driven only
    start_date=days_ago(1),
    catchup=False,
    tags=['emergency', 'stop_loss', 'saga', 'critical'],
    max_active_runs=1,
    priority_weight=1000,  # High priority
) as dag:

    # Listen for critical capital threshold alerts
    critical_capital_sensor = CapitalThresholdSensor(
        task_id='critical_capital_sensor',
        threshold_type='critical',
        poke_interval=5,  # More frequent checking
        timeout=60
    )
    
    # Listen for extreme market shock events
    extreme_market_shock_sensor = MarketShockSensor(
        task_id='extreme_market_shock_sensor',
        poke_interval=5,
        timeout=60
    )
    
    # Execute emergency liquidation with SAGA pattern
    execute_emergency_liquidation = EmergencyLiquidationOperator(
        task_id='execute_emergency_liquidation',
        trigger_rule='one_success'  # Trigger on any critical event
    )
    
    # Send immediate notification on completion
    send_emergency_notification = EmailOperator(
        task_id='send_emergency_notification',
        to=['risk@company.com', 'trading@company.com'],
        subject='EMERGENCY: Stop Loss Executed',
        html_content="""
        <h2>Emergency Stop Loss Executed</h2>
        <p>Emergency liquidation has been completed due to critical market conditions.</p>
        <p>Please review positions and assess next steps immediately.</p>
        <p>Execution Time: {{ ds }}</p>
        """
    )
    
    # Set task dependencies
    [critical_capital_sensor, extreme_market_shock_sensor] >> execute_emergency_liquidation >> send_emergency_notification

