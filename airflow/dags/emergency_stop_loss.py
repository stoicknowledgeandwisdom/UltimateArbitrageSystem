"""
Emergency Stop-Loss DAG
Liquidates all positions on critical alerts
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sensors.kafka_event_sensor import CapitalThresholdSensor
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'emergency_stop_loss',
    default_args=default_args,
    description='Emergency stop loss DAG that liquidates all positions on critical alerts',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['emergency', 'stop_loss'],
) as dag:

    # Define the task - Listen for capital threshold alerts
    listen_capital_threshold = CapitalThresholdSensor(
        task_id='listen_capital_threshold',
        threshold_type='critical'
    )

    def execute_stop_loss(**kwargs):
        """
        Function to liquidate all positions
        """
        kafka_event_data = kwargs.get('ti').xcom_pull(key='kafka_event_data', task_ids='listen_capital_threshold')
        
        if kafka_event_data:
            logger.info(f"Processing critical capital threshold event: {kafka_event_data}")

            # Placeholder implementation of stop loss logic
            result = {
                'status': 'success',
                'details': 'Stop loss executed.'
            }
            logger.info(f"Stop loss result: {result}")
            return result

    # Define the task - Execute stop loss
    execute_stop_loss = PythonOperator(
        task_id='execute_stop_loss_task',
        python_callable=execute_stop_loss,
        provide_context=True
    )

    # Set task dependencies
    listen_capital_threshold >> execute_stop_loss

