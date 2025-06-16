"""
DAG for Updating Funding-Rate Hedges
This DAG updates funding-rate hedges every 15 minutes or on >20 bps deviation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sensors.kafka_event_sensor import FundingRateDeviationSensor
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
    'funding_rate_hedge_update',
    default_args=default_args,
    description='Update funding-rate hedges',
    schedule_interval='*/15 * * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['funding', 'hedge'],
) as dag:

    # Define the task - Listen for funding-rate deviations
    listen_funding_deviation = FundingRateDeviationSensor(
        task_id='listen_funding_deviation'
    )

    def update_funding_hedge(**kwargs):
        """
        Function to update funding-rate hedges
        """
        kafka_event_data = kwargs.get('ti').xcom_pull(key='kafka_event_data', task_ids='listen_funding_deviation')
        
        if kafka_event_data:
            logger.info(f"Processing funding-rate deviation event: {kafka_event_data}")

            # Placeholder for logic to update funding-rate hedges
            result = {
                'status': 'success',
                'details': 'Funding-rate hedge updated.'
            }
            logger.info(f"Update result: {result}")
            return result

    # Define the task - Execute funding-rate hedge update
    execute_hedge_update = PythonOperator(
        task_id='execute_hedge_update',
        python_callable=update_funding_hedge,
        provide_context=True
    )

    # Set task dependencies
    listen_funding_deviation >> execute_hedge_update

