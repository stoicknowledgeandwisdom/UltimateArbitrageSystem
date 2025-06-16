"""
DAG for Auto-rebalancing Funds Across Exchanges
This DAG will listen to Kafka events and auto-rebalance funds using the Transferwise API
for the cheapest route.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sensors.kafka_event_sensor import MarketShockSensor
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
    'auto_rebalance_funds',
    default_args=default_args,
    description='Auto-rebalance funds across exchanges using Transferwise API',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['rebalance', 'finance'],
) as dag:

    # Define the task - Listen for market shocks
    listen_market_shock = MarketShockSensor(
        task_id='listen_market_shock'
    )

    def auto_rebalance(**kwargs):
        """
        Function to auto-rebalance funds using Transferwise API
        """
        kafka_event_data = kwargs.get('ti').xcom_pull(key='kafka_event_data', task_ids='listen_market_shock')
        
        if kafka_event_data:
            logger.info(f"Processing event: {kafka_event_data}")

            # Example placeholder logic for fund rebalancing using Transferwise API
            # Implement logic to calculate cheapest route and execute rebalance using Transferwise
            
            result = {
                'status': 'success',
                'details': 'Fund rebalancing initiated.'
            }
            logger.info(f"Rebalance result: {result}")
            return result

    # Define the task - Execute auto-rebalance
    execute_rebalance = PythonOperator(
        task_id='execute_auto_rebalance',
        python_callable=auto_rebalance,
        provide_context=True
    )

    # Set task dependencies
    listen_market_shock >> execute_rebalance

