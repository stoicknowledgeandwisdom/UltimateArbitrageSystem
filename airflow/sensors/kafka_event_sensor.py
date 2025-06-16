"""
Kafka Event Sensor for Market Events
Triggers DAGs based on Kafka topic events like market shocks and capital thresholds
"""

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from kafka import KafkaConsumer
import json
from typing import Any, Dict, Optional
from datetime import datetime
import logging

class KafkaEventSensor(BaseSensorOperator):
    """
    Sensor that listens to Kafka topics for specific events and triggers DAGs
    """
    
    template_fields = ['topic', 'bootstrap_servers']
    
    @apply_defaults
    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = 'localhost:9092',
        consumer_group: str = 'airflow-sensor',
        event_filter: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.consumer_group = consumer_group
        self.event_filter = event_filter or {}
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def poke(self, context: Dict[str, Any]) -> bool:
        """
        Check if the specified event has occurred in the Kafka topic
        """
        try:
            consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers.split(','),
                group_id=self.consumer_group,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=self.timeout * 1000
            )
            
            self.logger.info(f"Listening to Kafka topic: {self.topic}")
            
            for message in consumer:
                event_data = message.value
                self.logger.info(f"Received event: {event_data}")
                
                # Check if event matches filter criteria
                if self._matches_filter(event_data):
                    self.logger.info(f"Event matched filter criteria: {self.event_filter}")
                    # Store event data in XCom for downstream tasks
                    context['task_instance'].xcom_push(
                        key='kafka_event_data',
                        value=event_data
                    )
                    consumer.close()
                    return True
            
            consumer.close()
            return False
            
        except Exception as e:
            self.logger.error(f"Error in Kafka sensor: {e}")
            return False
    
    def _matches_filter(self, event_data: Dict[str, Any]) -> bool:
        """
        Check if event data matches the filter criteria
        """
        if not self.event_filter:
            return True
        
        for key, expected_value in self.event_filter.items():
            if key not in event_data:
                return False
            
            if isinstance(expected_value, dict):
                # Handle range filters
                if 'min' in expected_value or 'max' in expected_value:
                    value = event_data[key]
                    if 'min' in expected_value and value < expected_value['min']:
                        return False
                    if 'max' in expected_value and value > expected_value['max']:
                        return False
            elif event_data[key] != expected_value:
                return False
        
        return True

class MarketShockSensor(KafkaEventSensor):
    """
    Specialized sensor for market shock events
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super().__init__(
            topic='market-events',
            event_filter={
                'event_type': 'market_shock',
                'severity': {'min': 0.05}  # 5% or greater price movement
            },
            *args,
            **kwargs
        )

class CapitalThresholdSensor(KafkaEventSensor):
    """
    Specialized sensor for capital threshold events
    """
    
    @apply_defaults
    def __init__(self, threshold_type: str = 'low', *args, **kwargs):
        super().__init__(
            topic='capital-events',
            event_filter={
                'event_type': 'capital_threshold',
                'threshold_type': threshold_type
            },
            *args,
            **kwargs
        )

class FundingRateDeviationSensor(KafkaEventSensor):
    """
    Specialized sensor for funding rate deviation events
    """
    
    @apply_defaults
    def __init__(self, min_deviation_bps: int = 20, *args, **kwargs):
        super().__init__(
            topic='funding-rate-events',
            event_filter={
                'event_type': 'funding_rate_deviation',
                'deviation_bps': {'min': min_deviation_bps}
            },
            *args,
            **kwargs
        )

