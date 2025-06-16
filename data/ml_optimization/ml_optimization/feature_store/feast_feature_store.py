"""Feast-based Feature Store for ML optimization."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float64, Int64, String, Bytes
from feast.data_source import PushSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.feature_logging import LoggingConfig
import redis
from concurrent.futures import ThreadPoolExecutor


@dataclass
class FeatureConfig:
    """Configuration for feature store."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    batch_size: int = 1000
    cache_ttl: int = 300  # 5 minutes
    max_workers: int = 4


class FeastFeatureStore:
    """High-performance Feast-based feature store for trading features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=True
        )
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._feature_store = None
        self._initialize_feature_store()
    
    def _initialize_feature_store(self):
        """Initialize Feast feature store with trading-specific features."""
        try:
            # Define entities
            symbol_entity = Entity(
                name="symbol",
                description="Trading symbol identifier",
                value_type=String
            )
            
            exchange_entity = Entity(
                name="exchange",
                description="Exchange identifier",
                value_type=String
            )
            
            # Define push sources for real-time data
            price_push_source = PushSource(
                name="price_push_source",
                description="Real-time price data stream"
            )
            
            volume_push_source = PushSource(
                name="volume_push_source",
                description="Real-time volume data stream"
            )
            
            # Define feature views
            price_features = FeatureView(
                name="price_features",
                entities=[symbol_entity, exchange_entity],
                ttl=timedelta(minutes=5),
                schema=[
                    Field(name="bid_price", dtype=Float64),
                    Field(name="ask_price", dtype=Float64),
                    Field(name="mid_price", dtype=Float64),
                    Field(name="spread", dtype=Float64),
                    Field(name="spread_pct", dtype=Float64),
                    Field(name="price_volatility_1m", dtype=Float64),
                    Field(name="price_volatility_5m", dtype=Float64),
                    Field(name="price_change_1m", dtype=Float64),
                    Field(name="price_change_5m", dtype=Float64),
                ],
                source=price_push_source,
                description="Real-time price features"
            )
            
            volume_features = FeatureView(
                name="volume_features",
                entities=[symbol_entity, exchange_entity],
                ttl=timedelta(minutes=5),
                schema=[
                    Field(name="volume", dtype=Float64),
                    Field(name="volume_ma_1m", dtype=Float64),
                    Field(name="volume_ma_5m", dtype=Float64),
                    Field(name="volume_ratio_1m", dtype=Float64),
                    Field(name="volume_spike_indicator", dtype=Float64),
                    Field(name="bid_volume", dtype=Float64),
                    Field(name="ask_volume", dtype=Float64),
                    Field(name="volume_imbalance", dtype=Float64),
                ],
                source=volume_push_source,
                description="Real-time volume features"
            )
            
            # On-demand feature views for computed features
            @on_demand_feature_view(
                sources=[price_features, volume_features],
                schema=[
                    Field(name="momentum_score", dtype=Float64),
                    Field(name="liquidity_score", dtype=Float64),
                    Field(name="arbitrage_opportunity_score", dtype=Float64),
                    Field(name="risk_adjusted_return", dtype=Float64),
                ]
            )
            def computed_features(features_df: pd.DataFrame) -> pd.DataFrame:
                """Compute advanced features on-demand."""
                df = pd.DataFrame()
                
                # Momentum score combining price and volume signals
                df["momentum_score"] = (
                    features_df["price_change_1m"] * features_df["volume_ratio_1m"] +
                    features_df["price_change_5m"] * features_df["volume_ma_5m"] * 0.5
                )
                
                # Liquidity score based on spread and volume
                df["liquidity_score"] = (
                    1.0 / (1.0 + features_df["spread_pct"]) * 
                    np.log1p(features_df["volume"])
                )
                
                # Arbitrage opportunity score
                df["arbitrage_opportunity_score"] = (
                    features_df["spread_pct"] * features_df["volume_ratio_1m"] *
                    (1.0 / (1.0 + features_df["price_volatility_1m"]))
                )
                
                # Risk-adjusted return estimation
                df["risk_adjusted_return"] = (
                    features_df["price_change_1m"] / 
                    (features_df["price_volatility_1m"] + 1e-8)
                )
                
                return df
            
            # Initialize feature store
            self._feature_store = FeatureStore(repo_path=".")
            self.logger.info("Feast feature store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature store: {e}")
            raise
    
    async def push_features(self, features: Dict[str, Any], entity_keys: Dict[str, str]):
        """Push real-time features to the feature store."""
        try:
            # Prepare feature data
            feature_data = {
                **entity_keys,
                **features,
                "event_timestamp": datetime.utcnow()
            }
            
            # Push to Feast
            df = pd.DataFrame([feature_data])
            
            # Determine which feature view to update
            if any(key.startswith('price') or key in ['bid_price', 'ask_price', 'spread'] for key in features.keys()):
                await self._push_to_feature_store(df, "price_features")
            
            if any(key.startswith('volume') for key in features.keys()):
                await self._push_to_feature_store(df, "volume_features")
            
            # Cache in Redis for ultra-fast access
            cache_key = f"features:{entity_keys['symbol']}:{entity_keys['exchange']}"
            await self._cache_features(cache_key, features)
            
        except Exception as e:
            self.logger.error(f"Failed to push features: {e}")
            raise
    
    async def _push_to_feature_store(self, df: pd.DataFrame, feature_view_name: str):
        """Push data to specific feature view."""
        try:
            # Use thread executor for blocking Feast operations
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self._feature_store.push(
                    feature_view_name,
                    df
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to push to feature view {feature_view_name}: {e}")
    
    async def _cache_features(self, cache_key: str, features: Dict[str, Any]):
        """Cache features in Redis for fast access."""
        try:
            import json
            # Serialize features to JSON
            features_json = json.dumps(features, default=str)
            
            # Set with TTL
            self.redis_client.setex(
                cache_key,
                self.config.cache_ttl,
                features_json
            )
        except Exception as e:
            self.logger.error(f"Failed to cache features: {e}")
    
    async def get_online_features(
        self,
        entity_keys: Dict[str, str],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Get online features for real-time inference."""
        try:
            # Try cache first
            cache_key = f"features:{entity_keys['symbol']}:{entity_keys['exchange']}"
            cached_features = await self._get_cached_features(cache_key)
            
            if cached_features and all(fname in cached_features for fname in feature_names):
                return {fname: cached_features[fname] for fname in feature_names}
            
            # Fall back to Feast online store
            entity_df = pd.DataFrame([entity_keys])
            
            # Get features from Feast
            feature_vector = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self._feature_store.get_online_features(
                    features=feature_names,
                    entity_df=entity_df
                )
            )
            
            # Convert to dictionary
            result = feature_vector.to_dict()
            
            # Remove entity keys from result
            for key in entity_keys.keys():
                result.pop(key, None)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get online features: {e}")
            # Return default values
            return {fname: 0.0 for fname in feature_names}
    
    async def _get_cached_features(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get features from Redis cache."""
        try:
            import json
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get cached features: {e}")
            return None
    
    async def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: List[str],
        full_feature_names: bool = True
    ) -> pd.DataFrame:
        """Get historical features for training."""
        try:
            # Get historical features from Feast
            historical_features = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self._feature_store.get_historical_features(
                    entity_df=entity_df,
                    features=feature_names,
                    full_feature_names=full_feature_names
                )
            )
            
            return historical_features.to_df()
            
        except Exception as e:
            self.logger.error(f"Failed to get historical features: {e}")
            raise
    
    async def batch_push_features(
        self,
        features_batch: List[Dict[str, Any]],
        entity_keys_batch: List[Dict[str, str]]
    ):
        """Batch push features for better performance."""
        try:
            # Group features by feature view
            price_features_batch = []
            volume_features_batch = []
            
            for features, entity_keys in zip(features_batch, entity_keys_batch):
                feature_data = {
                    **entity_keys,
                    **features,
                    "event_timestamp": datetime.utcnow()
                }
                
                if any(key.startswith('price') or key in ['bid_price', 'ask_price', 'spread'] for key in features.keys()):
                    price_features_batch.append(feature_data)
                
                if any(key.startswith('volume') for key in features.keys()):
                    volume_features_batch.append(feature_data)
            
            # Push batches concurrently
            tasks = []
            
            if price_features_batch:
                df_price = pd.DataFrame(price_features_batch)
                tasks.append(self._push_to_feature_store(df_price, "price_features"))
            
            if volume_features_batch:
                df_volume = pd.DataFrame(volume_features_batch)
                tasks.append(self._push_to_feature_store(df_volume, "volume_features"))
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Failed to batch push features: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get all available feature names."""
        try:
            # Get feature names from all feature views
            feature_names = []
            
            # Add price features
            feature_names.extend([
                "price_features:bid_price",
                "price_features:ask_price",
                "price_features:mid_price",
                "price_features:spread",
                "price_features:spread_pct",
                "price_features:price_volatility_1m",
                "price_features:price_volatility_5m",
                "price_features:price_change_1m",
                "price_features:price_change_5m"
            ])
            
            # Add volume features
            feature_names.extend([
                "volume_features:volume",
                "volume_features:volume_ma_1m",
                "volume_features:volume_ma_5m",
                "volume_features:volume_ratio_1m",
                "volume_features:volume_spike_indicator",
                "volume_features:bid_volume",
                "volume_features:ask_volume",
                "volume_features:volume_imbalance"
            ])
            
            # Add computed features
            feature_names.extend([
                "computed_features:momentum_score",
                "computed_features:liquidity_score",
                "computed_features:arbitrage_opportunity_score",
                "computed_features:risk_adjusted_return"
            ])
            
            return feature_names
            
        except Exception as e:
            self.logger.error(f"Failed to get feature names: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.cleanup())

