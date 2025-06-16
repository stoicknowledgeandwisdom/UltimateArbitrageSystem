#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reinforcement Learning Optimizer
============================

Optimizes trading strategies using deep RL:
- Deep Q-Learning
- Policy Gradient
- Actor-Critic
- Multi-agent learning
- Reward engineering
- Experience replay
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    state_dim: int = 100
    action_dim: int = 10
    memory_size: int = 100000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.001
    learning_rate: float = 0.001
    update_frequency: int = 100
    warmup_steps: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

@dataclass
class TrainingMetrics:
    episode: int
    total_reward: float
    avg_q_value: float
    loss: float
    epsilon: float
    actions_taken: Dict[str, int]
    timestamp: datetime

class ReinforcementOptimizer:
    """Deep reinforcement learning optimizer"""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.epsilon = config.epsilon_start
        
        # Initialize networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Copy weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Experience replay memory
        self.memory = deque(maxlen=config.memory_size)
        
        # Training metrics
        self.metrics_history: List[TrainingMetrics] = []
        self.episode_count = 0
        self.step_count = 0
    
    def _build_actor(self) -> Model:
        """Build actor network"""
        state_input = Input(shape=(self.config.state_dim,))
        
        x = Dense(400, activation='relu')(state_input)
        x = Dense(300, activation='relu')(x)
        
        # Action outputs
        actions = Dense(
            self.config.action_dim,
            activation='tanh',
            kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003)
        )(x)
        
        model = Model(inputs=state_input, outputs=actions)
        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate))
        
        return model
    
    def _build_critic(self) -> Model:
        """Build critic network"""
        state_input = Input(shape=(self.config.state_dim,))
        action_input = Input(shape=(self.config.action_dim,))
        
        # State processing
        x1 = Dense(400, activation='relu')(state_input)
        x1 = Dense(300, activation='relu')(x1)
        
        # Action processing
        x2 = Dense(300, activation='relu')(action_input)
        
        # Combine state and action pathways
        x = Concatenate()([x1, x2])
        x = Dense(300, activation='relu')(x)
        
        # Q-value output
        q_value = Dense(
            1,
            kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003)
        )(x)
        
        model = Model(inputs=[state_input, action_input], outputs=q_value)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Random action
            return np.random.uniform(-1, 1, self.config.action_dim)
        
        # Get action from actor network
        return self.actor.predict(state.reshape(1, -1))[0]
    
    def train(self) -> Optional[TrainingMetrics]:
        """Train the networks"""
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.config.batch_size)
        
        # Unpack batch
        states = np.vstack([x[0] for x in batch])
        actions = np.vstack([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Get target actions
        target_actions = self.target_actor.predict(next_states)
        
        # Get target Q-values
        target_q = self.target_critic.predict([next_states, target_actions])
        
        # Calculate target values
        target = rewards + self.config.gamma * target_q.flatten() * (1 - dones)
        
        # Train critic
        critic_loss = self.critic.train_on_batch(
            [states, actions],
            target.reshape(-1, 1)
        )
        
        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q_values = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(q_values)
        
        actor_grads = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )
        self.actor.optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )
        
        # Update target networks
        if self.step_count % self.config.update_frequency == 0:
            self._update_target_networks()
        
        # Update epsilon
        if self.epsilon > self.config.epsilon_end:
            self.epsilon *= self.config.epsilon_decay
        
        # Update counters
        self.step_count += 1
        
        # Create metrics
        metrics = TrainingMetrics(
            episode=self.episode_count,
            total_reward=np.mean(rewards),
            avg_q_value=np.mean(target_q),
            loss=float(critic_loss),
            epsilon=self.epsilon,
            actions_taken=self._count_actions(actions),
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _update_target_networks(self) -> None:
        """Update target network weights"""
        # Actor update
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        
        for i in range(len(actor_weights)):
            target_actor_weights[i] = \
                self.config.tau * actor_weights[i] + \
                (1 - self.config.tau) * target_actor_weights[i]
        
        self.target_actor.set_weights(target_actor_weights)
        
        # Critic update
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        
        for i in range(len(critic_weights)):
            target_critic_weights[i] = \
                self.config.tau * critic_weights[i] + \
                (1 - self.config.tau) * target_critic_weights[i]
        
        self.target_critic.set_weights(target_critic_weights)
    
    def _count_actions(self, actions: np.ndarray) -> Dict[str, int]:
        """Count action types taken"""
        action_counts = {
            'buy': 0,
            'sell': 0,
            'hold': 0
        }
        
        for action in actions:
            if np.mean(action) > 0.5:
                action_counts['buy'] += 1
            elif np.mean(action) < -0.5:
                action_counts['sell'] += 1
            else:
                action_counts['hold'] += 1
        
        return action_counts
    
    def save_model(self, path: str) -> None:
        """Save model weights"""
        try:
            self.actor.save_weights(f"{path}/actor.h5")
            self.critic.save_weights(f"{path}/critic.h5")
            self.target_actor.save_weights(f"{path}/target_actor.h5")
            self.target_critic.save_weights(f"{path}/target_critic.h5")
            
            logger.info(f"Model weights saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model weights: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """Load model weights"""
        try:
            self.actor.load_weights(f"{path}/actor.h5")
            self.critic.load_weights(f"{path}/critic.h5")
            self.target_actor.load_weights(f"{path}/target_actor.h5")
            self.target_critic.load_weights(f"{path}/target_critic.h5")
            
            logger.info(f"Model weights loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
    
    def get_metrics_history(self) -> List[TrainingMetrics]:
        """Get training metrics history"""
        return self.metrics_history
    
    def reset_episode(self) -> None:
        """Reset episode counter and metrics"""
        self.episode_count += 1

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = OptimizerConfig(
        state_dim=100,
        action_dim=10,
        memory_size=100000,
        batch_size=64
    )
    
    # Initialize optimizer
    optimizer = ReinforcementOptimizer(config)
    
    # Example training loop
    total_episodes = 1000
    max_steps = 200
    
    for episode in range(total_episodes):
        state = np.random.random(config.state_dim)  # Example state
        total_reward = 0
        
        for step in range(max_steps):
            # Get action
            action = optimizer.act(state)
            
            # Example environment interaction
            next_state = np.random.random(config.state_dim)
            reward = np.random.normal(0, 1)
            done = step == max_steps - 1
            
            # Store experience and train
            optimizer.remember(state, action, reward, next_state, done)
            metrics = optimizer.train()
            
            if metrics:
                print(f"\nEpisode {episode}, Step {step}")
                print(f"Total Reward: {metrics.total_reward:.2f}")
                print(f"Average Q-Value: {metrics.avg_q_value:.2f}")
                print(f"Loss: {metrics.loss:.4f}")
                print(f"Epsilon: {metrics.epsilon:.4f}")
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        optimizer.reset_episode()
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode} completed")
            print(f"Total Reward: {total_reward:.2f}")

