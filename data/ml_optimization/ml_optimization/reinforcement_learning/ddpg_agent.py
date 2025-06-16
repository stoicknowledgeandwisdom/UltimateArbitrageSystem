"""Deep Deterministic Policy Gradient (DDPG) agent for trading."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
from collections import deque
import logging
import copy


@dataclass
class DDPGConfig:
    """Configuration for DDPG agent."""
    state_dim: int = 20
    action_dim: int = 3  # position_size, entry_price, exit_price
    hidden_dim: int = 256
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005  # soft update parameter
    buffer_size: int = 1000000
    batch_size: int = 64
    noise_scale: float = 0.1
    noise_decay: float = 0.995
    exploration_episodes: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration."""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        """Reset the noise state."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        self.fc4.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Output actions with appropriate bounds
        # position_size: [-1, 1], entry_price: [0, 1], exit_price: [0, 1]
        actions = torch.tanh(self.fc4(x))
        
        # Scale actions appropriately
        position_size = actions[:, 0:1]  # [-1, 1]
        entry_price = torch.sigmoid(actions[:, 1:2])  # [0, 1]
        exit_price = torch.sigmoid(actions[:, 2:3])  # [0, 1]
        
        return torch.cat([position_size, entry_price, exit_price], dim=1)


class Critic(nn.Module):
    """Critic network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # State processing
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # State + Action processing
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        self.fc4.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout(x)
        
        # Concatenate state and action
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        return self.fc4(x)


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent for trading optimization."""
    
    def __init__(self, config: DDPGConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Networks
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.actor_target = Actor(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Noise for exploration
        self.noise = OUNoise(config.action_dim)
        self.noise_scale = config.noise_scale
        
        # Training metrics
        self.total_steps = 0
        self.episode_count = 0
        self.training_mode = True
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using the actor network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise and self.training_mode and self.episode_count < self.config.exploration_episodes:
            # Add exploration noise
            noise = self.noise.sample() * self.noise_scale
            action = np.clip(action + noise, -1.0, 1.0)
            
            # Decay noise
            self.noise_scale *= self.config.noise_decay
        
        self.actor.train()
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update the agent's networks."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor, self.config.tau)
        self._soft_update(self.critic_target, self.critic, self.config.tau)
        
        self.total_steps += 1
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value_mean": current_q_values.mean().item()
        }
    
    def _soft_update(self, target_net: nn.Module, main_net: nn.Module, tau: float):
        """Soft update target network."""
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'noise_scale': self.noise_scale,
            'config': self.config
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.noise_scale = checkpoint['noise_scale']
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
    
    def reset_noise(self):
        """Reset exploration noise."""
        self.noise.reset()
    
    def get_action_info(self, action: np.ndarray) -> Dict[str, Any]:
        """Get interpretable action information."""
        return {
            'position_size': float(action[0]),  # [-1, 1] where -1 is max short, 1 is max long
            'entry_price_factor': float(action[1]),  # [0, 1] price adjustment factor
            'exit_price_factor': float(action[2]),  # [0, 1] price adjustment factor
            'action_magnitude': float(np.linalg.norm(action))
        }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'noise_scale': self.noise_scale,
            'buffer_size': len(self.replay_buffer),
            'training_mode': self.training_mode,
            'device': str(self.device)
        }
    
    def start_episode(self):
        """Start a new episode."""
        self.episode_count += 1
        self.reset_noise()
    
    def end_episode(self):
        """End current episode."""
        pass

