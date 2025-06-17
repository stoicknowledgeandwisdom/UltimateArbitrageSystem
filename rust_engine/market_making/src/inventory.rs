//! # Inventory Risk Management
//!
//! Advanced inventory management for market making strategies.

use ultimate_core::*;
use crate::{MarketMakingError, MarketMakingResult};

/// Inventory manager for market making
pub struct InventoryManager {
    max_position: Quantity,
    current_position: Quantity,
}

impl InventoryManager {
    /// Create new inventory manager
    pub fn new(max_position: Quantity) -> Self {
        Self {
            max_position,
            current_position: Quantity::ZERO,
        }
    }
    
    /// Check if we can take additional position
    pub fn can_take_position(&self, additional: Quantity) -> bool {
        self.current_position.saturating_add(additional) <= self.max_position
    }
    
    /// Update current position
    pub fn update_position(&mut self, trade_quantity: Quantity, side: Side) -> MarketMakingResult<()> {
        match side {
            Side::Buy => {
                self.current_position = self.current_position.saturating_add(trade_quantity);
            }
            Side::Sell => {
                self.current_position = self.current_position.saturating_sub(trade_quantity);
            }
        }
        
        if self.current_position > self.max_position {
            return Err(MarketMakingError::InventoryLimitExceeded);
        }
        
        Ok(())
    }
    
    /// Get current position
    pub fn current_position(&self) -> Quantity {
        self.current_position
    }
}

