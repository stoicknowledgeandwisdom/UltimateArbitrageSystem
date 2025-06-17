//! # Trading Primitives
//!
//! Ultra-high-performance trading primitives optimized for sub-millisecond execution.
//! All types are designed for zero-copy operations and maximum cache efficiency.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use crate::{Timestamp, CoreError, CoreResult};

/// Fixed-point decimal representation for ultra-fast price calculations
/// Stores prices as integer with implicit 8 decimal places (1e-8 precision)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Price(u64);

impl Price {
    /// Scale factor for 8 decimal places
    pub const SCALE: u64 = 100_000_000; // 1e8
    
    /// Maximum representable price
    pub const MAX: Price = Price(u64::MAX);
    
    /// Minimum representable price (1e-8)
    pub const MIN: Price = Price(1);
    
    /// Zero price
    pub const ZERO: Price = Price(0);
    
    /// Create a new price from a floating point value
    #[inline]
    pub fn from_f64(value: f64) -> CoreResult<Self> {
        if value < 0.0 || !value.is_finite() {
            return Err(CoreError::InvalidPrice(format!("Invalid price: {}", value)));
        }
        
        let scaled = (value * Self::SCALE as f64).round() as u64;
        Ok(Self(scaled))
    }
    
    /// Create a new price from integer with scale
    #[inline]
    pub const fn from_scaled(scaled_value: u64) -> Self {
        Self(scaled_value)
    }
    
    /// Create a new price from integer value (price * 1e8)
    #[inline]
    pub const fn from_raw(raw_value: u64) -> Self {
        Self(raw_value)
    }
    
    /// Get the raw scaled value
    #[inline]
    pub const fn raw(&self) -> u64 {
        self.0
    }
    
    /// Convert to floating point value
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }
    
    /// Add another price (saturating on overflow)
    #[inline]
    pub fn saturating_add(&self, other: Price) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    
    /// Subtract another price (saturating at zero)
    #[inline]
    pub fn saturating_sub(&self, other: Price) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
    
    /// Multiply by a quantity (with rounding)
    #[inline]
    pub fn multiply_quantity(&self, quantity: Quantity) -> u64 {
        // Multiply price by quantity, dividing by both scales
        let result = (self.0 as u128 * quantity.raw() as u128) / (Self::SCALE as u128);
        result as u64
    }
    
    /// Calculate percentage difference from another price
    #[inline]
    pub fn percentage_diff(&self, other: Price) -> f64 {
        if other.0 == 0 {
            return 0.0;
        }
        ((self.0 as f64 - other.0 as f64) / other.0 as f64) * 100.0
    }
    
    /// Check if price is valid (non-zero)
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.0 > 0
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.to_f64())
    }
}

impl FromStr for Price {
    type Err = CoreError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value: f64 = s.parse()
            .map_err(|_| CoreError::InvalidPrice(format!("Cannot parse price: {}", s)))?;
        Self::from_f64(value)
    }
}

/// Fixed-point decimal representation for quantities
/// Stores quantities as integer with implicit 8 decimal places
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Quantity(u64);

impl Quantity {
    /// Scale factor for 8 decimal places
    pub const SCALE: u64 = 100_000_000; // 1e8
    
    /// Maximum representable quantity
    pub const MAX: Quantity = Quantity(u64::MAX);
    
    /// Minimum representable quantity (1e-8)
    pub const MIN: Quantity = Quantity(1);
    
    /// Zero quantity
    pub const ZERO: Quantity = Quantity(0);
    
    /// Create a new quantity from a floating point value
    #[inline]
    pub fn from_f64(value: f64) -> CoreResult<Self> {
        if value < 0.0 || !value.is_finite() {
            return Err(CoreError::InvalidQuantity(format!("Invalid quantity: {}", value)));
        }
        
        let scaled = (value * Self::SCALE as f64).round() as u64;
        Ok(Self(scaled))
    }
    
    /// Create a new quantity from integer with scale
    #[inline]
    pub const fn from_scaled(scaled_value: u64) -> Self {
        Self(scaled_value)
    }
    
    /// Create a new quantity from raw value
    #[inline]
    pub const fn from_raw(raw_value: u64) -> Self {
        Self(raw_value)
    }
    
    /// Get the raw scaled value
    #[inline]
    pub const fn raw(&self) -> u64 {
        self.0
    }
    
    /// Convert to floating point value
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }
    
    /// Add another quantity (saturating on overflow)
    #[inline]
    pub fn saturating_add(&self, other: Quantity) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    
    /// Subtract another quantity (saturating at zero)
    #[inline]
    pub fn saturating_sub(&self, other: Quantity) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
    
    /// Check if quantity is valid (non-zero)
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.0 > 0
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.to_f64())
    }
}

impl FromStr for Quantity {
    type Err = CoreError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value: f64 = s.parse()
            .map_err(|_| CoreError::InvalidQuantity(format!("Cannot parse quantity: {}", s)))?;
        Self::from_f64(value)
    }
}

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Side {
    /// Buy order
    Buy = 0,
    /// Sell order
    Sell = 1,
}

impl Side {
    /// Get the opposite side
    #[inline]
    pub const fn opposite(&self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
    
    /// Check if this is a buy side
    #[inline]
    pub const fn is_buy(&self) -> bool {
        matches!(self, Side::Buy)
    }
    
    /// Check if this is a sell side
    #[inline]
    pub const fn is_sell(&self) -> bool {
        matches!(self, Side::Sell)
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderType {
    /// Market order - execute immediately at best available price
    Market = 0,
    /// Limit order - execute only at specified price or better
    Limit = 1,
    /// Stop order - become market order when stop price is reached
    Stop = 2,
    /// Stop-limit order - become limit order when stop price is reached
    StopLimit = 3,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::Stop => write!(f, "STOP"),
            OrderType::StopLimit => write!(f, "STOP_LIMIT"),
        }
    }
}

/// Order time in force enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TimeInForce {
    /// Good Till Cancelled - remains active until explicitly cancelled
    GTC = 0,
    /// Immediate Or Cancel - execute immediately, cancel remainder
    IOC = 1,
    /// Fill Or Kill - execute completely immediately or cancel
    FOK = 2,
    /// Good Till Date - remains active until specified time
    GTD = 3,
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeInForce::GTC => write!(f, "GTC"),
            TimeInForce::IOC => write!(f, "IOC"),
            TimeInForce::FOK => write!(f, "FOK"),
            TimeInForce::GTD => write!(f, "GTD"),
        }
    }
}

/// Unique order identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct OrderId(pub u64);

impl OrderId {
    /// Create a new order ID
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    #[inline]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Trading instrument identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    /// Symbol string (e.g., "BTC/USDT")
    inner: smallvec::SmallVec<[u8; 16]>,
}

impl Symbol {
    /// Create a new symbol from string
    pub fn new(symbol: &str) -> Self {
        Self {
            inner: symbol.bytes().collect(),
        }
    }
    
    /// Get symbol as string slice
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.inner).unwrap_or("INVALID")
    }
    
    /// Get symbol as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Symbol {
    type Err = CoreError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

/// Complete order structure optimized for high-frequency trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier
    pub id: OrderId,
    /// Trading symbol
    pub symbol: Symbol,
    /// Order side (buy/sell)
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Order quantity
    pub quantity: Quantity,
    /// Order price (for limit orders)
    pub price: Price,
    /// Stop price (for stop orders)
    pub stop_price: Option<Price>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Order creation timestamp
    pub timestamp: Timestamp,
    /// Order expiry timestamp (for GTD orders)
    pub expiry: Option<Timestamp>,
    /// Filled quantity
    pub filled_quantity: Quantity,
    /// Average fill price
    pub average_price: Price,
    /// Order status
    pub status: OrderStatus,
}

impl Order {
    /// Create a new market order
    pub fn new_market(
        id: OrderId,
        symbol: Symbol,
        side: Side,
        quantity: Quantity,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: Price::ZERO,
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            timestamp: Timestamp::now(),
            expiry: None,
            filled_quantity: Quantity::ZERO,
            average_price: Price::ZERO,
            status: OrderStatus::New,
        }
    }
    
    /// Create a new limit order
    pub fn new_limit(
        id: OrderId,
        symbol: Symbol,
        side: Side,
        quantity: Quantity,
        price: Price,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            order_type: OrderType::Limit,
            quantity,
            price,
            stop_price: None,
            time_in_force: TimeInForce::GTC,
            timestamp: Timestamp::now(),
            expiry: None,
            filled_quantity: Quantity::ZERO,
            average_price: Price::ZERO,
            status: OrderStatus::New,
        }
    }
    
    /// Get remaining quantity to fill
    #[inline]
    pub fn remaining_quantity(&self) -> Quantity {
        self.quantity.saturating_sub(self.filled_quantity)
    }
    
    /// Check if order is completely filled
    #[inline]
    pub fn is_filled(&self) -> bool {
        self.filled_quantity >= self.quantity
    }
    
    /// Check if order is active (can be filled)
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::New | OrderStatus::PartiallyFilled)
    }
    
    /// Calculate fill percentage
    #[inline]
    pub fn fill_percentage(&self) -> f64 {
        if self.quantity.raw() == 0 {
            return 0.0;
        }
        (self.filled_quantity.raw() as f64 / self.quantity.raw() as f64) * 100.0
    }
}

/// Order status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderStatus {
    /// Order created but not yet submitted
    New = 0,
    /// Order submitted to exchange
    Submitted = 1,
    /// Order partially filled
    PartiallyFilled = 2,
    /// Order completely filled
    Filled = 3,
    /// Order cancelled
    Cancelled = 4,
    /// Order rejected
    Rejected = 5,
    /// Order expired
    Expired = 6,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::New => write!(f, "NEW"),
            OrderStatus::Submitted => write!(f, "SUBMITTED"),
            OrderStatus::PartiallyFilled => write!(f, "PARTIALLY_FILLED"),
            OrderStatus::Filled => write!(f, "FILLED"),
            OrderStatus::Cancelled => write!(f, "CANCELLED"),
            OrderStatus::Rejected => write!(f, "REJECTED"),
            OrderStatus::Expired => write!(f, "EXPIRED"),
        }
    }
}

/// Trade execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: u64,
    /// Order ID that generated this trade
    pub order_id: OrderId,
    /// Trading symbol
    pub symbol: Symbol,
    /// Trade side
    pub side: Side,
    /// Trade quantity
    pub quantity: Quantity,
    /// Trade price
    pub price: Price,
    /// Trade timestamp
    pub timestamp: Timestamp,
    /// Trade fees
    pub fee: Price,
    /// Liquidity indicator (maker/taker)
    pub is_maker: bool,
}

impl Trade {
    /// Calculate trade value (quantity * price)
    #[inline]
    pub fn value(&self) -> u64 {
        self.price.multiply_quantity(self.quantity)
    }
    
    /// Calculate net value after fees
    #[inline]
    pub fn net_value(&self) -> u64 {
        let gross_value = self.value();
        let fee_value = self.fee.multiply_quantity(self.quantity);
        gross_value.saturating_sub(fee_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_price_operations() {
        let price1 = Price::from_f64(100.12345678).unwrap();
        let price2 = Price::from_f64(50.0).unwrap();
        
        assert_eq!(price1.to_f64(), 100.12345678);
        assert!(price1 > price2);
        
        let sum = price1.saturating_add(price2);
        assert_eq!(sum.to_f64(), 150.12345678);
        
        let diff = price1.saturating_sub(price2);
        assert_eq!(diff.to_f64(), 50.12345678);
    }
    
    #[test]
    fn test_quantity_operations() {
        let qty1 = Quantity::from_f64(10.5).unwrap();
        let qty2 = Quantity::from_f64(2.25).unwrap();
        
        assert_eq!(qty1.to_f64(), 10.5);
        assert!(qty1 > qty2);
        
        let sum = qty1.saturating_add(qty2);
        assert_eq!(sum.to_f64(), 12.75);
    }
    
    #[test]
    fn test_order_creation() {
        let symbol = Symbol::new("BTC/USDT");
        let order = Order::new_limit(
            OrderId::new(12345),
            symbol,
            Side::Buy,
            Quantity::from_f64(1.0).unwrap(),
            Price::from_f64(50000.0).unwrap(),
        );
        
        assert_eq!(order.id.raw(), 12345);
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.order_type, OrderType::Limit);
        assert!(order.is_active());
        assert!(!order.is_filled());
    }
    
    #[test]
    fn test_trade_calculations() {
        let trade = Trade {
            id: 1,
            order_id: OrderId::new(123),
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            quantity: Quantity::from_f64(1.0).unwrap(),
            price: Price::from_f64(50000.0).unwrap(),
            timestamp: Timestamp::now(),
            fee: Price::from_f64(0.1).unwrap(), // 0.1% fee
            is_maker: true,
        };
        
        let value = trade.value();
        let net_value = trade.net_value();
        
        assert!(value > 0);
        assert!(net_value < value); // After fees
    }
    
    #[test]
    fn test_side_operations() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
        assert!(Side::Buy.is_buy());
        assert!(Side::Sell.is_sell());
    }
    
    #[test]
    fn test_symbol_operations() {
        let symbol = Symbol::new("BTC/USDT");
        assert_eq!(symbol.as_str(), "BTC/USDT");
        assert_eq!(symbol.to_string(), "BTC/USDT");
    }
}

