//! Exchange & Liquidity Adapter Layer
//! 
//! Comprehensive adapter layer supporting:
//! - Centralized exchanges (Binance, Coinbase, Bybit, CME)
//! - Decentralized exchanges (Uniswap v4, 0x RFQ, dYdX v4)
//! - Cross-chain bridges (LayerZero, Wormhole)
//! - Unified order-book abstraction with depth-aware routing
//! - Adaptive rate limiting and session management
//! - Smart order routing with latency optimization

pub mod core;
pub mod centralized;
pub mod decentralized;
pub mod bridges;
pub mod order_book;
pub mod routing;
pub mod session;
pub mod rate_limiter;
pub mod websocket;
pub mod failover;
pub mod canary;

pub use core::*;
pub use order_book::*;
pub use routing::*;
pub use session::*;

