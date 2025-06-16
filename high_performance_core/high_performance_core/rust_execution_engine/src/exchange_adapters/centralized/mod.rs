//! Centralized Exchange Adapters
//!
//! This module contains adapters for major centralized exchanges:
//! - Binance (Spot & Futures)
//! - Coinbase Pro
//! - Bybit
//! - CME (Chicago Mercantile Exchange)

pub mod binance;
// pub mod coinbase;
// pub mod bybit;
// pub mod cme;

pub use binance::*;

