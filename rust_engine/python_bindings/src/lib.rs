//! # Python Bindings for Ultimate Core Trading Engine
//!
//! High-performance Python bindings using PyO3 for seamless integration
//! with the Master Orchestrator.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use ultimate_core::*;

/// Python wrapper for the core trading engine
#[pyclass]
struct PyOrderBook {
    inner: OrderBook,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    fn new(symbol: String) -> Self {
        Self {
            inner: OrderBook::new(Symbol::new(&symbol)),
        }
    }
    
    fn add_order(&self, order_id: u64, side: String, price: f64, quantity: f64) -> PyResult<bool> {
        let side = match side.as_str() {
            "buy" | "BUY" => Side::Buy,
            "sell" | "SELL" => Side::Sell,
            _ => return Ok(false),
        };
        
        let order = Order::new_limit(
            OrderId::new(order_id),
            self.inner.symbol().clone(),
            side,
            Quantity::from_f64(quantity).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
            Price::from_f64(price).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
        );
        
        match self.inner.add_order(order) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    fn remove_order(&self, order_id: u64) -> PyResult<bool> {
        match self.inner.remove_order(OrderId::new(order_id)) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    fn best_bid(&self) -> Option<f64> {
        self.inner.best_bid().map(|p| p.to_f64())
    }
    
    fn best_ask(&self) -> Option<f64> {
        self.inner.best_ask().map(|p| p.to_f64())
    }
    
    fn spread(&self) -> Option<f64> {
        self.inner.spread().map(|p| p.to_f64())
    }
    
    fn mid_price(&self) -> Option<f64> {
        self.inner.mid_price().map(|p| p.to_f64())
    }
    
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.statistics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("symbol", stats.symbol.as_str())?;
            dict.set_item("best_bid", stats.best_bid.map(|p| p.to_f64()))?;
            dict.set_item("best_ask", stats.best_ask.map(|p| p.to_f64()))?;
            dict.set_item("bid_volume", stats.bid_volume.to_f64())?;
            dict.set_item("ask_volume", stats.ask_volume.to_f64())?;
            dict.set_item("total_orders", stats.total_orders)?;
            dict.set_item("update_count", stats.update_count)?;
            Ok(dict.into())
        })
    }
}

/// Python wrapper for performance statistics
#[pyfunction]
fn get_performance_stats() -> PyResult<PyObject> {
    let stats = ultimate_core::get_performance_stats();
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("orders_processed", stats.orders_processed)?;
        dict.set_item("orders_per_second", stats.orders_per_second)?;
        dict.set_item("average_execution_time_ns", stats.average_execution_time_ns)?;
        dict.set_item("peak_memory_usage", stats.peak_memory_usage)?;
        dict.set_item("trades_executed", stats.trades_executed)?;
        dict.set_item("total_profit_bps", stats.total_profit_bps)?;
        Ok(dict.into())
    })
}

/// Initialize the core engine from Python
#[pyfunction]
fn initialize_engine(enable_simd: bool, enable_fpga: bool) -> PyResult<bool> {
    let config = CoreConfig {
        enable_simd,
        enable_fpga,
        ..CoreConfig::default()
    };
    
    match ultimate_core::initialize_core_engine(config) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Python module definition
#[pymodule]
fn ultimate_rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOrderBook>()?;
    m.add_function(wrap_pyfunction!(get_performance_stats, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_engine, m)?)?;
    Ok(())
}

