//! # FPGA Acceleration Layer
//!
//! Hardware acceleration interface for ultra-low-latency trading operations.

pub mod hal;
pub mod kernels;

use thiserror::Error;

/// FPGA-related errors
#[derive(Error, Debug)]
pub enum FpgaError {
    #[error("Hardware not available")]
    HardwareNotAvailable,
    
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Communication error: {0}")]
    CommunicationError(String),
}

/// FPGA result type
pub type FpgaResult<T> = Result<T, FpgaError>;

/// Initialize FPGA subsystem
pub fn initialize_fpga() -> FpgaResult<()> {
    #[cfg(feature = "hardware")]
    {
        log::info!("Initializing FPGA hardware acceleration");
        // Hardware initialization would go here
        Ok(())
    }
    
    #[cfg(not(feature = "hardware"))]
    {
        log::info!("FPGA hardware not available, using CPU fallback");
        Ok(())
    }
}

