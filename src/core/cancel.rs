//! Cooperative cancellation token shared across CLI, Desktop UI, Web UI, and tile workers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::error::{PipelineError, Result};

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    #[inline]
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    #[inline]
    pub fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(PipelineError::Cancelled)
        } else {
            Ok(())
        }
    }
}
