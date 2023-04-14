//! Detect when the user presses Control-C so we can exit gracefully. If
//! the user presses Control-C twice, we'll exit immediately.

use std::{
    process::exit,
    sync::atomic::{AtomicBool, Ordering},
};

use anyhow::Result;

/// The flag that indicates whether the user has pressed Control-C.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Initialize the signal handler.
pub fn initialize_signal_handlers() -> Result<()> {
    // We'll use the ctrlc crate to detect when the user presses Control-C. We
    // can use swap() to atomically set the INTERRUPTED flag to true and return
    // the previous value.
    ctrlc::set_handler(move || {
        let previously_interrupted = INTERRUPTED.swap(true, Ordering::SeqCst);
        if previously_interrupted {
            // The user pressed Control-C twice, so we'll exit immediately.
            exit(1);
        }
    })?;
    Ok(())
}

/// Has the user pressed Control-C?
pub fn interrupted_by_user() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}
