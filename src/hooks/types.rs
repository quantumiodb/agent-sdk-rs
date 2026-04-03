//! Hook system error types.

/// Errors that can occur in the hook system.
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    /// A hook handler returned an error or panicked.
    #[error("Hook handler failed: {0}")]
    HandlerFailed(String),

    /// The hook handler was expected to return within a timeout.
    #[error("Hook handler timed out after {0}ms")]
    Timeout(u64),

    /// A hook matcher pattern was invalid.
    #[error("Invalid hook matcher pattern: {0}")]
    InvalidPattern(String),

    /// Serialization or deserialization error in hook I/O.
    #[error("Hook serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Catch-all for unexpected errors.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}
