//! Hook system for intercepting tool execution lifecycle events.
//!
//! Gated on the `hooks` feature flag. Provides [`HookEngine`] for dispatching
//! [`HookEvent`]s to registered [`HookRule`] handlers, with tool-name pattern
//! matching supporting `"*"`, `"Bash|Edit"`, and `"mcp__*"` wildcards.
//!
//! # Example
//!
//! ```rust,ignore
//! use claude_agent_sdk::hooks::{HookConfig, HookEngine, HookRule, HookOutput};
//! use std::sync::Arc;
//!
//! let config = HookConfig {
//!     pre_tool_use: vec![HookRule {
//!         matcher: Some("Bash".into()),
//!         handler: Arc::new(|input| Box::pin(async move {
//!             // Block destructive commands
//!             if let Some(cmd) = input.tool_input.as_ref()
//!                 .and_then(|v| v["command"].as_str())
//!             {
//!                 if cmd.contains("rm -rf") {
//!                     return HookOutput {
//!                         r#continue: false,
//!                         reason: Some("Blocked".into()),
//!                         ..Default::default()
//!                     };
//!                 }
//!             }
//!             HookOutput::default()
//!         })),
//!     }],
//!     ..Default::default()
//! };
//!
//! let engine = HookEngine::new(config);
//! ```

pub mod engine;
pub mod types;

pub use engine::{
    matches_tool, HookConfig, HookEngine, HookEvent, HookFn, HookInput, HookOutput, HookRule,
};
pub use types::HookError;
