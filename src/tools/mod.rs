//! Tool module: registry, executor, and built-in tool implementations.
//!
//! This module provides:
//!
//! - [`ToolRegistry`] — a name-keyed, alias-aware container for `Arc<dyn Tool>`.
//! - [`execute_tools`] — a parallel/serial tool execution engine.
//! - Built-in tools (gated on the `built-in-tools` feature): `Bash`, `FileRead`,
//!   `FileEdit`, `FileWrite`, `Glob`, `Grep`.

pub mod registry;
pub mod executor;

// Built-in tool implementations
pub mod bash;
pub mod file_read;
pub mod file_edit;
pub mod file_write;
pub mod glob;
pub mod grep;

// Re-exports
pub use registry::ToolRegistry;
pub use executor::{execute_tools, execute_single, ToolUseInput};

pub use bash::BashTool;
pub use file_read::FileReadTool;
pub use file_edit::FileEditTool;
pub use file_write::FileWriteTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
