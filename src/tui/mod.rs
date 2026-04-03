//! Interactive TUI (Terminal User Interface) for the Claude Agent SDK.
//!
//! Built with [`ratatui`](https://ratatui.rs) + [`crossterm`](https://docs.rs/crossterm).
//!
//! Enable with the `tui` feature flag:
//!
//! ```toml
//! claude-agent-sdk = { version = "0.1", features = ["tui"] }
//! ```
//!
//! # Quick start
//!
//! ```rust,no_run
//! use claude_agent_sdk::{AgentOptions, ApiClientConfig};
//! use claude_agent_sdk::tui::TuiApp;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut app = TuiApp::new(AgentOptions {
//!         api: ApiClientConfig::FromEnv,
//!         ..Default::default()
//!     }).await?;
//!     app.run().await?;
//!     Ok(())
//! }
//! ```

pub mod app;
pub mod components;
pub mod state;

pub use app::TuiApp;
pub use state::{AgentStatus, AppState, ChatMessage, ChatRole};
