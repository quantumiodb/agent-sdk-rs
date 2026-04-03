//! Interactive TUI example: a full-featured terminal chat interface.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --features tui --example tui
//!
//! Or with Ollama:
//!   OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=llama3.2 cargo run --features tui --example tui
//!
//! Keyboard shortcuts:
//!   Enter       Send message
//!   Ctrl+C      Quit
//!   Ctrl+H      Toggle help overlay
//!   Ctrl+L      Clear messages
//!   PgUp/PgDn   Scroll messages
//!   Up/Down     Browse input history

use claude_agent_sdk::{AgentOptions, ApiClientConfig, PermissionMode};
use claude_agent_sdk::tui::TuiApp;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Use BypassPermissions for the TUI demo so tools run without prompts.
    let mut app = TuiApp::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        permission_mode: PermissionMode::BypassPermissions,
        ..Default::default()
    })
    .await?;

    app.run().await?;
    Ok(())
}
