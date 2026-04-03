//! Default binary entry point — launches the interactive TUI.
//!
//! Run with:
//!   cargo run

use claude_agent_sdk::{AgentOptions, ApiClientConfig, PermissionMode};
use claude_agent_sdk::tui::TuiApp;

#[tokio::main]
async fn main() {
    let result = run().await;
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    let mut app = TuiApp::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        permission_mode: PermissionMode::BypassPermissions,
        ..Default::default()
    })
    .await?;

    app.run().await?;
    Ok(())
}
