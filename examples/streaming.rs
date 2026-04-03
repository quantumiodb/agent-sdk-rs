//! Streaming example: process events from the agent in real time.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --example streaming

use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig, ContentDelta, SDKMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("claude_agent_sdk=info")
        .init();

    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        model: "claude-sonnet-4-6".into(),
        ..Default::default()
    })
    .await?;

    let (mut rx, handle) = agent
        .query("List the 5 most common Rust programming patterns and briefly explain each.")?;

    while let Some(msg) = rx.recv().await {
        match msg {
            SDKMessage::System { model, tools, .. } => {
                eprintln!("[system] model={model}, tools={}", tools.len());
            }
            SDKMessage::ContentDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
                print!("{text}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            SDKMessage::ToolProgress {
                tool_name,
                elapsed_ms,
                ..
            } => {
                eprintln!("\n[tool] {tool_name} running... ({elapsed_ms}ms)");
            }
            SDKMessage::ToolResult {
                tool_name,
                is_error,
                ..
            } => {
                let status = if is_error { "FAILED" } else { "OK" };
                eprintln!("[tool] {tool_name} → {status}");
            }
            SDKMessage::Result {
                num_turns,
                total_cost_usd,
                duration_ms,
                total_usage,
                ..
            } => {
                eprintln!(
                    "\n\n[done] {num_turns} turns | {} tokens | ${total_cost_usd:.6} | {duration_ms}ms",
                    total_usage.total_tokens()
                );
            }
            SDKMessage::Error { message, .. } => {
                eprintln!("\n[error] {message}");
            }
            _ => {}
        }
    }

    handle.await??;
    agent.close().await?;
    Ok(())
}
