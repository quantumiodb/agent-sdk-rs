//! Simple example: blocking query with the agent.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --example simple

use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for log output
    tracing_subscriber::fmt::init();

    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        ..Default::default()
    })
    .await?;

    let result = agent.prompt("What files are in the current directory?").await?;

    println!("{}", result.text);
    println!("\n--- Stats ---");
    println!("Turns: {}", result.num_turns);
    println!("Tokens: {}", result.usage.total_tokens());
    println!("Cost: ${:.6}", result.cost_usd);
    println!("Duration: {}ms", result.duration_ms);

    agent.close().await?;
    Ok(())
}
