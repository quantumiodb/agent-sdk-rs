//! NVIDIA NIM tool-use example — Grep & Bash.
//!
//! Tests the full tool-call round-trip using the built-in Grep and Bash tools
//! against the NVIDIA NIM OpenAI-compatible endpoint.
//!
//! Run with:
//!   NVIDIA_API_KEY=nvapi-... cargo run --example nvidia_nim

use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig, ContentDelta, SDKMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("NVIDIA_API_KEY").expect("NVIDIA_API_KEY not set");
    let base_url = std::env::var("NVIDIA_BASE_URL")
        .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".into());
    let model = std::env::var("NVIDIA_MODEL").unwrap_or_else(|_| "z-ai/glm4.7".into());

    println!("Provider : NVIDIA NIM");
    println!("Base URL : {base_url}");
    println!("Model    : {model}");
    println!("---");

    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::OpenAICompat {
            api_key,
            base_url,
            extra_body: None,
        },
        model,
        // Only expose Grep and Bash; block file-write tools for safety
        allowed_tools: Some(vec!["Grep".into()]),
        ..Default::default()
    })
    .await?;

    let prompt = "Use Grep to find all .rs files under src/tools/ that contain the word 'PermissionDenied'. List the matching files.";

    println!("User: {prompt}\n");
    println!("Assistant:");

    let (mut rx, handle) = agent.query(prompt)?;

    let mut in_thinking = false;
    while let Some(msg) = rx.recv().await {
        match msg {
            // New assistant turn starts — close any open <think> block.
            SDKMessage::Assistant { .. } => {
                if in_thinking {
                    println!("</think>");
                    in_thinking = false;
                }
            }
            SDKMessage::ContentDelta {
                delta: ContentDelta::ThinkingDelta { thinking },
                ..
            } => {
                if !in_thinking {
                    print!("<think>");
                    in_thinking = true;
                }
                print!("{thinking}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            SDKMessage::ContentDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
                if in_thinking {
                    print!("</think>\n");
                    in_thinking = false;
                }
                print!("{text}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            SDKMessage::Result {
                total_cost_usd,
                total_usage,
                num_turns,
                ..
            } => {
                println!(
                    "\n\n[done] turns={num_turns} input={} output={} cost=${total_cost_usd:.6}",
                    total_usage.input_tokens,
                    total_usage.output_tokens,
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
