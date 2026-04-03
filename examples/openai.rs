//! OpenAI provider example.
//!
//! Demonstrates using the SDK with OpenAI, Groq, or any OpenAI-compatible API.
//!
//! Run with:
//!   # OpenAI
//!   OPENAI_API_KEY=sk-... cargo run --example openai
//!
//!   # Groq (llama-3.1-70b-versatile)
//!   OPENAI_API_KEY=gsk_... OPENAI_BASE_URL=https://api.groq.com/openai \
//!     cargo run --example openai -- groq
//!
//!   # Local Ollama
//!   OPENAI_BASE_URL=http://localhost:11434 \
//!     cargo run --example openai -- ollama
//!
//!   # NVIDIA NIM (https://integrate.api.nvidia.com/v1)
//!   NVIDIA_API_KEY=nvapi-... cargo run --example openai -- nvidia

use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig, ContentDelta, SDKMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let provider = args.get(1).map(|s| s.as_str()).unwrap_or("auto");

    let api_config = match provider {
        "openai" => ApiClientConfig::OpenAI(
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
        ),
        "groq" => ApiClientConfig::OpenAICompat {
            api_key: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
            base_url: "https://api.groq.com/openai".into(),
            extra_body: None,
        },
        "ollama" => ApiClientConfig::OpenAICompat {
            api_key: String::new(), // Ollama doesn't require a real key
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".into()),
            extra_body: None,
        },
        "nvidia" => ApiClientConfig::OpenAICompat {
            api_key: std::env::var("NVIDIA_API_KEY").expect("NVIDIA_API_KEY not set"),
            base_url: std::env::var("NVIDIA_BASE_URL")
                .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".into()),
            extra_body: None,
        },
        _ => {
            // Auto-detect from environment (ANTHROPIC_API_KEY or OPENAI_API_KEY)
            ApiClientConfig::FromEnv
        }
    };

    // Use an OpenAI model name (or Claude model if using Anthropic).
    // For NVIDIA NIM and the auto path, honour OPENAI_MODEL if set.
    let model = match provider {
        "groq" => "llama-3.1-70b-versatile".to_string(),
        "ollama" => "llama3.2".to_string(),
        "openai" => "gpt-4o".to_string(),
        "nvidia" => std::env::var("NVIDIA_MODEL")
            .unwrap_or_else(|_| "z-ai/glm4.7".to_string()),
        _ => std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-6".to_string()),
    };

    let mut agent = Agent::new(AgentOptions {
        api: api_config,
        model: model.clone(),
        // Disable built-in tools for this simple demo
        disallowed_tools: vec![
            "Bash".into(),
            "Edit".into(),
            "Write".into(),
        ],
        ..Default::default()
    })
    .await?;

    println!("Using model: {model}");
    println!("---");

    let (mut rx, handle) = agent.query(
        "What is the capital of France, and what is its population? \
         Give a one-paragraph answer.",
    )?;

    while let Some(msg) = rx.recv().await {
        match msg {
            SDKMessage::ContentDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
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
                    "\n\n[done] turns={num_turns} tokens={} cost=${total_cost_usd:.6}",
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
