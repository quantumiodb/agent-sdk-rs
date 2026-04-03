# claude-agent-sdk

A Rust SDK for building autonomous AI agents. Implements the agentic loop in-process — no CLI subprocess required.

Supports **Anthropic Claude** natively, and any **OpenAI-compatible** API (Ollama, Groq, LM Studio, etc.) out of the box.

---

## Features

- **Agentic loop** — runs multi-turn conversations with automatic tool execution until the model stops or `max_turns` is reached
- **Streaming** — receive text deltas, tool progress, and cost events in real time via an async channel
- **Built-in tools** — `Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep` (mirrors Claude Code's tool set)
- **Custom tools** — implement the `Tool` trait and register with `AgentOptions::custom_tools`
- **Provider flexibility** — Anthropic native SSE or OpenAI Chat Completions (translated automatically)
- **Multi-turn memory** — conversation history persisted across `prompt()` calls on the same `Agent`
- **Permission system** — `BypassPermissions`, `Default`, or rule-based access control per tool
- **Cost tracking** — per-model token pricing, cumulative `$USD` reporting
- **MCP support** — connect external tool servers via the Model Context Protocol (optional feature)

---

## Installation

```toml
[dependencies]
claude-agent-sdk = { path = "." }          # local
# claude-agent-sdk = "0.1"                 # crates.io (coming soon)

tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Cargo features

| Feature | Default | Description |
|---|---|---|
| `built-in-tools` | ✓ | Bash, Read, Edit, Write, Glob, Grep tools |
| `mcp` | ✓ | Model Context Protocol client |
| `cost-tracking` | — | Per-model USD cost computation |
| `web-tools` | — | HTTP fetch / web scraping tools |
| `subagents` | — | Spawn child agents from within a tool |
| `hooks` | — | Pre/post-tool lifecycle hooks |
| `all` | — | Enable everything |

---

## Quick start

### Anthropic Claude

```rust
use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Reads ANTHROPIC_API_KEY from the environment
    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        ..Default::default()
    }).await?;

    let result = agent.prompt("What files are in the current directory?").await?;

    println!("{}", result.text);
    println!("Turns: {} | Tokens: {} | Cost: ${:.6}",
        result.num_turns, result.usage.total_tokens(), result.cost_usd);

    agent.close().await?;
    Ok(())
}
```

Set the environment variable before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cargo run --example simple
```

### OpenAI / Ollama / Groq

```rust
use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = Agent::new(AgentOptions {
        // Local Ollama
        api: ApiClientConfig::OpenAICompat {
            api_key: String::new(),                          // not required by Ollama
            base_url: "http://localhost:11434".into(),
        },
        model: "qwen3.5:latest".into(),
        ..Default::default()
    }).await?;

    let result = agent.prompt("Summarise the Rust ownership model in 3 bullet points.").await?;
    println!("{}", result.text);

    agent.close().await?;
    Ok(())
}
```

```bash
# Groq
OPENAI_API_KEY=gsk_... cargo run --example openai -- groq

# Ollama (local)
cargo run --example openai -- ollama

# OpenAI
OPENAI_API_KEY=sk-... cargo run --example openai -- openai
```

---

## Streaming

Use `Agent::query()` to receive events as they arrive:

```rust
use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig, ContentDelta, SDKMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        model: "claude-sonnet-4-6".into(),
        ..Default::default()
    }).await?;

    let (mut rx, handle) = agent.query("List the 5 most common Rust patterns.")?;

    while let Some(msg) = rx.recv().await {
        match msg {
            SDKMessage::ContentDelta {
                delta: ContentDelta::TextDelta { text }, ..
            } => print!("{text}"),

            SDKMessage::ToolResult { tool_name, is_error, .. } => {
                eprintln!("[tool] {tool_name} → {}", if is_error { "FAILED" } else { "OK" });
            }

            SDKMessage::Result { num_turns, total_usage, total_cost_usd, .. } => {
                eprintln!("\n[done] {num_turns} turns | {} tokens | ${total_cost_usd:.6}",
                    total_usage.total_tokens());
            }

            SDKMessage::Error { message, .. } => eprintln!("[error] {message}"),
            _ => {}
        }
    }

    handle.await??;
    agent.close().await?;
    Ok(())
}
```

---

## Custom tools

Implement the `Tool` trait and pass instances via `AgentOptions::custom_tools`:

```rust
use async_trait::async_trait;
use claude_agent_sdk::{
    Agent, AgentOptions, ApiClientConfig,
    Tool, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
    PermissionCheckResult,
};
use serde_json::Value;
use std::sync::Arc;

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str { "GetWeather" }

    fn description(&self) -> &str {
        "Get the current weather for a city."
    }

    fn input_schema(&self) -> ToolInputSchema {
        ToolInputSchema {
            schema_type: "object".into(),
            properties: serde_json::from_value(serde_json::json!({
                "city": { "type": "string", "description": "City name" }
            })).unwrap(),
            required: vec!["city".into()],
            additional_properties: false,
        }
    }

    fn is_read_only(&self, _: &Value) -> bool { true }
    fn is_concurrency_safe(&self, _: &Value) -> bool { true }

    async fn call(
        &self,
        input: Value,
        _ctx: &mut ToolUseContext,
        _progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let city = input["city"].as_str().unwrap_or("unknown");
        Ok(ToolResult::text(format!("{city}: 22°C, partly cloudy")))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        custom_tools: vec![Arc::new(WeatherTool)],
        ..Default::default()
    }).await?;

    let result = agent.prompt("What is the weather in Paris?").await?;
    println!("{}", result.text);

    agent.close().await?;
    Ok(())
}
```

---

## Multi-turn conversations

`Agent` retains conversation history across `prompt()` calls:

```rust
let mut agent = Agent::new(AgentOptions {
    api: ApiClientConfig::FromEnv,
    system_prompt: Some("You are a helpful assistant. Keep answers brief.".into()),
    ..Default::default()
}).await?;

let r1 = agent.prompt("My name is Alice.").await?;
println!("{}", r1.text);

let r2 = agent.prompt("What is my name?").await?;
println!("{}", r2.text);  // "Your name is Alice."

agent.close().await?;
```

---

## Configuration reference

### `AgentOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `api` | `ApiClientConfig` | `FromEnv` | API provider and credentials |
| `model` | `String` | `claude-sonnet-4-6` (or `ANTHROPIC_MODEL` env) | Model ID |
| `system_prompt` | `Option<String>` | `None` | System prompt prepended to every conversation |
| `max_turns` | `u32` | `10` | Maximum API calls before stopping |
| `permission_mode` | `PermissionMode` | `Default` | Tool permission enforcement level |
| `custom_tools` | `Vec<Arc<dyn Tool>>` | `[]` | Additional tools available to the model |
| `allowed_tools` | `Option<Vec<String>>` | `None` | Whitelist of tool names (all allowed if `None`) |
| `disallowed_tools` | `Vec<String>` | `[]` | Blacklist of tool names |
| `working_dir` | `PathBuf` | `cwd` | Working directory for file/bash tools |
| `thinking` | `Option<ThinkingConfig>` | `None` | Extended thinking (Claude 3.7+) |

### `ApiClientConfig`

```rust
pub enum ApiClientConfig {
    /// Auto-detect from ANTHROPIC_API_KEY / OPENAI_API_KEY environment variables
    FromEnv,
    /// Anthropic Messages API (native SSE)
    Anthropic(String),           // API key
    /// OpenAI Chat Completions API
    OpenAI(String),              // API key
    /// Any OpenAI-compatible endpoint (Ollama, Groq, LM Studio, …)
    OpenAICompat { api_key: String, base_url: String },
}
```

### Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_BASE_URL` | Override base URL → selects OpenAI-compat provider |
| `ANTHROPIC_MODEL` | Default model (overrides compile-time default) |
| `ANTHROPIC_AUTH_TOKEN` | Alternative to `ANTHROPIC_API_KEY` |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | Override OpenAI base URL |
| `NO_PROXY` | Comma-separated hosts that bypass the HTTP proxy |

For local development you can put these in `.cargo/config.toml`:

```toml
[env]
ANTHROPIC_BASE_URL   = "http://localhost:11434"
ANTHROPIC_MODEL      = "qwen3.5:latest"
```

---

## SDK message protocol (`SDKMessage`)

| Variant | When emitted |
|---|---|
| `System` | Once at start — model, tool list, session ID |
| `ContentDelta` | Each streamed text chunk from the model |
| `ToolUse` | Model requested a tool call |
| `ToolProgress` | Tool is still running (periodic heartbeat) |
| `ToolResult` | Tool finished (success or error) |
| `Result` | Final summary — turns, tokens, cost, full message history |
| `Error` | Unrecoverable error |

---

## Built-in tools

| Name | Description |
|---|---|
| `Bash` | Execute shell commands (sandboxed via `PermissionMode`) |
| `Read` | Read a file with line numbers (`cat -n` format) |
| `Edit` | Exact string replacement in a file (read-before-edit enforced) |
| `Write` | Create or overwrite a file |
| `Glob` | Find files by glob pattern |
| `Grep` | Search file contents with regex (ripgrep-style) |

Tools can be selectively enabled or disabled per agent:

```rust
AgentOptions {
    allowed_tools: Some(vec!["Bash".into(), "Read".into()]),  // only these two
    disallowed_tools: vec!["Write".into()],                   // or block specific ones
    ..Default::default()
}
```

---

## Permission modes

| Mode | Behaviour |
|---|---|
| `Default` | Prompts the user for dangerous operations (write, bash, etc.) |
| `BypassPermissions` | All tools run without prompts (use in trusted automation) |
| `Plan` | Read-only tools allowed; write tools blocked |

---

## Running the tests

```bash
# Unit tests
cargo test

# Integration tests against a local Ollama server
# (reads connection details from .cargo/config.toml automatically)
cargo test --test ollama_integration -- --nocapture

# Override for a different server
ANTHROPIC_BASE_URL=http://other:11434 ANTHROPIC_MODEL=llama3 \
  cargo test --test ollama_integration
```

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
