# claude-agent-sdk

A Rust SDK for building autonomous AI agents. Implements the agentic loop in-process — no CLI subprocess required.

Supports **Anthropic Claude** natively, plus any **OpenAI-compatible** API: Ollama, NVIDIA NIM, Groq, LM Studio, and more.

---

## Features

- **Agentic loop** — multi-turn conversations with automatic tool execution until the model stops or `max_turns` is reached
- **Streaming** — text deltas, tool progress, and cost events delivered in real time via an async channel
- **Built-in tools** — `Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep` (mirrors Claude Code's tool set)
- **Custom tools** — implement the `Tool` trait and register with `AgentOptions::custom_tools`
- **Multi-provider** — Anthropic native SSE, Ollama `/api/chat`, OpenAI Chat Completions (auto-translated)
- **Multi-turn memory** — conversation history persisted across `prompt()` calls on the same `Agent`
- **Permission system** — `BypassPermissions`, `Default`, or rule-based access control per tool
- **Cost tracking** — per-model token pricing, cumulative `$USD` reporting
- **MCP support** — connect external tool servers via the Model Context Protocol (optional feature)

---

## Installation

```toml
[dependencies]
claude-agent-sdk = { path = "." }

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

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cargo run --example simple
```

### Ollama (native `/api/chat`)

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3.5:latest
cargo run --example simple
```

`from_env()` detects `OLLAMA_BASE_URL` and selects the Ollama native provider automatically.

### Ollama via Anthropic Messages API (`/v1/messages`)

Some Ollama-compatible servers expose the Anthropic Messages API format. Set `ANTHROPIC_API_FORMAT=anthropic`:

```bash
export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_MODEL=qwen3.5:latest
export ANTHROPIC_NUM_CTX=65535
export ANTHROPIC_API_FORMAT=anthropic
cargo run --example simple
```

### NVIDIA NIM

```bash
# Credentials are read from NVIDIA_* env vars (set once in .cargo/config.toml)
cargo run --example nvidia_nim
```

Or inline:

```bash
NVIDIA_API_KEY=nvapi-... cargo run --example nvidia_nim
```

The `nvidia_nim` example uses `Grep` tool use end-to-end and displays `<think>…</think>` reasoning blocks from GLM4.7.

### OpenAI / Groq / LM Studio

```bash
# OpenAI
OPENAI_API_KEY=sk-... cargo run --example openai -- openai

# Groq
OPENAI_API_KEY=gsk_... cargo run --example openai -- groq

# LM Studio (local)
cargo run --example openai -- lmstudio
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
};
use serde_json::Value;
use std::sync::Arc;

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str { "GetWeather" }

    fn description(&self) -> &str { "Get the current weather for a city." }

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
| `model` | `String` | provider default | Model ID (overrides env-based default) |
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
    /// Auto-detect from environment variables (see table below)
    FromEnv,
    /// Anthropic Messages API (native SSE)
    Anthropic { api_key: String },
    /// Ollama native /api/chat
    Ollama { base_url: String, model: Option<String> },
    /// OpenAI Chat Completions (or any compatible endpoint)
    OpenAICompat { api_key: String, base_url: String, extra_body: Option<serde_json::Value> },
}
```

### Environment variables and `from_env()` detection order

`ApiClientConfig::FromEnv` detects the active provider in this order:

| Priority | Variable | Provider selected |
|---|---|---|
| 1 | `OLLAMA_BASE_URL` | Ollama native (`/api/chat`) |
| 2 | `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_FORMAT=anthropic` | Anthropic provider at custom URL |
| 2 | `ANTHROPIC_BASE_URL` (no format flag) | OpenAI-compat at that URL |
| 3 | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` | Anthropic (api.anthropic.com) |
| 4 | `OPENAI_API_KEY` | OpenAI-compat (api.openai.com) |

Full variable reference:

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL (e.g. `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model name for Ollama native provider |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_AUTH_TOKEN` | Alternative to `ANTHROPIC_API_KEY` |
| `ANTHROPIC_BASE_URL` | Custom Anthropic-compatible base URL |
| `ANTHROPIC_MODEL` | Model name when using an Anthropic provider |
| `ANTHROPIC_API_FORMAT` | Set to `anthropic` to force Anthropic Messages API format |
| `ANTHROPIC_NUM_CTX` | Context window size (forwarded as `options.num_ctx`) |
| `NVIDIA_API_KEY` | NVIDIA NIM API key (`nvapi-…`) |
| `NVIDIA_BASE_URL` | NIM endpoint (default: `https://integrate.api.nvidia.com/v1`) |
| `NVIDIA_MODEL` | Model name (default: `z-ai/glm4.7`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | Override OpenAI base URL |
| `NO_PROXY` | Comma-separated hosts that bypass the HTTP proxy |

### Persistent dev config (`.cargo/config.toml`)

Put always-on settings in `.cargo/config.toml` (gitignored alongside `.env.*`):

```toml
[env]
NO_PROXY        = { value = "localhost,127.0.0.1", force = true }
NVIDIA_API_KEY  = "nvapi-..."
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL    = "z-ai/glm4.7"
```

Per-provider settings go in `.env.*` files (sourced at test time):

```bash
# .env.ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3.5:latest

# .env.anthropic-ollama
export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_MODEL=qwen3.5:latest
export ANTHROPIC_NUM_CTX=65535
export ANTHROPIC_API_FORMAT=anthropic
```

---

## Running the examples and tests

A `Makefile` covers all provider configurations:

```bash
make test-unit                         # Unit tests (no network)
make test-ollama                       # Ollama native integration tests
make test-anthropic-ollama             # Ollama via Anthropic API format
make run-nvidia                        # NVIDIA NIM tool-use smoke test
make test-all                          # All of the above in sequence

# Run a specific example against a provider:
make run-ollama EXAMPLE=simple
make run-anthropic-ollama EXAMPLE=streaming
```

---

## SDK message protocol (`SDKMessage`)

| Variant | When emitted |
|---|---|
| `System` | Once at start — model, tool list, session ID |
| `Assistant` | Start of each assistant turn |
| `ContentDelta` | Each streamed text or thinking chunk |
| `ToolUse` | Model requested a tool call |
| `ToolProgress` | Tool still running (periodic heartbeat) |
| `ToolResult` | Tool finished (success or error) |
| `Result` | Final summary — turns, tokens, cost, full message history |
| `Error` | Unrecoverable error |

`ContentDelta` carries either `TextDelta { text }` or `ThinkingDelta { thinking }`. Both Ollama reasoning (`reasoning`) and NVIDIA NIM reasoning (`reasoning_content`) are normalised to `ThinkingDelta`.

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

Filter which tools the model can use:

```rust
AgentOptions {
    allowed_tools: Some(vec!["Grep".into(), "Read".into()]),  // only these two
    disallowed_tools: vec!["Bash".into()],                    // or block specific ones
    ..Default::default()
}
```

---

## Permission modes

| Mode | Behaviour |
|---|---|
| `Default` | Prompts for dangerous operations (write, bash, etc.) |
| `BypassPermissions` | All tools run without prompts (use in trusted automation) |
| `Plan` | Read-only tools allowed; write tools blocked |

---

## Provider quirks

See [`docs/provider-quirks.md`](docs/provider-quirks.md) for a detailed breakdown of non-standard behaviour discovered during integration testing, including GLM4.7 (NVIDIA NIM) specifics:

- `finish_reason` always `null` → SDK synthesises stop reason on `[DONE]`
- Complete tool call delivered in a single chunk
- Chain-of-thought in `reasoning_content` (not `reasoning`)
- `<tool_call>` tags echoed in `content` on multi-turn responses
- Usage object arrives in a separate empty-choices chunk

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
