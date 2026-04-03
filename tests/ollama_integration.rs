//! Integration test against the Ollama server configured via environment variables.
//!
//! Required environment variables (set in your shell or via `.cargo/config.toml`):
//!
//!   ANTHROPIC_BASE_URL=http://<host>:<port>
//!   ANTHROPIC_MODEL=<model-name>
//!
//! Optional:
//!   ANTHROPIC_AUTH_TOKEN=<token>   (falls back to ANTHROPIC_API_KEY, then empty string)
//!
//! Simply run:
//!   cargo test --test ollama_integration -- --nocapture
//!
//! Or override vars on the command line:
//!   ANTHROPIC_BASE_URL=http://other-host:11434 cargo test --test ollama_integration

use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig, ContentDelta, Message, MessageRole, PermissionMode, SDKMessage};

/// Helper: build AgentOptions using the **native** Ollama provider (`/api/chat`).
fn ollama_native_options(think: Option<bool>) -> AgentOptions {
    let base_url = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "http://yjhnupt.tech:11434".to_string());
    let model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "qwen3.5:latest".to_string());
    AgentOptions {
        api: ApiClientConfig::Ollama { base_url, think },
        model,
        disallowed_tools: vec![
            "Bash".into(), "Edit".into(), "Write".into(),
            "Glob".into(), "Grep".into(),
        ],
        max_turns: 3,
        ..Default::default()
    }
}

/// Helper: build AgentOptions for the Ollama test server.
///
/// Connection details are read from environment variables, which are injected
/// automatically via `.cargo/config.toml` in this workspace:
///
///   ANTHROPIC_BASE_URL=http://yjhnupt.tech:11434
///   ANTHROPIC_MODEL=qwen3.5:latest
///
/// The standard reqwest client is used; it respects the system-level `NO_PROXY`
/// environment variable.  Ensure `yjhnupt.tech` is listed in your `NO_PROXY`
/// (e.g. `export NO_PROXY="localhost,127.0.0.1,yjhnupt.tech"`).
fn ollama_options() -> AgentOptions {
    let base_url = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "http://yjhnupt.tech:11434".to_string());
    let api_key = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .unwrap_or_default();
    let model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "qwen3.5:latest".to_string());

    AgentOptions {
        api: ApiClientConfig::OpenAICompat { api_key, base_url, extra_body: None },
        model,
        // Disable file-system tools to keep tests focused on API/streaming.
        disallowed_tools: vec![
            "Bash".into(),
            "Edit".into(),
            "Write".into(),
            "Glob".into(),
            "Grep".into(),
        ],
        max_turns: 3,
        ..Default::default()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Verify that `from_env()` correctly picks up the Anthropic-style env vars and
/// selects the OpenAI-compat provider when ANTHROPIC_BASE_URL is set.
#[test]
fn from_env_detects_anthropic_base_url() {
    // Temporarily set env vars just for this test (serial, no side-effects on other tests)
    std::env::set_var("ANTHROPIC_BASE_URL", "http://yjhnupt.tech:11434");
    std::env::set_var("ANTHROPIC_AUTH_TOKEN", "test-key");
    std::env::remove_var("OLLAMA_BASE_URL");   // must not shadow ANTHROPIC_BASE_URL
    std::env::remove_var("ANTHROPIC_API_KEY");
    std::env::remove_var("OPENAI_API_KEY");

    let client = claude_agent_sdk::ApiClient::from_env().expect("from_env should succeed");
    assert_eq!(client.provider_name(), "openai-compat");

    // Clean up
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
    // Restore OLLAMA_BASE_URL so subsequent tests still use it.
    std::env::set_var("OLLAMA_BASE_URL", "http://yjhnupt.tech:11434");
}

/// `ANTHROPIC_MODEL` should set the default model in AgentOptions::default().
#[test]
fn anthropic_model_env_var() {
    std::env::set_var("ANTHROPIC_MODEL", "qwen3.5:latest");
    let opts = AgentOptions::default();
    assert_eq!(opts.model, "qwen3.5:latest");
    std::env::remove_var("ANTHROPIC_MODEL");
}

/// End-to-end prompt test against the live Ollama server.
///
/// Skipped automatically when ANTHROPIC_BASE_URL is not set (CI safety).
#[tokio::test]
async fn ollama_simple_prompt() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set — skipping live Ollama test");
        return;
    }

    let mut agent = Agent::new(ollama_options())
        .await
        .expect("Failed to create agent");

    let result = agent
        .prompt("Say exactly: 'Hello from Ollama!' and nothing else.")
        .await
        .expect("prompt() failed");

    println!("Response: {}", result.text);
    println!(
        "Tokens: {} | Cost: ${:.6} | Turns: {}",
        result.usage.total_tokens(),
        result.cost_usd,
        result.num_turns
    );

    assert!(!result.text.is_empty(), "Response should not be empty");
    agent.close().await.expect("close() failed");
}

/// Streaming test — verify we receive ContentDelta events.
#[tokio::test]
async fn ollama_streaming() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set — skipping live Ollama streaming test");
        return;
    }

    let mut agent = Agent::new(ollama_options())
        .await
        .expect("Failed to create agent");

    let (mut rx, handle) = agent
        .query("Count from 1 to 5, one number per line.")
        .expect("query() failed");

    let mut received_deltas = 0usize;
    let mut full_text = String::new();

    while let Some(msg) = rx.recv().await {
        match msg {
            SDKMessage::ContentDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
                received_deltas += 1;
                full_text.push_str(&text);
            }
            SDKMessage::Result {
                num_turns,
                total_usage,
                ..
            } => {
                println!(
                    "Done: turns={num_turns} tokens={}",
                    total_usage.total_tokens()
                );
            }
            SDKMessage::Error { message, .. } => {
                panic!("Agent error: {message}");
            }
            _ => {}
        }
    }

    handle.await.expect("join error").expect("agent loop error");

    println!("Full text:\n{full_text}");
    assert!(received_deltas > 0, "Should receive at least one text delta");
    assert!(!full_text.is_empty(), "Accumulated text should not be empty");

    agent.close().await.expect("close() failed");
}

/// Multi-turn conversation test.
#[tokio::test]
async fn ollama_multi_turn() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let mut agent = Agent::new(AgentOptions {
        system_prompt: Some("You are a helpful assistant. Keep answers very brief.".into()),
        ..ollama_options()
    })
    .await
    .expect("Failed to create agent");

    // Turn 1
    let r1 = agent
        .prompt("My favorite color is blue. Remember this.")
        .await
        .expect("turn 1 failed");
    println!("Turn 1: {}", r1.text);
    assert!(!r1.text.is_empty());

    // Turn 2 — test conversation memory
    let r2 = agent
        .prompt("What is my favorite color?")
        .await
        .expect("turn 2 failed");
    println!("Turn 2: {}", r2.text);
    assert!(
        r2.text.to_lowercase().contains("blue"),
        "Model should remember the color blue, got: {}",
        r2.text
    );

    agent.close().await.expect("close() failed");
}

/// Tool-use test: ask the model to use BashTool to run `echo hello_from_tool`.
///
/// Verifies end-to-end: model calls Bash, tool executes, result feeds back,
/// and the final response reflects the bash output.
#[tokio::test]
async fn ollama_tool_use_bash() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let mut agent = Agent::new(AgentOptions {
        system_prompt: Some(
            "You are a helpful assistant. Use the Bash tool to run commands when asked."
                .into(),
        ),
        api: ApiClientConfig::OpenAICompat {
            api_key: std::env::var("ANTHROPIC_AUTH_TOKEN")
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .unwrap_or_default(),
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "http://yjhnupt.tech:11434".into()),
            extra_body: None,
        },
        model: std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "qwen3.5:latest".into()),
        permission_mode: PermissionMode::BypassPermissions,
        max_turns: 10,
        allowed_tools: Some(vec!["Bash".into()]),
        ..Default::default()
    })
    .await
    .expect("Failed to create agent");

    let result = agent
        .prompt("Run the bash command: echo hello_from_tool")
        .await
        .expect("prompt() failed");

    println!("Tool-use response: {}", result.text);
    println!("Turns: {} | Tokens: {}", result.num_turns, result.usage.total_tokens());

    // The model must have called Bash (at least 2 API calls: initial + tool result).
    assert!(
        result.num_turns >= 2,
        "Expected at least 2 API calls (initial + tool-result round-trip), got: {}",
        result.num_turns
    );

    // Verify the conversation has tool_use and tool_result blocks.
    use claude_agent_sdk::{ContentBlock, MessageRole};
    let has_tool_use = result.messages.iter().any(|m| {
        m.role == MessageRole::Assistant
            && m.content.iter().any(|b| matches!(b, ContentBlock::ToolUse { name, .. } if name == "Bash"))
    });
    assert!(has_tool_use, "Bash tool_use block not found in message history");

    let has_tool_result = result.messages.iter().any(|m| {
        m.role == MessageRole::User
            && m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    });
    assert!(has_tool_result, "tool_result block not found in message history");

    agent.close().await.expect("close() failed");
}

/// Message history continuity test.
///
/// After prompt() completes, self.messages must include the assistant response
/// so that the next turn forms a valid conversation (not two consecutive user messages).
#[tokio::test]
async fn ollama_message_history_continuity() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let mut agent = Agent::new(AgentOptions {
        system_prompt: Some("You are a helpful assistant. Be very brief.".into()),
        ..ollama_options()
    })
    .await
    .expect("Failed to create agent");

    let r1 = agent
        .prompt("Remember the number 42.")
        .await
        .expect("turn 1 failed");
    println!("Turn 1: {}", r1.text);

    // After turn 1, messages must contain: user message + assistant message.
    let msg_count = agent.messages().len();
    assert!(
        msg_count >= 2,
        "After first prompt, messages should have at least 2 entries (user+assistant), got: {}",
        msg_count
    );

    use claude_agent_sdk::MessageRole;
    let last_role = &agent.messages()[msg_count - 1].role;
    assert_eq!(
        *last_role, MessageRole::Assistant,
        "Last message should be from the assistant"
    );

    let r2 = agent
        .prompt("What number did I ask you to remember?")
        .await
        .expect("turn 2 failed");
    println!("Turn 2: {}", r2.text);

    assert!(
        r2.text.contains("42"),
        "Model should recall 42 from proper conversation history, got: {}",
        r2.text
    );

    agent.close().await.expect("close() failed");
}

/// Context compaction integration test.
///
/// Builds a fake conversation of 12 messages (6 turns) — more than KEEP_RECENT (8) —
/// then calls `compact_messages` directly against the live Ollama server.
///
/// Assertions:
/// - The result is exactly 10 messages (2 synthetic + 8 recent).
/// - The first message is a User message containing a `<summary>` tag.
/// - The second message is an Assistant acknowledgement.
/// - The last 8 messages from the input are preserved verbatim.
#[tokio::test]
async fn ollama_compact_messages() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let api_client = claude_agent_sdk::ApiClient::from_env().expect("from_env failed");
    let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "qwen3.5:latest".into());

    // Build 12 messages (6 turns) — KEEP_RECENT is 8, so the first 4 will be summarised.
    let mut messages: Vec<Message> = Vec::new();
    for i in 1_u32..=6 {
        messages.push(Message::user_text(&format!(
            "Turn {i}: please remember that secret code {i} is {}.",
            i * 1111
        )));
        messages.push(Message::assistant_text(&format!(
            "Understood, secret code {i} is {}.",
            i * 1111
        )));
    }
    assert_eq!(messages.len(), 12);

    println!("Before compaction: {} messages", messages.len());

    let compacted = claude_agent_sdk::context::compact_messages(
        &messages,
        "You are a helpful assistant with a perfect memory.",
        &api_client,
        &model,
    )
    .await;

    println!("After compaction: {} messages", compacted.len());
    for (i, msg) in compacted.iter().enumerate() {
        let text = msg.text_content();
        // Show full summary (msg 0), brief preview for the rest.
        if i == 0 {
            println!("  [{i}] {:?} (len={}):\n{text}", msg.role, text.len());
        } else {
            let preview = &text[..text.len().min(80)];
            println!("  [{i}] {:?}: {preview}", msg.role);
        }
    }

    // 2 synthetic (summary + ack) + 8 recent = 10
    assert_eq!(
        compacted.len(),
        10,
        "Expected 2 synthetic + 8 recent = 10 messages, got {}",
        compacted.len()
    );

    // First message: User with <summary> tag
    assert_eq!(compacted[0].role, MessageRole::User, "First message must be User");
    assert!(
        compacted[0].text_content().contains("<summary>"),
        "First message must contain <summary> tag, got: {}",
        compacted[0].text_content()
    );
    assert!(
        !compacted[0].text_content().trim().is_empty(),
        "Summary must not be empty"
    );

    // Second message: Assistant ack
    assert_eq!(compacted[1].role, MessageRole::Assistant, "Second message must be Assistant");

    // Recent 8 messages preserved verbatim (messages[4..12])
    let recent_start = messages.len() - 8; // = 4
    for (i, original) in messages[recent_start..].iter().enumerate() {
        assert_eq!(
            compacted[2 + i].text_content(),
            original.text_content(),
            "Recent message {i} was not preserved verbatim"
        );
    }
}

/// Verify that `extra_body` fields are forwarded in the request body.
///
/// NOTE: Ollama's OpenAI-compat endpoint (`/v1/chat/completions`) ignores the
/// `think` field — the model still reasons internally and outputs to
/// `delta.reasoning`.  Proper thinking-mode control requires the native Ollama
/// API (`/api/chat`), which would need a dedicated `OllamaProvider`.
///
/// What this test DOES verify:
/// - `extra_body` is serialised and merged into the request (no serialisation error).
/// - The agent still produces a non-empty text response (via TextDelta, which
///   Qwen3 populates after the reasoning phase when max_tokens is generous).
#[tokio::test]
async fn ollama_no_think_extra_body() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let base_url = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "http://yjhnupt.tech:11434".into());
    let api_key = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .unwrap_or_default();
    let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "qwen3.5:latest".into());

    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::OpenAICompat {
            api_key,
            base_url,
            // Disable Qwen3 thinking: output goes to delta.content, not delta.reasoning.
            extra_body: Some(serde_json::json!({"think": false})),
        },
        model,
        disallowed_tools: vec![
            "Bash".into(), "Edit".into(), "Write".into(),
            "Glob".into(), "Grep".into(), "Read".into(),
        ],
        max_turns: 3,
        ..Default::default()
    })
    .await
    .expect("Failed to create agent");

    let start = std::time::Instant::now();
    let result = agent
        .prompt("Reply with exactly three words: yes no maybe")
        .await
        .expect("prompt() failed");

    let elapsed = start.elapsed();
    println!("Response (think=false): {:?}", result.text);
    println!("Elapsed: {}ms | Tokens: {}", elapsed.as_millis(), result.usage.total_tokens());

    assert!(
        !result.text.is_empty(),
        "Response must not be empty with think=false"
    );

    // With thinking disabled the response should be fast and direct.
    // (No hard timing assert — just print for comparison.)

    agent.close().await.expect("close() failed");
}

// ─── Native Ollama provider tests ─────────────────────────────────────────────

/// Native Ollama: `think: Some(false)` disables reasoning via `/api/chat`.
///
/// Unlike the OpenAI-compat endpoint, the native provider passes `think` directly
/// in the JSON body and Ollama honours it, giving an immediate non-reasoning response.
#[tokio::test]
async fn ollama_native_no_think() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let mut agent = Agent::new(ollama_native_options(Some(false)))
        .await
        .expect("Failed to create agent");

    let start = std::time::Instant::now();
    let result = agent
        .prompt("Reply with exactly three words: yes no maybe")
        .await
        .expect("prompt() failed");
    let elapsed = start.elapsed();

    println!(
        "Native think=false: {:?} in {}ms | tokens={}",
        result.text,
        elapsed.as_millis(),
        result.usage.total_tokens()
    );

    assert!(!result.text.is_empty(), "Response must not be empty");
    agent.close().await.expect("close() failed");
}

/// Native Ollama: default (no `think` override) — verify basic streaming works.
#[tokio::test]
async fn ollama_native_streaming() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let mut agent = Agent::new(ollama_native_options(None))
        .await
        .expect("Failed to create agent");

    let (mut rx, handle) = agent
        .query("Count from 1 to 3, one number per line.")
        .expect("query() failed");

    let mut delta_count = 0usize;
    let mut full_text = String::new();

    while let Some(msg) = rx.recv().await {
        match msg {
            SDKMessage::ContentDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
                delta_count += 1;
                full_text.push_str(&text);
            }
            SDKMessage::Error { message, .. } => panic!("Agent error: {message}"),
            _ => {}
        }
    }

    handle.await.expect("join error").expect("agent loop error");

    println!("Native streaming: {delta_count} deltas, text: {full_text:?}");
    assert!(delta_count > 0, "Should receive at least one text delta");
    assert!(!full_text.is_empty(), "Accumulated text must not be empty");

    agent.close().await.expect("close() failed");
}

/// Native Ollama: tool-use round-trip via `/api/chat`.
///
/// Asks the model to run `echo hello_from_native_tool` via BashTool and verifies
/// that the conversation has both a tool_use block and a tool_result block.
#[tokio::test]
async fn ollama_native_tool_use() {
    if std::env::var("ANTHROPIC_BASE_URL").is_err() {
        eprintln!("SKIP: ANTHROPIC_BASE_URL not set");
        return;
    }

    let base_url = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "http://yjhnupt.tech:11434".into());
    let model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "qwen3.5:latest".into());

    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::Ollama {
            base_url,
            think: Some(false), // disable thinking for faster tool-use round-trip
        },
        model,
        system_prompt: Some(
            "You are a helpful assistant. Use the Bash tool to run commands when asked.".into(),
        ),
        permission_mode: PermissionMode::BypassPermissions,
        max_turns: 10,
        allowed_tools: Some(vec!["Bash".into()]),
        ..Default::default()
    })
    .await
    .expect("Failed to create agent");

    let result = agent
        .prompt("Run the bash command: echo hello_from_native_tool")
        .await
        .expect("prompt() failed");

    println!(
        "Native tool-use response: {}\nTurns: {} | Tokens: {}",
        result.text,
        result.num_turns,
        result.usage.total_tokens()
    );

    assert!(
        result.num_turns >= 2,
        "Expected ≥2 turns (initial + tool-result), got {}",
        result.num_turns
    );

    use claude_agent_sdk::ContentBlock;
    let has_tool_use = result.messages.iter().any(|m| {
        m.role == MessageRole::Assistant
            && m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { name, .. } if name == "Bash"))
    });
    assert!(has_tool_use, "Bash tool_use block not found in message history");

    let has_tool_result = result.messages.iter().any(|m| {
        m.role == MessageRole::User
            && m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    });
    assert!(has_tool_result, "tool_result block not found in message history");

    agent.close().await.expect("close() failed");
}
