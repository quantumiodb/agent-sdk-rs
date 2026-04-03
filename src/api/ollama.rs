//! Native Ollama provider — calls `/api/chat` directly.
//!
//! Unlike [`super::openai_compat::OpenAICompatProvider`] (which uses
//! `/v1/chat/completions`), this provider uses Ollama's own REST API and
//! therefore supports Ollama-specific features:
//!
//! - `think: bool` — enable / disable extended thinking (Qwen3, DeepSeek-R1, …)
//! - `options` — model parameters (`temperature`, `num_ctx`, `seed`, …)
//!
//! # Wire format
//!
//! Ollama streams **NDJSON** — one JSON object per line, no `data:` prefix —
//! in contrast to the SSE format used by Anthropic and OpenAI.
//!
//! Each streaming chunk looks like:
//! ```json
//! {"model":"qwen3.5","message":{"role":"assistant","content":"Hello"},"done":false}
//! ```
//! The final chunk has `"done": true` and carries usage counters
//! (`prompt_eval_count`, `eval_count`).
//!
//! Tool calls arrive as a **complete object** in a single chunk:
//! ```json
//! {"message":{"role":"assistant","content":"",
//!   "tool_calls":[{"id":"call_x","function":{"name":"Bash","arguments":{"command":"ls"}}}]},
//!  "done":false}
//! ```

use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use super::client::{ApiError, ApiProvider, ApiStream};
use super::types::{
    ApiMessageStart, ApiStreamEvent, DeltaUsage, MessageDeltaBody, MessageRequest,
};
use crate::types::{ContentBlock, ContentDelta, StopReason};

// ─────────────────────────────────────────────────────────────────────────────
// Ollama request types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    /// Enable / disable extended thinking (`None` = let the model decide).
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    function: OllamaFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaFunction {
    #[serde(default)]
    name: String,
    /// Arguments as a JSON object (Ollama uses objects, not serialised strings).
    arguments: Value,
}

#[derive(Debug, Serialize)]
struct OllamaToolDef {
    #[serde(rename = "type")]
    tool_type: String, // "function"
    function: OllamaFunctionDef,
}

#[derive(Debug, Serialize)]
struct OllamaFunctionDef {
    name: String,
    description: String,
    parameters: Value,
}

/// Ollama model-level options.
///
/// Only the most common fields are listed; serde will forward any extra fields
/// set via the public `extra` map if needed.
#[derive(Debug, Default, Serialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ollama streaming response types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OllamaChunk {
    model: Option<String>,
    message: Option<OllamaChunkMessage>,
    done: bool,
    done_reason: Option<String>,
    /// Input tokens (present in the final `done: true` chunk).
    #[serde(default)]
    prompt_eval_count: u64,
    /// Output tokens (present in the final `done: true` chunk).
    #[serde(default)]
    eval_count: u64,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaChunkMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Request conversion
// ─────────────────────────────────────────────────────────────────────────────

fn convert_request(req: &MessageRequest, think: Option<bool>) -> OllamaRequest {
    let mut messages: Vec<OllamaMessage> = Vec::new();

    // System prompt
    if let Some(system) = &req.system {
        let text = match system {
            super::types::SystemPrompt::Text(t) => t.clone(),
            super::types::SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        messages.push(OllamaMessage {
            role: "system".into(),
            content: text,
            thinking: None,
            tool_calls: None,
        });
    }

    // Conversation messages
    for msg in &req.messages {
        let converted = convert_message(&msg.role, &msg.content);
        messages.extend(converted);
    }

    // Tool definitions
    let tools: Vec<OllamaToolDef> = req
        .tools
        .iter()
        .map(|t| OllamaToolDef {
            tool_type: "function".into(),
            function: OllamaFunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    let options = if req.temperature.is_some() || req.top_p.is_some() || req.top_k.is_some() {
        Some(OllamaOptions {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k.map(|v| v),
            ..Default::default()
        })
    } else {
        None
    };

    OllamaRequest {
        model: req.model.clone(),
        messages,
        stream: true,
        think,
        tools,
        options,
    }
}

/// Convert a single Anthropic-format message to Ollama messages.
///
/// The main cases:
/// - User text → `role: "user"`
/// - User tool_result → one `role: "tool"` message per result
/// - Assistant text → `role: "assistant"`
/// - Assistant tool_use → `role: "assistant"` with `tool_calls`
fn convert_message(role: &str, content: &Value) -> Vec<OllamaMessage> {
    match content {
        Value::String(text) => vec![OllamaMessage {
            role: role.to_string(),
            content: text.clone(),
            thinking: None,
            tool_calls: None,
        }],

        Value::Array(blocks) => {
            if role == "assistant" {
                convert_assistant_blocks(blocks)
            } else {
                convert_user_blocks(blocks)
            }
        }

        other => vec![OllamaMessage {
            role: role.to_string(),
            content: other.to_string(),
            thinking: None,
            tool_calls: None,
        }],
    }
}

fn convert_assistant_blocks(blocks: &[Value]) -> Vec<OllamaMessage> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OllamaToolCall> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(t.to_string());
                }
            }
            Some("thinking") => {
                if let Some(t) = block.get("thinking").and_then(|v| v.as_str()) {
                    thinking_parts.push(t.to_string());
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = block
                    .get("input")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                tool_calls.push(OllamaToolCall {
                    id,
                    function: OllamaFunction { name, arguments },
                });
            }
            _ => {}
        }
    }

    vec![OllamaMessage {
        role: "assistant".into(),
        content: text_parts.join(""),
        thinking: if thinking_parts.is_empty() {
            None
        } else {
            Some(thinking_parts.join(""))
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    }]
}

fn convert_user_blocks(blocks: &[Value]) -> Vec<OllamaMessage> {
    let mut messages: Vec<OllamaMessage> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(t.to_string());
                }
            }
            Some("tool_result") => {
                // Flush text before the tool result
                if !text_parts.is_empty() {
                    messages.push(OllamaMessage {
                        role: "user".into(),
                        content: text_parts.drain(..).collect::<Vec<_>>().join(""),
                        thinking: None,
                        tool_calls: None,
                    });
                }
                // Tool result → role: "tool"
                let result_text = if let Some(content) = block.get("content") {
                    match content {
                        Value::Array(items) => items
                            .iter()
                            .filter_map(|item| {
                                if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                                    item.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    }
                } else {
                    String::new()
                };
                messages.push(OllamaMessage {
                    role: "tool".into(),
                    content: result_text,
                    thinking: None,
                    tool_calls: None,
                });
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        messages.push(OllamaMessage {
            role: "user".into(),
            content: text_parts.join(""),
            thinking: None,
            tool_calls: None,
        });
    }

    if messages.is_empty() {
        messages.push(OllamaMessage {
            role: "user".into(),
            content: String::new(),
            thinking: None,
            tool_calls: None,
        });
    }

    messages
}

// ─────────────────────────────────────────────────────────────────────────────
// NDJSON stream → ApiStreamEvent
// ─────────────────────────────────────────────────────────────────────────────

struct OllamaNdjsonStream<S> {
    inner: S,
    buf: String,
    started: bool,
    next_block_idx: usize,
    /// Index of the current text content block (if open).
    text_block_idx: Option<usize>,
    /// Index of the current thinking content block (if open).
    thinking_block_idx: Option<usize>,
    pending: std::collections::VecDeque<Result<ApiStreamEvent, ApiError>>,
    done: bool,
    model: String,
}

impl<S> OllamaNdjsonStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    fn new(inner: S, model: String) -> Self {
        Self {
            inner,
            buf: String::new(),
            started: false,
            next_block_idx: 0,
            text_block_idx: None,
            thinking_block_idx: None,
            pending: Default::default(),
            done: false,
            model,
        }
    }

    fn alloc_block(&mut self) -> usize {
        let idx = self.next_block_idx;
        self.next_block_idx += 1;
        idx
    }

    fn parse_line(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }

        let chunk: OllamaChunk = match serde_json::from_str(line) {
            Ok(c) => c,
            Err(e) => {
                debug!(raw = line, error = %e, "Failed to parse Ollama chunk");
                return;
            }
        };

        // Capture model name
        if self.model.is_empty() {
            if let Some(m) = &chunk.model {
                self.model = m.clone();
            }
        }

        // Synthetic MessageStart on the first chunk
        if !self.started {
            self.pending.push_back(Ok(ApiStreamEvent::MessageStart {
                message: ApiMessageStart {
                    id: format!("ollama-{}", uuid::Uuid::new_v4()),
                    message_type: "message".into(),
                    role: "assistant".into(),
                    model: self.model.clone(),
                    usage: None,
                },
            }));
            self.started = true;
        }

        if let Some(msg) = &chunk.message {
            // ── thinking delta ─────────────────────────────────────────────
            if !msg.thinking.is_empty() {
                let idx = match self.thinking_block_idx {
                    Some(i) => i,
                    None => {
                        let i = self.alloc_block();
                        self.thinking_block_idx = Some(i);
                        self.pending.push_back(Ok(ApiStreamEvent::ContentBlockStart {
                            index: i,
                            content_block: ContentBlock::Thinking {
                                thinking: String::new(),
                                signature: None,
                            },
                        }));
                        i
                    }
                };
                self.pending.push_back(Ok(ApiStreamEvent::ContentBlockDelta {
                    index: idx,
                    delta: ContentDelta::ThinkingDelta {
                        thinking: msg.thinking.clone(),
                    },
                }));
            }

            // ── text delta ─────────────────────────────────────────────────
            if !msg.content.is_empty() {
                // Close thinking block before opening text block
                if let Some(ti) = self.thinking_block_idx.take() {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: ti }));
                }
                let idx = match self.text_block_idx {
                    Some(i) => i,
                    None => {
                        let i = self.alloc_block();
                        self.text_block_idx = Some(i);
                        self.pending.push_back(Ok(ApiStreamEvent::ContentBlockStart {
                            index: i,
                            content_block: ContentBlock::Text { text: String::new() },
                        }));
                        i
                    }
                };
                self.pending.push_back(Ok(ApiStreamEvent::ContentBlockDelta {
                    index: idx,
                    delta: ContentDelta::TextDelta {
                        text: msg.content.clone(),
                    },
                }));
            }

            // ── tool calls (arrive complete in one chunk) ──────────────────
            if let Some(tool_calls) = &msg.tool_calls {
                // Close any open text/thinking blocks
                if let Some(ti) = self.thinking_block_idx.take() {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: ti }));
                }
                if let Some(ti) = self.text_block_idx.take() {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: ti }));
                }

                for tc in tool_calls {
                    let idx = self.alloc_block();
                    let id = tc
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("call_{idx}"));

                    self.pending.push_back(Ok(ApiStreamEvent::ContentBlockStart {
                        index: idx,
                        content_block: ContentBlock::ToolUse {
                            id: id.clone(),
                            name: tc.function.name.clone(),
                            input: Value::Object(Default::default()),
                        },
                    }));

                    // Emit the full arguments as a single InputJsonDelta
                    let args_str = serde_json::to_string(&tc.function.arguments)
                        .unwrap_or_else(|_| "{}".into());
                    self.pending.push_back(Ok(ApiStreamEvent::ContentBlockDelta {
                        index: idx,
                        delta: ContentDelta::InputJsonDelta {
                            partial_json: args_str,
                        },
                    }));

                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: idx }));
                }
            }
        }

        // ── final chunk ────────────────────────────────────────────────────
        if chunk.done {
            // Close any still-open blocks
            if let Some(ti) = self.thinking_block_idx.take() {
                self.pending
                    .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: ti }));
            }
            if let Some(ti) = self.text_block_idx.take() {
                self.pending
                    .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: ti }));
            }

            let stop_reason = match chunk.done_reason.as_deref() {
                Some("stop") | None => Some(StopReason::EndTurn),
                Some("tool_calls") => Some(StopReason::ToolUse),
                Some("length") => Some(StopReason::MaxTokens),
                _ => Some(StopReason::EndTurn),
            };

            self.pending.push_back(Ok(ApiStreamEvent::MessageDelta {
                delta: MessageDeltaBody {
                    stop_reason,
                    stop_sequence: None,
                },
                usage: Some(DeltaUsage {
                    output_tokens: chunk.eval_count,
                }),
            }));
            self.pending.push_back(Ok(ApiStreamEvent::MessageStop));
            self.done = true;
        }
    }
}

impl<S> Stream for OllamaNdjsonStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ApiStreamEvent, ApiError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(event));
            }

            if self.done {
                return Poll::Ready(None);
            }

            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Stream ended without a done chunk — close gracefully
                    if let Some(idx) = self.thinking_block_idx.take() {
                        return Poll::Ready(Some(Ok(ApiStreamEvent::ContentBlockStop { index: idx })));
                    }
                    if let Some(idx) = self.text_block_idx.take() {
                        return Poll::Ready(Some(Ok(ApiStreamEvent::ContentBlockStop { index: idx })));
                    }
                    return Poll::Ready(Some(Ok(ApiStreamEvent::MessageStop)));
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ApiError::Network {
                        message: format!("Ollama stream error: {e}"),
                        source: e,
                    })));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    let text = match std::str::from_utf8(&bytes) {
                        Ok(s) => s.to_string(),
                        Err(_) => String::from_utf8_lossy(&bytes).to_string(),
                    };
                    self.buf.push_str(&text);

                    while let Some(pos) = self.buf.find('\n') {
                        let line = self.buf[..pos].to_string();
                        self.buf = self.buf[pos + 1..].to_string();
                        self.parse_line(&line);
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OllamaProvider
// ─────────────────────────────────────────────────────────────────────────────

/// API provider for Ollama's native `/api/chat` endpoint.
///
/// Supports Ollama-specific features unavailable through the OpenAI-compat layer:
/// - `think: bool` — toggle extended thinking per request
/// - `options` — model hyper-parameters (`temperature`, `num_ctx`, `seed`, …)
///
/// # Example
///
/// ```rust,ignore
/// use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig};
///
/// let mut agent = Agent::new(AgentOptions {
///     api: ApiClientConfig::Ollama {
///         base_url: "http://localhost:11434".into(),
///         think: Some(false),  // disable thinking for faster responses
///     },
///     model: "qwen3.5:latest".into(),
///     ..Default::default()
/// }).await?;
/// ```
pub struct OllamaProvider {
    base_url: String,
    think: Option<bool>,
    http_client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(base_url: impl Into<String>, think: Option<bool>) -> Self {
        let http_client = reqwest::ClientBuilder::new()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_default();
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            think,
            http_client,
        }
    }
}

impl std::fmt::Debug for OllamaProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaProvider")
            .field("base_url", &self.base_url)
            .field("think", &self.think)
            .finish()
    }
}

#[async_trait]
impl ApiProvider for OllamaProvider {
    async fn stream_message(&self, request: MessageRequest) -> Result<ApiStream, ApiError> {
        let url = format!("{}/api/chat", self.base_url);
        let model = request.model.clone();
        let ollama_req = convert_request(&request, self.think);

        debug!(
            model = %model,
            think = ?self.think,
            messages = ollama_req.messages.len(),
            tools = ollama_req.tools.len(),
            "Sending request to Ollama /api/chat"
        );

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let body = serde_json::to_vec(&ollama_req).map_err(|e| ApiError::InvalidResponse {
            message: format!("Failed to serialize Ollama request: {e}"),
            body: String::new(),
        })?;

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    ApiError::Network {
                        message: format!("Cannot connect to Ollama at {}: {e}", self.base_url),
                        source: e,
                    }
                } else {
                    ApiError::Network {
                        message: format!("Ollama request failed: {e}"),
                        source: e,
                    }
                }
            })?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::ApiError {
                status,
                message: body,
                error_type: "ollama_error".into(),
            });
        }

        let byte_stream = response.bytes_stream();
        Ok(Box::pin(OllamaNdjsonStream::new(byte_stream, model)))
    }

    fn name(&self) -> &str {
        "ollama"
    }

    fn supported_models(&self) -> &[&str] {
        &[]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_simple_user_message() {
        let req = MessageRequest {
            model: "qwen3.5".into(),
            max_tokens: 512,
            system: None,
            messages: vec![super::super::types::RequestMessage {
                role: "user".into(),
                content: Value::String("Hello".into()),
            }],
            tools: vec![],
            stream: true,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            metadata: None,
        };
        let ollama = convert_request(&req, None);
        assert_eq!(ollama.messages.len(), 1);
        assert_eq!(ollama.messages[0].role, "user");
        assert_eq!(ollama.messages[0].content, "Hello");
        assert!(ollama.think.is_none());
    }

    #[test]
    fn convert_with_system_prompt() {
        let req = MessageRequest {
            model: "qwen3.5".into(),
            max_tokens: 512,
            system: Some(super::super::types::SystemPrompt::Text("Be helpful.".into())),
            messages: vec![super::super::types::RequestMessage {
                role: "user".into(),
                content: Value::String("Hi".into()),
            }],
            tools: vec![],
            stream: true,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            metadata: None,
        };
        let ollama = convert_request(&req, Some(false));
        assert_eq!(ollama.messages[0].role, "system");
        assert_eq!(ollama.messages[0].content, "Be helpful.");
        assert_eq!(ollama.think, Some(false));
    }

    #[test]
    fn convert_tool_result_to_role_tool() {
        let blocks = serde_json::json!([
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": [{"type": "text", "text": "file list here"}]}
        ]);
        let msgs = convert_user_blocks(blocks.as_array().unwrap());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "tool");
        assert_eq!(msgs[0].content, "file list here");
    }

    #[test]
    fn convert_assistant_tool_use() {
        let blocks = serde_json::json!([
            {"type": "tool_use", "id": "tu_1", "name": "Bash",
             "input": {"command": "ls"}}
        ]);
        let msgs = convert_assistant_blocks(blocks.as_array().unwrap());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
        let calls = msgs[0].tool_calls.as_ref().expect("tool_calls");
        assert_eq!(calls[0].function.name, "Bash");
        assert_eq!(calls[0].function.arguments["command"], "ls");
    }

    #[test]
    fn think_false_sets_field() {
        let req = MessageRequest {
            model: "qwen3.5".into(),
            max_tokens: 512,
            system: None,
            messages: vec![super::super::types::RequestMessage {
                role: "user".into(),
                content: Value::String("hi".into()),
            }],
            tools: vec![],
            stream: true,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            metadata: None,
        };
        let ollama = convert_request(&req, Some(false));
        assert_eq!(ollama.think, Some(false));
        // Verify it serialises correctly
        let body = serde_json::to_string(&ollama).unwrap();
        assert!(body.contains("\"think\":false"));
    }
}
