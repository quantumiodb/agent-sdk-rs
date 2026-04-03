//! OpenAI-compatible API provider.
//!
//! Translates between the SDK's internal Anthropic-format types and the
//! OpenAI Chat Completions API wire format, enabling the agent to work with:
//!
//! - OpenAI (gpt-4o, gpt-4-turbo, o1, …)
//! - Azure OpenAI Service
//! - Groq, Together AI, Fireworks, Perplexity, …
//! - Any endpoint that speaks the OpenAI Chat Completions protocol
//!
//! # Request translation (Anthropic → OpenAI)
//!
//! | Anthropic field   | OpenAI field               |
//! |-------------------|-----------------------------|
//! | `model`           | `model`                     |
//! | `max_tokens`      | `max_tokens`                |
//! | `system`          | messages[0] with role=system|
//! | `messages`        | `messages`                  |
//! | `tools`           | `tools` (function format)   |
//! | `temperature`     | `temperature`               |
//! | `stream`          | `stream`                    |
//!
//! # Response translation (OpenAI SSE → internal ApiStreamEvent)
//!
//! OpenAI streaming chunks are translated to the equivalent Anthropic-style
//! `ApiStreamEvent` variants so the agent loop can remain provider-agnostic.

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
use super::types::{ApiMessageStart, ApiStreamEvent, DeltaUsage, MessageDeltaBody, MessageRequest};
use crate::types::{ContentBlock, ContentDelta, StopReason};

// ──────────────────────────────────────────────────────────────────────────────
// OpenAI request types
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct OaiRequest {
    model: String,
    messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OaiTool>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// Include usage in the final stream chunk (OpenAI ≥ 2024-10 preview).
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OaiStreamOptions>,
}

#[derive(Debug, Clone, Serialize)]
struct OaiStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OaiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OaiContent>,
    /// Tool calls produced by an assistant turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiToolCall>>,
    /// Tool result (role = "tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OaiContentPart {
    #[serde(rename = "type")]
    part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<OaiImageUrl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OaiImageUrl {
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OaiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OaiFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OaiFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OaiTool {
    #[serde(rename = "type")]
    tool_type: String, // "function"
    function: OaiFunctionDef,
}

#[derive(Debug, Clone, Serialize)]
struct OaiFunctionDef {
    name: String,
    description: String,
    parameters: Value,
}

// ──────────────────────────────────────────────────────────────────────────────
// OpenAI streaming response types
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OaiChunk {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<OaiChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
struct OaiChoice {
    #[allow(dead_code)]
    index: usize,
    delta: OaiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OaiDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    /// Thinking/reasoning output — field name varies by provider:
    /// - Ollama (Qwen3, DeepSeek-R1): `reasoning`
    /// - NVIDIA NIM (GLM4.7) and some OpenAI-compat endpoints: `reasoning_content`
    /// Both are mapped to ThinkingDelta; whichever is non-null wins.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OaiToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    #[serde(default)]
    #[allow(dead_code)]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<OaiFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OaiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OaiUsage {
    #[allow(dead_code)]
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    #[allow(dead_code)]
    total_tokens: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Return the index from which `s` ends with a non-empty proper prefix of `tag`.
///
/// Used to hold back the tail of a text chunk that might be the start of a
/// `<tool_call>` or `</tool_call>` tag that will complete in the next chunk.
fn find_partial_prefix(s: &str, tag: &str) -> usize {
    let bytes = s.as_bytes();
    let tag_bytes = tag.as_bytes();
    // Walk backwards: find the longest suffix of `s` that is a proper prefix of `tag`.
    for start in (0..s.len()).rev() {
        let suffix = &bytes[start..];
        if suffix.len() < tag_bytes.len() && tag_bytes.starts_with(suffix) {
            return start;
        }
    }
    s.len()
}

// ──────────────────────────────────────────────────────────────────────────────
// Request conversion: Anthropic MessageRequest → OpenAI OaiRequest
// ──────────────────────────────────────────────────────────────────────────────

fn convert_request(req: &MessageRequest) -> OaiRequest {
    let mut messages: Vec<OaiMessage> = Vec::new();

    // System prompt as first message
    if let Some(system) = &req.system {
        let text = match system {
            super::types::SystemPrompt::Text(t) => t.clone(),
            super::types::SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        messages.push(OaiMessage {
            role: "system".into(),
            content: Some(OaiContent::Text(text)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Conversation messages
    for msg in &req.messages {
        let role = msg.role.clone();
        let oai_msg = convert_message(&role, &msg.content);
        messages.extend(oai_msg);
    }

    // Tool definitions
    let tools: Vec<OaiTool> = req
        .tools
        .iter()
        .map(|t| OaiTool {
            tool_type: "function".into(),
            function: OaiFunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    OaiRequest {
        model: req.model.clone(),
        messages,
        max_tokens: Some(req.max_tokens),
        tools,
        stream: true,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences.clone(),
        stream_options: Some(OaiStreamOptions { include_usage: true }),
    }
}

/// Convert a single Anthropic-format message to one or more OpenAI messages.
///
/// The main complexity is `tool_use` + `tool_result` block handling:
/// - An assistant message with tool_use blocks → assistant message with tool_calls
/// - A user message with tool_result blocks → one "tool" message per result
fn convert_message(role: &str, content: &Value) -> Vec<OaiMessage> {
    match content {
        // Simple string content
        Value::String(text) => vec![OaiMessage {
            role: role.to_string(),
            content: Some(OaiContent::Text(text.clone())),
            tool_calls: None,
            tool_call_id: None,
        }],

        // Array of content blocks
        Value::Array(blocks) => {
            if role == "assistant" {
                convert_assistant_blocks(blocks)
            } else {
                convert_user_blocks(blocks)
            }
        }

        other => vec![OaiMessage {
            role: role.to_string(),
            content: Some(OaiContent::Text(other.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }],
    }
}

fn convert_assistant_blocks(blocks: &[Value]) -> Vec<OaiMessage> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OaiToolCall> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            Some("thinking") => {
                // Skip thinking blocks — not exposed to OpenAI
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Object(Default::default()));
                let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());

                tool_calls.push(OaiToolCall {
                    id,
                    call_type: "function".into(),
                    function: OaiFunction {
                        name,
                        arguments: Some(arguments),
                    },
                });
            }
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(OaiContent::Text(text_parts.join("")))
    };

    vec![OaiMessage {
        role: "assistant".into(),
        content,
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        tool_call_id: None,
    }]
}

fn convert_user_blocks(blocks: &[Value]) -> Vec<OaiMessage> {
    let mut messages: Vec<OaiMessage> = Vec::new();
    let mut text_parts: Vec<OaiContentPart> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(OaiContentPart {
                        part_type: "text".into(),
                        text: Some(text.to_string()),
                        image_url: None,
                    });
                }
            }
            Some("image") => {
                if let Some(source) = block.get("source") {
                    let media_type = source
                        .get("media_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("image/png");
                    let data = source
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    text_parts.push(OaiContentPart {
                        part_type: "image_url".into(),
                        text: None,
                        image_url: Some(OaiImageUrl {
                            url: format!("data:{media_type};base64,{data}"),
                        }),
                    });
                }
            }
            Some("tool_result") => {
                // Flush any accumulated text parts first
                if !text_parts.is_empty() {
                    let parts = std::mem::take(&mut text_parts);
                    messages.push(OaiMessage {
                        role: "user".into(),
                        content: Some(OaiContent::Parts(parts)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }

                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Extract text from tool_result content
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

                messages.push(OaiMessage {
                    role: "tool".into(),
                    content: Some(OaiContent::Text(result_text)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id),
                });
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        messages.push(OaiMessage {
            role: "user".into(),
            content: Some(OaiContent::Parts(text_parts)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // If nothing was emitted, return an empty user message
    if messages.is_empty() {
        messages.push(OaiMessage {
            role: "user".into(),
            content: Some(OaiContent::Text(String::new())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages
}

// ──────────────────────────────────────────────────────────────────────────────
// OpenAI SSE stream → internal ApiStreamEvent stream
// ──────────────────────────────────────────────────────────────────────────────

/// State machine for translating an OpenAI SSE byte stream into
/// `ApiStreamEvent` items matching the Anthropic streaming protocol.
///
/// OpenAI streams `chat.completion.chunk` objects; we map them onto:
/// - `MessageStart` (first delta with role=assistant)
/// - `ContentBlockStart` + `ContentBlockDelta` (text and tool_call deltas)
/// - `ContentBlockStop` (on finish_reason or tool call completion)
/// - `MessageDelta` (finish_reason → stop_reason)
/// - `MessageStop`
struct OaiSseStream<S> {
    inner: S,
    buf: String,
    /// Whether we've emitted the synthetic MessageStart event.
    started: bool,
    /// Highest content-block index seen so far (to emit ContentBlockStop).
    last_content_idx: Option<usize>,
    /// Content block index for the thinking/reasoning block.
    /// Used for models like Qwen3 / DeepSeek-R1 that expose reasoning via
    /// `reasoning_content` in the OpenAI-compat delta.
    thinking_block_idx: Option<usize>,
    /// Per-tool-call index → content-block index mapping.
    tool_block_map: std::collections::HashMap<usize, usize>,
    next_block_idx: usize,
    /// Buffer of events produced by the last parse step.
    pending: std::collections::VecDeque<Result<ApiStreamEvent, ApiError>>,
    done: bool,
    model: String,
    /// Some models (e.g. GLM4.7 via NVIDIA NIM) echo tool calls as plain text
    /// using `<tool_call>…</tool_call>` tags inside the content delta.
    /// We suppress these tags so they don't appear in the final text output.
    ///
    /// `in_tool_call_tag`: true while consuming text inside a tag block.
    in_tool_call_tag: bool,
    /// Partial-tag lookahead buffer: holds text that ends with a prefix of
    /// `<tool_call>` or `</tool_call>` until we know whether it's really a tag.
    text_hold: String,
}

impl<S> OaiSseStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    fn new(inner: S) -> Self {
        Self {
            inner,
            buf: String::new(),
            started: false,
            last_content_idx: None,
            thinking_block_idx: None,
            tool_block_map: Default::default(),
            next_block_idx: 0,
            pending: Default::default(),
            done: false,
            model: String::new(),
            in_tool_call_tag: false,
            text_hold: String::new(),
        }
    }

    /// Strip `<tool_call>…</tool_call>` blocks that some models (e.g. GLM4.7)
    /// echo verbatim inside the text content delta alongside the proper
    /// `tool_calls` JSON field.
    ///
    /// Handles tags split across SSE chunks via `self.text_hold` and
    /// `self.in_tool_call_tag`.  Returns the cleaned text ready to emit.
    fn clean_text(&mut self, incoming: &str) -> String {
        self.text_hold.push_str(incoming);

        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";

        let mut out = String::new();

        loop {
            if self.in_tool_call_tag {
                // Consume until </tool_call>
                if let Some(end) = self.text_hold.find(CLOSE) {
                    let rest = self.text_hold[end + CLOSE.len()..].to_string();
                    self.text_hold = rest;
                    self.in_tool_call_tag = false;
                    // continue to process remaining text
                } else {
                    // Still inside tag — discard buffer, wait for more chunks
                    self.text_hold.clear();
                    break;
                }
            } else {
                // Look for the next <tool_call>
                if let Some(start) = self.text_hold.find(OPEN) {
                    out.push_str(&self.text_hold[..start]);
                    let rest = self.text_hold[start + OPEN.len()..].to_string();
                    self.text_hold = rest;
                    self.in_tool_call_tag = true;
                    // continue to strip the tag body
                } else {
                    // No open tag — but the tail might be a partial prefix of OPEN.
                    // Hold back any trailing '<' that could begin a tag.
                    let hold_from = find_partial_prefix(&self.text_hold, OPEN);
                    out.push_str(&self.text_hold[..hold_from]);
                    self.text_hold = self.text_hold[hold_from..].to_string();
                    break;
                }
            }
        }

        out
    }

    /// Parse one SSE line and push translated events into `pending`.
    fn parse_line(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            return;
        }

        let data = if let Some(d) = line.strip_prefix("data: ") {
            d.trim()
        } else {
            return;
        };

        if data == "[DONE]" {
            // Close thinking block if open.
            if let Some(idx) = self.thinking_block_idx.take() {
                self.pending
                    .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: idx }));
            }
            // Close text block if open.
            if let Some(idx) = self.last_content_idx.take() {
                self.pending
                    .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: idx }));
            }
            // Close any tool-call blocks that were never closed by finish_reason.
            // Some models (e.g. GLM4.7 via NVIDIA NIM) omit finish_reason entirely
            // even when the response is a tool call.
            if !self.tool_block_map.is_empty() {
                for (_, &idx) in &self.tool_block_map {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: idx }));
                }
                self.tool_block_map.clear();
                // Synthesise the stop_reason the model forgot to send.
                self.pending.push_back(Ok(ApiStreamEvent::MessageDelta {
                    delta: MessageDeltaBody {
                        stop_reason: Some(StopReason::ToolUse),
                        stop_sequence: None,
                    },
                    usage: None,
                }));
            }
            self.pending.push_back(Ok(ApiStreamEvent::MessageStop));
            self.done = true;
            return;
        }

        let chunk: OaiChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(e) => {
                debug!(raw = data, error = %e, "Failed to parse OpenAI chunk");
                return;
            }
        };

        // Capture model name from first chunk
        if self.model.is_empty() {
            if let Some(m) = &chunk.model {
                self.model = m.clone();
            }
        }

        // Handle usage (usually in the last chunk when stream_options.include_usage=true)
        if let Some(usage) = &chunk.usage {
            let delta_usage = DeltaUsage {
                output_tokens: usage.completion_tokens,
            };
            self.pending
                .push_back(Ok(ApiStreamEvent::MessageDelta {
                    delta: MessageDeltaBody {
                        stop_reason: None,
                        stop_sequence: None,
                    },
                    usage: Some(delta_usage),
                }));
        }

        for choice in &chunk.choices {
            // Emit synthetic MessageStart on the first assistant role delta
            if !self.started {
                if choice.delta.role.as_deref() == Some("assistant") || !self.started {
                    let id = chunk.id.clone().unwrap_or_else(|| "oai-msg".to_string());
                    self.pending
                        .push_back(Ok(ApiStreamEvent::MessageStart {
                            message: ApiMessageStart {
                                id,
                                message_type: "message".into(),
                                role: "assistant".into(),
                                model: self.model.clone(),
                                usage: None,
                            },
                        }));
                    self.started = true;
                }
            }

            // Text content delta
            if let Some(text) = &choice.delta.content {
                if !text.is_empty() {
                    // Strip <tool_call>…</tool_call> echo tags (e.g. GLM4.7).
                    let clean = self.clean_text(text);
                    if !clean.is_empty() {
                        // Ensure a text content block is open (index 0)
                        if self.last_content_idx.is_none() {
                            let idx = self.next_block_idx;
                            self.next_block_idx += 1;
                            self.last_content_idx = Some(idx);
                            self.pending
                                .push_back(Ok(ApiStreamEvent::ContentBlockStart {
                                    index: idx,
                                    content_block: ContentBlock::Text { text: String::new() },
                                }));
                        }
                        let idx = self.last_content_idx.unwrap();
                        self.pending
                            .push_back(Ok(ApiStreamEvent::ContentBlockDelta {
                                index: idx,
                                delta: ContentDelta::TextDelta { text: clean },
                            }));
                    }
                }
            }

            // Thinking / reasoning content delta.
            // - Ollama (Qwen3, DeepSeek-R1): `reasoning` field
            // - NVIDIA NIM (GLM4.7) and similar: `reasoning_content` field
            // Prefer `reasoning_content`; fall back to `reasoning`.
            let reasoning_text: Option<String> = choice.delta.reasoning_content
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| choice.delta.reasoning.as_deref().filter(|s| !s.is_empty()))
                .map(|s| s.to_string());
            if let Some(reasoning) = reasoning_text {
                    let idx = match self.thinking_block_idx {
                        Some(i) => i,
                        None => {
                            let i = self.next_block_idx;
                            self.next_block_idx += 1;
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
                            thinking: reasoning,
                        },
                    }));
            }

            // Tool call deltas
            if let Some(tool_deltas) = &choice.delta.tool_calls {
                for td in tool_deltas {
                    // Get or create a content block for this tool call
                    let block_idx = *self
                        .tool_block_map
                        .entry(td.index)
                        .or_insert_with(|| {
                            // Close the text block if one was open
                            if let Some(text_idx) = self.last_content_idx.take() {
                                self.pending.push_back(Ok(ApiStreamEvent::ContentBlockStop {
                                    index: text_idx,
                                }));
                            }
                            let idx = self.next_block_idx;
                            self.next_block_idx += 1;
                            idx
                        });

                    // First delta for this tool call: emit ContentBlockStart with the tool name
                    if let Some(func) = &td.function {
                        if let Some(name) = &func.name {
                            // This is the first chunk for this tool call
                            let id = td.id.clone().unwrap_or_else(|| format!("call_{}", td.index));
                            self.pending.push_back(Ok(ApiStreamEvent::ContentBlockStart {
                                index: block_idx,
                                content_block: ContentBlock::ToolUse {
                                    id,
                                    name: name.clone(),
                                    input: Value::Object(Default::default()),
                                },
                            }));
                        }

                        // Arguments delta
                        if let Some(args) = &func.arguments {
                            if !args.is_empty() {
                                self.pending
                                    .push_back(Ok(ApiStreamEvent::ContentBlockDelta {
                                        index: block_idx,
                                        delta: ContentDelta::InputJsonDelta {
                                            partial_json: args.clone(),
                                        },
                                    }));
                            }
                        }
                    }
                }
            }

            // Finish reason → stop
            if let Some(reason) = &choice.finish_reason {
                // Close any open tool call blocks
                for (_, &idx) in &self.tool_block_map {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: idx }));
                }
                self.tool_block_map.clear();

                // Close text block if still open
                if let Some(idx) = self.last_content_idx.take() {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::ContentBlockStop { index: idx }));
                }

                let stop_reason = match reason.as_str() {
                    "stop" => Some(StopReason::EndTurn),
                    "tool_calls" => Some(StopReason::ToolUse),
                    "length" => Some(StopReason::MaxTokens),
                    "stop_sequence" => Some(StopReason::StopSequence),
                    _ => None,
                };

                if stop_reason.is_some() {
                    self.pending
                        .push_back(Ok(ApiStreamEvent::MessageDelta {
                            delta: MessageDeltaBody {
                                stop_reason,
                                stop_sequence: None,
                            },
                            usage: None,
                        }));
                }
            }
        }
    }
}

impl<S> Stream for OaiSseStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ApiStreamEvent, ApiError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Drain pending events first
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(event));
            }

            if self.done {
                return Poll::Ready(None);
            }

            // Pull next chunk from the HTTP response
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Stream ended without [DONE]; close gracefully
                    if let Some(idx) = self.last_content_idx.take() {
                        return Poll::Ready(Some(Ok(ApiStreamEvent::ContentBlockStop {
                            index: idx,
                        })));
                    }
                    return Poll::Ready(Some(Ok(ApiStreamEvent::MessageStop)));
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ApiError::Network {
                        message: format!("Stream read error: {e}"),
                        source: e,
                    })));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    // Append to line buffer and process complete lines
                    let text = match std::str::from_utf8(&bytes) {
                        Ok(s) => s.to_string(),
                        Err(_) => String::from_utf8_lossy(&bytes).to_string(),
                    };
                    self.buf.push_str(&text);

                    // Process all complete lines
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

// ──────────────────────────────────────────────────────────────────────────────
// OpenAICompatProvider
// ──────────────────────────────────────────────────────────────────────────────

/// API provider for OpenAI-compatible endpoints.
///
/// Works with OpenAI, Azure OpenAI, Groq, Together AI, Fireworks, and any
/// service that implements the OpenAI Chat Completions API.
pub struct OpenAICompatProvider {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    model_names: Vec<String>,
    /// Extra fields merged into every request body (provider-specific params).
    extra_body: Option<serde_json::Value>,
}

impl OpenAICompatProvider {
    /// Create a provider for the official OpenAI API.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com".to_string(),
            http_client: reqwest::Client::new(),
            model_names: vec![
                "gpt-4o".to_string(),
                "gpt-4o-mini".to_string(),
                "gpt-4-turbo".to_string(),
                "o1".to_string(),
                "o1-mini".to_string(),
                "o3".to_string(),
                "o3-mini".to_string(),
            ],
            extra_body: None,
        }
    }

    /// Create a provider for any OpenAI-compatible endpoint.
    ///
    /// Proxy settings are inherited from `http_proxy` / `HTTPS_PROXY` env vars.
    /// Use [`Self::no_proxy`] to bypass the system proxy.
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            http_client: reqwest::Client::new(),
            model_names: vec![],
            extra_body: None,
        }
    }

    /// Create a provider that bypasses the system proxy.
    ///
    /// Useful for local servers (Ollama, vLLM, LM Studio) that are reachable
    /// directly but inaccessible through a corporate/VPN proxy.
    pub fn no_proxy(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let client = reqwest::ClientBuilder::new()
            .no_proxy()
            .build()
            .unwrap_or_default();
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            http_client: client,
            model_names: vec![],
            extra_body: None,
        }
    }

    /// Create a provider that bypasses the system proxy, with extra body parameters.
    pub fn no_proxy_with_options(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        extra_body: Option<serde_json::Value>,
    ) -> Self {
        let client = reqwest::ClientBuilder::new()
            .no_proxy()
            .build()
            .unwrap_or_default();
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            http_client: client,
            model_names: vec![],
            extra_body,
        }
    }

    /// Add known model names (used by `supported_models()`).
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        self.model_names = models;
        self
    }
}

impl std::fmt::Debug for OpenAICompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatProvider")
            .field("base_url", &self.base_url)
            .field("api_key", &"[redacted]")
            .finish()
    }
}

#[async_trait]
impl ApiProvider for OpenAICompatProvider {
    async fn stream_message(&self, request: MessageRequest) -> Result<ApiStream, ApiError> {
        // Normalise base_url: some providers (NVIDIA NIM) already include /v1 in
        // their base URL, so avoid producing /v1/v1/chat/completions.
        let base = self.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{}/chat/completions", base)
        } else {
            format!("{}/v1/chat/completions", base)
        };

        let oai_req = convert_request(&request);

        // Serialize the request; merge extra_body fields if provided.
        let body_bytes = if let Some(extra) = &self.extra_body {
            let mut obj = serde_json::to_value(&oai_req)
                .map_err(|e| ApiError::InvalidResponse {
                    message: format!("Failed to serialize request: {e}"),
                    body: String::new(),
                })?
                .as_object()
                .cloned()
                .unwrap_or_default();
            if let Some(extra_obj) = extra.as_object() {
                for (k, v) in extra_obj {
                    obj.insert(k.clone(), v.clone());
                }
            }
            serde_json::to_vec(&serde_json::Value::Object(obj)).map_err(|e| {
                ApiError::InvalidResponse {
                    message: format!("Failed to serialize merged request: {e}"),
                    body: String::new(),
                }
            })?
        } else {
            serde_json::to_vec(&oai_req).map_err(|e| ApiError::InvalidResponse {
                message: format!("Failed to serialize request: {e}"),
                body: String::new(),
            })?
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|e| ApiError::Configuration(format!("Invalid API key: {e}")))?,
        );

        let response: reqwest::Response = self
            .http_client
            .post(&url)
            .headers(headers)
            .body(body_bytes)
            .send()
            .await
            .map_err(|e| ApiError::Network {
                message: format!("Failed to send request to {url}: {e}"),
                source: e,
            })?;

        let status = response.status().as_u16();

        if status == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v: &reqwest::header::HeaderValue| v.to_str().ok())
                .and_then(|v: &str| v.parse::<u64>().ok())
                .map(|s| s * 1000);
            return Err(ApiError::RateLimited {
                retry_after_ms: retry_after,
            });
        }
        if status == 401 || status == 403 {
            return Err(ApiError::AuthenticationFailed(format!(
                "Authentication failed ({status})"
            )));
        }
        if status >= 500 {
            return Err(ApiError::Overloaded);
        }
        if status >= 400 {
            let body: String = response.text().await.unwrap_or_default();
            return Err(ApiError::ApiError {
                status,
                message: body,
                error_type: "openai_error".to_string(),
            });
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = OaiSseStream::new(byte_stream);
        Ok(Box::pin(sse_stream))
    }

    fn name(&self) -> &str {
        "openai-compat"
    }

    fn supported_models(&self) -> &[&str] {
        // We can't easily return references to owned Strings, so return static slice
        &[]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(role: &str, content: Value) -> Vec<OaiMessage> {
        convert_message(role, &content)
    }

    #[test]
    fn simple_text_user_message() {
        let msgs = make_request("user", Value::String("Hello!".into()));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        matches!(&msgs[0].content, Some(OaiContent::Text(t)) if t == "Hello!");
    }

    #[test]
    fn assistant_tool_use_blocks() {
        let blocks = serde_json::json!([
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "ls"}}
        ]);
        let msgs = convert_assistant_blocks(blocks.as_array().unwrap());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
        let calls = msgs[0].tool_calls.as_ref().expect("tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "Bash");
    }

    #[test]
    fn user_tool_result_blocks() {
        let blocks = serde_json::json!([
            {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "result"}]}
        ]);
        let msgs = convert_user_blocks(blocks.as_array().unwrap());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "tool");
        assert_eq!(msgs[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn system_prompt_becomes_first_message() {
        use super::super::types::{RequestMessage, SystemPrompt};
        let req = MessageRequest {
            model: "gpt-4o".into(),
            max_tokens: 1024,
            system: Some(SystemPrompt::Text("You are helpful.".into())),
            messages: vec![RequestMessage {
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

        let oai = convert_request(&req);
        assert_eq!(oai.messages[0].role, "system");
        assert!(matches!(&oai.messages[0].content, Some(OaiContent::Text(t)) if t == "You are helpful."));
        assert_eq!(oai.messages[1].role, "user");
    }

    #[test]
    fn tools_converted_to_function_format() {
        use super::super::types::{RequestMessage, ToolDefinition as ApiToolDef};
        let req = MessageRequest {
            model: "gpt-4o".into(),
            max_tokens: 1024,
            system: None,
            messages: vec![RequestMessage {
                role: "user".into(),
                content: Value::String("Hi".into()),
            }],
            tools: vec![ApiToolDef {
                name: "Bash".into(),
                description: "Run bash commands".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
                cache_control: None,
            }],
            stream: true,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            metadata: None,
        };

        let oai = convert_request(&req);
        assert_eq!(oai.tools.len(), 1);
        assert_eq!(oai.tools[0].tool_type, "function");
        assert_eq!(oai.tools[0].function.name, "Bash");
    }

    #[test]
    fn openai_provider_name() {
        let p = OpenAICompatProvider::openai("sk-test");
        assert_eq!(p.name(), "openai-compat");
    }

    #[test]
    fn custom_provider_url() {
        let p = OpenAICompatProvider::new("key", "https://api.groq.com/openai");
        assert_eq!(p.base_url, "https://api.groq.com/openai");
    }
}
