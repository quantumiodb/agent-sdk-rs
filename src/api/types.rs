//! API request/response types for the Anthropic Messages API.
//!
//! These types model the wire format of the Anthropic streaming API and are
//! separate from the higher-level SDK types in [`crate::types`].

use serde::{Deserialize, Serialize};

use crate::types::{ContentBlock, ContentDelta, StopReason, Usage};

// ═══════════════════════════════════════════════════════════════════════════
// Request types
// ═══════════════════════════════════════════════════════════════════════════

/// A request to the Anthropic Messages API.
#[derive(Debug, Clone, Serialize)]
pub struct MessageRequest {
    /// Model identifier (e.g. "claude-sonnet-4-6").
    pub model: String,

    /// Maximum number of output tokens.
    pub max_tokens: u32,

    /// Conversation messages.
    pub messages: Vec<RequestMessage>,

    /// System prompt (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,

    /// Whether to stream the response. Always true for our usage.
    pub stream: bool,

    /// Temperature (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Stop sequences (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Extended thinking configuration (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Metadata (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,
}

/// A single message in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// System prompt — can be a plain string or an array of content blocks with
/// optional cache control.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple text system prompt.
    Text(String),
    /// Array of system prompt blocks (supports cache_control).
    Blocks(Vec<SystemPromptBlock>),
}

/// A block inside a structured system prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Cache control directive for prompt caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
}

/// Tool definition sent with the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Extended thinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// Optional request metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Response / Stream types
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level message response (non-streaming).
#[derive(Debug, Clone, Deserialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// The partial message included in a `message_start` SSE event.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiMessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// A streaming SSE event from the Anthropic Messages API.
///
/// Each variant corresponds to an SSE event type:
///   - `message_start`
///   - `content_block_start`
///   - `content_block_delta`
///   - `content_block_stop`
///   - `message_delta`
///   - `message_stop`
///   - `error`
///   - `ping`
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ApiStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: ApiMessageStart },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: Option<DeltaUsage>,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "error")]
    Error { error: ApiErrorBody },

    #[serde(rename = "ping")]
    Ping,
}

/// Body of a `message_delta` event.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageDeltaBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Usage counters included in a `message_delta` event.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeltaUsage {
    pub output_tokens: u64,
}

/// Error body returned in an `error` SSE event or HTTP error response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for exponential backoff retry logic.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Initial delay before the first retry, in milliseconds.
    pub initial_delay_ms: u64,
    /// Maximum delay between retries, in milliseconds.
    pub max_delay_ms: u64,
    /// Multiplier applied to the delay after each retry.
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 500,
            max_delay_ms: 30_000,
            backoff_multiplier: 2.0,
        }
    }
}
