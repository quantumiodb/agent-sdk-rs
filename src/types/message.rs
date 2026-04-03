//! Message types aligned with the Anthropic Messages API.
//!
//! These types correspond to the TypeScript SDK's message representations
//! and the Anthropic API content block formats.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// MessageRole
// ---------------------------------------------------------------------------

/// The role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// Internal message representation -- richer than the raw API format.
///
/// Carries an ID, role, content blocks, timestamps, and optional metadata
/// such as stop reason, token usage, and model identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message identifier.
    pub id: Uuid,

    /// Whether this is a user or assistant message.
    pub role: MessageRole,

    /// Ordered content blocks (text, tool_use, tool_result, thinking, image).
    pub content: Vec<ContentBlock>,

    /// When the message was created.
    pub timestamp: DateTime<Utc>,

    /// Why the model stopped generating (assistant messages only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    /// Token usage for this message (assistant messages only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// The model that produced this message (assistant messages only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// If this message is a tool result, the ID of the parent tool_use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tool_use_id: Option<String>,
}

impl Message {
    // -- Convenience constructors -------------------------------------------

    /// Create a user message containing a single text block.
    pub fn user_text(text: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            timestamp: Utc::now(),
            stop_reason: None,
            usage: None,
            model: None,
            parent_tool_use_id: None,
        }
    }

    /// Create an assistant message from content blocks.
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: MessageRole::Assistant,
            content,
            timestamp: Utc::now(),
            stop_reason: None,
            usage: None,
            model: None,
            parent_tool_use_id: None,
        }
    }

    /// Create an assistant message with a single text block.
    pub fn assistant_text(text: &str) -> Self {
        Self::assistant(vec![ContentBlock::Text {
            text: text.to_string(),
        }])
    }

    /// Returns all text content blocks concatenated.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns all tool-use blocks in this message.
    pub fn tool_use_blocks(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ContentBlock
// ---------------------------------------------------------------------------

/// A content block within a message -- 1:1 aligned with the Anthropic API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },

    /// A request from the model to invoke a tool.
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Unique ID for this tool invocation.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input (arbitrary JSON).
        input: serde_json::Value,
    },

    /// The result of a tool invocation, sent back as part of a user message.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// ID of the tool_use this result corresponds to.
        tool_use_id: String,
        /// Result content blocks (text and/or images).
        content: Vec<ToolResultContent>,
        /// Whether the tool call resulted in an error.
        #[serde(default)]
        is_error: bool,
    },

    /// Extended thinking content (model's chain-of-thought).
    #[serde(rename = "thinking")]
    Thinking {
        /// The thinking text.
        thinking: String,
        /// Optional cryptographic signature for verification.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    /// An image content block.
    #[serde(rename = "image")]
    Image { source: ImageSource },
}

// ---------------------------------------------------------------------------
// ToolResultContent
// ---------------------------------------------------------------------------

/// Content within a tool_result block -- either text or an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResultContent {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "image")]
    Image { source: ImageSource },
}

// ---------------------------------------------------------------------------
// ImageSource
// ---------------------------------------------------------------------------

/// Describes the source of an image (currently only base64-encoded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    /// Source type, e.g. `"base64"`.
    #[serde(rename = "type")]
    pub source_type: String,

    /// MIME type, e.g. `"image/png"`, `"image/jpeg"`.
    pub media_type: String,

    /// Base64-encoded image data.
    pub data: String,
}

// ---------------------------------------------------------------------------
// StopReason
// ---------------------------------------------------------------------------

/// Why the model stopped generating content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model finished its turn naturally.
    EndTurn,
    /// The model wants to invoke one or more tools.
    ToolUse,
    /// Hit the maximum output token limit.
    MaxTokens,
    /// Hit a custom stop sequence.
    StopSequence,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::EndTurn => write!(f, "end_turn"),
            StopReason::ToolUse => write!(f, "tool_use"),
            StopReason::MaxTokens => write!(f, "max_tokens"),
            StopReason::StopSequence => write!(f, "stop_sequence"),
        }
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage statistics for an API call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens consumed.
    pub input_tokens: u64,

    /// Number of output tokens generated.
    pub output_tokens: u64,

    /// Tokens written to the prompt cache.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,

    /// Tokens read from the prompt cache.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl Usage {
    /// Total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Accumulate another `Usage` into this one (mutable add).
    pub fn accumulate(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
    }

    /// Returns true when all counters are zero.
    pub fn is_empty(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_creation_input_tokens == 0
            && self.cache_read_input_tokens == 0
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;

    fn add(self, rhs: Self) -> Self::Output {
        Usage {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens
                + rhs.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens + rhs.cache_read_input_tokens,
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.accumulate(&rhs);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_text_creates_correct_message() {
        let msg = Message::user_text("hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.text_content(), "hello");
        assert!(msg.stop_reason.is_none());
    }

    #[test]
    fn assistant_text_creates_correct_message() {
        let msg = Message::assistant_text("hi there");
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.text_content(), "hi there");
    }

    #[test]
    fn usage_accumulate_works() {
        let mut u1 = Usage {
            input_tokens: 10,
            output_tokens: 20,
            cache_creation_input_tokens: 5,
            cache_read_input_tokens: 3,
        };
        let u2 = Usage {
            input_tokens: 1,
            output_tokens: 2,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 1,
        };
        u1.accumulate(&u2);
        assert_eq!(u1.total_tokens(), 33);
        assert_eq!(u1.cache_read_input_tokens, 4);
    }

    #[test]
    fn usage_add_trait() {
        let a = Usage {
            input_tokens: 5,
            output_tokens: 10,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 3,
            output_tokens: 7,
            ..Default::default()
        };
        let c = a + b;
        assert_eq!(c.total_tokens(), 25);
    }

    #[test]
    fn stop_reason_serialization() {
        let json = serde_json::to_string(&StopReason::EndTurn).unwrap();
        assert_eq!(json, "\"end_turn\"");
        let parsed: StopReason = serde_json::from_str("\"tool_use\"").unwrap();
        assert_eq!(parsed, StopReason::ToolUse);
    }

    #[test]
    fn content_block_roundtrip() {
        let block = ContentBlock::ToolUse {
            id: "tu_123".to_string(),
            name: "Bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        let parsed: ContentBlock = serde_json::from_str(&json).unwrap();
        match parsed {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_123");
                assert_eq!(name, "Bash");
                assert_eq!(input["command"], "ls");
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn message_role_display() {
        assert_eq!(format!("{}", MessageRole::User), "user");
        assert_eq!(format!("{}", MessageRole::Assistant), "assistant");
    }
}
