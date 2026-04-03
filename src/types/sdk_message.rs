//! SDK message types for the streaming event protocol.
//!
//! These types define the messages exchanged between the agent and its
//! consumer (the SDK host). They are aligned with the TypeScript SDK's
//! `SDKMessageSchema` union from `coreSchemas.ts`.
//!
//! Messages flow from the agent to the consumer via an `mpsc` channel,
//! providing real-time visibility into the agent's activity.

use serde::{Deserialize, Serialize};

use super::message::{Message, StopReason, ToolResultContent, Usage};
use super::mcp::McpServerStatus;
use super::tool::ToolDefinition;

// ---------------------------------------------------------------------------
// SDKMessage
// ---------------------------------------------------------------------------

/// SDK streaming event -- pushed from Agent to consumer via `mpsc` channel.
///
/// Aligned with the TypeScript SDK's `StdoutMessage` / `SDKMessage` union.
/// Each variant carries a `type` tag for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SDKMessage {
    /// Session initialization -- reports available tools, model, MCP servers.
    #[serde(rename = "system")]
    System {
        /// Unique session identifier.
        session_id: String,
        /// Tool definitions available in this session.
        tools: Vec<ToolDefinition>,
        /// The model being used.
        model: String,
        /// Status of connected MCP servers.
        mcp_servers: Vec<McpServerStatus>,
        /// Permission mode in effect.
        #[serde(skip_serializing_if = "Option::is_none")]
        permission_mode: Option<String>,
        /// SDK version string.
        #[serde(skip_serializing_if = "Option::is_none")]
        claude_code_version: Option<String>,
        /// Current working directory.
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },

    /// An assistant message (complete, with content blocks and usage).
    #[serde(rename = "assistant")]
    Assistant {
        /// The full assistant message.
        message: Message,
        /// Why the model stopped generating.
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<StopReason>,
        /// Session identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
    },

    /// Progress update during tool execution.
    #[serde(rename = "tool_progress")]
    ToolProgress {
        /// The tool_use ID this progress relates to.
        tool_use_id: String,
        /// Name of the tool being executed.
        tool_name: String,
        /// Parent tool_use ID (for nested tool calls).
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
        /// Tool-specific progress data.
        #[serde(skip_serializing_if = "Option::is_none")]
        progress_data: Option<serde_json::Value>,
        /// Milliseconds elapsed since tool execution started.
        elapsed_ms: u64,
    },

    /// Tool execution result.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// The tool_use ID this result corresponds to.
        tool_use_id: String,
        /// Name of the tool that was executed.
        tool_name: String,
        /// Result content blocks.
        content: Vec<ToolResultContent>,
        /// Whether this is an error result.
        is_error: bool,
    },

    /// Incremental content delta (real-time streaming).
    #[serde(rename = "content_delta")]
    ContentDelta {
        /// Index of the content block being updated.
        index: usize,
        /// The delta payload.
        delta: ContentDelta,
    },

    /// Final result of the agent's query execution.
    #[serde(rename = "result")]
    Result {
        /// Whether the execution ended with an error.
        is_error: bool,
        /// Cumulative token usage across all API calls.
        total_usage: Usage,
        /// Total cost in USD.
        total_cost_usd: f64,
        /// Wall-clock duration in milliseconds.
        duration_ms: u64,
        /// Number of agent loop turns (API round-trips).
        num_turns: u32,
        /// Session identifier.
        session_id: String,
        /// Final text result (concatenated assistant text).
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        /// Why the agent stopped.
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        /// Complete conversation history after this query (user + assistant + tool results).
        /// Used by [`Agent::prompt()`] to sync `self.messages` for multi-turn conversations.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        final_messages: Vec<Message>,
    },

    /// Context compaction event.
    #[serde(rename = "compact")]
    Compact {
        /// Token count before compaction.
        original_tokens: u64,
        /// Token count after compaction.
        compacted_tokens: u64,
    },

    /// Permission request -- the agent needs user approval to proceed.
    ///
    /// The consumer should respond via the control channel.
    #[serde(rename = "permission_request")]
    PermissionRequest {
        /// Unique ID for this permission request (for correlation).
        request_id: String,
        /// Name of the tool requesting permission.
        tool_name: String,
        /// The input the tool wants to execute with.
        tool_input: serde_json::Value,
        /// Human-readable description of what the tool wants to do.
        message: String,
    },

    /// Task/subagent lifecycle notification.
    #[serde(rename = "task_notification")]
    TaskNotification {
        /// Unique identifier for the task/subagent.
        agent_id: String,
        /// Current status.
        status: TaskStatus,
        /// Human-readable summary of the task's progress.
        summary: String,
        /// Final result text (when completed).
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        /// Token usage for this task.
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },

    /// Error event.
    #[serde(rename = "error")]
    Error {
        /// Error category/type.
        error_type: String,
        /// Human-readable error message.
        message: String,
        /// Whether this error is retryable.
        #[serde(skip_serializing_if = "Option::is_none")]
        retryable: Option<bool>,
    },

    /// Keep-alive for long-lived connections.
    #[serde(rename = "keep_alive")]
    KeepAlive,
}

// ---------------------------------------------------------------------------
// ContentDelta
// ---------------------------------------------------------------------------

/// An incremental update to a content block during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentDelta {
    /// Incremental text output.
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    /// Incremental thinking/reasoning output.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },

    /// Incremental JSON for tool input streaming.
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

// ---------------------------------------------------------------------------
// TaskStatus
// ---------------------------------------------------------------------------

/// Lifecycle status for a task or subagent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Started,
    Completed,
    Failed,
    Killed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Started => write!(f, "started"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
            TaskStatus::Killed => write!(f, "killed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_message_system_serialization() {
        let msg = SDKMessage::System {
            session_id: "sess_123".to_string(),
            tools: vec![],
            model: "claude-sonnet-4-6".to_string(),
            mcp_servers: vec![],
            permission_mode: Some("default".to_string()),
            claude_code_version: Some("0.1.0".to_string()),
            cwd: Some("/home/user".to_string()),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["type"], "system");
        assert_eq!(json["session_id"], "sess_123");
        assert_eq!(json["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn sdk_message_keep_alive_serialization() {
        let msg = SDKMessage::KeepAlive;
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["type"], "keep_alive");
    }

    #[test]
    fn content_delta_text_roundtrip() {
        let delta = ContentDelta::TextDelta {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: ContentDelta = serde_json::from_str(&json).unwrap();
        match parsed {
            ContentDelta::TextDelta { text } => assert_eq!(text, "hello"),
            _ => panic!("expected TextDelta"),
        }
    }

    #[test]
    fn content_delta_thinking_roundtrip() {
        let delta = ContentDelta::ThinkingDelta {
            thinking: "let me reason...".to_string(),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert_eq!(json["type"], "thinking_delta");
        assert_eq!(json["thinking"], "let me reason...");
    }

    #[test]
    fn task_status_serialization() {
        let json = serde_json::to_string(&TaskStatus::Completed).unwrap();
        assert_eq!(json, "\"completed\"");

        let parsed: TaskStatus = serde_json::from_str("\"failed\"").unwrap();
        assert_eq!(parsed, TaskStatus::Failed);
    }

    #[test]
    fn task_status_display() {
        assert_eq!(format!("{}", TaskStatus::Started), "started");
        assert_eq!(format!("{}", TaskStatus::Killed), "killed");
    }

    #[test]
    fn sdk_message_error_serialization() {
        let msg = SDKMessage::Error {
            error_type: "rate_limit".to_string(),
            message: "Too many requests".to_string(),
            retryable: Some(true),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error_type"], "rate_limit");
        assert_eq!(json["retryable"], true);
    }

    #[test]
    fn sdk_message_result_serialization() {
        let msg = SDKMessage::Result {
            is_error: false,
            total_usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            total_cost_usd: 0.0015,
            duration_ms: 2500,
            num_turns: 3,
            session_id: "sess_abc".to_string(),
            result: Some("Done!".to_string()),
            stop_reason: Some("end_turn".to_string()),
            final_messages: vec![],
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["type"], "result");
        assert_eq!(json["num_turns"], 3);
        assert_eq!(json["total_usage"]["input_tokens"], 100);
    }
}
