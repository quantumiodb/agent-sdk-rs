//! Tool trait and related types for the Rust Agent SDK.
//!
//! Defines the core [`Tool`] trait that all tools (built-in, custom, MCP)
//! must implement, along with supporting types for tool results, errors,
//! input schemas, execution context, and permission decisions.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use super::message::{ImageSource, Message, ToolResultContent};

// ---------------------------------------------------------------------------
// ToolInputSchema
// ---------------------------------------------------------------------------

/// JSON Schema representation for a tool's input parameters.
///
/// Corresponds to the `ToolInputJSONSchema` type in the TypeScript SDK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// Always `"object"` for tool input schemas.
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Property definitions keyed by parameter name.
    #[serde(default)]
    pub properties: serde_json::Map<String, Value>,

    /// Names of required properties.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,

    /// Whether additional properties beyond those listed are allowed.
    #[serde(default, skip_serializing_if = "is_false")]
    pub additional_properties: bool,
}

fn is_false(v: &bool) -> bool {
    !v
}

impl Default for ToolInputSchema {
    fn default() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: serde_json::Map::new(),
            required: Vec::new(),
            additional_properties: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolDefinition (wire format for API tool parameter)
// ---------------------------------------------------------------------------

/// Serializable tool definition sent to the Anthropic API.
///
/// This is the wire format included in the `tools` array of a Messages API
/// request. Corresponds to the TypeScript `ApiToolParam`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name as seen by the model.
    pub name: String,

    /// Human-readable description of what the tool does.
    pub description: String,

    /// JSON Schema for the tool's input parameters.
    pub input_schema: ToolInputSchema,
}

/// Alias for `ToolDefinition` matching the TypeScript naming convention.
pub type ApiToolParam = ToolDefinition;

// ---------------------------------------------------------------------------
// ToolResult
// ---------------------------------------------------------------------------

/// The result of executing a tool.
///
/// Contains content to return to the model plus optional side-effect messages
/// to inject into the conversation.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Content to return as the tool_result to the model.
    pub content: Vec<ToolResultContent>,

    /// Whether this result represents an error.
    pub is_error: bool,

    /// Extra messages to append to the conversation (side effects).
    pub extra_messages: Vec<Message>,
}

impl ToolResult {
    /// Create a successful text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text {
                text: text.into(),
            }],
            is_error: false,
            extra_messages: Vec::new(),
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text {
                text: message.into(),
            }],
            is_error: true,
            extra_messages: Vec::new(),
        }
    }

    /// Create an image result.
    pub fn image(media_type: impl Into<String>, base64_data: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: media_type.into(),
                    data: base64_data.into(),
                },
            }],
            is_error: false,
            extra_messages: Vec::new(),
        }
    }

    /// Attach extra messages to this result (builder pattern).
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.extra_messages = messages;
        self
    }
}

// ---------------------------------------------------------------------------
// ToolError
// ---------------------------------------------------------------------------

/// Errors that can occur during tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// A general execution error.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Invalid input was provided to the tool.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The operation was denied by the permission system.
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// An I/O error occurred.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The operation timed out.
    #[error("Timeout after {0}ms")]
    Timeout(u64),

    /// The operation was cancelled via the cancellation token.
    #[error("Aborted")]
    Aborted,

    /// Any other error (wraps `anyhow::Error`).
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// ToolUseContext
// ---------------------------------------------------------------------------

/// Runtime context provided to tools during execution.
///
/// Contains the working directory, cancellation support, shared file-read
/// state (for enforcing read-before-edit), and session identifiers.
#[derive(Debug, Clone)]
pub struct ToolUseContext {
    /// Current working directory for resolving relative paths.
    pub working_dir: PathBuf,

    /// Cancellation token -- tools should check this periodically and abort
    /// if cancelled.
    pub cancellation: CancellationToken,

    /// Tracks which files have been read in this session.
    /// Used by edit tools to enforce the read-before-edit requirement.
    pub read_file_state: Arc<RwLock<HashSet<String>>>,

    /// The session this tool execution belongs to.
    pub session_id: String,

    /// Set for subagent invocations; `None` on the main thread.
    pub agent_id: Option<String>,
}

impl ToolUseContext {
    /// Resolve a (possibly relative) path against the working directory.
    pub fn resolve_path(&self, path: &str) -> Result<PathBuf, ToolError> {
        let p = Path::new(path);
        if p.is_absolute() {
            Ok(p.to_path_buf())
        } else {
            Ok(self.working_dir.join(p))
        }
    }

    /// Record that a file has been read (for read-before-edit checks).
    pub async fn mark_file_read(&self, path: &str) {
        let mut state = self.read_file_state.write().await;
        state.insert(path.to_string());
    }

    /// Check whether a file has been read in this session.
    pub async fn was_file_read(&self, path: &str) -> bool {
        let state = self.read_file_state.read().await;
        state.contains(path)
    }

    /// Create a clone suitable for concurrent tool execution.
    ///
    /// The `read_file_state` and `cancellation` are shared (Arc/Clone),
    /// so concurrent tools see each other's reads and can be cancelled
    /// together.
    pub fn clone_for_concurrent(&self) -> Self {
        Self {
            working_dir: self.working_dir.clone(),
            cancellation: self.cancellation.clone(),
            read_file_state: self.read_file_state.clone(),
            session_id: self.session_id.clone(),
            agent_id: self.agent_id.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolProgressSender
// ---------------------------------------------------------------------------

/// Trait for sending progress updates during tool execution.
///
/// Implementors receive opaque JSON progress data that is forwarded to
/// the SDK message stream as `SDKMessage::ToolProgress`.
#[async_trait]
pub trait ToolProgressSender: Send + Sync {
    /// Send a progress update. The `data` value is tool-specific.
    async fn send(&self, data: Value);
}

// ---------------------------------------------------------------------------
// PermissionCheckResult
// ---------------------------------------------------------------------------

/// Result of a tool-specific permission check.
///
/// Returned by [`Tool::check_permissions`] to indicate whether the tool
/// may proceed, needs user confirmation, or is denied.
#[derive(Debug, Clone)]
pub enum PermissionCheckResult {
    /// The tool is allowed to execute (possibly with modified input).
    Allow {
        /// The (possibly modified) input to use for execution.
        updated_input: Value,
    },

    /// The tool needs explicit user approval before executing.
    AskUser {
        /// A human-readable explanation of why approval is needed.
        message: String,
    },

    /// The tool is denied execution.
    Deny {
        /// A human-readable reason for the denial.
        message: String,
    },
}

// ---------------------------------------------------------------------------
// InterruptBehavior
// ---------------------------------------------------------------------------

/// What should happen when the user interrupts while a tool is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterruptBehavior {
    /// Stop the tool immediately and discard its result.
    Cancel,
    /// Keep the tool running; the new user message waits until it finishes.
    Block,
}

// ---------------------------------------------------------------------------
// PermissionDecision
// ---------------------------------------------------------------------------

/// Final permission decision made by the permission system or the
/// `CanUseToolFn` callback.
#[derive(Debug, Clone)]
pub enum PermissionDecision {
    /// Allow the tool call as-is.
    Allow,

    /// Deny the tool call with a reason.
    Deny(String),

    /// Allow but use modified input instead of the original.
    AllowWithModifiedInput(Value),
}

// ---------------------------------------------------------------------------
// CanUseToolFn
// ---------------------------------------------------------------------------

/// Permission callback type used by the tool executor.
///
/// Receives the tool name and its input, and returns a [`PermissionDecision`].
/// Typically wraps the full permission-checking pipeline (rules, mode, hooks).
pub type CanUseToolFn =
    Arc<dyn Fn(&str, &Value) -> PermissionDecision + Send + Sync>;

// ---------------------------------------------------------------------------
// Tool Trait
// ---------------------------------------------------------------------------

/// Core tool trait -- all tools (built-in, custom, MCP) implement this.
///
/// # Design principles
///
/// - **Minimal required surface**: only `name`, `description`, `input_schema`,
///   and `call` are required.
/// - **Sensible defaults**: all other methods have safe default implementations.
/// - **API-aligned**: mirrors the TypeScript `Tool` interface's core subset.
///
/// Tools are always used behind `Arc<dyn Tool>` so they can be shared across
/// concurrent tasks.
#[async_trait]
pub trait Tool: Send + Sync + Debug {
    // === Required methods ===

    /// The tool's canonical name (e.g. `"Bash"`, `"Edit"`, `"mcp__server__tool"`).
    fn name(&self) -> &str;

    /// A human-readable description of the tool's capabilities.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    fn input_schema(&self) -> ToolInputSchema;

    /// Execute the tool with the given input.
    ///
    /// Implementations should:
    /// - Check `ctx.cancellation` periodically and return `ToolError::Aborted`
    ///   if cancelled.
    /// - Send progress updates via `progress` if the operation is long-running.
    /// - Return `ToolResult::error(...)` for recoverable failures that the
    ///   model should see, and `Err(ToolError::...)` for unrecoverable ones.
    async fn call(
        &self,
        input: Value,
        ctx: &mut ToolUseContext,
        progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError>;

    // === Optional methods with defaults ===

    /// Alternative names for this tool (for backwards compatibility when renamed).
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// Short keyword phrase for `ToolSearch` matching (3--10 words).
    fn search_hint(&self) -> Option<&str> {
        None
    }

    /// Whether this tool is currently enabled.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Whether this tool only reads state (never writes).
    fn is_read_only(&self, _input: &Value) -> bool {
        false
    }

    /// Whether this tool can safely run concurrently with other tools.
    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        false
    }

    /// Whether this tool performs irreversible/destructive operations.
    fn is_destructive(&self, _input: &Value) -> bool {
        false
    }

    /// Tool-specific permission check (runs after global permission rules).
    ///
    /// The default implementation allows execution with the unmodified input.
    async fn check_permissions(
        &self,
        input: &Value,
        _ctx: &ToolUseContext,
    ) -> PermissionCheckResult {
        PermissionCheckResult::Allow {
            updated_input: input.clone(),
        }
    }

    /// Validate the input before permission checks.
    ///
    /// Return `Ok(())` if valid, or `Err(message)` describing the problem.
    async fn validate_input(
        &self,
        _input: &Value,
        _ctx: &ToolUseContext,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Human-friendly name for display in UIs.
    fn user_facing_name(&self, _input: &Value) -> String {
        self.name().to_string()
    }

    /// If this tool operates on a file, return its path.
    fn get_path(&self, _input: &Value) -> Option<String> {
        None
    }

    /// Maximum result size in characters before truncation/persistence.
    fn max_result_size_chars(&self) -> usize {
        100_000
    }

    /// What happens when the user interrupts while this tool is running.
    fn interrupt_behavior(&self) -> InterruptBehavior {
        InterruptBehavior::Block
    }

    /// Whether this is an MCP-proxied tool.
    fn is_mcp(&self) -> bool {
        false
    }

    /// Whether this tool should be deferred (requires `ToolSearch` discovery).
    fn should_defer(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_text() {
        let r = ToolResult::text("hello");
        assert!(!r.is_error);
        assert_eq!(r.content.len(), 1);
        assert!(r.extra_messages.is_empty());
    }

    #[test]
    fn tool_result_error() {
        let r = ToolResult::error("something went wrong");
        assert!(r.is_error);
        assert_eq!(r.content.len(), 1);
    }

    #[test]
    fn tool_result_image() {
        let r = ToolResult::image("image/png", "aGVsbG8=");
        assert!(!r.is_error);
        match &r.content[0] {
            ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.source_type, "base64");
            }
            _ => panic!("expected Image"),
        }
    }

    #[test]
    fn tool_result_with_messages() {
        let msg = Message::user_text("injected");
        let r = ToolResult::text("ok").with_messages(vec![msg]);
        assert_eq!(r.extra_messages.len(), 1);
    }

    #[test]
    fn tool_input_schema_default() {
        let schema = ToolInputSchema::default();
        assert_eq!(schema.schema_type, "object");
        assert!(schema.properties.is_empty());
        assert!(schema.required.is_empty());
        assert!(!schema.additional_properties);
    }

    #[test]
    fn tool_definition_serialization() {
        let def = ToolDefinition {
            name: "Bash".to_string(),
            description: "Run a shell command".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: {
                    let mut m = serde_json::Map::new();
                    m.insert(
                        "command".to_string(),
                        serde_json::json!({"type": "string"}),
                    );
                    m
                },
                required: vec!["command".to_string()],
                additional_properties: false,
            },
        };
        let json = serde_json::to_value(&def).unwrap();
        assert_eq!(json["name"], "Bash");
        assert_eq!(json["input_schema"]["type"], "object");
        assert_eq!(json["input_schema"]["required"][0], "command");
    }

    #[test]
    fn permission_decision_variants() {
        let allow = PermissionDecision::Allow;
        assert!(matches!(allow, PermissionDecision::Allow));

        let deny = PermissionDecision::Deny("no".to_string());
        assert!(matches!(deny, PermissionDecision::Deny(ref s) if s == "no"));

        let modified = PermissionDecision::AllowWithModifiedInput(serde_json::json!({}));
        assert!(matches!(
            modified,
            PermissionDecision::AllowWithModifiedInput(_)
        ));
    }

    #[tokio::test]
    async fn tool_use_context_resolve_path() {
        let ctx = ToolUseContext {
            working_dir: PathBuf::from("/home/user/project"),
            cancellation: CancellationToken::new(),
            read_file_state: Arc::new(RwLock::new(HashSet::new())),
            session_id: "test".to_string(),
            agent_id: None,
        };

        // Absolute path passes through unchanged
        let abs = ctx.resolve_path("/etc/hosts").unwrap();
        assert_eq!(abs, PathBuf::from("/etc/hosts"));

        // Relative path is joined with working_dir
        let rel = ctx.resolve_path("src/main.rs").unwrap();
        assert_eq!(rel, PathBuf::from("/home/user/project/src/main.rs"));
    }

    #[tokio::test]
    async fn tool_use_context_file_read_tracking() {
        let ctx = ToolUseContext {
            working_dir: PathBuf::from("/tmp"),
            cancellation: CancellationToken::new(),
            read_file_state: Arc::new(RwLock::new(HashSet::new())),
            session_id: "test".to_string(),
            agent_id: None,
        };

        assert!(!ctx.was_file_read("foo.txt").await);
        ctx.mark_file_read("foo.txt").await;
        assert!(ctx.was_file_read("foo.txt").await);
    }
}
