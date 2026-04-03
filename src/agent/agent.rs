//! Agent struct and configuration types.
//!
//! [`Agent`] is the primary entry point for the SDK. It wraps an API client,
//! a tool registry, a permission context, and a cost tracker, exposing
//! `query()` (streaming) and `prompt()` (blocking) methods.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cost::{CostSummary, CostTracker};
use crate::permissions::PermissionContext;
use crate::tools::ToolRegistry;
use crate::types::{
    ContentBlock, McpServerConfig, Message, PermissionMode, PermissionRules,
    SDKMessage, ThinkingConfig, OutputFormat, Tool, Usage,
};

use super::agent_loop::{self, AgentLoopParams};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during agent creation or execution.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("API error: {0}")]
    Api(#[from] crate::api::ApiError),

    #[error("Channel send error: {0}")]
    Channel(String),

    #[error("Agent loop ended unexpectedly")]
    UnexpectedEnd,

    #[error("Join error: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("{0}")]
    Other(String),
}

impl<T> From<mpsc::error::SendError<T>> for AgentError {
    fn from(err: mpsc::error::SendError<T>) -> Self {
        AgentError::Channel(err.to_string())
    }
}

// ---------------------------------------------------------------------------
// ApiClientConfig
// ---------------------------------------------------------------------------

/// How to obtain or configure the API client.
#[derive(Debug, Clone)]
pub enum ApiClientConfig {
    /// Provide an Anthropic API key.
    ApiKey(String),
    /// Provide a pre-built [`ApiClient`] (wrapped in `Arc` for shared ownership).
    Client(Arc<ApiClient>),
    /// Use the official OpenAI API with the given key.
    OpenAI(String),
    /// Use any OpenAI-compatible endpoint (Groq, Together, Azure, Ollama, …).
    OpenAICompat {
        api_key: String,
        /// Base URL, e.g. `"https://api.groq.com/openai"`.
        base_url: String,
        /// Extra fields merged into every request body.
        ///
        /// Useful for provider-specific parameters that are not part of the
        /// standard OpenAI schema.  For example, to disable thinking on
        /// Qwen3 / DeepSeek-R1 via Ollama:
        ///
        /// ```rust,ignore
        /// extra_body: Some(serde_json::json!({"think": false}))
        /// ```
        extra_body: Option<serde_json::Value>,
    },
    /// Native Ollama provider using `/api/chat`.
    ///
    /// Unlike `OpenAICompat`, this uses Ollama's native NDJSON API which
    /// properly respects the `think` parameter for disabling/enabling reasoning.
    ///
    /// ```rust,ignore
    /// ApiClientConfig::Ollama {
    ///     base_url: "http://localhost:11434".into(),
    ///     think: Some(false),  // disable thinking for instant responses
    /// }
    /// ```
    Ollama {
        /// Ollama server URL, e.g. `"http://localhost:11434"`.
        base_url: String,
        /// Whether to enable (`Some(true)`) or disable (`Some(false)`) thinking,
        /// or leave it up to the model (`None`).
        think: Option<bool>,
    },
    /// Auto-detect from environment variables:
    /// `ANTHROPIC_API_KEY` first, then `OPENAI_API_KEY`.
    FromEnv,
}

// ---------------------------------------------------------------------------
// SubagentDefinition
// ---------------------------------------------------------------------------

/// Definition of a sub-agent type for multi-agent orchestration.
#[derive(Debug, Clone)]
pub struct SubagentDefinition {
    /// Agent type name (e.g. "researcher", "coder").
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// Specialised instructions for this sub-agent.
    pub instructions: Option<String>,
    /// Subset of tools this sub-agent may use.
    pub allowed_tools: Option<Vec<String>>,
    /// Model override (defaults to the parent agent's model).
    pub model: Option<String>,
    /// Permission mode override.
    pub permission_mode: Option<PermissionMode>,
}

// ---------------------------------------------------------------------------
// HooksConfig (placeholder)
// ---------------------------------------------------------------------------

/// Placeholder for hook configuration.
///
/// The full hook system is in `crate::hooks`; this type is used here so that
/// `AgentOptions` can carry an optional hooks configuration without pulling
/// in the entire hooks module at the type level.
#[derive(Debug, Clone, Default)]
pub struct HooksConfig {
    // TODO: Fill in when the hooks module is integrated.
}

// ---------------------------------------------------------------------------
// AgentOptions
// ---------------------------------------------------------------------------

/// Configuration for creating an [`Agent`].
///
/// All fields have sensible defaults via the [`Default`] implementation.
#[derive(Debug, Clone)]
pub struct AgentOptions {
    // --- API / Model ---
    /// How to obtain the API client.
    pub api: ApiClientConfig,
    /// Model identifier (e.g. `"claude-sonnet-4-6"`).
    pub model: String,
    /// Maximum output tokens per API call.
    pub max_tokens: u32,
    /// Extended thinking configuration.
    pub thinking: Option<ThinkingConfig>,
    /// Structured output JSON Schema.
    pub output_format: Option<OutputFormat>,

    // --- Prompting ---
    /// Custom system prompt (replaces the default, but env info is still prepended).
    pub system_prompt: Option<String>,
    /// Extra text appended to the system prompt.
    pub append_system_prompt: Option<String>,

    // --- Execution limits ---
    /// Maximum tool-use loop iterations (default: 100).
    pub max_turns: u32,
    /// Maximum spend in USD (None = unlimited).
    pub max_budget_usd: Option<f64>,

    // --- Tools ---
    /// Custom tools to register in addition to built-ins.
    pub custom_tools: Vec<Arc<dyn Tool>>,
    /// Whitelist: only these tools are available (None = all).
    pub allowed_tools: Option<Vec<String>>,
    /// Blacklist: these tools are removed.
    pub disallowed_tools: Vec<String>,

    // --- Permissions ---
    /// Permission mode.
    pub permission_mode: PermissionMode,
    /// Permission rules.
    pub permission_rules: PermissionRules,

    // --- MCP ---
    /// MCP server configurations.
    pub mcp_servers: Vec<McpServerConfig>,

    // --- Sub-agents ---
    /// Pre-defined sub-agent types.
    pub subagent_definitions: Vec<SubagentDefinition>,

    // --- Hooks ---
    /// Optional hooks configuration.
    pub hooks_config: Option<HooksConfig>,

    // --- Environment ---
    /// Working directory (defaults to `std::env::current_dir()`).
    pub cwd: Option<PathBuf>,

    // --- Network ---
    /// Extra HTTP headers sent with every API request.
    pub custom_headers: Vec<(String, String)>,
    /// HTTP/HTTPS proxy URL.
    pub proxy: Option<String>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        // Honour ANTHROPIC_MODEL / OPENAI_MODEL env vars.
        // ANTHROPIC_MODEL takes precedence (TypeScript SDK convention).
        let model = std::env::var("ANTHROPIC_MODEL")
            .or_else(|_| std::env::var("OPENAI_MODEL"))
            .unwrap_or_else(|_| "claude-sonnet-4-6".to_string());
        Self {
            api: ApiClientConfig::FromEnv,
            model,
            max_tokens: 16384,
            thinking: None,
            output_format: None,
            system_prompt: None,
            append_system_prompt: None,
            max_turns: 100,
            max_budget_usd: None,
            custom_tools: vec![],
            allowed_tools: None,
            disallowed_tools: vec![],
            permission_mode: PermissionMode::Default,
            permission_rules: PermissionRules::default(),
            mcp_servers: vec![],
            subagent_definitions: vec![],
            hooks_config: None,
            cwd: None,
            custom_headers: vec![],
            proxy: None,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// The result of a completed `prompt()` call.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Concatenated text output from the assistant.
    pub text: String,
    /// Total token usage across all API calls in this query.
    pub usage: Usage,
    /// Total estimated cost in USD.
    pub cost_usd: f64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Number of tool-use loop iterations.
    pub num_turns: u32,
    /// Full message history after the query.
    pub messages: Vec<Message>,
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// The primary user-facing struct for running Claude as an autonomous agent.
///
/// # Lifecycle
///
/// ```text
/// Agent::new(options) -> agent.query("...") / agent.prompt("...") -> agent.close()
/// ```
///
/// `query()` returns a streaming channel + join handle; `prompt()` is a
/// convenience wrapper that collects the stream into a [`QueryResult`].
pub struct Agent {
    /// API client (shared via Arc for use in spawned tasks).
    api_client: Arc<ApiClient>,
    /// Tool registry.
    tool_registry: Arc<ToolRegistry>,
    /// Conversation message history.
    messages: Vec<Message>,
    /// Cost tracker.
    cost_tracker: Arc<CostTracker>,
    /// Permission context.
    permission_ctx: Arc<PermissionContext>,
    /// Configuration snapshot.
    options: AgentOptions,
    /// Session identifier (UUID).
    session_id: String,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("session_id", &self.session_id)
            .field("model", &self.options.model)
            .field("tools", &self.tool_registry.len())
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl Agent {
    /// Create a new Agent instance.
    ///
    /// This is async because it may need to connect to MCP servers and
    /// perform other IO during initialization.
    pub async fn new(options: AgentOptions) -> Result<Self, AgentError> {
        // 1. Create API client.
        let api_client = match &options.api {
            ApiClientConfig::ApiKey(key) => Arc::new(ApiClient::anthropic(key)),
            ApiClientConfig::Client(c) => c.clone(),
            ApiClientConfig::OpenAI(key) => Arc::new(ApiClient::openai(key)),
            ApiClientConfig::OpenAICompat { api_key, base_url, extra_body } => {
                Arc::new(ApiClient::openai_compat_with_options(api_key, base_url, extra_body.clone()))
            }
            ApiClientConfig::Ollama { base_url, think } => {
                Arc::new(ApiClient::ollama(base_url, *think))
            }
            ApiClientConfig::FromEnv => Arc::new(ApiClient::from_env()?),
        };

        // 2. Build tool registry.
        let mut registry = ToolRegistry::default_registry();

        // Register custom tools.
        for tool in &options.custom_tools {
            registry.register(tool.clone());
        }

        // Apply whitelist.
        if let Some(allowed) = &options.allowed_tools {
            registry.retain(|name| allowed.iter().any(|a| a == name));
        }

        // Apply blacklist.
        for name in &options.disallowed_tools {
            registry.remove(name);
        }

        // 3. Build permission context.
        let cwd = options
            .cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")));

        let permission_ctx = Arc::new(PermissionContext::new(
            options.permission_mode,
            cwd,
            &options.permission_rules,
        ));

        Ok(Self {
            api_client,
            tool_registry: Arc::new(registry),
            messages: Vec::new(),
            cost_tracker: Arc::new(CostTracker::new(&options.model)),
            permission_ctx,
            session_id: uuid::Uuid::new_v4().to_string(),
            options,
        })
    }

    /// Start a streaming query.
    ///
    /// Appends a user message, spawns the agent loop in a tokio task, and
    /// returns the event receiver + task handle.
    pub fn query(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<(mpsc::Receiver<SDKMessage>, JoinHandle<Result<(), AgentError>>), AgentError> {
        let prompt = prompt.into();

        // Append user message.
        self.messages.push(Message::user_text(&prompt));

        let (tx, rx) = mpsc::channel(256);

        // Clone Arc handles for the spawned task.
        let api_client = self.api_client.clone();
        let tool_registry = self.tool_registry.clone();
        let cost_tracker = self.cost_tracker.clone();
        let permission_ctx = self.permission_ctx.clone();
        let messages = self.messages.clone();
        let options = self.options.clone();
        let session_id = self.session_id.clone();

        let handle = tokio::spawn(async move {
            agent_loop::run_loop(AgentLoopParams {
                api_client,
                tool_registry,
                cost_tracker,
                permission_ctx,
                messages,
                options,
                session_id,
                event_tx: tx,
            })
            .await
        });

        Ok((rx, handle))
    }

    /// Run a complete query and collect the result.
    ///
    /// This is a convenience wrapper around [`query()`](Self::query) that
    /// consumes the event stream and returns a [`QueryResult`].
    pub async fn prompt(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<QueryResult, AgentError> {
        let (mut rx, handle) = self.query(prompt)?;

        let mut final_text = String::new();
        let mut total_usage = Usage::default();

        while let Some(msg) = rx.recv().await {
            match &msg {
                SDKMessage::Assistant { message, .. } => {
                    for block in &message.content {
                        if let ContentBlock::Text { text } = block {
                            final_text.push_str(text);
                        }
                    }
                    if let Some(usage) = &message.usage {
                        total_usage.accumulate(usage);
                    }
                }
                SDKMessage::Result {
                    total_usage: u,
                    total_cost_usd,
                    duration_ms,
                    num_turns,
                    final_messages,
                    ..
                } => {
                    // Sync the agent's message history with the full conversation
                    // produced by the loop (includes assistant responses + tool results).
                    self.messages = final_messages.clone();
                    return Ok(QueryResult {
                        text: final_text,
                        usage: u.clone(),
                        cost_usd: *total_cost_usd,
                        duration_ms: *duration_ms,
                        num_turns: *num_turns,
                        messages: final_messages.clone(),
                    });
                }
                _ => {}
            }
        }

        // If the channel closed without a Result event, check the task.
        handle.await??;
        Err(AgentError::UnexpectedEnd)
    }

    /// Add a message to the conversation history (for multi-turn use).
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get a reference to the current conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get a snapshot of accumulated costs.
    pub fn cost_summary(&self) -> CostSummary {
        self.cost_tracker.summary()
    }

    /// Clean up resources (close MCP connections, etc.).
    pub async fn close(&mut self) -> Result<(), AgentError> {
        // TODO: Close MCP connections when MCP module is integrated.
        Ok(())
    }
}
