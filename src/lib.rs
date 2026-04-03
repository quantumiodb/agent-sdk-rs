//! # claude-agent-sdk
//!
//! A Rust SDK for building autonomous AI agents powered by Claude.
//!
//! The SDK provides a complete agentic loop that runs in-process — no CLI subprocess needed.
//! It implements the same protocol as the TypeScript Claude Code SDK (`coreSchemas.ts`).
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use claude_agent_sdk::{Agent, AgentOptions, ApiClientConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut agent = Agent::new(AgentOptions {
//!         api: ApiClientConfig::FromEnv,
//!         ..Default::default()
//!     }).await?;
//!
//!     let result = agent.prompt("What files are in the current directory?").await?;
//!     println!("{}", result.text);
//!     println!("Cost: ${:.4}", result.cost_usd);
//!
//!     agent.close().await?;
//!     Ok(())
//! }
//! ```

// ─── Core modules ────────────────────────────────────────────────────────────

pub mod agent;
pub mod api;
pub mod context;
pub mod cost;
pub mod permissions;
pub mod tools;
pub mod types;
pub mod utils;

// ─── Optional feature modules ─────────────────────────────────────────────────

#[cfg(feature = "hooks")]
pub mod hooks;

#[cfg(feature = "mcp")]
pub mod mcp;

// ─── Top-level re-exports ─────────────────────────────────────────────────────

// Agent
pub use agent::{Agent, AgentError, AgentOptions, ApiClientConfig, QueryResult, SubagentDefinition};

// API
pub use api::{
    ApiClient, ApiError, ApiProvider, ApiStream, ApiStreamEvent, MessageRequest,
    OpenAICompatProvider,
};

// Core message types
pub use types::{
    ContentBlock, ImageSource, Message, MessageRole, StopReason, ToolResultContent, Usage,
};

// Tool system
pub use types::{
    CanUseToolFn, InterruptBehavior, PermissionCheckResult, PermissionDecision, Tool,
    ToolDefinition, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
};

// SDK streaming protocol
pub use types::{ContentDelta, SDKMessage, TaskStatus};

// Permission types
pub use types::{PermissionMode, PermissionRule, PermissionRules};

// Configuration
pub use types::{OutputFormat, ThinkingConfig};

// MCP config types (always available)
pub use types::{McpServerConfig, McpServerStatus, McpToolDefinition};

// Tool registry
pub use tools::ToolRegistry;

// Cost tracking
pub use cost::{get_pricing, CostSummary, CostTracker, ModelPricing};

// Hooks (when feature is enabled)
#[cfg(feature = "hooks")]
pub use hooks::{
    HookConfig, HookEngine, HookError, HookEvent, HookFn, HookInput, HookOutput, HookRule,
};

// MCP client (when feature is enabled)
#[cfg(feature = "mcp")]
pub use mcp::{McpClient, McpError, McpToolAdapter};
