//! Core types for the Rust Agent SDK.
//!
//! This module re-exports all type definitions used throughout the SDK,
//! organized into submodules aligned with the TypeScript SDK's
//! `coreSchemas.ts` and `controlSchemas.ts`.

pub mod config;
pub mod mcp;
pub mod message;
pub mod permission;
pub mod sdk_message;
pub mod tool;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

// message.rs
pub use message::{
    ContentBlock, ImageSource, Message, MessageRole, StopReason, ToolResultContent, Usage,
};

// tool.rs
pub use tool::{
    ApiToolParam, CanUseToolFn, InterruptBehavior, PermissionCheckResult, PermissionDecision,
    Tool, ToolDefinition, ToolError, ToolInputSchema, ToolProgressSender, ToolResult,
    ToolUseContext,
};

// permission.rs
pub use permission::{
    PermissionBehavior, PermissionMode, PermissionResult, PermissionRule, PermissionRules,
    PermissionUpdate, PermissionUpdateDestination,
};

// sdk_message.rs
pub use sdk_message::{ContentDelta, SDKMessage, TaskStatus};

// config.rs
pub use config::{OutputFormat, ThinkingConfig};

// mcp.rs
pub use mcp::{
    McpConnectionStatus, McpServerConfig, McpServerInfo, McpServerStatus, McpToolAnnotations,
    McpToolCallResult, McpToolDefinition,
};
