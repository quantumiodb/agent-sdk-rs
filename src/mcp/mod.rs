//! MCP (Model Context Protocol) client integration.
//!
//! Gated on the `mcp` feature flag. Provides [`McpClient`] for connecting to
//! MCP servers over stdio (with HTTP/SSE as future transports), and
//! [`McpToolAdapter`] for exposing MCP tools through the standard [`Tool`](crate::types::tool::Tool) trait.
//!
//! MCP tools are named `mcp__{server}__{tool}` to avoid collisions with
//! built-in tools.
//!
//! # Example
//!
//! ```rust,ignore
//! use claude_agent_sdk::mcp::McpClient;
//! use claude_agent_sdk::McpServerConfig;
//! use std::collections::HashMap;
//!
//! let client = McpClient::new();
//! let server_name = client.connect(&McpServerConfig::Stdio {
//!     command: "npx".into(),
//!     args: vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
//!     env: HashMap::new(),
//! }).await?;
//!
//! let tools = client.list_all_tools().await?;
//! println!("Connected to {} with {} tools", server_name, tools.len());
//!
//! client.close_all().await?;
//! ```

pub mod client;
pub mod tool_adapter;
pub mod transport;

pub use client::{McpClient, McpError, McpToolCallResult, McpToolDefinition, McpToolAnnotations, McpContent};
pub use tool_adapter::McpToolAdapter;
pub use transport::{
    HttpTransport, JsonRpcError, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    StdioTransport, TransportError,
};
