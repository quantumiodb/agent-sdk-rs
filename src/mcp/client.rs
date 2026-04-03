//! MCP client: manages connections to multiple MCP servers.
//!
//! Each connection represents a running MCP server process (stdio) or a remote
//! endpoint (HTTP/SSE). The client handles the JSON-RPC initialization
//! handshake, tool discovery, and tool invocation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::transport::{
    JsonRpcNotification, JsonRpcRequest, StdioTransport, TransportError,
};
use crate::types::mcp::McpServerConfig;
use crate::types::tool::ToolInputSchema;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during MCP operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("MCP server not found: {0}")]
    ServerNotFound(String),

    #[error("MCP connection failed: {0}")]
    ConnectionFailed(String),

    #[error("MCP handshake failed: {0}")]
    HandshakeFailed(String),

    #[error("MCP tool call failed: {0}")]
    ToolCallFailed(String),

    #[error("MCP transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// MCP tool / result types
// ---------------------------------------------------------------------------

/// An MCP tool definition as reported by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// The tool's canonical name on the server.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input.
    pub input_schema: ToolInputSchema,
    /// The prefixed name used in the SDK: `mcp__{server}__{tool}`.
    #[serde(default)]
    pub prefixed_name: String,
    /// Optional MCP tool annotations.
    #[serde(default)]
    pub annotations: Option<McpToolAnnotations>,
}

/// MCP tool annotations (read-only hints, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpToolAnnotations {
    /// Whether the tool is read-only.
    #[serde(default)]
    pub read_only: Option<bool>,
    /// Whether the tool is destructive.
    #[serde(default)]
    pub destructive: Option<bool>,
}

/// The result of calling an MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallResult {
    /// Content blocks returned by the tool.
    pub content: Vec<McpContent>,
    /// Whether the call resulted in an error.
    #[serde(default)]
    pub is_error: bool,
}

/// A content item in an MCP tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "resource")]
    Resource { resource: Value },
}

impl McpContent {
    /// Extract the text value of this content item.
    pub fn text(&self) -> String {
        match self {
            McpContent::Text { text } => text.clone(),
            McpContent::Image { mime_type, .. } => format!("[Image: {mime_type}]"),
            McpContent::Resource { resource } => {
                serde_json::to_string(resource).unwrap_or_else(|_| "[Resource]".to_string())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal connection state
// ---------------------------------------------------------------------------

/// An active connection to a single MCP server.
struct McpConnection {
    /// Server name (from the initialize handshake).
    #[allow(dead_code)]
    name: String,
    /// The configuration used to establish this connection.
    #[allow(dead_code)]
    config: McpServerConfig,
    /// Tools advertised by this server.
    tools: Vec<McpToolDefinition>,
    /// The child process (for stdio transport).
    child: Option<tokio::process::Child>,
    /// Monotonically increasing JSON-RPC request ID.
    request_id: AtomicU64,
    /// Stdio transport handle (None for HTTP/SSE servers).
    transport: Option<Arc<StdioTransport>>,
}

// ---------------------------------------------------------------------------
// McpClient
// ---------------------------------------------------------------------------

/// Client managing connections to multiple MCP servers.
///
/// Thread-safe: all state is behind `RwLock` and can be shared via `Arc`.
pub struct McpClient {
    connections: RwLock<HashMap<String, McpConnection>>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient").finish_non_exhaustive()
    }
}

impl McpClient {
    /// Create a new MCP client with no connections.
    pub fn new() -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
        }
    }

    /// Connect to an MCP server using the given configuration.
    ///
    /// For stdio servers, this spawns the child process and performs the
    /// JSON-RPC `initialize` handshake followed by `tools/list`.
    ///
    /// Returns the server name (as reported by the server, or derived from
    /// the command).
    pub async fn connect(&self, config: &McpServerConfig) -> Result<String, McpError> {
        match config {
            McpServerConfig::Stdio { command, args, env } => {
                info!(command = %command, "Connecting to MCP server via stdio");

                // 1. Spawn the child process.
                let mut child = tokio::process::Command::new(command)
                    .args(args)
                    .envs(env)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .map_err(|e| {
                        McpError::ConnectionFailed(format!(
                            "Failed to spawn MCP server '{command}': {e}"
                        ))
                    })?;

                let stdin = child.stdin.take().ok_or_else(|| {
                    McpError::ConnectionFailed("Failed to capture child stdin".to_string())
                })?;
                let stdout = child.stdout.take().ok_or_else(|| {
                    McpError::ConnectionFailed("Failed to capture child stdout".to_string())
                })?;

                let transport = Arc::new(StdioTransport::new(stdin, stdout));

                // 2. MCP initialize handshake (JSON-RPC 2.0).
                let server_name = self
                    .perform_handshake(&transport)
                    .await
                    .map_err(|e| McpError::HandshakeFailed(e.to_string()))?;

                // 3. Discover tools.
                let tools = self
                    .discover_tools(&transport, &server_name)
                    .await
                    .map_err(|e| McpError::HandshakeFailed(e.to_string()))?;

                info!(
                    server = %server_name,
                    tool_count = tools.len(),
                    "MCP server connected"
                );

                let conn = McpConnection {
                    name: server_name.clone(),
                    config: config.clone(),
                    tools,
                    child: Some(child),
                    request_id: AtomicU64::new(100), // Start after handshake IDs
                    transport: Some(transport),
                };

                self.connections
                    .write()
                    .await
                    .insert(server_name.clone(), conn);

                Ok(server_name)
            }
            McpServerConfig::Sse { .. } | McpServerConfig::Http { .. } | McpServerConfig::Sdk { .. } => {
                // HTTP/SSE/SDK transports are not yet implemented.
                todo!("HTTP/SSE MCP transport not yet implemented")
            }
        }
    }

    /// Perform the MCP initialize handshake over the given transport.
    ///
    /// Sends an `initialize` request followed by an `initialized` notification.
    /// Returns the server name from the response.
    async fn perform_handshake(
        &self,
        transport: &StdioTransport,
    ) -> Result<String, McpError> {
        // Send initialize request
        let init_request = JsonRpcRequest::new(
            1,
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": { "listChanged": true }
                },
                "clientInfo": {
                    "name": "claude-agent-sdk",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
        );

        let response = transport.send_request(&init_request).await?;

        // Extract server name from response
        let server_name = response
            .result
            .as_ref()
            .and_then(|r| r.get("serverInfo"))
            .and_then(|si| si.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown")
            .to_string();

        if let Some(err) = &response.error {
            return Err(McpError::HandshakeFailed(format!(
                "Server returned error: {} (code {})",
                err.message, err.code
            )));
        }

        debug!(server = %server_name, "MCP handshake: initialize OK");

        // Send initialized notification (no response expected)
        let notification = JsonRpcNotification::new("notifications/initialized", None);
        transport.send_notification(&notification).await?;

        debug!("MCP handshake: initialized notification sent");

        Ok(server_name)
    }

    /// Discover tools from an MCP server via `tools/list`.
    async fn discover_tools(
        &self,
        transport: &StdioTransport,
        server_name: &str,
    ) -> Result<Vec<McpToolDefinition>, McpError> {
        let request = JsonRpcRequest::new(2, "tools/list", None);
        let response = transport.send_request(&request).await?;

        if let Some(err) = &response.error {
            return Err(McpError::ToolCallFailed(format!(
                "tools/list failed: {} (code {})",
                err.message, err.code
            )));
        }

        let tools_value = response
            .result
            .and_then(|r| r.get("tools").cloned())
            .unwrap_or(Value::Array(vec![]));

        // Parse raw tool definitions from the server
        let raw_tools: Vec<RawMcpTool> = serde_json::from_value(tools_value).map_err(|e| {
            McpError::ToolCallFailed(format!("Failed to parse tools/list response: {e}"))
        })?;

        let tools = raw_tools
            .into_iter()
            .map(|raw| {
                let prefixed_name = format!("mcp__{}__{}", server_name, raw.name);
                McpToolDefinition {
                    name: raw.name,
                    description: raw.description.unwrap_or_default(),
                    input_schema: raw.input_schema.unwrap_or_else(|| ToolInputSchema {
                        schema_type: "object".to_string(),
                        properties: serde_json::Map::new(),
                        required: vec![],
                        additional_properties: false,
                    }),
                    prefixed_name,
                    annotations: raw.annotations,
                }
            })
            .collect();

        Ok(tools)
    }

    /// Call a tool on a connected MCP server.
    ///
    /// `server_name` is the server identifier returned by [`connect`](Self::connect).
    /// `tool_name` is the tool's canonical name on the server (without the
    /// `mcp__` prefix).
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        input: Value,
    ) -> Result<McpToolCallResult, McpError> {
        let connections = self.connections.read().await;
        let conn = connections
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        let transport = conn.transport.as_ref().ok_or_else(|| {
            McpError::ToolCallFailed("No transport available for this connection".to_string())
        })?;

        let request_id = conn.request_id.fetch_add(1, Ordering::SeqCst);

        let request = JsonRpcRequest::new(
            request_id,
            "tools/call",
            Some(serde_json::json!({
                "name": tool_name,
                "arguments": input
            })),
        );

        debug!(
            server = %server_name,
            tool = %tool_name,
            request_id = request_id,
            "Calling MCP tool"
        );

        let response = transport.send_request(&request).await?;

        if let Some(err) = &response.error {
            return Err(McpError::ToolCallFailed(format!(
                "tools/call '{}' failed: {} (code {})",
                tool_name, err.message, err.code
            )));
        }

        let result = response.result.unwrap_or(serde_json::json!({
            "content": [{"type": "text", "text": ""}],
            "isError": false
        }));

        // Parse the MCP tool call result.
        // The MCP spec uses "isError" (camelCase) in the response.
        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let content: Vec<McpContent> = result
            .get("content")
            .and_then(|c| serde_json::from_value(c.clone()).ok())
            .unwrap_or_else(|| vec![McpContent::Text { text: String::new() }]);

        Ok(McpToolCallResult { content, is_error })
    }

    /// List all tools from all connected servers.
    ///
    /// Returns tuples of `(server_name, tool_definition)`.
    pub async fn list_all_tools(&self) -> Result<Vec<(String, McpToolDefinition)>, McpError> {
        let connections = self.connections.read().await;
        let mut all_tools = Vec::new();
        for (server_name, conn) in connections.iter() {
            for tool in &conn.tools {
                all_tools.push((server_name.clone(), tool.clone()));
            }
        }
        Ok(all_tools)
    }

    /// Close all MCP server connections.
    ///
    /// For stdio servers, this kills the child processes.
    pub async fn close_all(&self) -> Result<(), McpError> {
        let mut connections = self.connections.write().await;
        for (name, mut conn) in connections.drain() {
            info!(server = %name, "Closing MCP server connection");
            // Drop the transport first to close stdin, which signals the child.
            conn.transport.take();
            if let Some(mut child) = conn.child.take() {
                match child.kill().await {
                    Ok(()) => debug!(server = %name, "MCP server process killed"),
                    Err(e) => warn!(server = %name, error = %e, "Failed to kill MCP server process"),
                }
            }
        }
        Ok(())
    }
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helper types for parsing tools/list responses
// ---------------------------------------------------------------------------

/// Raw tool definition from the MCP server (before we add the prefixed name).
#[derive(Debug, Deserialize)]
struct RawMcpTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Option<ToolInputSchema>,
    #[serde(default)]
    annotations: Option<McpToolAnnotations>,
}
