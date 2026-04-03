//! MCP (Model Context Protocol) types for the Rust Agent SDK.
//!
//! Defines configuration and status types for MCP servers and tools.
//! Aligned with the TypeScript SDK's MCP schemas from `coreSchemas.ts`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// McpServerConfig
// ---------------------------------------------------------------------------

/// Configuration for connecting to an MCP server.
///
/// Aligned with the TypeScript SDK's `McpServerConfigForProcessTransportSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpServerConfig {
    /// Standard I/O transport -- launch a subprocess and communicate via
    /// stdin/stdout.
    #[serde(rename = "stdio")]
    Stdio {
        /// Command to execute.
        command: String,
        /// Command-line arguments.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        /// Environment variables to set for the subprocess.
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        env: HashMap<String, String>,
    },

    /// Server-Sent Events transport -- connect to an HTTP SSE endpoint.
    #[serde(rename = "sse")]
    Sse {
        /// SSE endpoint URL.
        url: String,
        /// Additional HTTP headers.
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
    },

    /// HTTP transport -- connect to an HTTP endpoint.
    #[serde(rename = "http")]
    Http {
        /// HTTP endpoint URL.
        url: String,
        /// Additional HTTP headers.
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
    },

    /// SDK-managed transport -- a named, in-process MCP server.
    #[serde(rename = "sdk")]
    Sdk {
        /// Logical name of the SDK-provided server.
        name: String,
    },
}

// ---------------------------------------------------------------------------
// McpConnectionStatus
// ---------------------------------------------------------------------------

/// Connection status of an MCP server.
///
/// Aligned with the TypeScript SDK's `McpServerStatusSchema` status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpConnectionStatus {
    Connected,
    Failed,
    NeedsAuth,
    Pending,
    Disabled,
}

impl std::fmt::Display for McpConnectionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpConnectionStatus::Connected => write!(f, "connected"),
            McpConnectionStatus::Failed => write!(f, "failed"),
            McpConnectionStatus::NeedsAuth => write!(f, "needs-auth"),
            McpConnectionStatus::Pending => write!(f, "pending"),
            McpConnectionStatus::Disabled => write!(f, "disabled"),
        }
    }
}

// ---------------------------------------------------------------------------
// McpServerInfo
// ---------------------------------------------------------------------------

/// Basic information about an MCP server (reported by the server itself).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    /// Server name as reported by the server.
    pub name: String,
    /// Server version.
    pub version: String,
}

// ---------------------------------------------------------------------------
// McpToolAnnotations
// ---------------------------------------------------------------------------

/// Annotations on an MCP tool that describe its behavior.
///
/// Aligned with the TypeScript SDK's tool annotations object in
/// `McpServerStatusSchema`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpToolAnnotations {
    /// Whether the tool only reads data (does not modify state).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,

    /// Whether the tool performs destructive/irreversible operations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destructive: Option<bool>,

    /// Whether the tool accesses external/open-world resources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_world: Option<bool>,
}

// ---------------------------------------------------------------------------
// McpToolDefinition
// ---------------------------------------------------------------------------

/// Definition of a tool provided by an MCP server.
///
/// This is the metadata received when listing tools from a connected MCP
/// server. It is used to register MCP tools in the tool registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Tool name (as reported by the MCP server).
    pub name: String,

    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Behavioral annotations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<McpToolAnnotations>,

    /// JSON Schema for the tool's input parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// McpServerStatus
// ---------------------------------------------------------------------------

/// Status information for an MCP server connection.
///
/// Aligned with the TypeScript SDK's `McpServerStatusSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    /// Server name as configured.
    pub name: String,

    /// Current connection status.
    pub status: McpConnectionStatus,

    /// Server information (available when connected).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info: Option<McpServerInfo>,

    /// Error message (available when status is `Failed`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Server configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<McpServerConfig>,

    /// Configuration scope (e.g. "project", "user", "local").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,

    /// Tools provided by this server (available when connected).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<McpToolDefinition>>,
}

impl McpServerStatus {
    /// Create a status for a connected server.
    pub fn connected(name: impl Into<String>, server_info: McpServerInfo) -> Self {
        Self {
            name: name.into(),
            status: McpConnectionStatus::Connected,
            server_info: Some(server_info),
            error: None,
            config: None,
            scope: None,
            tools: None,
        }
    }

    /// Create a status for a failed server.
    pub fn failed(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: McpConnectionStatus::Failed,
            server_info: None,
            error: Some(error.into()),
            config: None,
            scope: None,
            tools: None,
        }
    }

    /// Create a status for a pending server.
    pub fn pending(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: McpConnectionStatus::Pending,
            server_info: None,
            error: None,
            config: None,
            scope: None,
            tools: None,
        }
    }
}

// ---------------------------------------------------------------------------
// McpToolCallResult
// ---------------------------------------------------------------------------

/// Result from calling an MCP tool.
///
/// Wraps the content returned by the MCP server along with optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallResult {
    /// Whether the tool call resulted in an error.
    #[serde(default)]
    pub is_error: bool,

    /// Result content (text, images, etc.) as raw JSON values.
    pub content: Vec<serde_json::Value>,

    /// Optional metadata from the MCP server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<HashMap<String, serde_json::Value>>,

    /// Optional structured content for rich result rendering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<HashMap<String, serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_server_config_stdio_serialization() {
        let config = McpServerConfig::Stdio {
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
            ],
            env: HashMap::new(),
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "stdio");
        assert_eq!(json["command"], "npx");
        assert_eq!(json["args"][0], "-y");
    }

    #[test]
    fn mcp_server_config_sse_serialization() {
        let config = McpServerConfig::Sse {
            url: "https://mcp.example.com/events".to_string(),
            headers: {
                let mut h = HashMap::new();
                h.insert("Authorization".to_string(), "Bearer tok".to_string());
                h
            },
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "sse");
        assert_eq!(json["url"], "https://mcp.example.com/events");
        assert_eq!(json["headers"]["Authorization"], "Bearer tok");
    }

    #[test]
    fn mcp_server_config_http_roundtrip() {
        let config = McpServerConfig::Http {
            url: "https://api.example.com/mcp".to_string(),
            headers: HashMap::new(),
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: McpServerConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            McpServerConfig::Http { url, .. } => {
                assert_eq!(url, "https://api.example.com/mcp");
            }
            _ => panic!("expected Http variant"),
        }
    }

    #[test]
    fn mcp_server_config_sdk_serialization() {
        let config = McpServerConfig::Sdk {
            name: "builtin-fs".to_string(),
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "sdk");
        assert_eq!(json["name"], "builtin-fs");
    }

    #[test]
    fn mcp_connection_status_serialization() {
        let json = serde_json::to_string(&McpConnectionStatus::NeedsAuth).unwrap();
        assert_eq!(json, "\"needs-auth\"");

        let parsed: McpConnectionStatus = serde_json::from_str("\"connected\"").unwrap();
        assert_eq!(parsed, McpConnectionStatus::Connected);
    }

    #[test]
    fn mcp_connection_status_display() {
        assert_eq!(format!("{}", McpConnectionStatus::Connected), "connected");
        assert_eq!(format!("{}", McpConnectionStatus::NeedsAuth), "needs-auth");
    }

    #[test]
    fn mcp_server_status_constructors() {
        let connected = McpServerStatus::connected(
            "test-server",
            McpServerInfo {
                name: "test".to_string(),
                version: "1.0".to_string(),
            },
        );
        assert_eq!(connected.status, McpConnectionStatus::Connected);
        assert!(connected.server_info.is_some());
        assert!(connected.error.is_none());

        let failed = McpServerStatus::failed("broken-server", "connection refused");
        assert_eq!(failed.status, McpConnectionStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("connection refused"));

        let pending = McpServerStatus::pending("starting-server");
        assert_eq!(pending.status, McpConnectionStatus::Pending);
    }

    #[test]
    fn mcp_tool_definition_serialization() {
        let tool = McpToolDefinition {
            name: "read_file".to_string(),
            description: Some("Read a file".to_string()),
            annotations: Some(McpToolAnnotations {
                read_only: Some(true),
                destructive: Some(false),
                open_world: None,
            }),
            input_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                }
            })),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "read_file");
        assert_eq!(json["annotations"]["read_only"], true);
        assert_eq!(json["annotations"]["destructive"], false);
        assert!(json["annotations"].get("open_world").is_none());
    }

    #[test]
    fn mcp_tool_call_result_serialization() {
        let result = McpToolCallResult {
            is_error: false,
            content: vec![serde_json::json!({"type": "text", "text": "file contents"})],
            meta: None,
            structured_content: None,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["is_error"], false);
        assert_eq!(json["content"][0]["text"], "file contents");
    }

    #[test]
    fn mcp_tool_annotations_default() {
        let annotations = McpToolAnnotations::default();
        assert!(annotations.read_only.is_none());
        assert!(annotations.destructive.is_none());
        assert!(annotations.open_world.is_none());
    }
}
