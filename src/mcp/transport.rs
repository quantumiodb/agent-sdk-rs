//! MCP transport abstractions.
//!
//! Currently only stdio transport is fully implemented. HTTP and SSE transports
//! are defined as placeholders for future implementation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::Mutex;

/// Errors that can occur during MCP transport I/O.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Transport closed")]
    Closed,

    #[error("Invalid JSON-RPC response: {0}")]
    InvalidResponse(String),
}

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
}

/// A JSON-RPC 2.0 notification (no id field).
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    pub fn new(method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
        }
    }
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

// ---------------------------------------------------------------------------
// Stdio Transport
// ---------------------------------------------------------------------------

/// A transport that communicates with an MCP server over stdin/stdout of a
/// child process, using newline-delimited JSON-RPC messages.
pub struct StdioTransport {
    stdin: Mutex<ChildStdin>,
    reader: Mutex<BufReader<ChildStdout>>,
}

impl StdioTransport {
    /// Create a new stdio transport from a child process's stdin and stdout.
    pub fn new(stdin: ChildStdin, stdout: ChildStdout) -> Self {
        Self {
            stdin: Mutex::new(stdin),
            reader: Mutex::new(BufReader::new(stdout)),
        }
    }

    /// Send a JSON-RPC request and wait for the matching response.
    pub async fn send_request(
        &self,
        request: &JsonRpcRequest,
    ) -> Result<JsonRpcResponse, TransportError> {
        // Serialize and send
        let mut line = serde_json::to_string(request)?;
        line.push('\n');

        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await?;
        }

        // Read lines until we find a response with a matching id
        let mut reader = self.reader.lock().await;
        loop {
            let mut response_line = String::new();
            let bytes_read = reader.read_line(&mut response_line).await?;
            if bytes_read == 0 {
                return Err(TransportError::Closed);
            }

            let response_line = response_line.trim();
            if response_line.is_empty() {
                continue;
            }

            // Try to parse as a JSON-RPC response
            match serde_json::from_str::<JsonRpcResponse>(response_line) {
                Ok(response) => {
                    if response.id == Some(request.id) {
                        return Ok(response);
                    }
                    // Not our response (could be a notification from the server).
                    // In a full implementation we'd route these elsewhere.
                    tracing::trace!(
                        id = ?response.id,
                        "Received JSON-RPC message with non-matching id, skipping"
                    );
                }
                Err(_) => {
                    // Could be a server notification or log line; skip it.
                    tracing::trace!(
                        line = %response_line,
                        "Skipping non-JSON-RPC line from MCP server"
                    );
                }
            }
        }
    }

    /// Send a JSON-RPC notification (fire-and-forget, no response expected).
    pub async fn send_notification(
        &self,
        notification: &JsonRpcNotification,
    ) -> Result<(), TransportError> {
        let mut line = serde_json::to_string(notification)?;
        line.push('\n');

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }
}

impl std::fmt::Debug for StdioTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdioTransport").finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// HTTP / SSE Transport (placeholder)
// ---------------------------------------------------------------------------

/// HTTP transport for MCP servers (not yet implemented).
///
/// This will use `reqwest` to send JSON-RPC requests over HTTP POST and
/// optionally receive streaming responses via Server-Sent Events.
#[derive(Debug)]
pub struct HttpTransport {
    _url: String,
    _headers: std::collections::HashMap<String, String>,
}

impl HttpTransport {
    /// Create a new HTTP transport (placeholder).
    pub fn new(url: String, headers: std::collections::HashMap<String, String>) -> Self {
        Self {
            _url: url,
            _headers: headers,
        }
    }

    /// Send a JSON-RPC request over HTTP POST.
    pub async fn send_request(
        &self,
        _request: &JsonRpcRequest,
    ) -> Result<JsonRpcResponse, TransportError> {
        todo!("HTTP/SSE MCP transport not yet implemented")
    }

    /// Send a notification over HTTP POST.
    pub async fn send_notification(
        &self,
        _notification: &JsonRpcNotification,
    ) -> Result<(), TransportError> {
        todo!("HTTP/SSE MCP transport not yet implemented")
    }
}
