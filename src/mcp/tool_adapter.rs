//! MCP tool adapter: bridges MCP tools into the standard [`Tool`] trait.
//!
//! Each MCP tool is exposed with the naming convention `mcp__{server}__{tool}`,
//! and delegates execution to [`McpClient::call_tool`].

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::client::{McpClient, McpToolDefinition};
use crate::types::tool::{
    Tool, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
};

/// Adapts an MCP tool to the standard [`Tool`] trait.
///
/// Created automatically when connecting to an MCP server. The tool name
/// follows the convention `mcp__{server_name}__{tool_name}`.
#[derive(Debug)]
pub struct McpToolAdapter {
    /// Name of the MCP server this tool belongs to.
    server_name: String,
    /// The MCP tool definition (includes prefixed_name, description, schema).
    tool_def: McpToolDefinition,
    /// Shared reference to the MCP client for making tool calls.
    mcp_client: Arc<McpClient>,
}

impl McpToolAdapter {
    /// Create a new adapter for the given MCP tool.
    ///
    /// # Arguments
    ///
    /// * `server_name` - The MCP server name (as returned by `McpClient::connect`).
    /// * `tool_def` - The tool definition from the server.
    /// * `mcp_client` - A shared MCP client instance.
    pub fn new(
        server_name: String,
        tool_def: McpToolDefinition,
        mcp_client: Arc<McpClient>,
    ) -> Self {
        Self {
            server_name,
            tool_def,
            mcp_client,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.tool_def.prefixed_name
    }

    fn description(&self) -> &str {
        &self.tool_def.description
    }

    fn input_schema(&self) -> ToolInputSchema {
        self.tool_def.input_schema.clone()
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        self.tool_def
            .annotations
            .as_ref()
            .and_then(|a| a.read_only)
            .unwrap_or(false)
    }

    fn is_destructive(&self, _input: &Value) -> bool {
        self.tool_def
            .annotations
            .as_ref()
            .and_then(|a| a.destructive)
            .unwrap_or(false)
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        // MCP tools run in a separate process, so they are inherently
        // concurrency-safe from the SDK's perspective.
        true
    }

    fn is_mcp(&self) -> bool {
        true
    }

    async fn call(
        &self,
        input: Value,
        _ctx: &mut ToolUseContext,
        _progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let result = self
            .mcp_client
            .call_tool(&self.server_name, &self.tool_def.name, input)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;

        let text = result
            .content
            .iter()
            .map(|c| c.text())
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error {
            Ok(ToolResult::error(text))
        } else {
            Ok(ToolResult::text(text))
        }
    }
}
