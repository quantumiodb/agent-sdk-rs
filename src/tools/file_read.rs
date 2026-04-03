//! FileReadTool — read file contents with line numbers.
//!
//! Reads a file from disk, adds line numbers to each line (like `cat -n`),
//! and marks the file as read in the session context (for read-before-edit
//! enforcement).

use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;

use crate::types::tool::{
    Tool, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
};

// ---------------------------------------------------------------------------
// FileReadTool
// ---------------------------------------------------------------------------

/// Read a file from the local filesystem with optional offset and limit.
///
/// # Input schema
///
/// ```json
/// {
///   "file_path": "string (required) — absolute path to the file",
///   "offset": "number (optional) — 0-based line number to start reading from",
///   "limit": "number (optional) — maximum number of lines to read"
/// }
/// ```
///
/// # Output
///
/// File contents with 1-based line numbers prefixed to each line (tab-separated),
/// matching the `cat -n` format used in the TypeScript implementation.
#[derive(Debug, Clone, Copy)]
pub struct FileReadTool;

/// Default maximum number of lines to read.
const DEFAULT_LINE_LIMIT: usize = 2000;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "Read"
    }

    fn description(&self) -> &str {
        "Reads a file from the local filesystem. Returns file contents with line numbers. \
         You can optionally specify a line offset and limit for large files."
    }

    fn input_schema(&self) -> ToolInputSchema {
        ToolInputSchema {
            schema_type: "object".to_string(),
            properties: {
                let mut m = serde_json::Map::new();
                m.insert(
                    "file_path".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    }),
                );
                m.insert(
                    "offset".to_string(),
                    serde_json::json!({
                        "type": "number",
                        "description": "The 0-based line number to start reading from"
                    }),
                );
                m.insert(
                    "limit".to_string(),
                    serde_json::json!({
                        "type": "number",
                        "description": "The maximum number of lines to read (default 2000)"
                    }),
                );
                m
            },
            required: vec!["file_path".to_string()],
            additional_properties: false,
        }
    }

    fn aliases(&self) -> &[&str] {
        &["FileRead", "read", "cat"]
    }

    fn search_hint(&self) -> Option<&str> {
        Some("read file view contents cat display show")
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        true
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        true
    }

    fn get_path(&self, input: &Value) -> Option<String> {
        input
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn user_facing_name(&self, input: &Value) -> String {
        if let Some(path) = input.get("file_path").and_then(|v| v.as_str()) {
            // Show just the filename
            let filename = std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path);
            format!("Read({})", filename)
        } else {
            "Read".to_string()
        }
    }

    async fn validate_input(&self, input: &Value, _ctx: &ToolUseContext) -> Result<(), String> {
        let file_path = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field 'file_path' (must be a string)")?;

        if file_path.trim().is_empty() {
            return Err("'file_path' must not be empty".to_string());
        }

        // Validate offset if provided
        if let Some(offset) = input.get("offset") {
            if offset.as_u64().is_none() && offset.as_f64().is_none() {
                return Err("'offset' must be a non-negative number".to_string());
            }
        }

        // Validate limit if provided
        if let Some(limit) = input.get("limit") {
            match limit.as_u64() {
                Some(0) => return Err("'limit' must be greater than 0".to_string()),
                Some(_) => {}
                None => return Err("'limit' must be a positive number".to_string()),
            }
        }

        Ok(())
    }

    async fn call(
        &self,
        input: Value,
        ctx: &mut ToolUseContext,
        _progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let file_path_str = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'file_path' field".to_string()))?;

        let offset = input
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LINE_LIMIT);

        let resolved_path = ctx.resolve_path(file_path_str)?;
        let canonical_path = resolved_path
            .to_str()
            .unwrap_or(file_path_str)
            .to_string();

        debug!(
            path = %canonical_path,
            offset = %offset,
            limit = %limit,
            "Reading file"
        );

        // Check if file exists
        if !resolved_path.exists() {
            return Ok(ToolResult::error(format!(
                "File not found: {}",
                canonical_path
            )));
        }

        // Check if it is a directory
        if resolved_path.is_dir() {
            return Ok(ToolResult::error(format!(
                "'{}' is a directory, not a file. Use Bash with 'ls' to list directory contents.",
                canonical_path
            )));
        }

        // Read the file
        let content = tokio::fs::read_to_string(&resolved_path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                ToolError::PermissionDenied(format!(
                    "Cannot read file '{}': permission denied",
                    canonical_path
                ))
            } else {
                ToolError::Io(e)
            }
        })?;

        // Mark file as read for read-before-edit tracking
        ctx.mark_file_read(&canonical_path).await;

        // Apply offset and limit, adding line numbers
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        if offset >= total_lines && total_lines > 0 {
            return Ok(ToolResult::error(format!(
                "Offset {} is beyond the end of the file ({} lines)",
                offset, total_lines
            )));
        }

        let selected_lines = lines
            .iter()
            .skip(offset)
            .take(limit)
            .enumerate()
            .map(|(i, line)| {
                let line_num = offset + i + 1; // 1-based
                format!("{}\t{}", line_num, line)
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Add metadata if the file was truncated
        let mut result = selected_lines;
        if total_lines > offset + limit {
            result.push_str(&format!(
                "\n\n... ({} more lines not shown, {} total)",
                total_lines - offset - limit,
                total_lines,
            ));
        }

        if content.is_empty() {
            result = "(empty file)".to_string();
        }

        Ok(ToolResult::text(result))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tokio_util::sync::CancellationToken;

    fn test_ctx_with_dir(dir: &std::path::Path) -> ToolUseContext {
        ToolUseContext {
            working_dir: dir.to_path_buf(),
            cancellation: CancellationToken::new(),
            read_file_state: Arc::new(RwLock::new(HashSet::new())),
            session_id: "test".to_string(),
            agent_id: None,
        }
    }

    #[test]
    fn tool_metadata() {
        let tool = FileReadTool;
        assert_eq!(tool.name(), "Read");
        assert!(tool.is_read_only(&Value::Null));
        assert!(tool.is_concurrency_safe(&Value::Null));
    }

    #[tokio::test]
    async fn read_nonexistent_file() {
        let tool = FileReadTool;
        let mut ctx = test_ctx_with_dir(std::path::Path::new("/tmp"));
        let result = tool
            .call(
                serde_json::json!({"file_path": "/tmp/nonexistent_file_12345.txt"}),
                &mut ctx,
                None,
            )
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn read_directory() {
        let tool = FileReadTool;
        let mut ctx = test_ctx_with_dir(std::path::Path::new("/tmp"));
        let result = tool
            .call(
                serde_json::json!({"file_path": "/tmp"}),
                &mut ctx,
                None,
            )
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn read_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line one\nline two\nline three\n").unwrap();

        let tool = FileReadTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        let result = tool
            .call(
                serde_json::json!({"file_path": file_path.to_str().unwrap()}),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("1\tline one"));
                assert!(text.contains("2\tline two"));
                assert!(text.contains("3\tline three"));
            }
            _ => panic!("expected text"),
        }

        // Check that the file was marked as read
        assert!(
            ctx.was_file_read(file_path.to_str().unwrap()).await
        );
    }

    #[tokio::test]
    async fn read_with_offset_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("lines.txt");
        let content = (1..=10)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = FileReadTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "offset": 2,
                    "limit": 3
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                // offset=2 means skip first 2 lines, start from line 3
                assert!(text.contains("3\tline 3"));
                assert!(text.contains("4\tline 4"));
                assert!(text.contains("5\tline 5"));
                assert!(!text.contains("2\tline 2"));
                assert!(!text.contains("6\tline 6"));
            }
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn read_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty.txt");
        std::fs::write(&file_path, "").unwrap();

        let tool = FileReadTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        let result = tool
            .call(
                serde_json::json!({"file_path": file_path.to_str().unwrap()}),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("empty file"));
            }
            _ => panic!("expected text"),
        }
    }
}
