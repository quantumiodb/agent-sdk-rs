//! FileWriteTool — write or overwrite a file.
//!
//! Creates a new file or overwrites an existing one. Automatically creates
//! parent directories as needed. Not concurrency-safe.

use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;

use crate::types::tool::{
    Tool, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
};

// ---------------------------------------------------------------------------
// FileWriteTool
// ---------------------------------------------------------------------------

/// Write content to a file, creating it (and parent directories) if necessary.
///
/// # Input schema
///
/// ```json
/// {
///   "file_path": "string (required) — absolute path to the file",
///   "content": "string (required) — the content to write"
/// }
/// ```
///
/// # Behavior
///
/// - If the file exists, it is overwritten.
/// - If the file does not exist, it is created along with any missing parent
///   directories.
/// - After writing, the file is marked as read in the session context (so
///   subsequent edits do not require a separate read).
#[derive(Debug, Clone, Copy)]
pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "Write"
    }

    fn description(&self) -> &str {
        "Writes a file to the local filesystem. Creates parent directories as needed. \
         If the file exists, it is overwritten. Prefer the Edit tool for modifying \
         existing files — it only sends the diff."
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
                        "description": "The absolute path to the file to write"
                    }),
                );
                m.insert(
                    "content".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The content to write to the file"
                    }),
                );
                m
            },
            required: vec!["file_path".to_string(), "content".to_string()],
            additional_properties: false,
        }
    }

    fn aliases(&self) -> &[&str] {
        &["FileWrite", "write"]
    }

    fn search_hint(&self) -> Option<&str> {
        Some("write create file new content save")
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        false
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        false
    }

    fn get_path(&self, input: &Value) -> Option<String> {
        input
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn user_facing_name(&self, input: &Value) -> String {
        if let Some(path) = input.get("file_path").and_then(|v| v.as_str()) {
            let filename = std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path);
            format!("Write({})", filename)
        } else {
            "Write".to_string()
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

        // Content must be present (but can be empty for creating empty files)
        if input.get("content").and_then(|v| v.as_str()).is_none() {
            return Err("Missing required field 'content' (must be a string)".to_string());
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

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'content' field".to_string()))?;

        let resolved_path = ctx.resolve_path(file_path_str)?;
        let canonical_path = resolved_path
            .to_str()
            .unwrap_or(file_path_str)
            .to_string();

        debug!(path = %canonical_path, bytes = content.len(), "Writing file");

        // Create parent directories if they don't exist
        if let Some(parent) = resolved_path.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::Execution(format!(
                        "Failed to create parent directories for '{}': {}",
                        canonical_path, e
                    ))
                })?;
            }
        }

        // Determine if this is a new file or an overwrite
        let existed = resolved_path.exists();

        // Write the file
        tokio::fs::write(&resolved_path, content)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    ToolError::PermissionDenied(format!(
                        "Cannot write to '{}': permission denied",
                        canonical_path
                    ))
                } else {
                    ToolError::Io(e)
                }
            })?;

        // Mark the file as read (so subsequent edits don't require a separate read)
        ctx.mark_file_read(&canonical_path).await;

        let action = if existed { "updated" } else { "created" };
        let msg = format!(
            "File {} successfully at: {} ({} bytes written)",
            action,
            canonical_path,
            content.len(),
        );

        Ok(ToolResult::text(msg))
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
        let tool = FileWriteTool;
        assert_eq!(tool.name(), "Write");
        assert!(!tool.is_read_only(&Value::Null));
        assert!(!tool.is_concurrency_safe(&Value::Null));
    }

    #[tokio::test]
    async fn write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");

        let tool = FileWriteTool;
        let mut ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "content": "hello world"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello world");

        // File should be marked as read
        assert!(ctx.was_file_read(file_path.to_str().unwrap()).await);
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        std::fs::write(&file_path, "old content").unwrap();

        let tool = FileWriteTool;
        let mut ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "content": "new content"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "new content");
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("a").join("b").join("c").join("file.txt");

        let tool = FileWriteTool;
        let mut ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "content": "nested content"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "nested content"
        );
    }

    #[tokio::test]
    async fn write_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty.txt");

        let tool = FileWriteTool;
        let mut ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "content": ""
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "");
    }

    #[tokio::test]
    async fn validate_missing_content() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileWriteTool;
        let ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .validate_input(
                &serde_json::json!({"file_path": "/tmp/test.txt"}),
                &ctx,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn validate_missing_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileWriteTool;
        let ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .validate_input(
                &serde_json::json!({"content": "hello"}),
                &ctx,
            )
            .await;

        assert!(result.is_err());
    }
}
