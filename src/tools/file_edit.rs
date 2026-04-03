//! FileEditTool — exact string replacement in files.
//!
//! Performs precise string replacements in an existing file. Requires the file
//! to have been previously read in the same session (tracked via
//! `ToolUseContext::was_file_read`) to prevent blind edits.

use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;

use crate::types::tool::{
    Tool, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
};

// ---------------------------------------------------------------------------
// FileEditTool
// ---------------------------------------------------------------------------

/// Perform exact string replacement in a file.
///
/// # Input schema
///
/// ```json
/// {
///   "file_path": "string (required) — absolute path to the file",
///   "old_string": "string (required) — the exact text to find",
///   "new_string": "string (required) — the replacement text",
///   "replace_all": "boolean (optional, default false) — replace all occurrences"
/// }
/// ```
///
/// # Requirements
///
/// - The file must have been previously read via `FileReadTool` in this session.
/// - `old_string` must be found in the file (if not `replace_all`, it must be
///   unique — appearing exactly once).
/// - `old_string` and `new_string` must be different.
#[derive(Debug, Clone, Copy)]
pub struct FileEditTool;

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "Edit"
    }

    fn description(&self) -> &str {
        "Performs exact string replacements in files. You must use the Read tool at least \
         once before editing a file. The edit will fail if old_string is not unique in the \
         file — provide more surrounding context to make it unique, or use replace_all."
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
                        "description": "The absolute path to the file to modify"
                    }),
                );
                m.insert(
                    "old_string".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The exact text to replace"
                    }),
                );
                m.insert(
                    "new_string".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The text to replace it with"
                    }),
                );
                m.insert(
                    "replace_all".to_string(),
                    serde_json::json!({
                        "type": "boolean",
                        "description": "Replace all occurrences (default false)"
                    }),
                );
                m
            },
            required: vec![
                "file_path".to_string(),
                "old_string".to_string(),
                "new_string".to_string(),
            ],
            additional_properties: false,
        }
    }

    fn aliases(&self) -> &[&str] {
        &["FileEdit", "edit"]
    }

    fn search_hint(&self) -> Option<&str> {
        Some("edit modify replace string text file content")
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        false
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        // File edits are not safe to run concurrently (could corrupt files).
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
            format!("Edit({})", filename)
        } else {
            "Edit".to_string()
        }
    }

    async fn validate_input(&self, input: &Value, ctx: &ToolUseContext) -> Result<(), String> {
        let file_path = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field 'file_path' (must be a string)")?;

        if file_path.trim().is_empty() {
            return Err("'file_path' must not be empty".to_string());
        }

        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field 'old_string' (must be a string)")?;

        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field 'new_string' (must be a string)")?;

        if old_string == new_string {
            return Err("'old_string' and 'new_string' must be different".to_string());
        }

        // Check that the file was previously read
        let resolved = ctx.resolve_path(file_path).map_err(|e| e.to_string())?;
        let canonical = resolved.to_str().unwrap_or(file_path);

        if !ctx.was_file_read(canonical).await {
            return Err(format!(
                "File '{}' has not been read yet. Use the Read tool first before editing.",
                file_path
            ));
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

        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'old_string' field".to_string()))?;

        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'new_string' field".to_string()))?;

        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let resolved_path = ctx.resolve_path(file_path_str)?;
        let canonical_path = resolved_path
            .to_str()
            .unwrap_or(file_path_str)
            .to_string();

        debug!(
            path = %canonical_path,
            replace_all = %replace_all,
            "Editing file"
        );

        // Read current content
        if !resolved_path.exists() {
            return Ok(ToolResult::error(format!(
                "File not found: {}",
                canonical_path
            )));
        }

        let content = tokio::fs::read_to_string(&resolved_path)
            .await
            .map_err(ToolError::Io)?;

        // Count occurrences
        let count = content.matches(old_string).count();

        if count == 0 {
            return Ok(ToolResult::error(format!(
                "The old_string was not found in '{}'. Make sure it matches exactly, \
                 including whitespace and indentation.",
                canonical_path,
            )));
        }

        if !replace_all && count > 1 {
            return Ok(ToolResult::error(format!(
                "The old_string was found {} times in '{}'. It must be unique for a single \
                 replacement. Provide more surrounding context to make it unique, or set \
                 replace_all to true.",
                count, canonical_path,
            )));
        }

        // Perform replacement
        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        // Write back
        tokio::fs::write(&resolved_path, &new_content)
            .await
            .map_err(ToolError::Io)?;

        let msg = if replace_all {
            format!(
                "Successfully replaced {} occurrence(s) in '{}'.",
                count, canonical_path
            )
        } else {
            format!("Successfully edited '{}'.", canonical_path)
        };

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
        let tool = FileEditTool;
        assert_eq!(tool.name(), "Edit");
        assert!(!tool.is_read_only(&Value::Null));
        assert!(!tool.is_concurrency_safe(&Value::Null));
    }

    #[tokio::test]
    async fn edit_requires_prior_read() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = FileEditTool;
        let ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .validate_input(
                &serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "goodbye"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not been read"));
    }

    #[tokio::test]
    async fn edit_single_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = FileEditTool;
        let mut ctx = test_ctx_with_dir(dir.path());

        // Mark file as read
        ctx.mark_file_read(file_path.to_str().unwrap()).await;

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "goodbye"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "goodbye world");
    }

    #[tokio::test]
    async fn edit_replace_all() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "aaa bbb aaa ccc aaa").unwrap();

        let tool = FileEditTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        ctx.mark_file_read(file_path.to_str().unwrap()).await;

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "old_string": "aaa",
                    "new_string": "xxx",
                    "replace_all": true
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "xxx bbb xxx ccc xxx");
    }

    #[tokio::test]
    async fn edit_non_unique_without_replace_all() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "aaa bbb aaa").unwrap();

        let tool = FileEditTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        ctx.mark_file_read(file_path.to_str().unwrap()).await;

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "old_string": "aaa",
                    "new_string": "xxx"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(result.is_error);
        // File should be unchanged
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "aaa bbb aaa");
    }

    #[tokio::test]
    async fn edit_string_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = FileEditTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        ctx.mark_file_read(file_path.to_str().unwrap()).await;

        let result = tool
            .call(
                serde_json::json!({
                    "file_path": file_path.to_str().unwrap(),
                    "old_string": "nonexistent",
                    "new_string": "replacement"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(result.is_error);
    }

    #[tokio::test]
    async fn validate_same_strings() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileEditTool;
        let ctx = test_ctx_with_dir(dir.path());

        let result = tool
            .validate_input(
                &serde_json::json!({
                    "file_path": "/tmp/test.txt",
                    "old_string": "same",
                    "new_string": "same"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be different"));
    }
}
