//! GlobTool — fast file pattern matching.
//!
//! Supports glob patterns like `**/*.rs` or `src/**/*.ts`. Returns matching
//! file paths sorted by modification time (newest first).
//!
//! When the `built-in-tools` feature is enabled, uses the `globset` and `ignore`
//! crates for efficient matching. Otherwise, falls back to a simple
//! `std::fs`-based recursive walk.

use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;

use crate::types::tool::{
    Tool, ToolError, ToolInputSchema, ToolProgressSender, ToolResult, ToolUseContext,
};

// ---------------------------------------------------------------------------
// GlobTool
// ---------------------------------------------------------------------------

/// Find files matching a glob pattern.
///
/// # Input schema
///
/// ```json
/// {
///   "pattern": "string (required) — glob pattern, e.g. '**/*.rs'",
///   "path": "string (optional) — directory to search in, defaults to working dir"
/// }
/// ```
///
/// # Output
///
/// A newline-separated list of matching file paths, sorted by modification time
/// (newest first). Directories, hidden files inside `.git`, and common build
/// artifacts are excluded.
#[derive(Debug, Clone, Copy)]
pub struct GlobTool;

/// Maximum number of results to return.
const MAX_RESULTS: usize = 1000;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Fast file pattern matching tool that works with any codebase size. \
         Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\". \
         Returns matching file paths sorted by modification time (newest first)."
    }

    fn input_schema(&self) -> ToolInputSchema {
        ToolInputSchema {
            schema_type: "object".to_string(),
            properties: {
                let mut m = serde_json::Map::new();
                m.insert(
                    "pattern".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The glob pattern to match files against"
                    }),
                );
                m.insert(
                    "path".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The directory to search in (defaults to working directory)"
                    }),
                );
                m
            },
            required: vec!["pattern".to_string()],
            additional_properties: false,
        }
    }

    fn aliases(&self) -> &[&str] {
        &["glob", "find_files"]
    }

    fn search_hint(&self) -> Option<&str> {
        Some("find files glob pattern match search directory")
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        true
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        true
    }

    fn user_facing_name(&self, input: &Value) -> String {
        if let Some(pattern) = input.get("pattern").and_then(|v| v.as_str()) {
            format!("Glob({})", pattern)
        } else {
            "Glob".to_string()
        }
    }

    async fn validate_input(&self, input: &Value, _ctx: &ToolUseContext) -> Result<(), String> {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field 'pattern' (must be a string)")?;

        if pattern.trim().is_empty() {
            return Err("'pattern' must not be empty".to_string());
        }

        Ok(())
    }

    async fn call(
        &self,
        input: Value,
        ctx: &mut ToolUseContext,
        _progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'pattern' field".to_string()))?;

        let search_path = if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
            ctx.resolve_path(path)?
        } else {
            ctx.working_dir.clone()
        };

        debug!(pattern = %pattern, path = %search_path.display(), "Glob search");

        if !search_path.exists() {
            return Ok(ToolResult::error(format!(
                "Directory not found: {}",
                search_path.display()
            )));
        }

        if !search_path.is_dir() {
            return Ok(ToolResult::error(format!(
                "'{}' is not a directory",
                search_path.display()
            )));
        }

        // Run the glob in a blocking task to avoid blocking the async runtime.
        let pattern_owned = pattern.to_string();
        let search_path_clone = search_path.clone();
        let results = tokio::task::spawn_blocking(move || {
            glob_search(&pattern_owned, &search_path_clone)
        })
        .await
        .map_err(|e| ToolError::Execution(format!("Glob task failed: {}", e)))?
        .map_err(|e| ToolError::Execution(e))?;

        if results.is_empty() {
            return Ok(ToolResult::text(format!(
                "No files found matching pattern '{}' in '{}'",
                pattern,
                search_path.display()
            )));
        }

        let total = results.len();
        let displayed: Vec<_> = results.into_iter().take(MAX_RESULTS).collect();
        let mut output = displayed.join("\n");

        if total > MAX_RESULTS {
            output.push_str(&format!(
                "\n\n... ({} more files not shown, {} total)",
                total - MAX_RESULTS,
                total
            ));
        }

        Ok(ToolResult::text(output))
    }
}

// ---------------------------------------------------------------------------
// Glob implementation (feature-gated)
// ---------------------------------------------------------------------------

/// Perform the glob search. Returns file paths as strings, sorted by mtime
/// (newest first).
#[cfg(feature = "built-in-tools")]
fn glob_search(
    pattern: &str,
    search_path: &std::path::Path,
) -> Result<Vec<String>, String> {
    use ignore::WalkBuilder;
    use globset::{Glob, GlobMatcher};

    let glob = Glob::new(pattern)
        .map_err(|e| format!("Invalid glob pattern '{}': {}", pattern, e))?;
    let matcher: GlobMatcher = glob.compile_matcher();

    let mut entries: Vec<(String, std::time::SystemTime)> = Vec::new();

    let walker = WalkBuilder::new(search_path)
        .hidden(false) // include hidden files
        .git_ignore(true) // respect .gitignore
        .git_global(true)
        .git_exclude(true)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip directories
        if entry.file_type().map_or(true, |ft| ft.is_dir()) {
            continue;
        }

        let path = entry.path();

        // Match against the pattern using the relative path from search_path
        let relative = path
            .strip_prefix(search_path)
            .unwrap_or(path);

        if matcher.is_match(relative) || matcher.is_match(path) {
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

            entries.push((path.to_string_lossy().to_string(), mtime));
        }
    }

    // Sort by modification time, newest first
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(entries.into_iter().map(|(path, _)| path).collect())
}

/// Fallback glob search using `std::fs` when the `built-in-tools` feature is
/// not enabled. Only supports simple `*` and `**` patterns.
#[cfg(not(feature = "built-in-tools"))]
fn glob_search(
    pattern: &str,
    search_path: &std::path::Path,
) -> Result<Vec<String>, String> {
    let mut entries: Vec<(String, std::time::SystemTime)> = Vec::new();

    // Simple pattern matching: convert glob to a basic check
    let pattern_parts: Vec<&str> = pattern.split('/').collect();

    fn walk_dir(
        dir: &std::path::Path,
        base: &std::path::Path,
        pattern: &str,
        entries: &mut Vec<(String, std::time::SystemTime)>,
    ) {
        let read_dir = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let relative = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            // Skip .git directories
            if relative.contains(".git/") || relative == ".git" {
                continue;
            }

            if path.is_dir() {
                walk_dir(&path, base, pattern, entries);
            } else if simple_glob_match(pattern, &relative) {
                let mtime = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                entries.push((path.to_string_lossy().to_string(), mtime));
            }
        }
    }

    walk_dir(search_path, search_path, pattern, &mut entries);

    // Sort by modification time, newest first
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(entries.into_iter().map(|(path, _)| path).collect())
}

/// Very simple glob matching. Supports `*` (any within segment) and `**`
/// (any path segments). This is intentionally basic — the full `globset`
/// crate is used when the feature is enabled.
#[cfg(not(feature = "built-in-tools"))]
fn simple_glob_match(pattern: &str, path: &str) -> bool {
    // Handle ** pattern (match any path)
    if pattern.starts_with("**/") {
        let suffix = &pattern[3..];
        // Try matching the suffix against the path and all sub-paths
        if simple_glob_match(suffix, path) {
            return true;
        }
        // Try stripping directory components
        let mut remaining = path;
        while let Some(pos) = remaining.find('/') {
            remaining = &remaining[pos + 1..];
            if simple_glob_match(suffix, remaining) {
                return true;
            }
        }
        return false;
    }

    // Handle trailing /**
    if pattern.ends_with("/**") {
        let prefix = &pattern[..pattern.len() - 3];
        return path.starts_with(prefix) || path == prefix;
    }

    // Handle * within a single segment
    if pattern.contains('*') && !pattern.contains("**") {
        // Split on * and check if path matches the parts in order
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            let (prefix, suffix) = (parts[0], parts[1]);
            if prefix.contains('/') || suffix.contains('/') {
                // Multi-segment pattern with single *
                return path.starts_with(prefix) && path.ends_with(suffix);
            }
            // Single segment: match filename
            let filename = path.rsplit('/').next().unwrap_or(path);
            return filename.starts_with(prefix) && filename.ends_with(suffix);
        }
    }

    // Exact match
    path == pattern
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
        let tool = GlobTool;
        assert_eq!(tool.name(), "Glob");
        assert!(tool.is_read_only(&Value::Null));
        assert!(tool.is_concurrency_safe(&Value::Null));
    }

    #[tokio::test]
    async fn glob_nonexistent_dir() {
        let tool = GlobTool;
        let mut ctx = test_ctx_with_dir(std::path::Path::new("/tmp"));
        let result = tool
            .call(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": "/tmp/nonexistent_dir_xyz"
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("foo.txt"), "").unwrap();
        std::fs::write(dir.path().join("bar.txt"), "").unwrap();
        std::fs::write(dir.path().join("baz.rs"), "").unwrap();

        let tool = GlobTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        let result = tool
            .call(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": dir.path().to_str().unwrap()
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("foo.txt"));
                assert!(text.contains("bar.txt"));
                assert!(!text.contains("baz.rs"));
            }
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn glob_recursive_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(dir.path().join("top.rs"), "").unwrap();
        std::fs::write(sub.join("nested.rs"), "").unwrap();

        let tool = GlobTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        let result = tool
            .call(
                serde_json::json!({
                    "pattern": "**/*.rs",
                    "path": dir.path().to_str().unwrap()
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("top.rs"));
                assert!(text.contains("nested.rs"));
            }
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn glob_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();

        let tool = GlobTool;
        let mut ctx = test_ctx_with_dir(dir.path());
        let result = tool
            .call(
                serde_json::json!({
                    "pattern": "*.xyz",
                    "path": dir.path().to_str().unwrap()
                }),
                &mut ctx,
                None,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("No files found"));
            }
            _ => panic!("expected text"),
        }
    }

    #[cfg(not(feature = "built-in-tools"))]
    mod fallback_tests {
        use super::super::simple_glob_match;

        #[test]
        fn simple_glob_star() {
            assert!(simple_glob_match("*.rs", "main.rs"));
            assert!(simple_glob_match("*.rs", "lib.rs"));
            assert!(!simple_glob_match("*.rs", "main.txt"));
        }

        #[test]
        fn simple_glob_double_star() {
            assert!(simple_glob_match("**/*.rs", "src/main.rs"));
            assert!(simple_glob_match("**/*.rs", "src/tools/mod.rs"));
            assert!(simple_glob_match("**/*.rs", "main.rs"));
            assert!(!simple_glob_match("**/*.rs", "main.txt"));
        }
    }
}
