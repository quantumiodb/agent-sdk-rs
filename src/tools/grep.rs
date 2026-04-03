//! GrepTool — search file contents using regex patterns.
//!
//! Uses the `grep-regex` and `grep-searcher` crates when the `built-in-tools`
//! feature is enabled; falls back to a pure-Rust line-by-line scan otherwise.

use std::path::Path;

use async_trait::async_trait;
use serde_json::Value;

use crate::types::{
    InterruptBehavior, PermissionCheckResult, Tool, ToolError, ToolInputSchema, ToolResult,
    ToolUseContext,
};

/// Search file contents using a regular expression.
#[derive(Debug)]
pub struct GrepTool;

impl GrepTool {
    fn schema() -> serde_json::Map<String, Value> {
        serde_json::from_value(serde_json::json!({
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "Directory or file path to search (default: current working directory)"
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. \"*.rs\", \"**/*.ts\")"
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode: 'content' (matching lines), 'files_with_matches' (paths only), 'count'"
            },
            "context": {
                "type": "integer",
                "description": "Number of context lines to show before and after each match (output_mode=content only)"
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Whether to match case-insensitively"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 250)"
            }
        }))
        .unwrap()
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        "Search file contents using a regular expression. Returns matching lines, \
         file paths, or counts depending on output_mode. Supports glob filtering."
    }

    fn input_schema(&self) -> ToolInputSchema {
        ToolInputSchema {
            schema_type: "object".into(),
            properties: Self::schema(),
            required: vec!["pattern".into()],
            additional_properties: false,
        }
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        true
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        true
    }

    fn interrupt_behavior(&self) -> InterruptBehavior {
        InterruptBehavior::Cancel
    }

    fn search_hint(&self) -> Option<&str> {
        Some("search file contents regex pattern match")
    }

    async fn check_permissions(
        &self,
        input: &Value,
        _ctx: &ToolUseContext,
    ) -> PermissionCheckResult {
        PermissionCheckResult::Allow {
            updated_input: input.clone(),
        }
    }

    async fn call(
        &self,
        input: Value,
        ctx: &mut ToolUseContext,
        _progress: Option<&dyn crate::types::ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let pattern = input["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("pattern is required".into()))?;

        let search_path = input["path"]
            .as_str()
            .map(|p| ctx.resolve_path(p))
            .transpose()?
            .unwrap_or_else(|| ctx.working_dir.clone());

        let glob_pattern = input["glob"].as_str().unwrap_or("*");
        let output_mode = input["output_mode"].as_str().unwrap_or("content");
        let context_lines = input["context"].as_u64().unwrap_or(0) as usize;
        let case_insensitive = input["case_insensitive"].as_bool().unwrap_or(false);
        let max_results = input["max_results"].as_u64().unwrap_or(250) as usize;

        #[cfg(feature = "built-in-tools")]
        {
            grep_with_ripgrep(
                pattern,
                &search_path,
                glob_pattern,
                output_mode,
                context_lines,
                case_insensitive,
                max_results,
            )
            .await
        }

        #[cfg(not(feature = "built-in-tools"))]
        {
            grep_fallback(
                pattern,
                &search_path,
                glob_pattern,
                output_mode,
                context_lines,
                case_insensitive,
                max_results,
            )
            .await
        }
    }
}

// ---------------------------------------------------------------------------
// ripgrep-based implementation (built-in-tools feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "built-in-tools")]
async fn grep_with_ripgrep(
    pattern: &str,
    search_path: &Path,
    glob_pattern: &str,
    output_mode: &str,
    context_lines: usize,
    case_insensitive: bool,
    max_results: usize,
) -> Result<ToolResult, ToolError> {
    use grep_regex::RegexMatcherBuilder;
    use grep_searcher::{SearcherBuilder, Sink, SinkContext, SinkContextKind, SinkMatch};
    use ignore::WalkBuilder;

    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(case_insensitive)
        .build(pattern)
        .map_err(|e| ToolError::InvalidInput(format!("Invalid regex: {e}")))?;

    let glob_set = {
        let mut builder = globset::GlobSetBuilder::new();
        builder
            .add(
                globset::Glob::new(glob_pattern)
                    .map_err(|e| ToolError::InvalidInput(format!("Invalid glob: {e}")))?,
            )
            .build()
            .map_err(|e| ToolError::InvalidInput(format!("Glob build error: {e}")))?
    };

    let mut results: Vec<String> = Vec::new();
    let mut match_count = 0usize;

    let walker = WalkBuilder::new(search_path)
        .hidden(false)
        .git_ignore(true)
        .build();

    'walk: for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.file_type().map(|ft| !ft.is_file()).unwrap_or(true) {
            continue;
        }

        let file_name = entry
            .file_name()
            .to_str()
            .unwrap_or("")
            .to_string();

        // Apply glob filter
        if glob_pattern != "*" && !glob_set.is_match(&file_name) {
            // Also try matching the full path
            let path_str = entry.path().to_string_lossy();
            if !glob_set.is_match(path_str.as_ref()) {
                continue;
            }
        }

        let path = entry.path().to_path_buf();

        match output_mode {
            "files_with_matches" => {
                // Check if the file has any match
                struct FileSink {
                    matched: bool,
                }
                impl Sink for FileSink {
                    type Error = std::io::Error;
                    fn matched(
                        &mut self,
                        _: &grep_searcher::Searcher,
                        _: &SinkMatch<'_>,
                    ) -> Result<bool, Self::Error> {
                        self.matched = true;
                        Ok(false) // stop after first match
                    }
                }

                let mut sink = FileSink { matched: false };
                let mut searcher = SearcherBuilder::new().build();
                let _ = searcher.search_path(&matcher, &path, &mut sink);
                if sink.matched {
                    results.push(path.display().to_string());
                    match_count += 1;
                    if match_count >= max_results {
                        break 'walk;
                    }
                }
            }
            "count" => {
                struct CountSink(usize);
                impl Sink for CountSink {
                    type Error = std::io::Error;
                    fn matched(
                        &mut self,
                        _: &grep_searcher::Searcher,
                        _: &SinkMatch<'_>,
                    ) -> Result<bool, Self::Error> {
                        self.0 += 1;
                        Ok(true)
                    }
                }

                let mut sink = CountSink(0);
                let mut searcher = SearcherBuilder::new().build();
                let _ = searcher.search_path(&matcher, &path, &mut sink);
                if sink.0 > 0 {
                    results.push(format!("{}: {}", path.display(), sink.0));
                    match_count += 1;
                    if match_count >= max_results {
                        break 'walk;
                    }
                }
            }
            _ => {
                // "content" mode with optional context
                struct ContentSink {
                    path: String,
                    lines: Vec<String>,
                    context_lines: usize,
                    match_count: usize,
                    max_results: usize,
                }

                impl Sink for ContentSink {
                    type Error = std::io::Error;

                    fn matched(
                        &mut self,
                        _searcher: &grep_searcher::Searcher,
                        mat: &SinkMatch<'_>,
                    ) -> Result<bool, Self::Error> {
                        if self.match_count >= self.max_results {
                            return Ok(false);
                        }
                        let line_num = mat.line_number().unwrap_or(0);
                        let text = String::from_utf8_lossy(mat.bytes()).trim_end().to_string();
                        self.lines
                            .push(format!("{}:{}:{}", self.path, line_num, text));
                        self.match_count += 1;
                        Ok(self.match_count < self.max_results)
                    }

                    fn context(
                        &mut self,
                        _searcher: &grep_searcher::Searcher,
                        ctx: &SinkContext<'_>,
                    ) -> Result<bool, Self::Error> {
                        if self.context_lines > 0 {
                            let line_num = ctx.line_number().unwrap_or(0);
                            let text =
                                String::from_utf8_lossy(ctx.bytes()).trim_end().to_string();
                            let kind = match ctx.kind() {
                                SinkContextKind::Before => "before",
                                SinkContextKind::After => "after",
                                _ => "ctx",
                            };
                            self.lines.push(format!(
                                "{}:{}:({}) {}",
                                self.path, line_num, kind, text
                            ));
                        }
                        Ok(true)
                    }
                }

                let mut sink = ContentSink {
                    path: path.display().to_string(),
                    lines: Vec::new(),
                    context_lines,
                    match_count: 0,
                    max_results,
                };

                let mut searcher = SearcherBuilder::new()
                    .line_number(true)
                    .before_context(context_lines)
                    .after_context(context_lines)
                    .build();

                let _ = searcher.search_path(&matcher, &path, &mut sink);
                if !sink.lines.is_empty() {
                    match_count += sink.lines.len();
                    results.extend(sink.lines);
                    if match_count >= max_results {
                        break 'walk;
                    }
                }
            }
        }
    }

    if results.is_empty() {
        return Ok(ToolResult::text("No matches found."));
    }

    let mut output = results.join("\n");
    if match_count >= max_results {
        output.push_str(&format!(
            "\n... (results truncated at {} matches)",
            max_results
        ));
    }

    Ok(ToolResult::text(output))
}

// ---------------------------------------------------------------------------
// Fallback implementation (no built-in-tools feature)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "built-in-tools"))]
async fn grep_fallback(
    pattern: &str,
    search_path: &Path,
    glob_pattern: &str,
    output_mode: &str,
    _context_lines: usize,
    case_insensitive: bool,
    max_results: usize,
) -> Result<ToolResult, ToolError> {
    use std::io::{BufRead, BufReader};

    let re = {
        let p = if case_insensitive {
            format!("(?i){pattern}")
        } else {
            pattern.to_string()
        };
        regex::Regex::new(&p)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid regex: {e}")))?
    };

    let files = collect_files(search_path, glob_pattern)?;
    let mut results: Vec<String> = Vec::new();
    let mut match_count = 0usize;

    'files: for file_path in &files {
        let file = match std::fs::File::open(file_path) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let reader = BufReader::new(file);
        let mut file_matched = false;
        let mut file_count = 0usize;

        for (line_no, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };

            if re.is_match(&line) {
                file_matched = true;
                file_count += 1;

                if output_mode == "content" {
                    results.push(format!("{}:{}:{}", file_path.display(), line_no + 1, line));
                    match_count += 1;
                    if match_count >= max_results {
                        break 'files;
                    }
                }
            }
        }

        match output_mode {
            "files_with_matches" if file_matched => {
                results.push(file_path.display().to_string());
                match_count += 1;
                if match_count >= max_results {
                    break 'files;
                }
            }
            "count" if file_count > 0 => {
                results.push(format!("{}: {}", file_path.display(), file_count));
                match_count += 1;
                if match_count >= max_results {
                    break 'files;
                }
            }
            _ => {}
        }
    }

    if results.is_empty() {
        return Ok(ToolResult::text("No matches found."));
    }

    Ok(ToolResult::text(results.join("\n")))
}

/// Simple recursive file collector respecting a glob-like extension filter.
#[allow(dead_code)]
fn collect_files(
    root: &Path,
    glob_pattern: &str,
) -> Result<Vec<std::path::PathBuf>, ToolError> {
    let mut files = Vec::new();

    if root.is_file() {
        files.push(root.to_path_buf());
        return Ok(files);
    }

    fn walk(dir: &Path, glob: &str, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden directories and common vendor dirs
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.starts_with('.') || name == "node_modules" || name == "target" {
                    continue;
                }
                walk(&path, glob, out)?;
            } else if path.is_file() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if glob == "*" || name.ends_with(&glob.replace("*.", ".")) {
                    out.push(path);
                }
            }
        }
        Ok(())
    }

    walk(root, glob_pattern, &mut files).map_err(ToolError::Io)?;
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grep_tool_name() {
        assert_eq!(GrepTool.name(), "Grep");
    }

    #[test]
    fn grep_tool_is_read_only() {
        assert!(GrepTool.is_read_only(&Value::Null));
    }

    #[test]
    fn grep_tool_is_concurrency_safe() {
        assert!(GrepTool.is_concurrency_safe(&Value::Null));
    }

    #[test]
    fn grep_schema_has_pattern() {
        let schema = GrepTool.input_schema();
        assert!(schema.properties.contains_key("pattern"));
        assert!(schema.required.contains(&"pattern".to_string()));
    }

    #[tokio::test]
    async fn grep_no_match_returns_friendly_message() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "hello world\nfoo bar\n")
            .await
            .unwrap();

        let mut ctx = ToolUseContext {
            working_dir: dir.path().to_path_buf(),
            cancellation: tokio_util::sync::CancellationToken::new(),
            read_file_state: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::<String>::new(),
            )),
            session_id: "test".into(),
            agent_id: None,
        };

        let input = serde_json::json!({
            "pattern": "zzznomatch",
            "path": dir.path().to_str().unwrap()
        });

        let result = GrepTool.call(input, &mut ctx, None).await.unwrap();
        assert!(!result.is_error);
        let text = match &result.content[0] {
            crate::types::ToolResultContent::Text { text } => text.clone(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("No matches found"));
    }

    #[tokio::test]
    async fn grep_finds_match_in_file() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let file = dir.path().join("src.rs");
        tokio::fs::write(&file, "fn main() {\n    println!(\"hello\");\n}\n")
            .await
            .unwrap();

        let mut ctx = ToolUseContext {
            working_dir: dir.path().to_path_buf(),
            cancellation: tokio_util::sync::CancellationToken::new(),
            read_file_state: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::<String>::new(),
            )),
            session_id: "test".into(),
            agent_id: None,
        };

        let input = serde_json::json!({
            "pattern": "println",
            "path": file.to_str().unwrap()
        });

        let result = GrepTool.call(input, &mut ctx, None).await.unwrap();
        assert!(!result.is_error);
        let text = match &result.content[0] {
            crate::types::ToolResultContent::Text { text } => text.clone(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("println"));
    }
}
