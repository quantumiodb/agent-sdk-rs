//! BashTool — execute shell commands via `/bin/bash -c`.
//!
//! This tool runs arbitrary shell commands in the agent's working directory.
//! It captures stdout and stderr, respects an optional timeout (default 120s),
//! and is **not** concurrency-safe (commands may have arbitrary side effects).

use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;

use crate::types::tool::{
    InterruptBehavior, PermissionCheckResult, Tool, ToolError, ToolInputSchema,
    ToolProgressSender, ToolResult, ToolUseContext,
};

// ---------------------------------------------------------------------------
// BashTool
// ---------------------------------------------------------------------------

/// Execute a shell command via `/bin/bash -c`.
///
/// # Input schema
///
/// ```json
/// {
///   "command": "string (required) — the shell command to run",
///   "timeout": "number (optional) — timeout in milliseconds, default 120000"
/// }
/// ```
///
/// # Behavior
///
/// - The command runs in the context's `working_dir`.
/// - stdout and stderr are merged in the output.
/// - The tool respects cancellation via the context's cancellation token.
/// - If the command exits with a non-zero code, the result includes the code.
#[derive(Debug, Clone, Copy)]
pub struct BashTool;

/// Default timeout in milliseconds (120 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// Maximum output size in bytes before truncation.
const MAX_OUTPUT_BYTES: usize = 512_000;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }

    fn description(&self) -> &str {
        "Executes a given bash command and returns its output. \
         The working directory persists between commands. \
         Use for running shell commands, scripts, and system operations."
    }

    fn input_schema(&self) -> ToolInputSchema {
        ToolInputSchema {
            schema_type: "object".to_string(),
            properties: {
                let mut m = serde_json::Map::new();
                m.insert(
                    "command".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The bash command to execute"
                    }),
                );
                m.insert(
                    "timeout".to_string(),
                    serde_json::json!({
                        "type": "number",
                        "description": "Optional timeout in milliseconds (max 600000, default 120000)"
                    }),
                );
                m
            },
            required: vec!["command".to_string()],
            additional_properties: false,
        }
    }

    fn aliases(&self) -> &[&str] {
        &["bash", "shell"]
    }

    fn search_hint(&self) -> Option<&str> {
        Some("run execute shell bash command terminal")
    }

    fn is_read_only(&self, input: &Value) -> bool {
        // Heuristic: certain commands are read-only.
        if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
            let trimmed = cmd.trim();
            // Simple read-only command detection
            let read_only_prefixes = [
                "ls", "cat", "head", "tail", "echo", "pwd", "which", "whoami",
                "date", "env", "printenv", "uname", "hostname", "id", "df",
                "du", "wc", "sort", "uniq", "find", "grep", "rg", "ag",
                "git status", "git log", "git diff", "git show", "git branch",
                "git remote", "git tag",
            ];
            for prefix in &read_only_prefixes {
                if trimmed.starts_with(prefix) {
                    // Make sure it is not piped to a write command
                    if !trimmed.contains('>') && !trimmed.contains("rm ")
                        && !trimmed.contains("mv ") && !trimmed.contains("cp ")
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        // Shell commands can have arbitrary side effects.
        false
    }

    fn interrupt_behavior(&self) -> InterruptBehavior {
        InterruptBehavior::Cancel
    }

    async fn validate_input(&self, input: &Value, _ctx: &ToolUseContext) -> Result<(), String> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field 'command' (must be a string)")?;

        if command.trim().is_empty() {
            return Err("The 'command' field must not be empty".to_string());
        }

        // Validate timeout if provided
        if let Some(timeout) = input.get("timeout") {
            if let Some(ms) = timeout.as_u64() {
                if ms > 600_000 {
                    return Err("Timeout cannot exceed 600000ms (10 minutes)".to_string());
                }
            } else if let Some(ms) = timeout.as_f64() {
                if ms < 0.0 || ms > 600_000.0 {
                    return Err("Timeout must be between 0 and 600000ms".to_string());
                }
            } else {
                return Err("'timeout' must be a number".to_string());
            }
        }

        Ok(())
    }

    async fn check_permissions(
        &self,
        input: &Value,
        _ctx: &ToolUseContext,
    ) -> PermissionCheckResult {
        // Read-only commands can be auto-allowed; write commands need review.
        // The actual permission decision is made at the global layer; here
        // we just pass through.
        PermissionCheckResult::Allow {
            updated_input: input.clone(),
        }
    }

    fn user_facing_name(&self, input: &Value) -> String {
        if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
            let short = if cmd.len() > 60 {
                format!("{}...", &cmd[..57])
            } else {
                cmd.to_string()
            };
            format!("Bash({})", short)
        } else {
            "Bash".to_string()
        }
    }

    async fn call(
        &self,
        input: Value,
        ctx: &mut ToolUseContext,
        _progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'command' field".to_string()))?;

        let timeout_ms = input
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_MS);

        debug!(command = %command, timeout_ms = %timeout_ms, "Executing bash command");

        // Build the child process
        let child = tokio::process::Command::new("/bin/bash")
            .arg("-c")
            .arg(command)
            .current_dir(&ctx.working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Avoid inheriting the agent's stdin
            .stdin(std::process::Stdio::null())
            .spawn()
            .map_err(|e| ToolError::Execution(format!("Failed to spawn bash: {}", e)))?;

        // Wait for completion with timeout and cancellation support
        let timeout_duration = std::time::Duration::from_millis(timeout_ms);
        let cancellation = ctx.cancellation.clone();

        // Capture PID before consuming child
        let pid = child.id();

        let kill_pid = |pid: Option<u32>| {
            if let Some(pid) = pid {
                let _ = std::process::Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
        };

        let output = tokio::select! {
            result = tokio::time::timeout(timeout_duration, child.wait_with_output()) => {
                match result {
                    Ok(Ok(output)) => output,
                    Ok(Err(e)) => return Err(ToolError::Execution(
                        format!("Failed to read command output: {}", e)
                    )),
                    Err(_) => {
                        kill_pid(pid);
                        return Err(ToolError::Timeout(timeout_ms));
                    }
                }
            }
            _ = cancellation.cancelled() => {
                kill_pid(pid);
                return Err(ToolError::Aborted);
            }
        };

        // Build result text
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        let mut result_text = String::new();

        if !stdout.is_empty() {
            let stdout_str = if stdout.len() > MAX_OUTPUT_BYTES {
                format!(
                    "{}...\n[stdout truncated: {} bytes total]",
                    &stdout[..MAX_OUTPUT_BYTES],
                    stdout.len()
                )
            } else {
                stdout.to_string()
            };
            result_text.push_str(&stdout_str);
        }

        if !stderr.is_empty() {
            if !result_text.is_empty() {
                result_text.push('\n');
            }
            let stderr_str = if stderr.len() > MAX_OUTPUT_BYTES {
                format!(
                    "{}...\n[stderr truncated: {} bytes total]",
                    &stderr[..MAX_OUTPUT_BYTES],
                    stderr.len()
                )
            } else {
                stderr.to_string()
            };
            result_text.push_str(&stderr_str);
        }

        if exit_code != 0 {
            if !result_text.is_empty() {
                result_text.push('\n');
            }
            result_text.push_str(&format!("(exit code: {})", exit_code));
        }

        if result_text.is_empty() {
            result_text = "(no output)".to_string();
        }

        if exit_code != 0 {
            Ok(ToolResult::error(result_text))
        } else {
            Ok(ToolResult::text(result_text))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tokio_util::sync::CancellationToken;

    fn test_ctx() -> ToolUseContext {
        ToolUseContext {
            working_dir: PathBuf::from("/tmp"),
            cancellation: CancellationToken::new(),
            read_file_state: Arc::new(RwLock::new(HashSet::new())),
            session_id: "test".to_string(),
            agent_id: None,
        }
    }

    #[test]
    fn tool_metadata() {
        let tool = BashTool;
        assert_eq!(tool.name(), "Bash");
        assert!(!tool.description().is_empty());
        assert!(tool.aliases().contains(&"bash"));
        assert!(tool.aliases().contains(&"shell"));
    }

    #[test]
    fn read_only_heuristic() {
        let tool = BashTool;
        assert!(tool.is_read_only(&serde_json::json!({"command": "ls -la"})));
        assert!(tool.is_read_only(&serde_json::json!({"command": "git status"})));
        assert!(tool.is_read_only(&serde_json::json!({"command": "cat foo.txt"})));
        assert!(!tool.is_read_only(&serde_json::json!({"command": "rm -rf /"})));
        assert!(!tool.is_read_only(&serde_json::json!({"command": "echo hello > file"})));
        // Piping ls to a file is not read-only
        assert!(!tool.is_read_only(&serde_json::json!({"command": "ls > out.txt"})));
    }

    #[tokio::test]
    async fn validate_missing_command() {
        let tool = BashTool;
        let ctx = test_ctx();
        let result = tool.validate_input(&serde_json::json!({}), &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn validate_empty_command() {
        let tool = BashTool;
        let ctx = test_ctx();
        let result = tool
            .validate_input(&serde_json::json!({"command": "  "}), &ctx)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn validate_timeout_too_large() {
        let tool = BashTool;
        let ctx = test_ctx();
        let result = tool
            .validate_input(
                &serde_json::json!({"command": "echo hi", "timeout": 700000}),
                &ctx,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_echo() {
        let tool = BashTool;
        let mut ctx = test_ctx();
        let result = tool
            .call(
                serde_json::json!({"command": "echo hello world"}),
                &mut ctx,
                None,
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("hello world"));
            }
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn execute_nonzero_exit() {
        let tool = BashTool;
        let mut ctx = test_ctx();
        let result = tool
            .call(
                serde_json::json!({"command": "exit 42"}),
                &mut ctx,
                None,
            )
            .await
            .unwrap();
        assert!(result.is_error);
        match &result.content[0] {
            crate::types::message::ToolResultContent::Text { text } => {
                assert!(text.contains("exit code: 42"));
            }
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn execute_with_timeout() {
        let tool = BashTool;
        let mut ctx = test_ctx();
        // Very short timeout for a sleep command
        let result = tool
            .call(
                serde_json::json!({"command": "sleep 30", "timeout": 100}),
                &mut ctx,
                None,
            )
            .await;
        assert!(matches!(result, Err(ToolError::Timeout(100))));
    }

    #[tokio::test]
    async fn execute_with_cancellation() {
        let tool = BashTool;
        let mut ctx = test_ctx();
        let cancel = ctx.cancellation.clone();

        // Cancel after a small delay
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            cancel.cancel();
        });

        let result = tool
            .call(
                serde_json::json!({"command": "sleep 30"}),
                &mut ctx,
                None,
            )
            .await;
        assert!(matches!(result, Err(ToolError::Aborted)));
    }
}
