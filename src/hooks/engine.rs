//! Hook engine: event dispatch and tool-name pattern matching.
//!
//! The hook system lets callers intercept tool execution at four lifecycle
//! points: before a tool runs, after it succeeds, after it fails, and when
//! the agent is about to stop.

use std::sync::Arc;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::types::HookError;
use crate::types::tool::ToolResult;

// ---------------------------------------------------------------------------
// HookEvent
// ---------------------------------------------------------------------------

/// Hook event types, aligned with the core hook events in coreSchemas.ts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    /// Fires before a tool is executed. The handler can block execution.
    PreToolUse,
    /// Fires after a tool executes successfully.
    PostToolUse,
    /// Fires after a tool execution fails.
    PostToolUseFailure,
    /// Fires when the agent is about to stop (end_turn).
    Stop,
}

// ---------------------------------------------------------------------------
// HookInput / HookOutput
// ---------------------------------------------------------------------------

/// Data passed into a hook handler.
#[derive(Debug, Clone, Serialize)]
pub struct HookInput {
    /// Which lifecycle event triggered this hook.
    pub event: HookEvent,
    /// The active session identifier.
    pub session_id: String,
    /// The name of the tool being invoked (None for Stop events).
    pub tool_name: Option<String>,
    /// The tool's input JSON (None for Stop events).
    pub tool_input: Option<Value>,
    /// The tool's result JSON (only for PostToolUse / PostToolUseFailure).
    pub tool_result: Option<Value>,
}

/// Data returned from a hook handler.
#[derive(Debug, Clone, Deserialize)]
pub struct HookOutput {
    /// If `false`, the triggering action is blocked (tool is not executed,
    /// or the agent does not stop).
    #[serde(default = "default_true")]
    pub r#continue: bool,
    /// An optional system message to inject into the conversation.
    pub system_message: Option<String>,
    /// Human-readable reason for blocking (when `continue` is false).
    pub reason: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Default for HookOutput {
    fn default() -> Self {
        Self {
            r#continue: true,
            system_message: None,
            reason: None,
        }
    }
}

// ---------------------------------------------------------------------------
// HookFn / HookRule / HookConfig
// ---------------------------------------------------------------------------

/// An async hook handler function.
///
/// Receives a [`HookInput`] and returns a [`HookOutput`] that may block the
/// action or inject a system message.
pub type HookFn = Arc<dyn Fn(HookInput) -> BoxFuture<'static, HookOutput> + Send + Sync>;

/// A single hook rule: a tool-name matcher paired with a handler function.
pub struct HookRule {
    /// Tool name pattern to match against. Supports:
    /// - `None` — matches all tools
    /// - `"*"` — matches all tools
    /// - `"Bash"` — exact match
    /// - `"Bash|Edit|Write"` — pipe-separated alternatives
    /// - `"mcp__*"` — prefix wildcard
    pub matcher: Option<String>,
    /// The handler to invoke when the matcher hits.
    pub handler: HookFn,
}

// Manual Debug because HookFn is not Debug.
impl std::fmt::Debug for HookRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRule")
            .field("matcher", &self.matcher)
            .field("handler", &"<HookFn>")
            .finish()
    }
}

/// Configuration holding all registered hooks, grouped by event type.
pub struct HookConfig {
    /// Hooks that fire before a tool executes.
    pub pre_tool_use: Vec<HookRule>,
    /// Hooks that fire after a tool executes successfully.
    pub post_tool_use: Vec<HookRule>,
    /// Hooks that fire after a tool execution fails.
    pub post_tool_use_failure: Vec<HookRule>,
    /// Hooks that fire when the agent is about to stop.
    pub stop: Vec<HookRule>,
}

impl std::fmt::Debug for HookConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookConfig")
            .field("pre_tool_use", &self.pre_tool_use.len())
            .field("post_tool_use", &self.post_tool_use.len())
            .field("post_tool_use_failure", &self.post_tool_use_failure.len())
            .field("stop", &self.stop.len())
            .finish()
    }
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            pre_tool_use: Vec::new(),
            post_tool_use: Vec::new(),
            post_tool_use_failure: Vec::new(),
            stop: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// HookEngine
// ---------------------------------------------------------------------------

/// The hook engine dispatches lifecycle events to registered hook handlers
/// and evaluates tool-name patterns.
pub struct HookEngine {
    config: HookConfig,
}

impl std::fmt::Debug for HookEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookEngine")
            .field("config", &self.config)
            .finish()
    }
}

impl HookEngine {
    /// Create a new hook engine from the given configuration.
    pub fn new(config: HookConfig) -> Self {
        Self { config }
    }

    /// Run all `PreToolUse` hooks for the given tool invocation.
    ///
    /// Returns the first blocking output (where `continue` is `false`), or a
    /// default "continue" output if all hooks pass.
    pub async fn run_pre_tool_use(
        &self,
        session_id: &str,
        tool_name: &str,
        tool_input: &Value,
    ) -> Result<HookOutput, HookError> {
        let input = HookInput {
            event: HookEvent::PreToolUse,
            session_id: session_id.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_result: None,
        };

        for rule in &self.config.pre_tool_use {
            if matches_tool(&rule.matcher, tool_name) {
                let output = (rule.handler)(input.clone()).await;
                if !output.r#continue {
                    return Ok(output);
                }
            }
        }

        Ok(HookOutput::default())
    }

    /// Run all `PostToolUse` hooks for a successful tool execution.
    pub async fn run_post_tool_use(
        &self,
        session_id: &str,
        tool_name: &str,
        tool_input: &Value,
        tool_result: &ToolResult,
    ) -> Result<HookOutput, HookError> {
        let result_value = serde_json::to_value(tool_result).unwrap_or_default();
        let input = HookInput {
            event: HookEvent::PostToolUse,
            session_id: session_id.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_result: Some(result_value),
        };

        for rule in &self.config.post_tool_use {
            if matches_tool(&rule.matcher, tool_name) {
                let output = (rule.handler)(input.clone()).await;
                if !output.r#continue {
                    return Ok(output);
                }
            }
        }

        Ok(HookOutput::default())
    }

    /// Run all `PostToolUseFailure` hooks for a failed tool execution.
    pub async fn run_post_tool_use_failure(
        &self,
        session_id: &str,
        tool_name: &str,
        tool_input: &Value,
        error_message: &str,
    ) -> Result<HookOutput, HookError> {
        let input = HookInput {
            event: HookEvent::PostToolUseFailure,
            session_id: session_id.to_string(),
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_result: Some(serde_json::json!({ "error": error_message })),
        };

        for rule in &self.config.post_tool_use_failure {
            if matches_tool(&rule.matcher, tool_name) {
                let output = (rule.handler)(input.clone()).await;
                if !output.r#continue {
                    return Ok(output);
                }
            }
        }

        Ok(HookOutput::default())
    }

    /// Run all `Stop` hooks. If any returns `continue: false`, the agent
    /// should not stop and instead continue the conversation.
    pub async fn run_stop(&self, session_id: &str) -> Result<HookOutput, HookError> {
        for rule in &self.config.stop {
            let input = HookInput {
                event: HookEvent::Stop,
                session_id: session_id.to_string(),
                tool_name: None,
                tool_input: None,
                tool_result: None,
            };
            let output = (rule.handler)(input).await;
            if !output.r#continue {
                return Ok(output);
            }
        }

        Ok(HookOutput::default())
    }
}

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------

/// Match a tool name against a hook matcher pattern.
///
/// Supported patterns:
/// - `None` — matches everything
/// - `"*"` — matches everything
/// - `"Bash"` — exact match on a single name
/// - `"Bash|Edit|Write"` — pipe-separated list of exact names
/// - `"mcp__*"` — prefix wildcard (matches any name starting with `"mcp__"`)
///
/// Patterns are case-sensitive.
pub fn matches_tool(matcher: &Option<String>, tool_name: &str) -> bool {
    match matcher {
        None => true,
        Some(pattern) => {
            if pattern == "*" {
                return true;
            }
            for part in pattern.split('|') {
                let part = part.trim();
                if part == tool_name {
                    return true;
                }
                if part.ends_with('*') && tool_name.starts_with(&part[..part.len() - 1]) {
                    return true;
                }
            }
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_tool_none_matches_all() {
        assert!(matches_tool(&None, "Bash"));
        assert!(matches_tool(&None, "anything"));
    }

    #[test]
    fn matches_tool_star_matches_all() {
        let pat = Some("*".to_string());
        assert!(matches_tool(&pat, "Bash"));
        assert!(matches_tool(&pat, "Edit"));
        assert!(matches_tool(&pat, "mcp__server__tool"));
    }

    #[test]
    fn matches_tool_exact() {
        let pat = Some("Bash".to_string());
        assert!(matches_tool(&pat, "Bash"));
        assert!(!matches_tool(&pat, "Edit"));
        assert!(!matches_tool(&pat, "BashExtra"));
    }

    #[test]
    fn matches_tool_pipe_separated() {
        let pat = Some("Bash|Edit|Write".to_string());
        assert!(matches_tool(&pat, "Bash"));
        assert!(matches_tool(&pat, "Edit"));
        assert!(matches_tool(&pat, "Write"));
        assert!(!matches_tool(&pat, "Read"));
        assert!(!matches_tool(&pat, "Glob"));
    }

    #[test]
    fn matches_tool_prefix_wildcard() {
        let pat = Some("mcp__*".to_string());
        assert!(matches_tool(&pat, "mcp__server__tool"));
        assert!(matches_tool(&pat, "mcp__fs__read"));
        assert!(!matches_tool(&pat, "Bash"));
        assert!(!matches_tool(&pat, "mcp_bad"));
    }

    #[test]
    fn matches_tool_pipe_with_wildcard() {
        let pat = Some("Bash | mcp__*".to_string());
        assert!(matches_tool(&pat, "Bash"));
        assert!(matches_tool(&pat, "mcp__server__tool"));
        assert!(!matches_tool(&pat, "Edit"));
    }

    #[tokio::test]
    async fn hook_engine_pre_tool_use_allows_by_default() {
        let engine = HookEngine::new(HookConfig::default());
        let output = engine
            .run_pre_tool_use("sess", "Bash", &serde_json::json!({"command": "ls"}))
            .await
            .unwrap();
        assert!(output.r#continue);
    }

    #[tokio::test]
    async fn hook_engine_pre_tool_use_can_block() {
        let config = HookConfig {
            pre_tool_use: vec![HookRule {
                matcher: Some("Bash".to_string()),
                handler: Arc::new(|_input| {
                    Box::pin(async {
                        HookOutput {
                            r#continue: false,
                            system_message: None,
                            reason: Some("Blocked by test".to_string()),
                        }
                    })
                }),
            }],
            ..Default::default()
        };
        let engine = HookEngine::new(config);

        let output = engine
            .run_pre_tool_use("sess", "Bash", &serde_json::json!({"command": "rm -rf /"}))
            .await
            .unwrap();

        assert!(!output.r#continue);
        assert_eq!(output.reason.as_deref(), Some("Blocked by test"));
    }

    #[tokio::test]
    async fn hook_engine_pre_tool_use_skips_non_matching() {
        let config = HookConfig {
            pre_tool_use: vec![HookRule {
                matcher: Some("Bash".to_string()),
                handler: Arc::new(|_| {
                    Box::pin(async {
                        HookOutput {
                            r#continue: false,
                            reason: Some("Should not fire".to_string()),
                            ..Default::default()
                        }
                    })
                }),
            }],
            ..Default::default()
        };
        let engine = HookEngine::new(config);

        // "Edit" should not match the "Bash" rule.
        let output = engine
            .run_pre_tool_use("sess", "Edit", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(output.r#continue);
    }

    #[tokio::test]
    async fn hook_engine_stop_continues_by_default() {
        let engine = HookEngine::new(HookConfig::default());
        let output = engine.run_stop("sess").await.unwrap();
        assert!(output.r#continue);
    }

    #[tokio::test]
    async fn hook_engine_stop_can_prevent_stopping() {
        let config = HookConfig {
            stop: vec![HookRule {
                matcher: None,
                handler: Arc::new(|_| {
                    Box::pin(async {
                        HookOutput {
                            r#continue: false,
                            system_message: Some("Keep going".to_string()),
                            reason: None,
                        }
                    })
                }),
            }],
            ..Default::default()
        };
        let engine = HookEngine::new(config);
        let output = engine.run_stop("sess").await.unwrap();
        assert!(!output.r#continue);
        assert_eq!(output.system_message.as_deref(), Some("Keep going"));
    }
}
