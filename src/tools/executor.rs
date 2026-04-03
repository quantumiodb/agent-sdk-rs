//! Tool execution engine: parallel/serial execution with permission checks.
//!
//! The main entry point is [`execute_tools`], which partitions tool-use blocks
//! into concurrency-safe and sequential groups, runs the former in parallel
//! via `tokio::spawn`, and the latter serially.

use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::types::sdk_message::SDKMessage;
use crate::types::tool::{
    CanUseToolFn, PermissionCheckResult, PermissionDecision, Tool,
    ToolResult, ToolUseContext,
};

use super::registry::ToolRegistry;

// ---------------------------------------------------------------------------
// ToolUseInput — a single tool invocation request
// ---------------------------------------------------------------------------

/// Describes a single tool invocation extracted from an assistant message.
#[derive(Debug, Clone)]
pub struct ToolUseInput {
    /// Unique ID for this tool-use (matches the API `tool_use.id`).
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool input (arbitrary JSON).
    pub input: Value,
}

// ---------------------------------------------------------------------------
// Result truncation helper
// ---------------------------------------------------------------------------

/// Truncate the text content of a [`ToolResult`] if it exceeds `max_chars`.
fn truncate_result(result: &mut ToolResult, max_chars: usize) {
    use crate::types::message::ToolResultContent;

    for block in &mut result.content {
        if let ToolResultContent::Text { ref mut text } = block {
            if text.len() > max_chars {
                let truncated_msg = format!(
                    "\n\n... [truncated: {} chars total, showing first {}]",
                    text.len(),
                    max_chars,
                );
                text.truncate(max_chars);
                text.push_str(&truncated_msg);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// execute_tools — main entry point
// ---------------------------------------------------------------------------

/// Execute a batch of tool invocations, running concurrency-safe tools in
/// parallel and the rest serially.
///
/// # Arguments
///
/// * `tool_uses` — the tool invocations to execute.
/// * `ctx` — the shared execution context (working dir, cancellation, etc.).
/// * `registry` — the tool registry to look up tools by name.
/// * `can_use_tool` — global permission callback.
/// * `event_tx` — channel for streaming SDK events back to the caller.
///
/// # Returns
///
/// A `Vec<(String, ToolResult)>` where each entry pairs a tool-use ID with
/// its result, in the same order as `tool_uses`.
pub async fn execute_tools(
    tool_uses: &[ToolUseInput],
    ctx: &mut ToolUseContext,
    registry: &ToolRegistry,
    can_use_tool: &CanUseToolFn,
    event_tx: &mpsc::Sender<SDKMessage>,
) -> Vec<(String, ToolResult)> {
    // Partition into concurrent-safe and sequential.
    let (concurrent, sequential): (Vec<_>, Vec<_>) =
        tool_uses.iter().enumerate().partition(|(_, tu)| {
            registry
                .find(&tu.name)
                .map(|t| t.is_concurrency_safe(&tu.input))
                .unwrap_or(false)
        });

    let mut results: Vec<Option<(String, ToolResult)>> = vec![None; tool_uses.len()];

    // ─── Phase 1: run concurrency-safe tools in parallel ───
    if !concurrent.is_empty() {
        let mut handles = Vec::with_capacity(concurrent.len());

        for (idx, tu) in &concurrent {
            let idx = *idx;
            let tool = registry.find(&tu.name).cloned();
            let input = tu.input.clone();
            let id = tu.id.clone();
            let name = tu.name.clone();
            let ctx_clone = ctx.clone_for_concurrent();
            let can_use = can_use_tool.clone();
            let tx = event_tx.clone();

            handles.push((
                idx,
                id.clone(),
                tokio::spawn(async move {
                    execute_single(tool, &name, &id, input, ctx_clone, &can_use, &tx).await
                }),
            ));
        }

        for (idx, id, handle) in handles {
            let result = match handle.await {
                Ok(r) => r,
                Err(e) => {
                    warn!(tool_use_id = %id, "tokio task join error: {}", e);
                    ToolResult::error(format!("Internal error: task panicked: {}", e))
                }
            };
            results[idx] = Some((id, result));
        }
    }

    // ─── Phase 2: run non-concurrent-safe tools serially ───
    for (idx, tu) in sequential {
        let tool = registry.find(&tu.name).cloned();
        let result = execute_single(
            tool,
            &tu.name,
            &tu.id,
            tu.input.clone(),
            ctx.clone_for_concurrent(),
            can_use_tool,
            event_tx,
        )
        .await;
        results[idx] = Some((tu.id.clone(), result));
    }

    // Unwrap all Options (every slot was filled).
    results.into_iter().map(|r| r.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// execute_single — process one tool invocation
// ---------------------------------------------------------------------------

/// Execute a single tool invocation with full permission and validation checks.
///
/// The pipeline is:
/// 1. Global permission check (`can_use_tool`)
/// 2. Input validation (`tool.validate_input`)
/// 3. Tool-specific permission check (`tool.check_permissions`)
/// 4. Actual execution via [`execute_tool_inner`]
pub async fn execute_single(
    tool: Option<Arc<dyn Tool>>,
    name: &str,
    tool_use_id: &str,
    input: Value,
    mut ctx: ToolUseContext,
    can_use_tool: &CanUseToolFn,
    event_tx: &mpsc::Sender<SDKMessage>,
) -> ToolResult {
    let start = Instant::now();

    // ─── Unknown tool ───
    let tool = match tool {
        Some(t) => t,
        None => {
            warn!(tool_name = %name, "Unknown tool requested");
            return ToolResult::error(format!(
                "Unknown tool '{}'. Use ToolSearch to find available tools.",
                name,
            ));
        }
    };

    // ─── Step 1: global permission check ───
    match can_use_tool(name, &input) {
        PermissionDecision::Deny(msg) => {
            debug!(tool = %name, "Permission denied: {}", msg);
            return ToolResult::error(msg);
        }
        PermissionDecision::AllowWithModifiedInput(new_input) => {
            return execute_tool_inner(
                &tool,
                tool_use_id,
                new_input,
                &mut ctx,
                event_tx,
                start,
            )
            .await;
        }
        PermissionDecision::Allow => {}
    }

    // ─── Step 2: input validation ───
    if let Err(msg) = tool.validate_input(&input, &ctx).await {
        return ToolResult::error(format!("Validation failed: {}", msg));
    }

    // ─── Step 3: tool-specific permission check ───
    match tool.check_permissions(&input, &ctx).await {
        PermissionCheckResult::Deny { message } => {
            debug!(tool = %name, "Tool permission denied: {}", message);
            return ToolResult::error(message);
        }
        PermissionCheckResult::AskUser { message } => {
            // Send a permission request event and deny for now.
            // Full bidirectional permission protocol is implemented at the
            // Agent level; here we simply signal the denial.
            let _ = event_tx
                .send(SDKMessage::PermissionRequest {
                    request_id: uuid::Uuid::new_v4().to_string(),
                    tool_name: name.to_string(),
                    tool_input: input.clone(),
                    message: message.clone(),
                })
                .await;
            return ToolResult::error(format!("Permission required: {}", message));
        }
        PermissionCheckResult::Allow { updated_input } => {
            return execute_tool_inner(
                &tool,
                tool_use_id,
                updated_input,
                &mut ctx,
                event_tx,
                start,
            )
            .await;
        }
    }
}

// ---------------------------------------------------------------------------
// execute_tool_inner — run the tool and emit progress
// ---------------------------------------------------------------------------

/// Send a progress event and invoke the tool.
async fn execute_tool_inner(
    tool: &Arc<dyn Tool>,
    tool_use_id: &str,
    input: Value,
    ctx: &mut ToolUseContext,
    event_tx: &mpsc::Sender<SDKMessage>,
    start: Instant,
) -> ToolResult {
    // Emit progress event
    let _ = event_tx
        .send(SDKMessage::ToolProgress {
            tool_use_id: tool_use_id.to_string(),
            tool_name: tool.name().to_string(),
            parent_tool_use_id: None,
            progress_data: None,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
        .await;

    // Execute
    let result = tool.call(input, ctx, None).await;

    match result {
        Ok(mut r) => {
            truncate_result(&mut r, tool.max_result_size_chars());

            // Emit tool result event
            let _ = event_tx
                .send(SDKMessage::ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    tool_name: tool.name().to_string(),
                    content: r.content.clone(),
                    is_error: r.is_error,
                })
                .await;

            r
        }
        Err(e) => {
            let err_result = ToolResult::error(e.to_string());

            let _ = event_tx
                .send(SDKMessage::ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    tool_name: tool.name().to_string(),
                    content: err_result.content.clone(),
                    is_error: true,
                })
                .await;

            err_result
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tool::*;
    use async_trait::async_trait;
    use std::collections::HashSet;
    use std::path::PathBuf;
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

    fn allow_all() -> CanUseToolFn {
        Arc::new(|_, _| PermissionDecision::Allow)
    }

    #[derive(Debug)]
    struct ConcurrentTool;

    #[async_trait]
    impl Tool for ConcurrentTool {
        fn name(&self) -> &str { "ConcTool" }
        fn description(&self) -> &str { "A concurrent-safe tool" }
        fn input_schema(&self) -> ToolInputSchema { ToolInputSchema::default() }
        fn is_concurrency_safe(&self, _input: &Value) -> bool { true }
        fn is_read_only(&self, _input: &Value) -> bool { true }
        async fn call(
            &self,
            _input: Value,
            _ctx: &mut ToolUseContext,
            _progress: Option<&dyn ToolProgressSender>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("concurrent result"))
        }
    }

    #[derive(Debug)]
    struct SequentialTool;

    #[async_trait]
    impl Tool for SequentialTool {
        fn name(&self) -> &str { "SeqTool" }
        fn description(&self) -> &str { "A sequential tool" }
        fn input_schema(&self) -> ToolInputSchema { ToolInputSchema::default() }
        async fn call(
            &self,
            _input: Value,
            _ctx: &mut ToolUseContext,
            _progress: Option<&dyn ToolProgressSender>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("sequential result"))
        }
    }

    #[tokio::test]
    async fn execute_single_unknown_tool() {
        let (tx, _rx) = mpsc::channel(16);
        let result = execute_single(
            None,
            "unknown",
            "tu_1",
            Value::Null,
            test_ctx(),
            &allow_all(),
            &tx,
        )
        .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn execute_single_permission_denied() {
        let (tx, _rx) = mpsc::channel(16);
        let deny_all: CanUseToolFn =
            Arc::new(|_, _| PermissionDecision::Deny("nope".to_string()));
        let tool: Arc<dyn Tool> = Arc::new(ConcurrentTool);
        let result = execute_single(
            Some(tool),
            "ConcTool",
            "tu_2",
            Value::Null,
            test_ctx(),
            &deny_all,
            &tx,
        )
        .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn execute_tools_partitions_correctly() {
        use super::super::registry::ToolRegistry;

        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(ConcurrentTool));
        reg.register(Arc::new(SequentialTool));

        let tool_uses = vec![
            ToolUseInput {
                id: "tu_c".to_string(),
                name: "ConcTool".to_string(),
                input: Value::Null,
            },
            ToolUseInput {
                id: "tu_s".to_string(),
                name: "SeqTool".to_string(),
                input: Value::Null,
            },
        ];

        let (tx, _rx) = mpsc::channel(64);
        let mut ctx = test_ctx();
        let results = execute_tools(&tool_uses, &mut ctx, &reg, &allow_all(), &tx).await;

        assert_eq!(results.len(), 2);
        // Verify order matches input order
        assert_eq!(results[0].0, "tu_c");
        assert_eq!(results[1].0, "tu_s");
        assert!(!results[0].1.is_error);
        assert!(!results[1].1.is_error);
    }
}
