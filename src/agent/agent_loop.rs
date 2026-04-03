//! Core agentic loop: stream API -> parse response -> execute tools -> repeat.
//!
//! This module implements the heart of the agent: a loop that calls the Claude
//! API, processes tool_use blocks, executes tools, and feeds results back
//! until the model signals `end_turn`, hits a budget/turn limit, or encounters
//! an unrecoverable error.

use std::collections::HashSet;
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::api::{ApiClient, ApiStreamEvent, MessageRequest};
use crate::api::types::{RequestMessage, SystemPrompt};
use crate::context::{build_system_prompt, compact_messages};
use crate::cost::CostTracker;
use crate::permissions::PermissionContext;
use crate::tools::ToolRegistry;
use crate::types::{
    CanUseToolFn, ContentBlock, ContentDelta, Message, MessageRole, PermissionDecision,
    SDKMessage, StopReason, ToolResultContent, Usage,
};
use crate::utils::tokens::{estimate_messages_tokens, get_max_context_tokens};

use super::agent::AgentError;
use super::agent::AgentOptions;

// ---------------------------------------------------------------------------
// AgentLoopParams
// ---------------------------------------------------------------------------

/// Parameters passed into the agent loop task.
pub struct AgentLoopParams {
    pub api_client: Arc<ApiClient>,
    pub tool_registry: Arc<ToolRegistry>,
    pub cost_tracker: Arc<CostTracker>,
    pub permission_ctx: Arc<PermissionContext>,
    pub messages: Vec<Message>,
    pub options: AgentOptions,
    pub session_id: String,
    pub event_tx: mpsc::Sender<SDKMessage>,
}

// ---------------------------------------------------------------------------
// ToolUseBlock
// ---------------------------------------------------------------------------

/// Accumulated state for a single tool_use block during streaming.
#[derive(Debug, Clone)]
struct ToolUseBlock {
    id: String,
    name: String,
    input: Value,
    input_json_buf: String,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of auto-continue attempts when the model hits max_tokens.
const MAX_TOKENS_RECOVERY_LIMIT: u32 = 3;

/// Fraction of context window that triggers auto-compact.
const COMPACT_THRESHOLD: f64 = 0.8;

// ---------------------------------------------------------------------------
// run_loop
// ---------------------------------------------------------------------------

/// Run the core agentic loop.
///
/// Flow:
/// 1. Send `System` event.
/// 2. Build system prompt.
/// 3. Loop:
///    a. Auto-compact check.
///    b. Build API request and stream response.
///    c. Parse response into content blocks + tool_use blocks.
///    d. If tool_use: execute tools -> append results -> continue.
///    e. If end_turn: break.
///    f. If max_tokens: auto-continue up to 3 times.
/// 4. Send `Result` event.
pub async fn run_loop(params: AgentLoopParams) -> Result<(), AgentError> {
    let start_time = std::time::Instant::now();
    let mut api_calls: u32 = 0;
    let mut max_tokens_recovery: u32 = 0;
    let mut messages = params.messages;

    // --- Send System event ---
    let _ = params
        .event_tx
        .send(SDKMessage::System {
            session_id: params.session_id.clone(),
            tools: params.tool_registry.tool_definitions(),
            model: params.options.model.clone(),
            mcp_servers: vec![],
            permission_mode: None,
            claude_code_version: None,
            cwd: params.options.cwd.as_ref().map(|p| p.display().to_string()),
        })
        .await;

    // --- Build system prompt ---
    let cwd = params
        .options
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| "/".into()));
    let system_prompt = build_system_prompt(
        &params.options.system_prompt,
        &params.options.append_system_prompt,
        &cwd,
    );

    // --- Build permission callback ---
    let permission_ctx = params.permission_ctx.clone();
    let can_use_tool: CanUseToolFn = Arc::new(move |tool_name, input| {
        permission_ctx.quick_check(tool_name, input)
    });

    // --- Main loop ---
    loop {
        // (a) Auto-compact check.
        let token_estimate = estimate_messages_tokens(&messages);
        let max_context = get_max_context_tokens(&params.options.model);
        if token_estimate > (max_context as f64 * COMPACT_THRESHOLD) as u64 {
            info!(
                tokens = token_estimate,
                max = max_context,
                "Auto-compacting context"
            );
            let original_tokens = token_estimate;
            messages = compact_messages(
                    &messages,
                    &system_prompt,
                    &params.api_client,
                    &params.options.model,
                )
                .await;
            let compacted_tokens = estimate_messages_tokens(&messages);
            let _ = params
                .event_tx
                .send(SDKMessage::Compact {
                    original_tokens,
                    compacted_tokens,
                })
                .await;
        }

        // Count every API call as a turn.
        api_calls += 1;

        // (b) Build API request.
        let api_messages = normalize_messages_for_api(&messages);
        let tool_defs = params.tool_registry.api_tool_params();
        let request = MessageRequest {
            model: params.options.model.clone(),
            max_tokens: params.options.max_tokens,
            system: Some(SystemPrompt::Text(system_prompt.clone())),
            messages: api_messages,
            tools: tool_defs
                .into_iter()
                .map(|tp| crate::api::types::ToolDefinition {
                    name: tp.name,
                    description: tp.description,
                    input_schema: serde_json::to_value(&tp.input_schema)
                        .unwrap_or(Value::Object(Default::default())),
                    cache_control: None,
                })
                .collect(),
            stream: true,
            thinking: params.options.thinking.as_ref().and_then(|t| {
                match t {
                    crate::types::ThinkingConfig::Enabled { budget_tokens } => {
                        Some(crate::api::types::ThinkingConfig {
                            thinking_type: "enabled".to_string(),
                            budget_tokens: *budget_tokens,
                        })
                    }
                    crate::types::ThinkingConfig::Adaptive => {
                        Some(crate::api::types::ThinkingConfig {
                            thinking_type: "enabled".to_string(),
                            budget_tokens: None,
                        })
                    }
                    crate::types::ThinkingConfig::Disabled => None,
                }
            }),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None
        };

        // (c) Stream API response.
        let api_start = std::time::Instant::now();
        let mut stream = params.api_client.stream_message(request).await?;
        let mut assistant_content: Vec<ContentBlock> = Vec::new();
        let mut tool_uses: Vec<ToolUseBlock> = Vec::new();
        let mut stop_reason: Option<StopReason> = None;
        let mut turn_usage = Usage::default();

        while let Some(event) = stream.next().await {
            match event {
                Ok(ApiStreamEvent::MessageStart { message: msg_start }) => {
                    if let Some(usage) = &msg_start.usage {
                        turn_usage.input_tokens = usage.input_tokens;
                        turn_usage.cache_creation_input_tokens = usage.cache_creation_input_tokens;
                        turn_usage.cache_read_input_tokens = usage.cache_read_input_tokens;
                    }
                }

                Ok(ApiStreamEvent::ContentBlockStart {
                    content_block,
                    index: _,
                }) => {
                    if let ContentBlock::ToolUse { id, name, .. } = &content_block {
                        tool_uses.push(ToolUseBlock {
                            id: id.clone(),
                            name: name.clone(),
                            input: Value::Null,
                            input_json_buf: String::new(),
                        });
                    }
                    assistant_content.push(content_block);
                }

                Ok(ApiStreamEvent::ContentBlockDelta { delta, index }) => {
                    // Push delta to caller.
                    let _ = params
                        .event_tx
                        .send(SDKMessage::ContentDelta {
                            index,
                            delta: delta.clone(),
                        })
                        .await;

                    // Accumulate into assistant_content.
                    apply_delta(&mut assistant_content, index, &delta);

                    // Accumulate tool_use JSON.
                    if let ContentDelta::InputJsonDelta { partial_json } = &delta {
                        if let Some(tu) = tool_uses.last_mut() {
                            tu.input_json_buf.push_str(partial_json);
                        }
                    }
                }

                Ok(ApiStreamEvent::MessageDelta { delta, usage }) => {
                    if let Some(sr) = delta.stop_reason {
                        stop_reason = Some(sr);
                    }
                    if let Some(u) = usage {
                        turn_usage.output_tokens += u.output_tokens;
                    }
                }

                Ok(ApiStreamEvent::Error { error }) => {
                    let _ = params
                        .event_tx
                        .send(SDKMessage::Error {
                            error_type: error.error_type.clone(),
                            message: error.message.clone(),
                            retryable: None,
                        })
                        .await;
                    return Err(AgentError::Other(format!(
                        "API stream error: {}",
                        error.message
                    )));
                }

                Ok(_) => {
                    // ContentBlockStop, MessageStop, Ping -- no action needed.
                }

                Err(e) => {
                    return Err(AgentError::Api(e));
                }
            }
        }

        // Record API duration.
        params.cost_tracker.add_api_duration(api_start.elapsed());

        // Parse accumulated tool_use JSON.
        for tu in &mut tool_uses {
            if !tu.input_json_buf.is_empty() {
                tu.input = serde_json::from_str(&tu.input_json_buf)
                    .unwrap_or(Value::Object(Default::default()));
            }
        }

        // Sync parsed tool inputs back into assistant_content so that the
        // SDKMessage::Assistant carries fully-parsed ToolUse.input objects
        // (streaming accumulates them as Value::String).
        for tu in &tool_uses {
            for block in assistant_content.iter_mut() {
                if let ContentBlock::ToolUse { id, input, .. } = block {
                    if *id == tu.id {
                        *input = tu.input.clone();
                    }
                }
            }
        }

        // Build and record assistant message.
        let assistant_msg = Message {
            id: uuid::Uuid::new_v4(),
            role: MessageRole::Assistant,
            content: assistant_content.clone(),
            timestamp: chrono::Utc::now(),
            stop_reason,
            usage: Some(turn_usage.clone()),
            model: Some(params.options.model.clone()),
            parent_tool_use_id: None,
        };
        messages.push(assistant_msg.clone());

        // Record cost.
        params
            .cost_tracker
            .add_usage(&params.options.model, &turn_usage);

        // Send Assistant event.
        let _ = params
            .event_tx
            .send(SDKMessage::Assistant {
                message: assistant_msg,
                stop_reason,
                session_id: Some(params.session_id.clone()),
            })
            .await;

        // (d) Handle stop reason + tool execution.
        match (stop_reason, tool_uses.is_empty()) {
            // Normal end, no tool calls.
            (Some(StopReason::EndTurn), true) => break,

            // Tool calls present.
            (_, false) => {
                let tool_results =
                    execute_tools(&tool_uses, &params.tool_registry, &can_use_tool).await;

                // Build tool_result user message.
                let tool_result_message = build_tool_results_message(&tool_uses, &tool_results);
                messages.push(tool_result_message);

                // Send ToolResult events.
                for (tu, result) in tool_uses.iter().zip(&tool_results) {
                    let _ = params
                        .event_tx
                        .send(SDKMessage::ToolResult {
                            tool_use_id: tu.id.clone(),
                            tool_name: tu.name.clone(),
                            content: result.0.clone(),
                            is_error: result.1,
                        })
                        .await;
                }

                // Budget check.
                if let Some(max_budget) = params.options.max_budget_usd {
                    if params.cost_tracker.total_cost() > max_budget {
                        warn!(
                            cost = params.cost_tracker.total_cost(),
                            max = max_budget,
                            "Budget exceeded"
                        );
                        break;
                    }
                }

                // Turn limit check (api_calls incremented at top of loop).
                if api_calls >= params.options.max_turns {
                    warn!(
                        turns = api_calls,
                        max = params.options.max_turns,
                        "Max turns reached"
                    );
                    break;
                }
            }

            // Max tokens -- auto-continue.
            (Some(StopReason::MaxTokens), true) => {
                if max_tokens_recovery < MAX_TOKENS_RECOVERY_LIMIT {
                    max_tokens_recovery += 1;
                    debug!(
                        attempt = max_tokens_recovery,
                        "Auto-continuing after max_tokens"
                    );
                    messages.push(Message::user_text(
                        "Please continue from where you left off.",
                    ));
                    continue;
                }
                break;
            }

            // Any other case (StopSequence, unexpected None, etc.)
            _ => break,
        }
    }

    // --- Send final Result event ---
    let elapsed = start_time.elapsed();
    let _ = params
        .event_tx
        .send(SDKMessage::Result {
            is_error: false,
            total_usage: params.cost_tracker.total_usage(),
            total_cost_usd: params.cost_tracker.total_cost(),
            duration_ms: elapsed.as_millis() as u64,
            num_turns: api_calls,
            session_id: params.session_id,
            result: None,
            stop_reason: None,
            final_messages: messages,
        })
        .await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: normalize messages for the API
// ---------------------------------------------------------------------------

/// Convert internal `Message` list to the API wire format.
fn normalize_messages_for_api(messages: &[Message]) -> Vec<RequestMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };
            RequestMessage {
                role: role.to_string(),
                content: serde_json::to_value(&msg.content).unwrap_or(Value::Array(vec![])),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: apply a content delta to accumulated content
// ---------------------------------------------------------------------------

/// Apply a streaming delta to the in-progress content blocks.
fn apply_delta(content: &mut [ContentBlock], index: usize, delta: &ContentDelta) {
    if index >= content.len() {
        return;
    }
    match (&mut content[index], delta) {
        (ContentBlock::Text { text }, ContentDelta::TextDelta { text: new_text }) => {
            text.push_str(new_text);
        }
        (
            ContentBlock::Thinking { thinking, .. },
            ContentDelta::ThinkingDelta {
                thinking: new_thinking,
            },
        ) => {
            thinking.push_str(new_thinking);
        }
        (ContentBlock::ToolUse { input, .. }, ContentDelta::InputJsonDelta { partial_json }) => {
            // Tool use input is accumulated in the ToolUseBlock struct, not here.
            // But we also keep a string in the content block for completeness.
            if let Value::Null = input {
                *input = Value::String(partial_json.clone());
            } else if let Value::String(s) = input {
                s.push_str(partial_json);
            }
        }
        _ => {
            // Mismatched delta/block type -- ignore.
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: execute tools
// ---------------------------------------------------------------------------

/// Execute tool calls and return (content, is_error) pairs.
///
/// This is a simplified executor. The full parallel/serial executor from
/// `crate::tools::executor` can be integrated later.
async fn execute_tools(
    tool_uses: &[ToolUseBlock],
    registry: &ToolRegistry,
    can_use_tool: &CanUseToolFn,
) -> Vec<(Vec<ToolResultContent>, bool)> {
    let mut results = Vec::with_capacity(tool_uses.len());

    for tu in tool_uses {
        // Permission check.
        match can_use_tool(&tu.name, &tu.input) {
            PermissionDecision::Deny(msg) => {
                results.push((
                    vec![ToolResultContent::Text { text: msg }],
                    true,
                ));
                continue;
            }
            PermissionDecision::AllowWithModifiedInput(_new_input) => {
                // TODO: use modified input
            }
            PermissionDecision::Allow => {}
        }

        // Find and execute tool.
        match registry.find(&tu.name) {
            Some(tool) => {
                let mut ctx = crate::types::ToolUseContext {
                    working_dir: std::env::current_dir()
                        .unwrap_or_else(|_| "/".into()),
                    cancellation: tokio_util::sync::CancellationToken::new(),
                    read_file_state: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
                    session_id: String::new(),
                    agent_id: None,
                };

                let tool_start = std::time::Instant::now();
                match tool.call(tu.input.clone(), &mut ctx, None).await {
                    Ok(result) => {
                        debug!(
                            tool = tu.name,
                            elapsed_ms = tool_start.elapsed().as_millis() as u64,
                            is_error = result.is_error,
                            "Tool completed"
                        );
                        results.push((result.content, result.is_error));
                    }
                    Err(e) => {
                        warn!(tool = tu.name, error = %e, "Tool execution failed");
                        results.push((
                            vec![ToolResultContent::Text {
                                text: format!("Error: {e}"),
                            }],
                            true,
                        ));
                    }
                }
            }
            None => {
                results.push((
                    vec![ToolResultContent::Text {
                        text: format!(
                            "Unknown tool '{}'. Use ToolSearch to find available tools.",
                            tu.name
                        ),
                    }],
                    true,
                ));
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Helper: build tool results message
// ---------------------------------------------------------------------------

/// Assemble a user message containing tool_result blocks.
fn build_tool_results_message(
    tool_uses: &[ToolUseBlock],
    results: &[(Vec<ToolResultContent>, bool)],
) -> Message {
    let content: Vec<ContentBlock> = tool_uses
        .iter()
        .zip(results)
        .map(|(tu, (content, is_error))| ContentBlock::ToolResult {
            tool_use_id: tu.id.clone(),
            content: content.clone(),
            is_error: *is_error,
        })
        .collect();

    Message {
        id: uuid::Uuid::new_v4(),
        role: MessageRole::User,
        content,
        timestamp: chrono::Utc::now(),
        stop_reason: None,
        usage: None,
        model: None,
        parent_tool_use_id: None,
    }
}
