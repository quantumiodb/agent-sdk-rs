//! Context compaction: summarise old messages to free up context window space.
//!
//! When a conversation grows close to the model's context window limit the agent
//! loop calls [`compact_messages`].  The function:
//!
//! 1. Keeps the most recent [`KEEP_RECENT`] messages verbatim.
//! 2. Formats the older messages as a readable transcript.
//! 3. Calls the same model to produce a concise summary of that transcript.
//! 4. Replaces the old messages with two synthetic messages:
//!    - `user`      — the generated summary wrapped in `<summary>` tags.
//!    - `assistant` — a brief acknowledgment so the conversation stays valid
//!                    (the API requires alternating user / assistant turns).
//! 5. Appends the preserved recent messages.
//!
//! If the summarisation API call fails the function falls back to returning
//! the original messages unchanged, so the agent can continue even if
//! compaction is unavailable.

use futures::StreamExt;
use tracing::{debug, warn};

use crate::api::{ApiClient, ApiStreamEvent, MessageRequest};
use crate::api::types::{RequestMessage, SystemPrompt};
use crate::types::{ContentBlock, ContentDelta, Message, MessageRole, ToolResultContent};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of recent messages kept verbatim after compaction.
/// Must be even so we always preserve complete user / assistant turn pairs.
const KEEP_RECENT: usize = 8;

/// Maximum characters of a tool result included in the transcript.
/// Long tool outputs (e.g. grep results, file reads) are truncated.
const MAX_TOOL_RESULT_CHARS: usize = 600;

/// Maximum output tokens requested from the summarisation call.
const SUMMARY_MAX_TOKENS: u32 = 1024;

/// System prompt sent to the model for summarisation.
const SUMMARIZER_SYSTEM: &str = "\
You are a conversation summarizer. Your task is to produce a concise but \
complete summary of the conversation provided. The summary will replace the \
earlier part of an ongoing conversation, so it must preserve all important \
facts, decisions, code snippets, file paths, error messages, and any \
unresolved questions or tasks. Write in third person (e.g. \"The user asked \
…\", \"The assistant ran …\"). Do not add commentary or headings — output \
only the summary text.";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Replace old messages with a model-generated summary, keeping the most
/// recent [`KEEP_RECENT`] messages verbatim.
///
/// Returns the original slice unchanged if:
/// * the history is too short to need compaction, or
/// * the summarisation API call fails (logs a warning but does not panic).
pub async fn compact_messages(
    messages: &[Message],
    system_prompt: &str,
    api_client: &ApiClient,
    model: &str,
) -> Vec<Message> {
    // Nothing to compact.
    if messages.len() <= KEEP_RECENT {
        debug!(
            messages = messages.len(),
            keep = KEEP_RECENT,
            "History short enough — skipping compaction"
        );
        return messages.to_vec();
    }

    // Find the split: keep last KEEP_RECENT messages, adjusting so that
    // the first retained message is always a User turn.
    let mut split = messages.len().saturating_sub(KEEP_RECENT);
    while split < messages.len() && messages[split].role != MessageRole::User {
        split += 1;
    }

    // Degenerate: couldn't find a user-role start in the recent window.
    if split >= messages.len() || messages.len() - split < 2 {
        debug!("Could not find a clean split point — skipping compaction");
        return messages.to_vec();
    }

    let old_messages = &messages[..split];
    let recent_messages = &messages[split..];

    // At least 2 old messages are needed to make a meaningful summary.
    if old_messages.len() < 2 {
        return messages.to_vec();
    }

    debug!(
        old = old_messages.len(),
        recent = recent_messages.len(),
        "Compacting context"
    );

    // Build a human-readable transcript of the old messages.
    let transcript = format_transcript(old_messages, system_prompt);

    // Ask the model to summarise the transcript.
    match call_summarizer(api_client, model, &transcript).await {
        Ok(summary) => {
            build_compacted_messages(&summary, recent_messages)
        }
        Err(e) => {
            warn!(error = %e, "Summarisation failed — keeping original messages");
            messages.to_vec()
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Assemble the compacted message list from the summary + recent messages.
fn build_compacted_messages(summary: &str, recent: &[Message]) -> Vec<Message> {
    let mut result = Vec::with_capacity(2 + recent.len());

    // Synthetic user message containing the summary.
    result.push(Message::user_text(&format!(
        "<summary>\n\
         The following is a summary of the conversation that occurred before \
         this point:\n\n\
         {summary}\n\
         </summary>"
    )));

    // Synthetic assistant acknowledgement — required to keep the
    // user / assistant turn alternation valid for the API.
    result.push(Message::assistant_text(
        "I have reviewed the conversation summary above and will continue \
         from where we left off.",
    ));

    result.extend_from_slice(recent);
    result
}

/// Format a slice of messages as a plain-text transcript suitable for the
/// summarisation prompt.
fn format_transcript(messages: &[Message], system_prompt: &str) -> String {
    let mut buf = String::new();

    if !system_prompt.is_empty() {
        buf.push_str("<system_prompt>\n");
        buf.push_str(system_prompt);
        buf.push_str("\n</system_prompt>\n\n");
    }

    for msg in messages {
        let label = match msg.role {
            MessageRole::User => "Human",
            MessageRole::Assistant => "Assistant",
        };
        buf.push_str(label);
        buf.push_str(": ");

        let mut first = true;
        for block in &msg.content {
            if !first {
                buf.push(' ');
            }
            first = false;

            match block {
                ContentBlock::Text { text } => {
                    buf.push_str(text);
                }

                ContentBlock::ToolUse { name, input, .. } => {
                    let input_str = serde_json::to_string(input).unwrap_or_default();
                    // Truncate very long inputs.
                    let input_preview = if input_str.len() > 300 {
                        format!("{}…", &input_str[..300])
                    } else {
                        input_str
                    };
                    buf.push_str(&format!("[Called {name}({input_preview})]"));
                }

                ContentBlock::ToolResult { content, is_error, .. } => {
                    let text: String = content
                        .iter()
                        .filter_map(|c| match c {
                            ToolResultContent::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let preview = if text.len() > MAX_TOOL_RESULT_CHARS {
                        format!("{}… (truncated)", &text[..MAX_TOOL_RESULT_CHARS])
                    } else {
                        text
                    };
                    let tag = if *is_error { "tool_error" } else { "tool_result" };
                    buf.push_str(&format!("[{tag}: {preview}]"));
                }

                ContentBlock::Thinking { thinking, .. } => {
                    // Only include a brief hint that thinking occurred.
                    let preview = if thinking.len() > 100 {
                        format!("{}…", &thinking[..100])
                    } else {
                        thinking.clone()
                    };
                    buf.push_str(&format!("[Thinking: {preview}]"));
                }

                ContentBlock::Image { .. } => {
                    buf.push_str("[Image]");
                }
            }
        }

        buf.push('\n');
    }

    buf
}

/// Call the model to summarise `transcript`.  Collects the full streaming
/// response and returns the accumulated text.
///
/// Some models (e.g. Qwen3 with extended thinking enabled) emit their
/// reasoning exclusively as `ThinkingDelta` and produce an empty `TextDelta`.
/// In that case the thinking text is used as the summary instead.
async fn call_summarizer(
    api_client: &ApiClient,
    model: &str,
    transcript: &str,
) -> Result<String, crate::api::ApiError> {
    let request = MessageRequest {
        model: model.to_string(),
        max_tokens: SUMMARY_MAX_TOKENS,
        system: Some(SystemPrompt::Text(SUMMARIZER_SYSTEM.to_string())),
        messages: vec![RequestMessage {
            role: "user".to_string(),
            content: serde_json::json!([{
                "type": "text",
                "text": format!(
                    "Summarise the following conversation:\n\n{transcript}\n\n\
                     Remember: output only the summary text, no headings or commentary."
                )
            }]),
        }],
        tools: vec![],
        stream: true,
        temperature: Some(0.3), // lower temperature for factual summary
        top_p: None,
        top_k: None,
        stop_sequences: None,
        thinking: None,
        metadata: None,
    };

    let mut stream = api_client.stream_message(request).await?;
    let mut text_summary = String::new();
    let mut thinking_summary = String::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(ApiStreamEvent::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            }) => {
                text_summary.push_str(&text);
            }
            Ok(ApiStreamEvent::ContentBlockDelta {
                delta: ContentDelta::ThinkingDelta { thinking },
                ..
            }) => {
                // Collect thinking as a fallback for models that output reasoning
                // instead of plain text (e.g. Qwen3, DeepSeek-R1).
                thinking_summary.push_str(&thinking);
            }
            _ => {}
        }
    }

    // Prefer the text response; fall back to thinking if text is empty.
    let summary = if text_summary.trim().is_empty() {
        if !thinking_summary.is_empty() {
            warn!(
                thinking_chars = thinking_summary.len(),
                "Model returned no text — using thinking block as summary"
            );
        }
        thinking_summary
    } else {
        text_summary
    };

    debug!(chars = summary.len(), "Summarisation complete");
    Ok(summary)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    fn make_conversation(pairs: &[(&str, &str)]) -> Vec<Message> {
        let mut msgs = Vec::new();
        for (user, assistant) in pairs {
            msgs.push(Message::user_text(user));
            msgs.push(Message::assistant_text(assistant));
        }
        msgs
    }

    #[test]
    fn short_history_returned_unchanged() {
        // 4 messages (2 turns) — below KEEP_RECENT (8)
        let msgs = make_conversation(&[
            ("Hello", "Hi there!"),
            ("How are you?", "Great, thanks!"),
        ]);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // We pass a dummy api_client — it should not be called.
        // We can't easily construct a real ApiClient in unit tests without a server,
        // so we verify the short-circuit path by checking the length.
        // (The function returns early before touching api_client.)
        //
        // Use a real ApiClient pointed at a dummy URL — the short-circuit
        // happens before any network call.
        let api_client = crate::api::ApiClient::openai_compat("dummy", "http://127.0.0.1:1");
        let result = rt.block_on(compact_messages(&msgs, "", &api_client, "test-model"));
        assert_eq!(result.len(), msgs.len(), "Short history should be returned unchanged");
    }

    #[test]
    fn format_transcript_includes_roles() {
        let msgs = vec![
            Message::user_text("What is Rust?"),
            Message::assistant_text("Rust is a systems programming language."),
        ];
        let transcript = format_transcript(&msgs, "");
        assert!(transcript.contains("Human: What is Rust?"));
        assert!(transcript.contains("Assistant: Rust is a systems programming language."));
    }

    #[test]
    fn format_transcript_includes_system_prompt() {
        let msgs = vec![Message::user_text("Hello")];
        let transcript = format_transcript(&msgs, "You are helpful.");
        assert!(transcript.contains("<system_prompt>"));
        assert!(transcript.contains("You are helpful."));
    }

    #[test]
    fn format_transcript_truncates_tool_result() {
        use crate::types::{ContentBlock, MessageRole, ToolResultContent};
        use uuid::Uuid;

        let long_output = "x".repeat(MAX_TOOL_RESULT_CHARS + 100);
        let msg = Message {
            id: Uuid::new_v4(),
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: vec![ToolResultContent::Text { text: long_output }],
                is_error: false,
            }],
            timestamp: chrono::Utc::now(),
            stop_reason: None,
            usage: None,
            model: None,
            parent_tool_use_id: None,
        };
        let transcript = format_transcript(&[msg], "");
        assert!(transcript.contains("truncated"));
    }

    #[test]
    fn build_compacted_messages_structure() {
        let recent = vec![
            Message::user_text("What's next?"),
            Message::assistant_text("We continue here."),
        ];
        let result = build_compacted_messages("Summary of old conversation.", &recent);

        assert_eq!(result.len(), 4, "2 synthetic + 2 recent");
        assert_eq!(result[0].role, MessageRole::User);
        assert!(result[0].text_content().contains("<summary>"));
        assert_eq!(result[1].role, MessageRole::Assistant);
        assert_eq!(result[2].role, MessageRole::User);
        assert_eq!(result[2].text_content(), "What's next?");
    }

    #[test]
    fn split_lands_on_user_message() {
        // Create 10 messages (5 turns). split = 10 - 8 = 2 → messages[2] is assistant.
        // The adjustment should bump split to 2 (user) — it's already a user message
        // at index 2 if we start with user.
        let msgs = make_conversation(&[
            ("u0", "a0"), // 0,1
            ("u1", "a1"), // 2,3  ← split=2, messages[2]=user ✓
            ("u2", "a2"), // 4,5
            ("u3", "a3"), // 6,7
            ("u4", "a4"), // 8,9
        ]);
        assert_eq!(msgs.len(), 10);

        // split = 10 - 8 = 2, messages[2].role == User → no adjustment needed.
        let mut split = msgs.len().saturating_sub(KEEP_RECENT);
        while split < msgs.len() && msgs[split].role != MessageRole::User {
            split += 1;
        }
        assert_eq!(split, 2);
        assert_eq!(msgs[split].role, MessageRole::User);
    }
}
