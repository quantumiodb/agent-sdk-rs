//! Token estimation and context window utilities.
//!
//! These are rough heuristics used for auto-compact decisions.
//! They intentionally over-estimate slightly to provide a safety margin.

use crate::types::Message;

/// Estimate the token count for a string using the chars/4 heuristic.
///
/// This is a fast, zero-dependency approximation. Real tokenisation would
/// require a BPE implementation, but for auto-compact thresholds the rough
/// estimate is sufficient.
pub fn estimate_token_count(text: &str) -> u64 {
    // English text averages ~4 chars per token with Claude's tokenizer.
    // Non-ASCII text (CJK, etc.) is closer to 1-2 chars per token, but the
    // chars/4 heuristic still works as a conservative lower bound.
    let char_count = text.len() as u64;
    // Add 1 to avoid returning 0 for very short strings.
    (char_count / 4).max(1)
}

/// Estimate total tokens across a slice of messages.
///
/// Adds a small per-message overhead to account for role/formatting tokens.
pub fn estimate_messages_tokens(messages: &[Message]) -> u64 {
    let mut total: u64 = 0;
    for msg in messages {
        // ~4 tokens per message for role/delimiter overhead
        total += 4;
        for block in &msg.content {
            total += estimate_content_block_tokens(block);
        }
    }
    total
}

/// Estimate tokens for a single content block.
fn estimate_content_block_tokens(block: &crate::types::ContentBlock) -> u64 {
    match block {
        crate::types::ContentBlock::Text { text } => estimate_token_count(text),
        crate::types::ContentBlock::ToolUse { name, input, .. } => {
            // Tool name + JSON input
            let input_str = serde_json::to_string(input).unwrap_or_default();
            estimate_token_count(name) + estimate_token_count(&input_str) + 10
        }
        crate::types::ContentBlock::ToolResult { content, .. } => {
            let mut tokens: u64 = 5; // overhead for tool_result structure
            for c in content {
                match c {
                    crate::types::ToolResultContent::Text { text } => {
                        tokens += estimate_token_count(text);
                    }
                    crate::types::ToolResultContent::Image { .. } => {
                        // Images are billed separately and use a fixed token count.
                        // A rough estimate for a typical screenshot.
                        tokens += 1600;
                    }
                }
            }
            tokens
        }
        crate::types::ContentBlock::Thinking { thinking, .. } => {
            estimate_token_count(thinking)
        }
        crate::types::ContentBlock::Image { .. } => 1600,
    }
}

/// Get the maximum context window size (in tokens) for a model.
///
/// Returns a conservative default for unknown models.
pub fn get_max_context_tokens(model: &str) -> u64 {
    let key = model.to_lowercase();

    if key.contains("opus") {
        200_000
    } else if key.contains("sonnet") {
        200_000
    } else if key.contains("haiku") {
        200_000
    } else {
        // Conservative default for unknown models.
        128_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn estimate_short_text() {
        // "hello" = 5 chars -> 5/4 = 1 token
        assert_eq!(estimate_token_count("hello"), 1);
    }

    #[test]
    fn estimate_longer_text() {
        let text = "The quick brown fox jumps over the lazy dog.";
        // 44 chars / 4 = 11 tokens
        let est = estimate_token_count(text);
        assert_eq!(est, 11);
    }

    #[test]
    fn estimate_empty_text() {
        // Empty string: 0/4 = 0, clamped to 1
        assert_eq!(estimate_token_count(""), 1);
    }

    #[test]
    fn estimate_messages() {
        let messages = vec![
            Message::user_text("Hello, how are you?"),
            Message::assistant_text("I'm doing well, thanks for asking!"),
        ];
        let tokens = estimate_messages_tokens(&messages);
        // Should be > 0 and roughly in the right ballpark
        assert!(tokens > 5);
        assert!(tokens < 100);
    }

    #[test]
    fn max_context_tokens_known_models() {
        assert_eq!(get_max_context_tokens("claude-opus-4-6"), 200_000);
        assert_eq!(get_max_context_tokens("claude-sonnet-4-6"), 200_000);
        assert_eq!(get_max_context_tokens("claude-3-5-haiku-20241022"), 200_000);
    }

    #[test]
    fn max_context_tokens_unknown_model() {
        assert_eq!(get_max_context_tokens("gpt-4o"), 128_000);
    }
}
