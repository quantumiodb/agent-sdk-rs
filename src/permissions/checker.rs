//! Permission checker implementing the deny -> allow -> mode-default pipeline.
//!
//! Aligns with the TypeScript SDK's permission resolution order:
//! 1. Deny rules -- if matched, reject immediately.
//! 2. Allow rules -- if matched, permit immediately.
//! 3. Mode-based default -- depends on the configured [`PermissionMode`].

use std::path::PathBuf;

use serde_json::Value;

use crate::types::{PermissionDecision, PermissionMode, PermissionRules};
#[cfg(test)]
use crate::types::PermissionRule;

// ---------------------------------------------------------------------------
// PermissionContext
// ---------------------------------------------------------------------------

/// Encapsulates permission state and exposes a synchronous `quick_check`
/// method suitable for use as a [`CanUseToolFn`](crate::types::CanUseToolFn).
#[derive(Debug)]
pub struct PermissionContext {
    /// The active permission mode.
    pub mode: PermissionMode,
    /// The working directory (used for path-based rules).
    pub cwd: PathBuf,
    /// Additional directories that are considered safe.
    pub additional_dirs: Vec<PathBuf>,
    /// The configured permission rules.
    pub rules: PermissionRules,
}

impl PermissionContext {
    /// Create a new permission context.
    pub fn new(mode: PermissionMode, cwd: PathBuf, rules: &PermissionRules) -> Self {
        Self {
            mode,
            cwd,
            additional_dirs: vec![],
            rules: rules.clone(),
        }
    }

    /// Fast synchronous permission check.
    ///
    /// Evaluation order (matching the TypeScript SDK):
    /// 1. **Deny rules** -- if a deny rule matches, return `Deny`.
    /// 2. **Allow rules** -- if an allow rule matches, return `Allow`.
    /// 3. **Mode default** -- behavior depends on the configured mode:
    ///    - `BypassPermissions` -> `Allow`
    ///    - `DontAsk` -> `Deny`
    ///    - `AcceptEdits` -> `Allow`
    ///    - `Plan` -> `Allow` (actual read-only check is in the executor)
    ///    - `Default` -> `Allow` (interactive confirmation is not modeled here)
    pub fn quick_check(&self, tool_name: &str, input: &Value) -> PermissionDecision {
        // 1. Check deny rules.
        for rule in &self.rules.deny {
            if matches_rule(&rule.tool_name, tool_name)
                && matches_content(&rule.rule_content, input)
            {
                return PermissionDecision::Deny(format!(
                    "Blocked by deny rule: {}",
                    rule.tool_name
                ));
            }
        }

        // 2. Check allow rules.
        for rule in &self.rules.allow {
            if matches_rule(&rule.tool_name, tool_name)
                && matches_content(&rule.rule_content, input)
            {
                return PermissionDecision::Allow;
            }
        }

        // 3. Mode-based default.
        match self.mode {
            PermissionMode::BypassPermissions => PermissionDecision::Allow,
            PermissionMode::DontAsk => {
                PermissionDecision::Deny("Not pre-authorized (DontAsk mode)".into())
            }
            PermissionMode::Plan => {
                // Plan mode: the executor layer uses tool.is_read_only() to
                // decide whether to actually execute. We return Allow here
                // so the executor can make that finer-grained decision.
                PermissionDecision::Allow
            }
            PermissionMode::AcceptEdits => PermissionDecision::Allow,
            PermissionMode::Default => {
                // In a non-interactive SDK context, default mode allows
                // everything. A real interactive UI would prompt the user.
                PermissionDecision::Allow
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rule matching helpers
// ---------------------------------------------------------------------------

/// Match a rule pattern against a tool name.
///
/// Supports:
/// - Exact match: `"Bash"` matches `"Bash"`
/// - Trailing wildcard: `"mcp__*"` matches `"mcp__github__search"`
pub fn matches_rule(pattern: &str, tool_name: &str) -> bool {
    if pattern == tool_name {
        return true;
    }
    if pattern.ends_with('*') {
        let prefix = &pattern[..pattern.len() - 1];
        return tool_name.starts_with(prefix);
    }
    false
}

/// Match optional rule content against tool input.
///
/// - `None` -> unconditional match.
/// - `Some(pattern)` -> the serialized JSON input must contain the pattern
///   as a substring.
pub fn matches_content(rule_content: &Option<String>, input: &Value) -> bool {
    match rule_content {
        None => true,
        Some(pattern) => {
            let input_str = serde_json::to_string(input).unwrap_or_default();
            input_str.contains(pattern.as_str())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_ctx(mode: PermissionMode, rules: PermissionRules) -> PermissionContext {
        PermissionContext::new(mode, PathBuf::from("/tmp"), &rules)
    }

    // --- matches_rule ---

    #[test]
    fn exact_match() {
        assert!(matches_rule("Bash", "Bash"));
        assert!(!matches_rule("Bash", "Edit"));
    }

    #[test]
    fn wildcard_match() {
        assert!(matches_rule("mcp__*", "mcp__github__search"));
        assert!(matches_rule("mcp__*", "mcp__"));
        assert!(!matches_rule("mcp__*", "Bash"));
    }

    #[test]
    fn wildcard_alone_matches_everything() {
        assert!(matches_rule("*", "Bash"));
        assert!(matches_rule("*", "mcp__github__search"));
    }

    // --- matches_content ---

    #[test]
    fn none_matches_everything() {
        assert!(matches_content(&None, &json!({"command": "ls"})));
    }

    #[test]
    fn substring_match() {
        assert!(matches_content(
            &Some("git ".to_string()),
            &json!({"command": "git status"})
        ));
        assert!(!matches_content(
            &Some("rm -rf".to_string()),
            &json!({"command": "git status"})
        ));
    }

    // --- quick_check: deny rules ---

    #[test]
    fn deny_rule_blocks() {
        let rules = PermissionRules {
            deny: vec![PermissionRule {
                tool_name: "Bash".to_string(),
                rule_content: Some("rm -rf".to_string()),
            }],
            ..Default::default()
        };
        let ctx = make_ctx(PermissionMode::Default, rules);

        let result = ctx.quick_check("Bash", &json!({"command": "rm -rf /"}));
        assert!(matches!(result, PermissionDecision::Deny(_)));
    }

    #[test]
    fn deny_rule_does_not_block_non_matching() {
        let rules = PermissionRules {
            deny: vec![PermissionRule {
                tool_name: "Bash".to_string(),
                rule_content: Some("rm -rf".to_string()),
            }],
            ..Default::default()
        };
        let ctx = make_ctx(PermissionMode::Default, rules);

        let result = ctx.quick_check("Bash", &json!({"command": "ls -la"}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    // --- quick_check: allow rules ---

    #[test]
    fn allow_rule_permits() {
        let rules = PermissionRules {
            allow: vec![PermissionRule {
                tool_name: "Bash".to_string(),
                rule_content: Some("git ".to_string()),
            }],
            ..Default::default()
        };
        let ctx = make_ctx(PermissionMode::DontAsk, rules);

        // Even in DontAsk mode, an explicit allow rule overrides.
        let result = ctx.quick_check("Bash", &json!({"command": "git status"}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    // --- quick_check: mode defaults ---

    #[test]
    fn bypass_mode_allows_all() {
        let ctx = make_ctx(PermissionMode::BypassPermissions, PermissionRules::default());
        let result = ctx.quick_check("Bash", &json!({"command": "rm -rf /"}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    #[test]
    fn dontask_mode_denies_unmatched() {
        let ctx = make_ctx(PermissionMode::DontAsk, PermissionRules::default());
        let result = ctx.quick_check("Bash", &json!({"command": "ls"}));
        assert!(matches!(result, PermissionDecision::Deny(_)));
    }

    #[test]
    fn default_mode_allows() {
        let ctx = make_ctx(PermissionMode::Default, PermissionRules::default());
        let result = ctx.quick_check("Edit", &json!({"file": "/tmp/foo.txt"}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    #[test]
    fn plan_mode_allows() {
        let ctx = make_ctx(PermissionMode::Plan, PermissionRules::default());
        let result = ctx.quick_check("Read", &json!({"file": "/tmp/foo.txt"}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    // --- deny takes priority over allow ---

    #[test]
    fn deny_overrides_allow() {
        let rules = PermissionRules {
            allow: vec![PermissionRule {
                tool_name: "Bash".to_string(),
                rule_content: None,
            }],
            deny: vec![PermissionRule {
                tool_name: "Bash".to_string(),
                rule_content: Some("rm -rf".to_string()),
            }],
            ..Default::default()
        };
        let ctx = make_ctx(PermissionMode::Default, rules);

        // The deny rule for "rm -rf" should win even though there's a blanket allow.
        let result = ctx.quick_check("Bash", &json!({"command": "rm -rf /tmp"}));
        assert!(matches!(result, PermissionDecision::Deny(_)));
    }

    // --- wildcard deny ---

    #[test]
    fn wildcard_deny_blocks_mcp_tools() {
        let rules = PermissionRules {
            deny: vec![PermissionRule {
                tool_name: "mcp__*".to_string(),
                rule_content: None,
            }],
            ..Default::default()
        };
        let ctx = make_ctx(PermissionMode::Default, rules);

        let result = ctx.quick_check("mcp__github__issues", &json!({}));
        assert!(matches!(result, PermissionDecision::Deny(_)));

        // Non-MCP tool should still be allowed.
        let result = ctx.quick_check("Bash", &json!({}));
        assert!(matches!(result, PermissionDecision::Allow));
    }
}
