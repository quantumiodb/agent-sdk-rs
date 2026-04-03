//! Permission system types for the Rust Agent SDK.
//!
//! Defines permission modes, rules, and rule sets that control how tool
//! executions are authorized. Aligned with the TypeScript SDK's
//! `PermissionModeSchema`, `PermissionBehaviorSchema`, and
//! `PermissionRuleValueSchema` from `coreSchemas.ts`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PermissionMode
// ---------------------------------------------------------------------------

/// Controls how tool executions are handled.
///
/// Aligned with the TypeScript SDK's `PermissionModeSchema`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    /// Standard behavior -- prompts for dangerous operations.
    Default,

    /// Auto-accept file edit operations.
    AcceptEdits,

    /// Bypass all permission checks.
    ///
    /// Requires `allowDangerouslySkipPermissions` to be set.
    BypassPermissions,

    /// Planning mode -- no actual tool execution.
    Plan,

    /// Don't prompt for permissions; deny anything not pre-approved.
    #[serde(rename = "dontAsk")]
    DontAsk,
}

impl Default for PermissionMode {
    fn default() -> Self {
        PermissionMode::Default
    }
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PermissionMode::Default => write!(f, "default"),
            PermissionMode::AcceptEdits => write!(f, "acceptEdits"),
            PermissionMode::BypassPermissions => write!(f, "bypassPermissions"),
            PermissionMode::Plan => write!(f, "plan"),
            PermissionMode::DontAsk => write!(f, "dontAsk"),
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionBehavior
// ---------------------------------------------------------------------------

/// The behavior to apply when a permission rule matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionBehavior {
    /// Allow the tool call.
    Allow,
    /// Deny the tool call.
    Deny,
    /// Prompt the user for a decision.
    Ask,
}

// ---------------------------------------------------------------------------
// PermissionRule
// ---------------------------------------------------------------------------

/// A single permission rule that matches a tool invocation.
///
/// Rules consist of a tool name and an optional content pattern.
/// For example, `Bash(git *)` matches Bash tool calls whose command
/// starts with `git `.
///
/// Aligned with the TypeScript SDK's `PermissionRuleValueSchema`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    /// The tool name this rule applies to (e.g. `"Bash"`, `"Edit"`).
    pub tool_name: String,

    /// Optional content/pattern to further scope the rule.
    ///
    /// For example, `"git *"` in `Bash(git *)` means this rule only
    /// applies to Bash calls whose command matches `git *`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_content: Option<String>,
}

impl PermissionRule {
    /// Create a rule that matches any invocation of the named tool.
    pub fn tool(name: impl Into<String>) -> Self {
        Self {
            tool_name: name.into(),
            rule_content: None,
        }
    }

    /// Create a rule that matches a tool invocation with a specific pattern.
    pub fn tool_with_content(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_name: name.into(),
            rule_content: Some(content.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionRules
// ---------------------------------------------------------------------------

/// A collection of permission rules organized by behavior.
///
/// Rules are evaluated in order: deny rules first, then allow rules,
/// then ask rules. If no rule matches, the permission mode's default
/// behavior applies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionRules {
    /// Rules that always allow the matched tool call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<PermissionRule>,

    /// Rules that always deny the matched tool call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<PermissionRule>,

    /// Rules that always prompt the user for a decision.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ask: Vec<PermissionRule>,
}

impl PermissionRules {
    /// Create an empty rule set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an allow rule.
    pub fn add_allow(mut self, rule: PermissionRule) -> Self {
        self.allow.push(rule);
        self
    }

    /// Add a deny rule.
    pub fn add_deny(mut self, rule: PermissionRule) -> Self {
        self.deny.push(rule);
        self
    }

    /// Add an ask rule.
    pub fn add_ask(mut self, rule: PermissionRule) -> Self {
        self.ask.push(rule);
        self
    }

    /// Returns true if there are no rules in any category.
    pub fn is_empty(&self) -> bool {
        self.allow.is_empty() && self.deny.is_empty() && self.ask.is_empty()
    }

    /// Total number of rules across all categories.
    pub fn len(&self) -> usize {
        self.allow.len() + self.deny.len() + self.ask.len()
    }
}

// ---------------------------------------------------------------------------
// PermissionUpdateDestination
// ---------------------------------------------------------------------------

/// Where a permission update should be persisted.
///
/// Aligned with the TypeScript SDK's `PermissionUpdateDestinationSchema`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionUpdateDestination {
    UserSettings,
    ProjectSettings,
    LocalSettings,
    Session,
    CliArg,
}

// ---------------------------------------------------------------------------
// PermissionUpdate
// ---------------------------------------------------------------------------

/// A mutation to apply to the permission rules.
///
/// Aligned with the TypeScript SDK's `PermissionUpdateSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PermissionUpdate {
    AddRules {
        rules: Vec<PermissionRule>,
        behavior: PermissionBehavior,
        destination: PermissionUpdateDestination,
    },
    ReplaceRules {
        rules: Vec<PermissionRule>,
        behavior: PermissionBehavior,
        destination: PermissionUpdateDestination,
    },
    RemoveRules {
        rules: Vec<PermissionRule>,
        behavior: PermissionBehavior,
        destination: PermissionUpdateDestination,
    },
    SetMode {
        mode: PermissionMode,
        destination: PermissionUpdateDestination,
    },
    AddDirectories {
        directories: Vec<String>,
        destination: PermissionUpdateDestination,
    },
    RemoveDirectories {
        directories: Vec<String>,
        destination: PermissionUpdateDestination,
    },
}

// ---------------------------------------------------------------------------
// PermissionResult
// ---------------------------------------------------------------------------

/// The result of a full permission evaluation (global rules + tool-specific).
///
/// Aligned with the TypeScript SDK's `PermissionResultSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "behavior", rename_all = "lowercase")]
pub enum PermissionResult {
    /// The tool call is permitted.
    Allow {
        /// Optionally modified input to use instead of the original.
        #[serde(skip_serializing_if = "Option::is_none")]
        updated_input: Option<serde_json::Value>,

        /// Permission changes to persist (e.g. "always allow this").
        #[serde(skip_serializing_if = "Option::is_none")]
        updated_permissions: Option<Vec<PermissionUpdate>>,

        /// The tool_use ID this decision applies to.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<String>,
    },

    /// The tool call is denied.
    Deny {
        /// Reason for denial.
        message: String,

        /// If true, this denial should interrupt the agent loop.
        #[serde(skip_serializing_if = "Option::is_none")]
        interrupt: Option<bool>,

        /// The tool_use ID this decision applies to.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_mode_default_is_default() {
        assert_eq!(PermissionMode::default(), PermissionMode::Default);
    }

    #[test]
    fn permission_mode_display() {
        assert_eq!(format!("{}", PermissionMode::Default), "default");
        assert_eq!(format!("{}", PermissionMode::DontAsk), "dontAsk");
        assert_eq!(
            format!("{}", PermissionMode::BypassPermissions),
            "bypassPermissions"
        );
    }

    #[test]
    fn permission_mode_serialization() {
        let json = serde_json::to_string(&PermissionMode::AcceptEdits).unwrap();
        assert_eq!(json, "\"acceptEdits\"");

        let parsed: PermissionMode = serde_json::from_str("\"plan\"").unwrap();
        assert_eq!(parsed, PermissionMode::Plan);
    }

    #[test]
    fn permission_rule_constructors() {
        let r1 = PermissionRule::tool("Bash");
        assert_eq!(r1.tool_name, "Bash");
        assert!(r1.rule_content.is_none());

        let r2 = PermissionRule::tool_with_content("Bash", "git *");
        assert_eq!(r2.tool_name, "Bash");
        assert_eq!(r2.rule_content.as_deref(), Some("git *"));
    }

    #[test]
    fn permission_rules_builder() {
        let rules = PermissionRules::new()
            .add_allow(PermissionRule::tool("Read"))
            .add_deny(PermissionRule::tool_with_content("Bash", "rm -rf *"))
            .add_ask(PermissionRule::tool("Edit"));

        assert_eq!(rules.len(), 3);
        assert!(!rules.is_empty());
        assert_eq!(rules.allow.len(), 1);
        assert_eq!(rules.deny.len(), 1);
        assert_eq!(rules.ask.len(), 1);
    }

    #[test]
    fn permission_rules_default_is_empty() {
        let rules = PermissionRules::default();
        assert!(rules.is_empty());
        assert_eq!(rules.len(), 0);
    }

    #[test]
    fn permission_result_allow_roundtrip() {
        let result = PermissionResult::Allow {
            updated_input: None,
            updated_permissions: None,
            tool_use_id: Some("tu_123".to_string()),
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["behavior"], "allow");
        assert_eq!(json["tool_use_id"], "tu_123");
    }

    #[test]
    fn permission_result_deny_roundtrip() {
        let result = PermissionResult::Deny {
            message: "not allowed".to_string(),
            interrupt: Some(true),
            tool_use_id: None,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["behavior"], "deny");
        assert_eq!(json["message"], "not allowed");
        assert_eq!(json["interrupt"], true);
    }
}
