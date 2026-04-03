//! Configuration types for the Rust Agent SDK.
//!
//! Defines thinking configuration and output format types, aligned with
//! the TypeScript SDK's `ThinkingConfigSchema` and `OutputFormatSchema`
//! from `coreSchemas.ts`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ThinkingConfig
// ---------------------------------------------------------------------------

/// Controls the model's extended thinking/reasoning behavior.
///
/// Aligned with the TypeScript SDK's `ThinkingConfigSchema`.
///
/// When set, this takes precedence over a raw `max_thinking_tokens` value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingConfig {
    /// Claude decides when and how much to think (Opus 4.6+).
    Adaptive,

    /// Fixed thinking token budget (older models).
    Enabled {
        /// Maximum number of tokens the model may use for thinking.
        /// If `None`, the server picks a default budget.
        #[serde(skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
    },

    /// Extended thinking is disabled.
    Disabled,
}

impl ThinkingConfig {
    /// Create an adaptive thinking config.
    pub fn adaptive() -> Self {
        ThinkingConfig::Adaptive
    }

    /// Create an enabled thinking config with a specific budget.
    pub fn enabled(budget_tokens: u32) -> Self {
        ThinkingConfig::Enabled {
            budget_tokens: Some(budget_tokens),
        }
    }

    /// Create an enabled thinking config with the server default budget.
    pub fn enabled_default() -> Self {
        ThinkingConfig::Enabled {
            budget_tokens: None,
        }
    }

    /// Create a disabled thinking config.
    pub fn disabled() -> Self {
        ThinkingConfig::Disabled
    }

    /// Whether thinking is active in any form.
    pub fn is_active(&self) -> bool {
        !matches!(self, ThinkingConfig::Disabled)
    }
}

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// Structured output format for constraining model output to a JSON schema.
///
/// Aligned with the TypeScript SDK's `OutputFormatSchema` /
/// `JsonSchemaOutputFormatSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputFormat {
    /// Constrain the model's output to match a JSON Schema.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The JSON Schema that the model's output must conform to.
        schema: HashMap<String, serde_json::Value>,
    },
}

impl OutputFormat {
    /// Create a JSON schema output format from a schema map.
    pub fn json_schema(schema: HashMap<String, serde_json::Value>) -> Self {
        OutputFormat::JsonSchema { schema }
    }

    /// Create a JSON schema output format from a `serde_json::Value`.
    ///
    /// The value must be an object; this will panic otherwise.
    pub fn from_value(value: serde_json::Value) -> Self {
        let schema = value
            .as_object()
            .expect("OutputFormat::from_value requires a JSON object")
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        OutputFormat::JsonSchema { schema }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_config_adaptive_serialization() {
        let config = ThinkingConfig::adaptive();
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "adaptive");
        assert!(config.is_active());
    }

    #[test]
    fn thinking_config_enabled_serialization() {
        let config = ThinkingConfig::enabled(8192);
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "enabled");
        assert_eq!(json["budget_tokens"], 8192);
        assert!(config.is_active());
    }

    #[test]
    fn thinking_config_enabled_default_omits_budget() {
        let config = ThinkingConfig::enabled_default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("budget_tokens"));
        assert!(config.is_active());
    }

    #[test]
    fn thinking_config_disabled_serialization() {
        let config = ThinkingConfig::disabled();
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "disabled");
        assert!(!config.is_active());
    }

    #[test]
    fn thinking_config_roundtrip() {
        let configs = vec![
            ThinkingConfig::Adaptive,
            ThinkingConfig::Enabled {
                budget_tokens: Some(4096),
            },
            ThinkingConfig::Disabled,
        ];
        for config in configs {
            let json = serde_json::to_string(&config).unwrap();
            let parsed: ThinkingConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, config);
        }
    }

    #[test]
    fn output_format_json_schema_serialization() {
        let mut schema = HashMap::new();
        schema.insert("type".to_string(), serde_json::json!("object"));
        schema.insert(
            "properties".to_string(),
            serde_json::json!({
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }),
        );

        let format = OutputFormat::json_schema(schema);
        let json = serde_json::to_value(&format).unwrap();
        assert_eq!(json["type"], "json_schema");
        assert_eq!(json["schema"]["type"], "object");
    }

    #[test]
    fn output_format_from_value() {
        let value = serde_json::json!({
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        });
        let format = OutputFormat::from_value(value);
        match &format {
            OutputFormat::JsonSchema { schema } => {
                assert_eq!(schema["type"], serde_json::json!("object"));
            }
        }
    }

    #[test]
    fn output_format_roundtrip() {
        let mut schema = HashMap::new();
        schema.insert("type".to_string(), serde_json::json!("object"));
        let format = OutputFormat::json_schema(schema);
        let json = serde_json::to_string(&format).unwrap();
        let parsed: OutputFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, format);
    }
}
