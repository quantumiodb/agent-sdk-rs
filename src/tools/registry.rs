//! Tool registry: a `HashMap`-based container for tools with alias support.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::tool::{ApiToolParam, Tool, ToolDefinition};

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// A name-keyed registry of tools with alias support.
///
/// Tools are stored as `Arc<dyn Tool>` so they can be cheaply shared across
/// concurrent tasks. Each tool's canonical name is its primary key; aliases map
/// to that canonical name.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    /// Canonical name → tool.
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Alias → canonical name.
    aliases: HashMap<String, String>,
}

impl ToolRegistry {
    // === Construction ===

    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in tools.
    ///
    /// Available only when the `built-in-tools` feature is enabled.
    #[cfg(feature = "built-in-tools")]
    pub fn default_registry() -> Self {
        use super::{
            BashTool, FileEditTool, FileReadTool, FileWriteTool, GlobTool, GrepTool,
        };

        let mut reg = Self::new();
        reg.register(Arc::new(BashTool));
        reg.register(Arc::new(FileReadTool));
        reg.register(Arc::new(FileEditTool));
        reg.register(Arc::new(FileWriteTool));
        reg.register(Arc::new(GlobTool));
        reg.register(Arc::new(GrepTool));
        reg
    }

    /// Create a registry with no built-in tools (fallback when the feature is
    /// disabled).
    #[cfg(not(feature = "built-in-tools"))]
    pub fn default_registry() -> Self {
        Self::new()
    }

    // === Registration ===

    /// Register a tool. Also registers any aliases it declares.
    ///
    /// If a tool with the same name is already registered, it is replaced.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        for alias in tool.aliases() {
            self.aliases.insert(alias.to_string(), name.clone());
        }
        self.tools.insert(name, tool);
    }

    // === Lookup ===

    /// Find a tool by its canonical name or any of its aliases.
    pub fn find(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name).or_else(|| {
            self.aliases
                .get(name)
                .and_then(|canonical| self.tools.get(canonical))
        })
    }

    /// Return references to all registered tools.
    pub fn all(&self) -> Vec<&Arc<dyn Tool>> {
        self.tools.values().collect()
    }

    /// Return a sorted list of canonical tool names.
    ///
    /// Sorting ensures stable output for prompt caching.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    // === API / SDK parameter generation ===

    /// Generate API tool parameters for sending to the Anthropic Messages API.
    ///
    /// Includes only enabled, non-deferred tools.
    pub fn api_tool_params(&self) -> Vec<ApiToolParam> {
        self.tools
            .values()
            .filter(|t| t.is_enabled() && !t.should_defer())
            .map(|t| ApiToolParam {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Generate SDK tool definitions (richer metadata for consumers).
    ///
    /// Includes all enabled tools (even deferred ones).
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .filter(|t| t.is_enabled())
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    // === Mutation ===

    /// Retain only tools whose canonical name satisfies the predicate.
    pub fn retain(&mut self, f: impl Fn(&str) -> bool) {
        self.tools.retain(|name, _| f(name));
        // Also clean up aliases pointing to removed tools.
        self.aliases.retain(|_, canonical| self.tools.contains_key(canonical));
    }

    /// Remove a tool by its canonical name.
    pub fn remove(&mut self, name: &str) {
        self.tools.remove(name);
        self.aliases.retain(|_, canonical| canonical != name);
    }

    // === Introspection ===

    /// Number of tools in the registry.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
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
    use serde_json::Value;

    #[derive(Debug)]
    struct DummyTool {
        tool_name: &'static str,
        tool_aliases: &'static [&'static str],
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.tool_name
        }
        fn description(&self) -> &str {
            "A dummy tool for testing"
        }
        fn input_schema(&self) -> ToolInputSchema {
            ToolInputSchema::default()
        }
        fn aliases(&self) -> &[&str] {
            self.tool_aliases
        }
        async fn call(
            &self,
            _input: Value,
            _ctx: &mut ToolUseContext,
            _progress: Option<&dyn ToolProgressSender>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("ok"))
        }
    }

    #[test]
    fn register_and_find() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            tool_name: "MyTool",
            tool_aliases: &["mt", "my_tool"],
        }));

        assert!(reg.find("MyTool").is_some());
        assert!(reg.find("mt").is_some());
        assert!(reg.find("my_tool").is_some());
        assert!(reg.find("Nonexistent").is_none());
    }

    #[test]
    fn names_are_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            tool_name: "Zeta",
            tool_aliases: &[],
        }));
        reg.register(Arc::new(DummyTool {
            tool_name: "Alpha",
            tool_aliases: &[],
        }));
        reg.register(Arc::new(DummyTool {
            tool_name: "Mu",
            tool_aliases: &[],
        }));

        let names = reg.names();
        assert_eq!(names, vec!["Alpha", "Mu", "Zeta"]);
    }

    #[test]
    fn retain_removes_tools_and_aliases() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            tool_name: "Keep",
            tool_aliases: &["k"],
        }));
        reg.register(Arc::new(DummyTool {
            tool_name: "Drop",
            tool_aliases: &["d"],
        }));

        reg.retain(|name| name == "Keep");
        assert_eq!(reg.len(), 1);
        assert!(reg.find("Keep").is_some());
        assert!(reg.find("k").is_some());
        assert!(reg.find("Drop").is_none());
        assert!(reg.find("d").is_none());
    }

    #[test]
    fn remove_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            tool_name: "Foo",
            tool_aliases: &["f"],
        }));

        reg.remove("Foo");
        assert!(reg.find("Foo").is_none());
        assert!(reg.find("f").is_none());
        assert!(reg.is_empty());
    }

    #[test]
    fn api_tool_params_generation() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            tool_name: "Enabled",
            tool_aliases: &[],
        }));

        let params = reg.api_tool_params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "Enabled");
    }

    #[test]
    fn tool_definitions_generation() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool {
            tool_name: "TestTool",
            tool_aliases: &[],
        }));

        let defs = reg.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "TestTool");
    }

    #[cfg(feature = "built-in-tools")]
    #[test]
    fn default_registry_has_built_in_tools() {
        let reg = ToolRegistry::default_registry();
        assert!(reg.len() >= 6);
        assert!(reg.find("Bash").is_some());
        assert!(reg.find("Read").is_some());
        assert!(reg.find("Edit").is_some());
        assert!(reg.find("Write").is_some());
        assert!(reg.find("Glob").is_some());
        assert!(reg.find("Grep").is_some());
    }
}
