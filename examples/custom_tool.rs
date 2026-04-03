//! Custom tool example: define and register a domain-specific tool.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --example custom_tool

use std::sync::Arc;

use async_trait::async_trait;
use claude_agent_sdk::{
    Agent, AgentOptions, ApiClientConfig, PermissionCheckResult, Tool, ToolError, ToolInputSchema,
    ToolProgressSender, ToolResult, ToolUseContext,
};
use serde_json::Value;

/// A mock "database query" tool for demonstration purposes.
#[derive(Debug)]
struct MockDatabaseTool;

#[async_trait]
impl Tool for MockDatabaseTool {
    fn name(&self) -> &str {
        "DatabaseQuery"
    }

    fn description(&self) -> &str {
        "Execute a read-only SQL query against the application database. \
         Only SELECT statements are permitted."
    }

    fn input_schema(&self) -> ToolInputSchema {
        ToolInputSchema {
            schema_type: "object".into(),
            properties: serde_json::from_value(serde_json::json!({
                "query": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default: 100)"
                }
            }))
            .unwrap(),
            required: vec!["query".into()],
            additional_properties: false,
        }
    }

    fn is_read_only(&self, _input: &Value) -> bool {
        true
    }

    fn is_concurrency_safe(&self, _input: &Value) -> bool {
        true
    }

    async fn validate_input(&self, input: &Value, _ctx: &ToolUseContext) -> Result<(), String> {
        let query = input["query"].as_str().unwrap_or("");
        let trimmed = query.trim().to_uppercase();
        if !trimmed.starts_with("SELECT") {
            return Err(format!(
                "Only SELECT queries are allowed, got: {}",
                &query[..query.len().min(50)]
            ));
        }
        Ok(())
    }

    async fn check_permissions(
        &self,
        input: &Value,
        _ctx: &ToolUseContext,
    ) -> PermissionCheckResult {
        PermissionCheckResult::Allow {
            updated_input: input.clone(),
        }
    }

    async fn call(
        &self,
        input: Value,
        _ctx: &mut ToolUseContext,
        _progress: Option<&dyn ToolProgressSender>,
    ) -> Result<ToolResult, ToolError> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("query is required".into()))?;
        let limit = input["limit"].as_u64().unwrap_or(100);

        // Mock response — in a real tool, execute against your database
        let mock_response = format!(
            "Query: {query}\nLimit: {limit}\n\n\
             Results:\n\
             id | name       | signup_date\n\
             ---+------------+------------\n\
             1  | Alice Smith | 2026-03-28\n\
             2  | Bob Jones   | 2026-03-29\n\
             3  | Carol White | 2026-03-30\n\
             \n(3 rows)"
        );

        Ok(ToolResult::text(mock_response))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let db_tool = Arc::new(MockDatabaseTool);

    let mut agent = Agent::new(AgentOptions {
        api: ApiClientConfig::FromEnv,
        custom_tools: vec![db_tool],
        system_prompt: Some(
            "You are a data analyst. Use the DatabaseQuery tool to answer questions \
             about user data. Always explain your SQL query before running it."
                .into(),
        ),
        ..Default::default()
    })
    .await?;

    let result = agent.prompt("How many users signed up this week?").await?;

    println!("{}", result.text);
    println!("\nCost: ${:.6}", result.cost_usd);

    agent.close().await?;
    Ok(())
}
