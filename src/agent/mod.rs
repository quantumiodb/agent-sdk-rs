//! Core Agent: agentic loop, session management, and orchestration.
//!
//! The [`Agent`] struct is the main entry point for running an autonomous agent.
//! It manages the conversation loop, tool execution, MCP connections, and
//! streaming events.

mod agent;
mod agent_loop;

pub use agent::{Agent, AgentError, AgentOptions, ApiClientConfig, QueryResult, SubagentDefinition};
