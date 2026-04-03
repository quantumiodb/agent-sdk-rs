//! API client for communicating with the Anthropic Messages API.
//!
//! Provides the [`ApiClient`] (unified client with retry logic),
//! [`ApiProvider`] trait (abstraction over Anthropic/Bedrock/Vertex/OpenAI),
//! and SSE stream parsing.

pub mod client;
pub mod ollama;
pub mod openai_compat;
pub mod streaming;
pub mod types;

pub use client::{ApiClient, ApiError, ApiProvider, ApiStream};
pub use ollama::OllamaProvider;
pub use openai_compat::OpenAICompatProvider;
pub use streaming::SseStream;
pub use types::{
    ApiErrorBody, ApiMessageStart, ApiStreamEvent, DeltaUsage, MessageDeltaBody, MessageRequest,
    MessageResponse, RetryConfig,
};
