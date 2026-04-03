//! Unified API client with provider abstraction and retry logic.
//!
//! [`ApiClient`] wraps an [`ApiProvider`] implementation and adds exponential
//! backoff retry, usage tracking, and convenience constructors for common
//! providers (Anthropic, OpenAI-compatible, Bedrock, Vertex).

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use tracing::{debug, warn};

use super::streaming::SseStream;
use super::types::{ApiStreamEvent, MessageRequest, RetryConfig};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during API operations.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Network error: {message}")]
    Network {
        message: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("Invalid response: {message}")]
    InvalidResponse { message: String, body: String },

    #[error("Rate limited (retry after {retry_after_ms:?}ms)")]
    RateLimited { retry_after_ms: Option<u64> },

    #[error("Server overloaded")]
    Overloaded,

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("API error ({status}): {message}")]
    ApiError {
        status: u16,
        message: String,
        error_type: String,
    },

    #[error("Configuration error: {0}")]
    Configuration(String),
}

// ---------------------------------------------------------------------------
// ApiStream type alias
// ---------------------------------------------------------------------------

/// A stream of parsed SSE events from the API.
pub type ApiStream = Pin<Box<dyn Stream<Item = Result<ApiStreamEvent, ApiError>> + Send>>;

// ---------------------------------------------------------------------------
// ApiProvider trait
// ---------------------------------------------------------------------------

/// Abstraction over different API providers (Anthropic, Bedrock, Vertex, etc.).
///
/// Implementations handle provider-specific authentication, endpoints, and
/// request formatting.
#[async_trait]
pub trait ApiProvider: Send + Sync {
    /// Send a streaming message request and return an event stream.
    async fn stream_message(&self, request: MessageRequest) -> Result<ApiStream, ApiError>;

    /// Provider name for logging and debugging.
    fn name(&self) -> &str;

    /// Models supported by this provider.
    fn supported_models(&self) -> &[&str];
}

// ---------------------------------------------------------------------------
// ApiClient
// ---------------------------------------------------------------------------

/// Unified API client wrapping a provider with retry logic and usage tracking.
pub struct ApiClient {
    provider: Box<dyn ApiProvider>,
    retry_config: RetryConfig,
    #[allow(dead_code)]
    http_client: reqwest::Client,
}

impl std::fmt::Debug for ApiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiClient")
            .field("provider", &self.provider.name())
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl ApiClient {
    /// Create an ApiClient for the Anthropic API with the given API key.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            provider: Box::new(AnthropicProvider::new(api_key)),
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient for the official OpenAI API.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            provider: Box::new(super::openai_compat::OpenAICompatProvider::openai(api_key)),
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient for any OpenAI-compatible endpoint.
    ///
    /// Works with Groq, Together AI, Fireworks, Azure OpenAI, local Ollama, etc.
    /// Proxy is inherited from environment variables (`http_proxy`, etc.).
    pub fn openai_compat(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            provider: Box::new(super::openai_compat::OpenAICompatProvider::new(
                api_key, base_url,
            )),
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient for any OpenAI-compatible endpoint, bypassing the system proxy.
    ///
    /// Use this for local servers (Ollama, vLLM, LM Studio) when a system proxy
    /// is configured but cannot reach the local endpoint.
    pub fn openai_compat_no_proxy(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            provider: Box::new(super::openai_compat::OpenAICompatProvider::no_proxy(
                api_key, base_url,
            )),
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient for any OpenAI-compatible endpoint with extra body parameters,
    /// bypassing the system proxy.
    ///
    /// `extra_body` fields are merged into every request body.  Use this for
    /// provider-specific options not covered by the standard OpenAI schema, e.g.:
    ///
    /// ```rust,ignore
    /// ApiClient::openai_compat_with_options(key, url, Some(json!({"think": false})))
    /// ```
    pub fn openai_compat_with_options(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        extra_body: Option<serde_json::Value>,
    ) -> Self {
        Self {
            provider: Box::new(
                super::openai_compat::OpenAICompatProvider::no_proxy_with_options(
                    api_key, base_url, extra_body,
                ),
            ),
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient for Ollama using the native `/api/chat` endpoint.
    ///
    /// `think` controls whether the model reasons before responding:
    /// - `Some(true)`  → enable thinking
    /// - `Some(false)` → disable thinking (fast, no reasoning tokens)
    /// - `None`        → let the model decide (default)
    pub fn ollama(base_url: impl Into<String>, think: Option<bool>) -> Self {
        Self {
            provider: Box::new(super::ollama::OllamaProvider::new(base_url, think)),
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient for AWS Bedrock.
    pub fn bedrock(_region: impl Into<String>) -> Self {
        todo!("Bedrock provider not yet implemented")
    }

    /// Create an ApiClient for Google Vertex AI.
    pub fn vertex(_project_id: impl Into<String>, _region: impl Into<String>) -> Self {
        todo!("Vertex provider not yet implemented")
    }

    /// Create an ApiClient with a custom provider.
    pub fn custom(provider: Box<dyn ApiProvider>) -> Self {
        Self {
            provider,
            retry_config: RetryConfig::default(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create an ApiClient from environment variables.
    ///
    /// Checks `ANTHROPIC_API_KEY` first, then falls back to other provider
    /// environment variables.
    /// Create an ApiClient from environment variables.
    ///
    /// Detection order:
    /// 1. `ANTHROPIC_BASE_URL` is set → OpenAI-compat provider pointed at that URL
    ///    (covers Ollama, LiteLLM proxy, vLLM, etc.)
    ///    Auth key from `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` (default: `"ollama"`).
    /// 2. `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` → native Anthropic provider
    /// 3. `OPENAI_API_KEY` → OpenAI-compat provider (uses `OPENAI_BASE_URL` if set)
    pub fn from_env() -> Result<Self, ApiError> {
        // ── Provider detection order ──────────────────────────────────────────
        //
        // 1. OLLAMA_BASE_URL            → native OllamaProvider (/api/chat)
        // 2. ANTHROPIC_BASE_URL
        //      + ANTHROPIC_API_FORMAT=anthropic → AnthropicProvider (custom URL)
        //      (default)                        → OpenAICompatProvider
        // 3. ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN (no BASE_URL)
        //                               → AnthropicProvider (api.anthropic.com)
        // 4. OPENAI_API_KEY             → OpenAICompatProvider
        //                                 (OPENAI_BASE_URL if set, else api.openai.com)
        //
        // ── Ollama (native /api/chat protocol) ───────────────────────────────
        if let Ok(base_url) = std::env::var("OLLAMA_BASE_URL") {
            let think = std::env::var("OLLAMA_THINK")
                .ok()
                .and_then(|v| match v.to_lowercase().as_str() {
                    "true" | "1"  => Some(true),
                    "false" | "0" => Some(false),
                    _ => None,
                });
            return Ok(Self::ollama(base_url, think));
        }

        // Custom base URL handling.
        //
        // ANTHROPIC_API_FORMAT controls the wire format sent to the custom URL:
        //   "anthropic" → native Anthropic Messages API format (/v1/messages)
        //   "openai"    → OpenAI Chat Completions format (/v1/chat/completions) [default]
        //
        // Example — LiteLLM proxy speaking Anthropic format:
        //   ANTHROPIC_BASE_URL=http://proxy:4000
        //   ANTHROPIC_AUTH_TOKEN=sk-...
        //   ANTHROPIC_API_FORMAT=anthropic
        if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
            let api_key = std::env::var("ANTHROPIC_AUTH_TOKEN")
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .unwrap_or_else(|_| "ollama".to_string());

            let format = std::env::var("ANTHROPIC_API_FORMAT")
                .unwrap_or_default();

            if format.eq_ignore_ascii_case("anthropic") {
                return Ok(Self {
                    provider: Box::new(AnthropicProvider::with_base_url(api_key, base_url)),
                    retry_config: RetryConfig::default(),
                    http_client: reqwest::Client::new(),
                });
            }

            // Default: OpenAI-compat mode (Ollama / LiteLLM / vLLM)
            let no_proxy = std::env::var("ANTHROPIC_NO_PROXY")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false)
                || is_local_or_private_url(&base_url);

            return if no_proxy {
                Ok(Self::openai_compat_no_proxy(api_key, base_url))
            } else {
                Ok(Self::openai_compat(api_key, base_url))
            };
        }

        // Native Anthropic
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_AUTH_TOKEN"))
        {
            return Ok(Self::anthropic(key));
        }

        // OpenAI / compatible
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".to_string());
            return Ok(Self::openai_compat(key, base_url));
        }

        Err(ApiError::Configuration(
            "No API key found. Set ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or OPENAI_API_KEY."
                .to_string(),
        ))
    }

    /// Set the retry configuration.
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Send a streaming message request with retry logic.
    pub async fn stream_message(&self, request: MessageRequest) -> Result<ApiStream, ApiError> {
        let mut last_error = None;
        let mut delay = Duration::from_millis(self.retry_config.initial_delay_ms);

        for attempt in 0..=self.retry_config.max_retries {
            if attempt > 0 {
                debug!(attempt, delay_ms = delay.as_millis(), "Retrying API request");
                tokio::time::sleep(delay).await;
                delay = Duration::from_millis(
                    (delay.as_millis() as f64 * self.retry_config.backoff_multiplier) as u64,
                )
                .min(Duration::from_millis(self.retry_config.max_delay_ms));
            }

            match self.provider.stream_message(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    let retryable = matches!(
                        &e,
                        ApiError::RateLimited { .. }
                            | ApiError::Overloaded
                            | ApiError::Network { .. }
                    );
                    if !retryable || attempt == self.retry_config.max_retries {
                        return Err(e);
                    }
                    warn!(
                        attempt,
                        error = %e,
                        "API request failed, will retry"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(ApiError::Configuration(
            "No retry attempts made".to_string(),
        )))
    }

    /// Get the provider name.
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

// ---------------------------------------------------------------------------
// AnthropicProvider (default provider)
// ---------------------------------------------------------------------------

/// Built-in provider for the Anthropic Messages API.
struct AnthropicProvider {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl AnthropicProvider {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.anthropic.com".to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ApiProvider for AnthropicProvider {
    async fn stream_message(&self, request: MessageRequest) -> Result<ApiStream, ApiError> {
        let url = format!("{}/v1/messages", self.base_url);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|e| ApiError::Configuration(format!("Invalid API key header: {e}")))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );

        let response: reqwest::Response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::Network {
                message: format!("Failed to send request: {e}"),
                source: e,
            })?;

        let status = response.status().as_u16();
        if status == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v: &reqwest::header::HeaderValue| v.to_str().ok())
                .and_then(|v: &str| v.parse::<u64>().ok())
                .map(|s| s * 1000);
            return Err(ApiError::RateLimited {
                retry_after_ms: retry_after,
            });
        }
        if status == 529 {
            return Err(ApiError::Overloaded);
        }
        if status == 401 {
            return Err(ApiError::AuthenticationFailed(
                "Invalid API key".to_string(),
            ));
        }
        if status >= 400 {
            let body: String = response.text().await.unwrap_or_default();
            return Err(ApiError::ApiError {
                status,
                message: body,
                error_type: "api_error".to_string(),
            });
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream);
        Ok(Box::pin(sse_stream))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn supported_models(&self) -> &[&str] {
        &[
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-3-5",
        ]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true if the URL looks like a local or private-network address
/// that should bypass any system proxy.
fn is_local_or_private_url(url: &str) -> bool {
    // Strip scheme
    let host = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    // Remove path/port
    let host = host.split('/').next().unwrap_or(host);
    let host = host.split(':').next().unwrap_or(host);

    matches!(
        host,
        "localhost" | "127.0.0.1" | "::1" | "0.0.0.0"
    ) || host.starts_with("192.168.")
        || host.starts_with("10.")
        || host.starts_with("172.")
}
