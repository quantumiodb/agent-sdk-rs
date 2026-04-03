//! Anthropic native provider implementation.
//!
//! Sends streaming requests to `https://api.anthropic.com/v1/messages` and
//! parses the response as SSE events.

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use tracing::{debug, trace};

use super::client::{ApiError, ApiProvider, ApiStream};
use super::retry::parse_retry_after;
use super::streaming::SseStream;
use super::types::MessageRequest;

/// The default Anthropic API base URL.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// The Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Models known to be supported by the Anthropic API.
const SUPPORTED_MODELS: &[&str] = &[
    "claude-sonnet-4-6",
    "claude-sonnet-4-20250514",
    "claude-opus-4-6",
    "claude-opus-4-20250514",
    "claude-haiku-4-5-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
];

/// Anthropic Messages API provider.
///
/// Formats requests with Anthropic-specific headers and authentication,
/// sends them to the Messages API endpoint, and wraps the response body
/// in an SSE stream parser.
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string())
    }

    /// Create a new provider with a custom base URL (useful for proxies or testing).
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5-minute timeout for streaming
            .build()
            .expect("Failed to build HTTP client");

        Self {
            api_key,
            base_url,
            http_client,
        }
    }

    /// Build the request headers.
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).expect("Invalid API key header value"),
        );

        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );

        // Enable streaming beta features.
        headers.insert(
            "anthropic-beta",
            HeaderValue::from_static("prompt-caching-2024-07-31"),
        );

        headers
    }

    /// Build the full endpoint URL.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url.trim_end_matches('/'))
    }
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

#[async_trait]
impl ApiProvider for AnthropicProvider {
    async fn stream_message(&self, request: MessageRequest) -> Result<ApiStream, ApiError> {
        let url = self.messages_url();
        let headers = self.build_headers();

        debug!(
            model = %request.model,
            max_tokens = request.max_tokens,
            messages = request.messages.len(),
            tools = request.tools.len(),
            "Sending streaming request to Anthropic API"
        );

        // Serialize the request body.
        let body = serde_json::to_string(&request).map_err(|e| ApiError::Other {
            message: format!("Failed to serialize request: {e}"),
            source: Some(Box::new(e)),
        })?;

        trace!(url = %url, body_len = body.len(), "POST request");

        // Send the HTTP request.
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ApiError::Timeout {
                        duration: std::time::Duration::from_secs(300),
                    }
                } else if e.is_connect() {
                    ApiError::Network {
                        message: format!("Connection failed: {e}"),
                        source: e,
                    }
                } else {
                    ApiError::Network {
                        message: format!("Request failed: {e}"),
                        source: e,
                    }
                }
            })?;

        let status = response.status();

        // Extract Retry-After header before consuming the response.
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after);

        // Handle non-success status codes.
        if !status.is_success() {
            let status_code = status.as_u16();
            let error_body = response.text().await.unwrap_or_default();

            debug!(
                status = status_code,
                body = %error_body,
                "Anthropic API returned error"
            );

            return Err(ApiError::from_status(status_code, &error_body, retry_after));
        }

        // Wrap the response body stream in the SSE parser.
        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream);

        Ok(Box::pin(sse_stream))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn supported_models(&self) -> &[&str] {
        SUPPORTED_MODELS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_url_default() {
        let provider = AnthropicProvider::new("test-key".into());
        assert_eq!(
            provider.messages_url(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_messages_url_custom() {
        let provider =
            AnthropicProvider::with_base_url("test-key".into(), "http://localhost:8080/".into());
        assert_eq!(
            provider.messages_url(),
            "http://localhost:8080/v1/messages"
        );
    }

    #[test]
    fn test_messages_url_no_trailing_slash() {
        let provider =
            AnthropicProvider::with_base_url("test-key".into(), "http://localhost:8080".into());
        assert_eq!(
            provider.messages_url(),
            "http://localhost:8080/v1/messages"
        );
    }

    #[test]
    fn test_headers_contain_required_fields() {
        let provider = AnthropicProvider::new("sk-ant-test-key".into());
        let headers = provider.build_headers();

        assert_eq!(
            headers.get("content-type").unwrap(),
            "application/json"
        );
        assert_eq!(headers.get("x-api-key").unwrap(), "sk-ant-test-key");
        assert_eq!(headers.get("anthropic-version").unwrap(), "2023-06-01");
    }

    #[test]
    fn test_supported_models_non_empty() {
        let provider = AnthropicProvider::new("test-key".into());
        assert!(!provider.supported_models().is_empty());
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let provider = AnthropicProvider::new("super-secret-key".into());
        let debug_output = format!("{provider:?}");
        assert!(!debug_output.contains("super-secret-key"));
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::new("test-key".into());
        assert_eq!(provider.name(), "anthropic");
    }
}
