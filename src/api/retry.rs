//! Exponential backoff retry logic for transient API errors.
//!
//! Handles HTTP 429 (rate limit), 529 (overloaded), and 500 (server error)
//! responses. Respects the `Retry-After` header when present.

use std::time::Duration;
use tracing::{debug, warn};

use super::client::ApiError;
use super::types::RetryConfig;

/// Determines whether an error is retryable.
pub fn is_retryable(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::RateLimit { .. } | ApiError::Overloaded { .. } | ApiError::ServerError { .. }
    )
}

/// Determines whether an HTTP status code is retryable.
pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 529)
}

/// Computes the delay before the next retry attempt using exponential backoff.
///
/// If `retry_after` is provided (from the `Retry-After` header), it takes
/// precedence over the computed delay.
pub fn compute_delay(config: &RetryConfig, attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(server_delay) = retry_after {
        // Respect Retry-After header, but cap at max_delay.
        let max = Duration::from_millis(config.max_delay_ms);
        return server_delay.min(max);
    }

    let base_ms = config.initial_delay_ms as f64;
    let multiplier = config.backoff_multiplier.powi(attempt as i32);
    let delay_ms = (base_ms * multiplier).min(config.max_delay_ms as f64) as u64;

    Duration::from_millis(delay_ms)
}

/// Parses a `Retry-After` header value into a [`Duration`].
///
/// The header can be either:
/// - An integer number of seconds (e.g. `"30"`)
/// - An HTTP-date (e.g. `"Wed, 21 Oct 2015 07:28:00 GMT"`) — we only support
///   the integer form for simplicity.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Execute an async operation with retry logic.
///
/// Calls `operation` up to `config.max_retries + 1` times (one initial attempt
/// plus retries). Only retries on errors that satisfy [`is_retryable`].
///
/// The `retry_after_fn` closure extracts an optional `Retry-After` duration from
/// the error, allowing the caller to pass server-specified backoff durations.
pub async fn with_retry<F, Fut, T>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T, ApiError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, ApiError>>,
{
    let mut attempt: u32 = 0;

    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if !is_retryable(&err) || attempt >= config.max_retries {
                    if attempt > 0 {
                        warn!(
                            attempts = attempt + 1,
                            retryable = is_retryable(&err),
                            "Giving up after {} attempt(s): {}",
                            attempt + 1,
                            err
                        );
                    }
                    return Err(err);
                }

                let retry_after = err.retry_after();
                let delay = compute_delay(config, attempt, retry_after);

                debug!(
                    attempt = attempt + 1,
                    max_retries = config.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    error = %err,
                    "Retrying after transient error"
                );

                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_delay_exponential() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay_ms: 100,
            max_delay_ms: 10_000,
            backoff_multiplier: 2.0,
        };

        assert_eq!(compute_delay(&config, 0, None), Duration::from_millis(100));
        assert_eq!(compute_delay(&config, 1, None), Duration::from_millis(200));
        assert_eq!(compute_delay(&config, 2, None), Duration::from_millis(400));
        assert_eq!(compute_delay(&config, 3, None), Duration::from_millis(800));
    }

    #[test]
    fn test_compute_delay_caps_at_max() {
        let config = RetryConfig {
            max_retries: 10,
            initial_delay_ms: 1000,
            max_delay_ms: 5000,
            backoff_multiplier: 3.0,
        };

        // 1000 * 3^3 = 27000, but capped at 5000
        assert_eq!(compute_delay(&config, 3, None), Duration::from_millis(5000));
    }

    #[test]
    fn test_compute_delay_respects_retry_after() {
        let config = RetryConfig::default();
        let retry_after = Some(Duration::from_secs(5));

        assert_eq!(
            compute_delay(&config, 0, retry_after),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_compute_delay_caps_retry_after_at_max() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 2000,
            backoff_multiplier: 2.0,
        };
        let retry_after = Some(Duration::from_secs(60));

        assert_eq!(
            compute_delay(&config, 0, retry_after),
            Duration::from_millis(2000)
        );
    }

    #[test]
    fn test_parse_retry_after_seconds() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after(" 5 "), Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_parse_retry_after_invalid() {
        assert_eq!(parse_retry_after("not-a-number"), None);
        // HTTP-date format is not supported; returns None.
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"),
            None
        );
    }

    #[test]
    fn test_is_retryable_status() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(529));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(403));
        assert!(!is_retryable_status(404));
    }

    #[tokio::test]
    async fn test_with_retry_succeeds_first_try() {
        let config = RetryConfig::default();
        let result = with_retry(&config, || async { Ok::<_, ApiError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_retry_non_retryable_fails_immediately() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
        };

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();

        let result: Result<(), ApiError> = with_retry(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err(ApiError::Authentication {
                    message: "bad key".into(),
                })
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_retries_on_rate_limit() {
        let config = RetryConfig {
            max_retries: 2,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
        };

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();

        let result: Result<i32, ApiError> = with_retry(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n < 2 {
                    Err(ApiError::RateLimit {
                        message: "rate limited".into(),
                        retry_after: None,
                    })
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
