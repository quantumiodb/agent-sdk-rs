//! Thread-safe cost tracker.
//!
//! Accumulates token usage, API call durations, and tool durations across
//! the lifetime of an agent session.

use std::sync::RwLock;
use std::time::Duration;

use crate::types::Usage;

use super::pricing::get_pricing;

// ---------------------------------------------------------------------------
// CostSummary
// ---------------------------------------------------------------------------

/// A snapshot of accumulated costs and usage.
#[derive(Debug, Clone)]
pub struct CostSummary {
    /// Total token usage across all API calls.
    pub total_usage: Usage,
    /// Total estimated cost in USD.
    pub total_cost_usd: f64,
    /// Number of API calls made.
    pub api_call_count: u32,
    /// Total time spent waiting for API responses.
    pub total_api_duration: Duration,
    /// Total time spent executing tools.
    pub total_tool_duration: Duration,
}

// ---------------------------------------------------------------------------
// CostTracker
// ---------------------------------------------------------------------------

/// Internal mutable state protected by an `RwLock`.
struct TrackerState {
    #[allow(dead_code)]
    model: String,
    total_usage: Usage,
    total_cost_usd: f64,
    api_call_count: u32,
    total_api_duration: Duration,
    total_tool_duration: Duration,
}

/// Thread-safe cost tracker for an agent session.
///
/// Uses `std::sync::RwLock` (not `tokio::sync`) because all mutations are
/// fast (no async work under the lock) and this allows use from both sync
/// and async contexts.
pub struct CostTracker {
    state: RwLock<TrackerState>,
}

impl std::fmt::Debug for CostTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostTracker")
            .field("total_cost_usd", &self.total_cost())
            .finish()
    }
}

impl CostTracker {
    /// Create a new tracker for the given model.
    pub fn new(model: &str) -> Self {
        Self {
            state: RwLock::new(TrackerState {
                model: model.to_string(),
                total_usage: Usage::default(),
                total_cost_usd: 0.0,
                api_call_count: 0,
                total_api_duration: Duration::ZERO,
                total_tool_duration: Duration::ZERO,
            }),
        }
    }

    /// Record token usage from a single API call and update the cost estimate.
    pub fn add_usage(&self, model: &str, usage: &Usage) {
        let pricing = get_pricing(model);
        let cost = pricing
            .map(|p| {
                (usage.input_tokens as f64 * p.input_per_token)
                    + (usage.output_tokens as f64 * p.output_per_token)
                    + (usage.cache_creation_input_tokens as f64 * p.cache_creation_per_token)
                    + (usage.cache_read_input_tokens as f64 * p.cache_read_per_token)
            })
            .unwrap_or(0.0);

        if let Ok(mut state) = self.state.write() {
            state.total_usage.accumulate(usage);
            state.total_cost_usd += cost;
            state.api_call_count += 1;
        }
    }

    /// Record the duration of a single API call.
    pub fn add_api_duration(&self, duration: Duration) {
        if let Ok(mut state) = self.state.write() {
            state.total_api_duration += duration;
        }
    }

    /// Record the duration of a single tool execution.
    pub fn add_tool_duration(&self, duration: Duration) {
        if let Ok(mut state) = self.state.write() {
            state.total_tool_duration += duration;
        }
    }

    /// Get the total estimated cost in USD.
    pub fn total_cost(&self) -> f64 {
        self.state
            .read()
            .map(|s| s.total_cost_usd)
            .unwrap_or(0.0)
    }

    /// Get the total accumulated token usage.
    pub fn total_usage(&self) -> Usage {
        self.state
            .read()
            .map(|s| s.total_usage.clone())
            .unwrap_or_default()
    }

    /// Get a full cost summary snapshot.
    pub fn summary(&self) -> CostSummary {
        self.state
            .read()
            .map(|s| CostSummary {
                total_usage: s.total_usage.clone(),
                total_cost_usd: s.total_cost_usd,
                api_call_count: s.api_call_count,
                total_api_duration: s.total_api_duration,
                total_tool_duration: s.total_tool_duration,
            })
            .unwrap_or_else(|_| CostSummary {
                total_usage: Usage::default(),
                total_cost_usd: 0.0,
                api_call_count: 0,
                total_api_duration: Duration::ZERO,
                total_tool_duration: Duration::ZERO,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_has_zero_cost() {
        let tracker = CostTracker::new("claude-sonnet-4-6");
        assert_eq!(tracker.total_cost(), 0.0);
        assert!(tracker.total_usage().is_empty());
    }

    #[test]
    fn add_usage_accumulates() {
        let tracker = CostTracker::new("claude-sonnet-4-6");

        let u1 = Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let u2 = Usage {
            input_tokens: 200,
            output_tokens: 100,
            ..Default::default()
        };

        tracker.add_usage("claude-sonnet-4-6", &u1);
        tracker.add_usage("claude-sonnet-4-6", &u2);

        let usage = tracker.total_usage();
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
    }

    #[test]
    fn add_usage_computes_cost() {
        let tracker = CostTracker::new("claude-sonnet-4-6");

        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        tracker.add_usage("claude-sonnet-4-6", &usage);

        // Sonnet: $3/M input + $15/M output = $18
        let cost = tracker.total_cost();
        assert!((cost - 18.0).abs() < 0.01, "Expected ~$18, got ${cost}");
    }

    #[test]
    fn add_usage_unknown_model_zero_cost() {
        let tracker = CostTracker::new("unknown-model");

        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        tracker.add_usage("unknown-model", &usage);

        assert_eq!(tracker.total_cost(), 0.0);
        // But usage is still tracked.
        assert_eq!(tracker.total_usage().input_tokens, 1000);
    }

    #[test]
    fn summary_includes_durations() {
        let tracker = CostTracker::new("claude-sonnet-4-6");
        tracker.add_api_duration(Duration::from_millis(500));
        tracker.add_api_duration(Duration::from_millis(300));
        tracker.add_tool_duration(Duration::from_millis(100));

        let summary = tracker.summary();
        assert_eq!(summary.total_api_duration, Duration::from_millis(800));
        assert_eq!(summary.total_tool_duration, Duration::from_millis(100));
    }

    #[test]
    fn summary_api_call_count() {
        let tracker = CostTracker::new("claude-sonnet-4-6");
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };

        tracker.add_usage("claude-sonnet-4-6", &usage);
        tracker.add_usage("claude-sonnet-4-6", &usage);
        tracker.add_usage("claude-sonnet-4-6", &usage);

        assert_eq!(tracker.summary().api_call_count, 3);
    }
}
