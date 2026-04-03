//! Cost tracking for Claude API usage.
//!
//! Provides [`CostTracker`] for thread-safe accumulation of token usage and
//! cost data across multiple API calls, and [`get_pricing`] for looking up
//! per-model pricing.

pub mod tracker;
pub mod pricing;

pub use tracker::{CostTracker, CostSummary};
pub use pricing::{ModelPricing, get_pricing};
