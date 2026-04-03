//! Model pricing data.
//!
//! Prices are in USD per token (not per 1K or per 1M tokens) so that cost
//! computation is a simple `tokens * price_per_token`.

/// Per-model pricing information.
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Cost per input token in USD.
    pub input_per_token: f64,
    /// Cost per output token in USD.
    pub output_per_token: f64,
    /// Cost per cache-write input token in USD.
    pub cache_creation_per_token: f64,
    /// Cost per cache-read input token in USD.
    pub cache_read_per_token: f64,
}

/// Look up pricing for a model by name.
///
/// Returns `None` for unknown models. The caller can fall back to a default
/// (e.g. Sonnet pricing) or return zero cost.
///
/// Pricing as of 2026-03-31 (USD per token).
pub fn get_pricing(model: &str) -> Option<ModelPricing> {
    // Normalize: strip date suffixes like "claude-sonnet-4-6-20260301"
    let key = normalize_model_name(model);

    match key {
        // --- Opus family ---
        s if s.contains("opus") => Some(ModelPricing {
            input_per_token: 15.0 / 1_000_000.0,
            output_per_token: 75.0 / 1_000_000.0,
            cache_creation_per_token: 18.75 / 1_000_000.0,
            cache_read_per_token: 1.50 / 1_000_000.0,
        }),

        // --- Sonnet family ---
        s if s.contains("sonnet") => Some(ModelPricing {
            input_per_token: 3.0 / 1_000_000.0,
            output_per_token: 15.0 / 1_000_000.0,
            cache_creation_per_token: 3.75 / 1_000_000.0,
            cache_read_per_token: 0.30 / 1_000_000.0,
        }),

        // --- Haiku family ---
        s if s.contains("haiku") => Some(ModelPricing {
            input_per_token: 0.25 / 1_000_000.0,
            output_per_token: 1.25 / 1_000_000.0,
            cache_creation_per_token: 0.30 / 1_000_000.0,
            cache_read_per_token: 0.03 / 1_000_000.0,
        }),

        _ => None,
    }
}

/// Strip date suffixes and whitespace from model identifiers to simplify
/// matching. For example:
///   "claude-sonnet-4-6-20260301" -> "claude-sonnet-4-6"
///   "claude-3-5-haiku-20241022" -> "claude-3-5-haiku"
fn normalize_model_name(model: &str) -> String {
    let trimmed = model.trim().to_lowercase();
    // Remove trailing date segment: "-YYYYMMDD"
    if let Some(pos) = trimmed.rfind('-') {
        let suffix = &trimmed[pos + 1..];
        if suffix.len() == 8 && suffix.chars().all(|c| c.is_ascii_digit()) {
            return trimmed[..pos].to_string();
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_pricing() {
        let p = get_pricing("claude-opus-4-6").unwrap();
        assert!((p.input_per_token - 15.0 / 1_000_000.0).abs() < f64::EPSILON);
        assert!((p.output_per_token - 75.0 / 1_000_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sonnet_pricing() {
        let p = get_pricing("claude-sonnet-4-6").unwrap();
        assert!((p.input_per_token - 3.0 / 1_000_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn haiku_pricing() {
        let p = get_pricing("claude-3-5-haiku-20241022").unwrap();
        assert!((p.input_per_token - 0.25 / 1_000_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(get_pricing("gpt-4o").is_none());
    }

    #[test]
    fn normalize_strips_date_suffix() {
        assert_eq!(
            normalize_model_name("claude-sonnet-4-6-20260301"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            normalize_model_name("claude-3-5-haiku-20241022"),
            "claude-3-5-haiku"
        );
    }

    #[test]
    fn normalize_preserves_name_without_date() {
        assert_eq!(
            normalize_model_name("claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
    }
}
