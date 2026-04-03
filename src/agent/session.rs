//! Session persistence (placeholder).
//!
//! This module will provide session serialization and deserialization so that
//! an agent's conversation state can be saved to disk and resumed later.
//!
//! TODO: Implement session save/load:
//! - Serialize messages, tool state, and cost tracker to JSON.
//! - Deserialize and reconstruct an Agent from saved state.
//! - Support incremental append for long-running sessions.
