//! Context assembly: system prompt construction and context compaction.

pub mod system_prompt;
pub mod compact;

pub use system_prompt::build_system_prompt;
pub use compact::compact_messages;
