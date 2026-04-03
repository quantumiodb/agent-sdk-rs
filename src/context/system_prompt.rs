//! System prompt assembly.
//!
//! Combines environment info, the user's system prompt, and any appended
//! instructions into a single string sent to the model.

/// Build the complete system prompt from agent options.
///
/// The structure mirrors the TypeScript SDK:
///
/// 1. **Environment block** -- current date, working directory, OS.
/// 2. **User system prompt** -- replaces the environment block if provided
///    (but the environment info is still prepended as a preamble).
/// 3. **Appended prompt** -- always added at the end, after a separator.
pub fn build_system_prompt(
    system_prompt: &Option<String>,
    append_system_prompt: &Option<String>,
    cwd: &std::path::Path,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // 1. Environment preamble.
    let env_info = build_env_info(cwd);
    parts.push(env_info);

    // 2. User-provided system prompt.
    if let Some(prompt) = system_prompt {
        parts.push(prompt.clone());
    }

    // 3. Appended instructions.
    if let Some(append) = append_system_prompt {
        parts.push(append.clone());
    }

    parts.join("\n\n")
}

/// Build the environment information block.
fn build_env_info(cwd: &std::path::Path) -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();

    format!(
        "Environment:\n\
         - Date: {date}\n\
         - OS: {os} ({arch})\n\
         - Working directory: {}",
        cwd.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn env_only() {
        let prompt = build_system_prompt(&None, &None, &PathBuf::from("/tmp/test"));
        assert!(prompt.contains("Environment:"));
        assert!(prompt.contains("/tmp/test"));
    }

    #[test]
    fn with_user_prompt() {
        let prompt = build_system_prompt(
            &Some("You are a helpful assistant.".to_string()),
            &None,
            &PathBuf::from("/home/user"),
        );
        assert!(prompt.contains("Environment:"));
        assert!(prompt.contains("You are a helpful assistant."));
    }

    #[test]
    fn with_append_prompt() {
        let prompt = build_system_prompt(
            &None,
            &Some("Always respond in JSON.".to_string()),
            &PathBuf::from("/tmp"),
        );
        assert!(prompt.contains("Always respond in JSON."));
    }

    #[test]
    fn with_both_prompts() {
        let prompt = build_system_prompt(
            &Some("Be concise.".to_string()),
            &Some("Output JSON.".to_string()),
            &PathBuf::from("/tmp"),
        );
        assert!(prompt.contains("Be concise."));
        assert!(prompt.contains("Output JSON."));
        // Env info should come first.
        let env_pos = prompt.find("Environment:").unwrap();
        let user_pos = prompt.find("Be concise.").unwrap();
        let append_pos = prompt.find("Output JSON.").unwrap();
        assert!(env_pos < user_pos);
        assert!(user_pos < append_pos);
    }
}
