//! Main TUI application with async event loop.
//!
//! Integrates crossterm terminal events with the Agent's streaming SDK messages
//! via tokio select. Renders the UI using ratatui.

use std::io;

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    Terminal,
};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentError, AgentOptions};
use crate::types::{ContentBlock, ContentDelta, SDKMessage, ToolResultContent};

use super::components::{self, Header, HelpOverlay, InputBox, MessageList, StatusBar};
use super::state::{AgentStatus, AppState, ChatRole};

/// The interactive TUI application.
pub struct TuiApp {
    terminal: Terminal<CrosstermBackend<io::Stderr>>,
    state: AppState,
    agent: Agent,
    /// Receiver for agent streaming events (active during a query).
    agent_rx: Option<mpsc::Receiver<SDKMessage>>,
    /// Join handle for the running agent task.
    agent_handle: Option<tokio::task::JoinHandle<Result<(), AgentError>>>,
}

impl TuiApp {
    /// Create a new TUI application from agent options.
    pub async fn new(options: AgentOptions) -> Result<Self, AgentError> {
        let model = options.model.clone();
        let provider = detect_provider(&options);
        let agent = Agent::new(options).await?;

        // Terminal setup.
        enable_raw_mode().map_err(|e| AgentError::Other(e.to_string()))?;
        let mut stderr = io::stderr();
        execute!(stderr, EnterAlternateScreen, EnableMouseCapture)
            .map_err(|e| AgentError::Other(e.to_string()))?;
        let backend = CrosstermBackend::new(stderr);
        let terminal =
            Terminal::new(backend).map_err(|e| AgentError::Other(e.to_string()))?;

        let session_id = uuid::Uuid::new_v4().to_string();

        let state = AppState::new(model, provider, session_id);

        Ok(Self {
            terminal,
            state,
            agent,
            agent_rx: None,
            agent_handle: None,
        })
    }

    /// Run the TUI event loop. Returns when the user quits.
    pub async fn run(&mut self) -> Result<(), AgentError> {
        // Welcome message.
        self.state.push_message(
            ChatRole::System,
            format!(
                "Welcome to Claude Agent TUI! Model: {} | Type /help for commands.",
                self.state.model
            ),
        );

        // Initial draw.
        self.draw()?;

        loop {
            if self.state.should_quit {
                break;
            }

            // Use tokio select to handle both terminal events and agent messages.
            tokio::select! {
                // Poll crossterm events with a small timeout.
                event_result = tokio::task::spawn_blocking(|| {
                    if event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
                        Some(event::read())
                    } else {
                        None
                    }
                }) => {
                    if let Ok(Some(Ok(event))) = event_result {
                        self.handle_event(event);
                    }
                }

                // Receive agent streaming messages if there's an active query.
                msg = async {
                    if let Some(rx) = &mut self.agent_rx {
                        rx.recv().await
                    } else {
                        // No active query -- park this future forever.
                        std::future::pending().await
                    }
                } => {
                    match msg {
                        Some(sdk_msg) => self.handle_sdk_message(sdk_msg),
                        None => {
                            // Channel closed -- agent task finished.
                            self.finalize_agent_task().await;
                        }
                    }
                }
            }

            self.draw()?;
        }

        self.cleanup()?;
        Ok(())
    }

    /// Draw the full UI.
    fn draw(&mut self) -> Result<(), AgentError> {
        let state = &self.state;
        self.terminal
            .draw(|frame| {
                let area = frame.area();

                // Layout: header(2) | messages(flex) | status(2) | input(3).
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(2),  // Header
                        Constraint::Min(4),     // Messages
                        Constraint::Length(2),  // Status bar
                        Constraint::Length(3),  // Input
                    ])
                    .split(area);

                // Render each zone.
                frame.render_widget(Header { state }, chunks[0]);
                frame.render_widget(MessageList { state }, chunks[1]);
                frame.render_widget(StatusBar { state }, chunks[2]);
                frame.render_widget(InputBox { state }, chunks[3]);

                // Cursor position in the input box.
                let is_active = state.status == AgentStatus::Idle
                    || state.status == AgentStatus::Done;
                if is_active && !state.show_help {
                    let (cx, cy) = components::input_cursor_position(state, chunks[3]);
                    frame.set_cursor_position((cx, cy));
                }

                // Help overlay (centered).
                if state.show_help {
                    let overlay = centered_rect(60, 80, area);
                    // Clear the overlay area first.
                    frame.render_widget(ratatui::widgets::Clear, overlay);
                    frame.render_widget(HelpOverlay, overlay);
                }
            })
            .map_err(|e| AgentError::Other(e.to_string()))?;
        Ok(())
    }

    /// Handle a crossterm terminal event.
    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                // Help overlay dismissal: any key closes it.
                if self.state.show_help {
                    self.state.show_help = false;
                    return;
                }

                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

                match key.code {
                    // Quit.
                    KeyCode::Char('c') if ctrl => {
                        self.state.should_quit = true;
                    }
                    // Interrupt the running agent loop.
                    KeyCode::Esc => {
                        self.interrupt_agent();
                    }
                    // Toggle help.
                    KeyCode::Char('h') if ctrl => {
                        self.state.show_help = !self.state.show_help;
                    }
                    // Clear messages.
                    KeyCode::Char('l') if ctrl => {
                        self.state.messages.clear();
                        self.state.scroll_offset = 0;
                    }
                    // Kill to end of line.
                    KeyCode::Char('k') if ctrl => {
                        self.state.kill_line();
                    }
                    // Kill to start of line.
                    KeyCode::Char('u') if ctrl => {
                        self.state.kill_to_start();
                    }
                    // Toggle copy mode (disable/enable mouse capture).
                    KeyCode::Char('x') if ctrl => {
                        self.toggle_copy_mode();
                    }
                    // Cursor to start.
                    KeyCode::Char('a') if ctrl => {
                        self.state.cursor_home();
                    }
                    // Cursor to end.
                    KeyCode::Char('e') if ctrl => {
                        self.state.cursor_end();
                    }
                    // Submit input.
                    KeyCode::Enter => {
                        self.submit();
                    }
                    // Backspace.
                    KeyCode::Backspace => {
                        self.state.backspace();
                    }
                    // Delete.
                    KeyCode::Delete => {
                        self.state.delete();
                    }
                    // Cursor movement.
                    KeyCode::Left => self.state.cursor_left(),
                    KeyCode::Right => self.state.cursor_right(),
                    KeyCode::Home => self.state.cursor_home(),
                    KeyCode::End => self.state.cursor_end(),
                    // History navigation.
                    KeyCode::Up => self.state.navigate_history(true),
                    KeyCode::Down => self.state.navigate_history(false),
                    // Scroll messages.
                    KeyCode::PageUp => {
                        self.state.scroll_offset = self.state.scroll_offset.saturating_add(10);
                        self.state.user_scrolled = true;
                    }
                    KeyCode::PageDown => {
                        self.state.scroll_offset = self.state.scroll_offset.saturating_sub(10);
                        if self.state.scroll_offset == 0 {
                            self.state.user_scrolled = false;
                        }
                    }
                    // Character input.
                    KeyCode::Char(c) => {
                        self.state.insert_char(c);
                    }
                    KeyCode::Tab => {
                        self.state.insert_char('\t');
                    }
                    _ => {}
                }
            }
            Event::Mouse(mouse) => {
                use crossterm::event::MouseEventKind;
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        self.state.scroll_offset = self.state.scroll_offset.saturating_add(3);
                        self.state.user_scrolled = true;
                    }
                    MouseEventKind::ScrollDown => {
                        self.state.scroll_offset = self.state.scroll_offset.saturating_sub(3);
                        if self.state.scroll_offset == 0 {
                            self.state.user_scrolled = false;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Submit user input (or handle slash command).
    fn submit(&mut self) {
        let input = match self.state.submit_input() {
            Some(s) => s,
            None => return,
        };

        // Slash commands.
        if input.starts_with('/') {
            self.handle_slash_command(&input);
            return;
        }

        // Don't allow sending while agent is busy.
        if self.state.status != AgentStatus::Idle && self.state.status != AgentStatus::Done {
            self.state
                .push_message(ChatRole::System, "Agent is busy. Please wait...");
            return;
        }

        // Show user message and start agent query.
        self.state.push_message(ChatRole::User, &input);
        self.state.status = AgentStatus::Thinking;
        self.state.streaming_text.clear();

        match self.agent.query(&input) {
            Ok((rx, handle)) => {
                self.agent_rx = Some(rx);
                self.agent_handle = Some(handle);
            }
            Err(e) => {
                self.state.push_message(ChatRole::Error, format!("Failed to start query: {e}"));
                self.state.status = AgentStatus::Idle;
            }
        }
    }

    /// Handle a slash command.
    fn handle_slash_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        match parts[0] {
            "/quit" | "/exit" | "/q" => {
                self.state.should_quit = true;
            }
            "/help" | "/h" => {
                self.state.show_help = true;
            }
            "/clear" => {
                self.state.messages.clear();
                self.state.scroll_offset = 0;
                self.state.push_message(ChatRole::System, "Chat history cleared.");
            }
            "/model" => {
                self.state.push_message(
                    ChatRole::System,
                    format!(
                        "Model: {} | Provider: {}",
                        self.state.model, self.state.provider
                    ),
                );
            }
            "/cost" => {
                self.state.push_message(
                    ChatRole::System,
                    format!(
                        "Cost: ${:.6} | Tokens: {} | Turns: {}",
                        self.state.cost_usd, self.state.total_tokens, self.state.num_turns
                    ),
                );
            }
            _ => {
                self.state.push_message(
                    ChatRole::System,
                    format!("Unknown command: {}", parts[0]),
                );
            }
        }
    }

    /// Process an incoming SDK message from the agent.
    fn handle_sdk_message(&mut self, msg: SDKMessage) {
        match msg {
            SDKMessage::System { model, .. } => {
                self.state.model = model;
            }
            SDKMessage::ContentDelta { delta, .. } => match delta {
                ContentDelta::TextDelta { text } => {
                    self.state.status = AgentStatus::Thinking;
                    self.state.append_streaming_text(&text);
                    // Auto-scroll to bottom.
                    if !self.state.user_scrolled {
                        self.state.scroll_offset = 0;
                    }
                }
                ContentDelta::ThinkingDelta { thinking } => {
                    self.state.status = AgentStatus::Thinking;
                    self.state.append_streaming_thinking(&thinking);
                    if !self.state.user_scrolled {
                        self.state.scroll_offset = 0;
                    }
                }
                ContentDelta::InputJsonDelta { .. } => {
                    // Tool input streaming -- ignore in the UI.
                }
            },
            SDKMessage::Assistant { message, .. } => {
                // Full assistant message -- capture tool_use blocks with their inputs.
                for block in &message.content {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        self.state.pending_tools.insert(
                            id.clone(),
                            (name.clone(), input.clone()),
                        );
                        self.state.status = AgentStatus::RunningTool(name.clone());
                    }
                }
            }
            SDKMessage::ToolProgress {
                tool_name, ..
            } => {
                self.state.status = AgentStatus::RunningTool(tool_name.clone());
            }
            SDKMessage::ToolResult {
                tool_use_id,
                tool_name,
                content,
                is_error,
            } => {
                let status = if is_error { "FAILED" } else { "OK" };

                // Build header: [ToolName] OK  + input parameters
                let mut result_text = format!("[{tool_name}] {status}");

                // Look up the original tool invocation input.
                if let Some((_, input)) = self.state.pending_tools.remove(&tool_use_id) {
                    let params = format_tool_input(&tool_name, &input);
                    if !params.is_empty() {
                        result_text.push('\n');
                        result_text.push_str(&params);
                    }
                }

                // Separator before output.
                result_text.push_str("\n───");

                for c in &content {
                    match c {
                        ToolResultContent::Text { text } => {
                            if !text.is_empty() {
                                result_text.push('\n');
                                result_text.push_str(text);
                            }
                        }
                        ToolResultContent::Image { .. } => {
                            result_text.push_str("\n[image]");
                        }
                    }
                }

                let role = if is_error { ChatRole::Error } else { ChatRole::Tool };
                self.state.push_message(role, result_text);
                self.state.status = AgentStatus::Thinking;
            }
            SDKMessage::Result {
                total_usage,
                total_cost_usd,
                num_turns,
                ..
            } => {
                self.state.total_tokens = total_usage.total_tokens();
                self.state.cost_usd = total_cost_usd;
                self.state.num_turns += num_turns;
                self.state.finish_streaming();
                self.state.status = AgentStatus::Done;

                // Sync agent message history.
                // We need to update the agent's internal state, but since we
                // own it mutably, we'll do this in finalize_agent_task.
            }
            SDKMessage::Error { message, .. } => {
                self.state.push_message(ChatRole::Error, message);
            }
            SDKMessage::Compact {
                original_tokens,
                compacted_tokens,
            } => {
                self.state.push_message(
                    ChatRole::System,
                    format!(
                        "Context compacted: {} -> {} tokens",
                        original_tokens, compacted_tokens
                    ),
                );
            }
            SDKMessage::PermissionRequest {
                tool_name, message, ..
            } => {
                self.state.push_message(
                    ChatRole::System,
                    format!("Permission request for {tool_name}: {message}"),
                );
            }
            SDKMessage::TaskNotification {
                status, summary, ..
            } => {
                self.state.push_message(
                    ChatRole::System,
                    format!("Task {status}: {summary}"),
                );
            }
            SDKMessage::KeepAlive => {}
        }
    }

    /// Toggle copy mode: disable mouse capture so terminal-native selection works.
    fn toggle_copy_mode(&mut self) {
        self.state.copy_mode = !self.state.copy_mode;
        if self.state.copy_mode {
            let _ = execute!(self.terminal.backend_mut(), DisableMouseCapture);
        } else {
            let _ = execute!(self.terminal.backend_mut(), EnableMouseCapture);
        }
    }

    /// Interrupt a running agent loop (Esc key).
    ///
    /// Aborts the tokio task, drains the receiver, and resets state to Idle.
    /// Does nothing if no agent is currently running.
    fn interrupt_agent(&mut self) {
        let was_running = self.agent_handle.is_some() || self.agent_rx.is_some();
        if let Some(handle) = self.agent_handle.take() {
            handle.abort();
        }
        self.agent_rx = None;
        if was_running {
            self.state.finish_streaming();
            self.state.status = AgentStatus::Idle;
            self.state.push_message(ChatRole::System, "Interrupted.");
        }
    }

    /// Finalize the agent task after the channel closes.
    async fn finalize_agent_task(&mut self) {
        self.agent_rx = None;
        if let Some(handle) = self.agent_handle.take() {
            if let Err(e) = handle.await {
                self.state
                    .push_message(ChatRole::Error, format!("Agent task error: {e}"));
            }
        }
        if self.state.status != AgentStatus::Done {
            self.state.status = AgentStatus::Idle;
        }
        // Reset to idle so user can send the next message.
        self.state.status = AgentStatus::Idle;
    }

    /// Restore terminal state.
    fn cleanup(&mut self) -> Result<(), AgentError> {
        disable_raw_mode().map_err(|e| AgentError::Other(e.to_string()))?;
        execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )
        .map_err(|e| AgentError::Other(e.to_string()))?;
        self.terminal
            .show_cursor()
            .map_err(|e| AgentError::Other(e.to_string()))?;
        Ok(())
    }
}

impl Drop for TuiApp {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Detect the provider name from agent options.
fn detect_provider(options: &AgentOptions) -> String {
    match &options.api {
        crate::agent::ApiClientConfig::ApiKey(_) => "Anthropic".into(),
        crate::agent::ApiClientConfig::Client(_) => "Custom".into(),
        crate::agent::ApiClientConfig::OpenAI(_) => "OpenAI".into(),
        crate::agent::ApiClientConfig::OpenAICompat { base_url, .. } => {
            if base_url.contains("groq") {
                "Groq".into()
            } else if base_url.contains("together") {
                "Together".into()
            } else if base_url.contains("nvidia") {
                "NVIDIA NIM".into()
            } else {
                "OpenAI-compat".into()
            }
        }
        crate::agent::ApiClientConfig::Ollama { .. } => "Ollama".into(),
        crate::agent::ApiClientConfig::FromEnv => {
            if std::env::var("OLLAMA_BASE_URL").is_ok() {
                "Ollama".into()
            } else if std::env::var("ANTHROPIC_BASE_URL").is_ok() {
                "Anthropic-compat".into()
            } else if std::env::var("ANTHROPIC_API_KEY").is_ok()
                || std::env::var("ANTHROPIC_AUTH_TOKEN").is_ok()
            {
                "Anthropic".into()
            } else if std::env::var("NVIDIA_API_KEY").is_ok() {
                "NVIDIA NIM".into()
            } else if std::env::var("OPENAI_API_KEY").is_ok() {
                "OpenAI".into()
            } else {
                "Unknown".into()
            }
        }
    }
}

/// Format tool input parameters for display.
///
/// Produces a human-readable summary such as:
///   $ ls -la /tmp          (Bash)
///   file: /src/main.rs     (Read)
///   pattern: "TODO" path: src/  (Grep)
fn format_tool_input(tool_name: &str, input: &serde_json::Value) -> String {
    // The input may be a proper JSON object, or a JSON string that needs parsing
    // (OpenAI-compat streaming accumulates input as Value::String).
    let parsed: Option<serde_json::Value>;
    let obj = match input.as_object() {
        Some(o) => o,
        None => {
            // Try parsing a string value as JSON.
            if let Some(s) = input.as_str() {
                parsed = serde_json::from_str(s).ok();
                match parsed.as_ref().and_then(|v| v.as_object()) {
                    Some(o) => o,
                    None => return s.to_string(),
                }
            } else {
                return format!("{input}");
            }
        }
    };

    match tool_name {
        "Bash" => {
            let cmd = obj.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let timeout = obj.get("timeout").and_then(|v| v.as_u64());
            let desc = obj.get("description").and_then(|v| v.as_str());
            let mut out = format!("$ {cmd}");
            if let Some(t) = timeout {
                out.push_str(&format!("  (timeout: {t}ms)"));
            }
            if let Some(d) = desc {
                out.push_str(&format!("  # {d}"));
            }
            out
        }
        "Read" => {
            let path = obj.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
            let mut out = format!("file: {path}");
            if let Some(offset) = obj.get("offset").and_then(|v| v.as_u64()) {
                out.push_str(&format!("  offset: {offset}"));
            }
            if let Some(limit) = obj.get("limit").and_then(|v| v.as_u64()) {
                out.push_str(&format!("  limit: {limit}"));
            }
            out
        }
        "Write" => {
            let path = obj.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
            let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let lines = content.lines().count();
            format!("file: {path}  ({lines} lines)")
        }
        "Edit" => {
            let path = obj.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
            let old = obj.get("old_string").and_then(|v| v.as_str()).unwrap_or("");
            let new = obj.get("new_string").and_then(|v| v.as_str()).unwrap_or("");
            format_edit_diff(path, old, new)
        }
        "Glob" => {
            let pattern = obj.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            let path = obj.get("path").and_then(|v| v.as_str());
            let mut out = format!("pattern: {pattern}");
            if let Some(p) = path {
                out.push_str(&format!("  path: {p}"));
            }
            out
        }
        "Grep" => {
            let pattern = obj.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            let path = obj.get("path").and_then(|v| v.as_str());
            let glob = obj.get("glob").and_then(|v| v.as_str());
            let mut out = format!("pattern: \"{pattern}\"");
            if let Some(p) = path {
                out.push_str(&format!("  path: {p}"));
            }
            if let Some(g) = glob {
                out.push_str(&format!("  glob: {g}"));
            }
            out
        }
        // Generic fallback: show all key=value pairs.
        _ => {
            let mut parts: Vec<String> = Vec::new();
            for (key, val) in obj {
                let display = match val {
                    serde_json::Value::String(s) => truncate_str(s, 80),
                    other => {
                        let s = other.to_string();
                        truncate_str(&s, 80)
                    }
                };
                parts.push(format!("{key}: {display}"));
            }
            parts.join("\n")
        }
    }
}

/// Generate a git-diff–style unified diff for the Edit tool.
/// Requires the `built-in-tools` feature (which includes the `similar` crate).
#[cfg(feature = "built-in-tools")]
fn format_edit_diff(path: &str, old: &str, new: &str) -> String {
    use similar::TextDiff;
    let diff = TextDiff::from_lines(old, new);
    let diff_text = diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    if diff_text.trim().is_empty() {
        format!("file: {path}  (no changes)")
    } else {
        format!("file: {path}\n{}", diff_text.trim_end())
    }
}

/// Fallback when `built-in-tools` is not enabled.
#[cfg(not(feature = "built-in-tools"))]
fn format_edit_diff(path: &str, old: &str, new: &str) -> String {
    let old_preview = truncate_str(old, 60);
    let new_preview = truncate_str(new, 60);
    format!("file: {path}\n  - {old_preview}\n  + {new_preview}")
}

/// Truncate a string for display, appending "…" if it exceeds `max_chars`.
fn truncate_str(s: &str, max_chars: usize) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    if first_line.chars().count() <= max_chars {
        first_line.to_string()
    } else {
        let truncated: String = first_line.chars().take(max_chars).collect();
        format!("{truncated}…")
    }
}

/// Compute a centered rectangle within the given area.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
