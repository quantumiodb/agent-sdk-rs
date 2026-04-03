//! TUI application state.

use std::collections::VecDeque;

/// A single chat message displayed in the TUI.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Who sent this message.
    pub role: ChatRole,
    /// The displayed text content.
    pub text: String,
}

/// The role/origin of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    /// Model's chain-of-thought / reasoning.
    Thinking,
    System,
    Tool,
    Error,
}

/// Status of the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    /// Waiting for user input.
    Idle,
    /// Thinking / calling the LLM.
    Thinking,
    /// Executing a tool.
    RunningTool(String),
    /// The agent has finished and is ready for the next prompt.
    Done,
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentStatus::Idle => write!(f, "Ready"),
            AgentStatus::Thinking => write!(f, "Thinking..."),
            AgentStatus::RunningTool(name) => write!(f, "Running {name}..."),
            AgentStatus::Done => write!(f, "Done"),
        }
    }
}

/// Central state for the TUI application.
pub struct AppState {
    /// Chat message history.
    pub messages: VecDeque<ChatMessage>,
    /// Current input buffer.
    pub input: String,
    /// Cursor position within the input buffer.
    pub cursor_pos: usize,
    /// Vertical scroll offset for the messages area (lines from bottom).
    pub scroll_offset: u16,
    /// Whether the user has manually scrolled up (disables auto-scroll).
    pub user_scrolled: bool,
    /// Current agent status.
    pub status: AgentStatus,
    /// Model name.
    pub model: String,
    /// Provider name.
    pub provider: String,
    /// Session ID (shortened for display).
    pub session_id: String,
    /// Accumulated cost in USD.
    pub cost_usd: f64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Number of turns.
    pub num_turns: u32,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// Whether the help overlay is visible.
    pub show_help: bool,
    /// Streaming text accumulator for the current assistant response.
    pub streaming_text: String,
    /// Streaming text accumulator for the current thinking block.
    pub streaming_thinking: String,
    /// Whether we are currently inside a thinking block.
    pub in_thinking: bool,
    /// Input history for up/down navigation.
    pub input_history: Vec<String>,
    /// Current position in input history (-1 = current input).
    pub history_index: Option<usize>,
    /// Saved current input when browsing history.
    pub saved_input: String,
}

impl AppState {
    pub fn new(model: String, provider: String, session_id: String) -> Self {
        Self {
            messages: VecDeque::new(),
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            user_scrolled: false,
            status: AgentStatus::Idle,
            model,
            provider,
            session_id,
            cost_usd: 0.0,
            total_tokens: 0,
            num_turns: 0,
            should_quit: false,
            show_help: false,
            streaming_text: String::new(),
            streaming_thinking: String::new(),
            in_thinking: false,
            input_history: Vec::new(),
            history_index: None,
            saved_input: String::new(),
        }
    }

    /// Push a chat message and reset scroll to bottom.
    pub fn push_message(&mut self, role: ChatRole, text: impl Into<String>) {
        self.messages.push_back(ChatMessage {
            role,
            text: text.into(),
        });
        if !self.user_scrolled {
            self.scroll_offset = 0;
        }
    }

    /// Append thinking text to the current streaming thinking message.
    pub fn append_streaming_thinking(&mut self, text: &str) {
        // If we were streaming a text response, finalize the thinking transition.
        if !self.in_thinking {
            self.in_thinking = true;
            self.streaming_thinking.clear();
        }
        self.streaming_thinking.push_str(text);
        // Update the last message if it's a thinking message, otherwise create one.
        if let Some(last) = self.messages.back_mut() {
            if last.role == ChatRole::Thinking {
                last.text = self.streaming_thinking.clone();
                return;
            }
        }
        self.messages.push_back(ChatMessage {
            role: ChatRole::Thinking,
            text: self.streaming_thinking.clone(),
        });
    }

    /// Append text to the current streaming assistant message.
    /// If there is no in-progress assistant message, create one.
    pub fn append_streaming_text(&mut self, text: &str) {
        // If we were in a thinking block, finalize it.
        if self.in_thinking {
            self.in_thinking = false;
            self.streaming_thinking.clear();
        }
        self.streaming_text.push_str(text);
        // Update the last message if it's an assistant message, otherwise create one.
        if let Some(last) = self.messages.back_mut() {
            if last.role == ChatRole::Assistant {
                last.text = self.streaming_text.clone();
                return;
            }
        }
        self.messages.push_back(ChatMessage {
            role: ChatRole::Assistant,
            text: self.streaming_text.clone(),
        });
    }

    /// Finalize the current streaming response.
    pub fn finish_streaming(&mut self) {
        self.streaming_text.clear();
        self.streaming_thinking.clear();
        self.in_thinking = false;
    }

    /// Submit the current input, returning it and clearing the buffer.
    pub fn submit_input(&mut self) -> Option<String> {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return None;
        }
        self.input_history.push(text.clone());
        self.history_index = None;
        self.saved_input.clear();
        self.input.clear();
        self.cursor_pos = 0;
        Some(text)
    }

    /// Navigate input history (up = true, down = false).
    pub fn navigate_history(&mut self, up: bool) {
        if self.input_history.is_empty() {
            return;
        }
        match self.history_index {
            None => {
                if up {
                    self.saved_input = self.input.clone();
                    let idx = self.input_history.len() - 1;
                    self.history_index = Some(idx);
                    self.input = self.input_history[idx].clone();
                    self.cursor_pos = self.input.len();
                }
            }
            Some(idx) => {
                if up {
                    if idx > 0 {
                        let new_idx = idx - 1;
                        self.history_index = Some(new_idx);
                        self.input = self.input_history[new_idx].clone();
                        self.cursor_pos = self.input.len();
                    }
                } else {
                    if idx + 1 < self.input_history.len() {
                        let new_idx = idx + 1;
                        self.history_index = Some(new_idx);
                        self.input = self.input_history[new_idx].clone();
                        self.cursor_pos = self.input.len();
                    } else {
                        self.history_index = None;
                        self.input = self.saved_input.clone();
                        self.cursor_pos = self.input.len();
                    }
                }
            }
        }
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor_pos, c);
        self.cursor_pos += c.len_utf8();
    }

    /// Delete the character before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            // Find the previous char boundary.
            let prev = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.remove(prev);
            self.cursor_pos = prev;
        }
    }

    /// Delete the character at the cursor.
    pub fn delete(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.input.remove(self.cursor_pos);
        }
    }

    /// Move cursor left.
    pub fn cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    /// Move cursor right.
    pub fn cursor_right(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.cursor_pos = self.input[self.cursor_pos..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor_pos + i)
                .unwrap_or(self.input.len());
        }
    }

    /// Move cursor to beginning.
    pub fn cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    /// Move cursor to end.
    pub fn cursor_end(&mut self) {
        self.cursor_pos = self.input.len();
    }

    /// Delete from cursor to end of line.
    pub fn kill_line(&mut self) {
        self.input.truncate(self.cursor_pos);
    }

    /// Delete from cursor to beginning of line.
    pub fn kill_to_start(&mut self) {
        self.input = self.input[self.cursor_pos..].to_string();
        self.cursor_pos = 0;
    }
}
