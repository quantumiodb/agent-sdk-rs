//! Reusable ratatui widgets for the TUI.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
};

use super::state::{AgentStatus, AppState, ChatRole};

// ─── Color palette (dark theme) ─────────────────────────────────────────────

const BG_DARK: Color = Color::Rgb(20, 20, 30);
const BG_MID: Color = Color::Rgb(25, 25, 35);
const BG_LIGHT: Color = Color::Rgb(30, 30, 46);
const FG_DIM: Color = Color::Rgb(108, 112, 134);
const FG_TEXT: Color = Color::Rgb(205, 214, 244);
const ACCENT_BLUE: Color = Color::Rgb(137, 180, 250);
const ACCENT_GREEN: Color = Color::Rgb(166, 227, 161);
const ACCENT_MAUVE: Color = Color::Rgb(203, 166, 247);
const ACCENT_PEACH: Color = Color::Rgb(250, 179, 135);
const ACCENT_RED: Color = Color::Rgb(243, 139, 168);
const ACCENT_YELLOW: Color = Color::Rgb(249, 226, 175);
const ACCENT_TEAL: Color = Color::Rgb(148, 226, 213);

/// Dimmed foreground for thinking text.
const FG_THINKING: Color = Color::Rgb(147, 153, 178);

fn role_color(role: ChatRole) -> Color {
    match role {
        ChatRole::User => ACCENT_BLUE,
        ChatRole::Assistant => ACCENT_GREEN,
        ChatRole::Thinking => ACCENT_YELLOW,
        ChatRole::System => ACCENT_YELLOW,
        ChatRole::Tool => ACCENT_MAUVE,
        ChatRole::Error => ACCENT_RED,
    }
}

fn role_label(role: ChatRole) -> &'static str {
    match role {
        ChatRole::User => "You",
        ChatRole::Assistant => "Claude",
        ChatRole::Thinking => "Thinking",
        ChatRole::System => "System",
        ChatRole::Tool => "Tool",
        ChatRole::Error => "Error",
    }
}

// ─── Header ─────────────────────────────────────────────────────────────────

pub struct Header<'a> {
    pub state: &'a AppState,
}

impl Widget for Header<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(FG_DIM))
            .style(Style::default().bg(BG_DARK));

        let inner = block.inner(area);
        block.render(area, buf);

        let title = Line::from(vec![
            Span::styled(
                " Claude Agent ",
                Style::default()
                    .fg(ACCENT_GREEN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("│", Style::default().fg(FG_DIM)),
            Span::raw(" "),
            Span::styled(&self.state.model, Style::default().fg(ACCENT_BLUE)),
            Span::raw(" "),
            Span::styled("│", Style::default().fg(FG_DIM)),
            Span::raw(" "),
            Span::styled(
                &self.state.provider,
                Style::default().fg(ACCENT_MAUVE),
            ),
            Span::raw(" "),
            Span::styled("│", Style::default().fg(FG_DIM)),
            Span::styled(
                format!(" {}", self.state.session_id.get(..8).unwrap_or(&self.state.session_id)),
                Style::default().fg(FG_DIM),
            ),
        ]);

        Paragraph::new(title).render(inner, buf);
    }
}

// ─── Messages ───────────────────────────────────────────────────────────────

pub struct MessageList<'a> {
    pub state: &'a AppState,
}

impl MessageList<'_> {
    /// Render messages into styled lines, wrapping at the given width.
    fn render_lines(&self, width: u16) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        let w = width as usize;

        if self.state.messages.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Type a message and press Enter to start chatting...",
                Style::default().fg(FG_DIM).add_modifier(Modifier::ITALIC),
            )));
            return lines;
        }

        for (i, msg) in self.state.messages.iter().enumerate() {
            if i > 0 {
                lines.push(Line::from(""));
            }

            let color = role_color(msg.role);
            let label = role_label(msg.role);
            let is_thinking = msg.role == ChatRole::Thinking;

            // Text style: italic + dimmed for thinking, normal for others.
            let text_style = if is_thinking {
                Style::default().fg(FG_THINKING).add_modifier(Modifier::ITALIC)
            } else {
                Style::default().fg(FG_TEXT)
            };

            // Role header line.  Thinking gets a <think> tag style.
            if is_thinking {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {label} "),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "<think>",
                        Style::default().fg(FG_DIM).add_modifier(Modifier::ITALIC),
                    ),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {label}"),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                ]));
            }

            // Message text, wrapped by character width (CJK-safe).
            for text_line in msg.text.lines() {
                if text_line.is_empty() {
                    lines.push(Line::from(""));
                    continue;
                }
                // Indent: thinking uses "│ " gutter, others use "  ".
                let indent_display_width: usize = if is_thinking { 4 } else { 2 };
                let effective_width = w.saturating_sub(indent_display_width + 1);
                if effective_width == 0 {
                    if is_thinking {
                        lines.push(Line::from(vec![
                            Span::styled("  │ ", Style::default().fg(FG_DIM)),
                            Span::styled(text_line.to_string(), text_style),
                        ]));
                    } else {
                        lines.push(Line::from(Span::styled(
                            format!("  {text_line}"),
                            text_style,
                        )));
                    }
                    continue;
                }
                for chunk in wrap_line(text_line, effective_width) {
                    if is_thinking {
                        lines.push(Line::from(vec![
                            Span::styled("  │ ", Style::default().fg(FG_DIM)),
                            Span::styled(chunk, text_style),
                        ]));
                    } else {
                        lines.push(Line::from(Span::styled(
                            format!("  {chunk}"),
                            text_style,
                        )));
                    }
                }
            }
        }

        lines
    }
}

impl Widget for MessageList<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .style(Style::default().bg(BG_MID));
        let inner = block.inner(area);
        block.render(area, buf);

        let all_lines = self.render_lines(inner.width);
        let total = all_lines.len() as u16;
        let visible = inner.height;

        // Compute scroll: scroll_offset=0 means bottom-aligned.
        let max_scroll = total.saturating_sub(visible);
        let scroll = max_scroll.saturating_sub(self.state.scroll_offset);

        let paragraph = Paragraph::new(all_lines)
            .scroll((scroll, 0));
        paragraph.render(inner, buf);
    }
}

// ─── Input box ──────────────────────────────────────────────────────────────

pub struct InputBox<'a> {
    pub state: &'a AppState,
}

impl Widget for InputBox<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let is_active = self.state.status == AgentStatus::Idle
            || self.state.status == AgentStatus::Done;

        let border_color = if is_active { ACCENT_BLUE } else { FG_DIM };

        let title = if is_active {
            " > "
        } else {
            " Waiting... "
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(Span::styled(
                title,
                Style::default().fg(border_color).add_modifier(Modifier::BOLD),
            ))
            .style(Style::default().bg(BG_LIGHT));

        let inner = block.inner(area);
        block.render(area, buf);

        // Render input text with cursor.
        let input_text = &self.state.input;
        let cursor_pos = self.state.cursor_pos;

        // Calculate horizontal scroll if input is longer than visible width.
        let visible_width = inner.width as usize;
        let char_pos = input_text[..cursor_pos].chars().count();
        let h_scroll = if char_pos >= visible_width {
            char_pos - visible_width + 1
        } else {
            0
        };

        let display: String = input_text.chars().skip(h_scroll).collect();

        let paragraph = Paragraph::new(Line::from(Span::styled(
            display,
            Style::default().fg(FG_TEXT),
        )));
        paragraph.render(inner, buf);
    }
}

/// Returns the cursor position within the input area for `Frame::set_cursor_position`.
pub fn input_cursor_position(state: &AppState, input_area: Rect) -> (u16, u16) {
    let inner_x = input_area.x + 1; // +1 for border
    let inner_y = input_area.y + 1;
    let visible_width = input_area.width.saturating_sub(2) as usize;
    let char_pos = state.input[..state.cursor_pos].chars().count();
    let h_scroll = if char_pos >= visible_width {
        char_pos - visible_width + 1
    } else {
        0
    };
    let display_pos = (char_pos - h_scroll) as u16;
    (inner_x + display_pos, inner_y)
}

// ─── Status bar ─────────────────────────────────────────────────────────────

pub struct StatusBar<'a> {
    pub state: &'a AppState,
}

impl Widget for StatusBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(FG_DIM))
            .style(Style::default().bg(BG_DARK));
        let inner = block.inner(area);
        block.render(area, buf);

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(20),
            ])
            .split(inner);

        // Status
        let (status_text, status_color) = match &self.state.status {
            AgentStatus::Idle => ("Ready", ACCENT_GREEN),
            AgentStatus::Thinking => ("Thinking...", ACCENT_YELLOW),
            AgentStatus::RunningTool(_) => {
                ("", ACCENT_PEACH)
            }
            AgentStatus::Done => ("Done", ACCENT_TEAL),
        };

        let status_line = if let AgentStatus::RunningTool(name) = &self.state.status {
            Line::from(vec![
                Span::styled(" ● ", Style::default().fg(ACCENT_PEACH)),
                Span::styled(
                    format!("Running {name}"),
                    Style::default().fg(ACCENT_PEACH),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled(" ● ", Style::default().fg(status_color)),
                Span::styled(status_text, Style::default().fg(status_color)),
            ])
        };
        Paragraph::new(status_line).render(chunks[0], buf);

        // Tokens
        let tokens_line = Line::from(vec![
            Span::styled(" Tokens: ", Style::default().fg(FG_DIM)),
            Span::styled(
                format_tokens(self.state.total_tokens),
                Style::default().fg(ACCENT_BLUE),
            ),
        ]);
        Paragraph::new(tokens_line).render(chunks[1], buf);

        // Cost
        let cost_line = Line::from(vec![
            Span::styled(" Cost: ", Style::default().fg(FG_DIM)),
            Span::styled(
                format!("${:.4}", self.state.cost_usd),
                Style::default().fg(ACCENT_MAUVE),
            ),
        ]);
        Paragraph::new(cost_line).render(chunks[2], buf);

        // Help hint / copy mode indicator
        let help_line = if self.state.copy_mode {
            Line::from(vec![
                Span::styled(" COPY MODE ", Style::default().fg(Color::Rgb(20, 20, 30)).bg(ACCENT_YELLOW).add_modifier(Modifier::BOLD)),
                Span::styled(" Ctrl+X exit ", Style::default().fg(ACCENT_YELLOW)),
            ])
        } else {
            Line::from(vec![
                Span::styled(" Ctrl+H help ", Style::default().fg(FG_DIM)),
            ])
        };
        Paragraph::new(help_line).render(chunks[3], buf);
    }
}

// ─── Help overlay ───────────────────────────────────────────────────────────

pub struct HelpOverlay;

impl Widget for HelpOverlay {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Semi-transparent background
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT_BLUE))
            .title(Span::styled(
                " Help ",
                Style::default().fg(ACCENT_BLUE).add_modifier(Modifier::BOLD),
            ))
            .style(Style::default().bg(BG_DARK));
        let inner = block.inner(area);
        block.render(area, buf);

        let help_text = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Keyboard Shortcuts", Style::default().fg(ACCENT_GREEN).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Enter       ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Send message", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+C      ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Quit", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+H      ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Toggle this help", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+X      ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Toggle copy mode (mouse select)", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+L      ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Clear messages", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+U      ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Delete to start of line", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+K      ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Delete to end of line", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Up/Down     ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Browse input history", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  PgUp/PgDn   ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Scroll messages", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  Home/End    ", Style::default().fg(ACCENT_BLUE)),
                Span::styled("Cursor to start/end", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Slash Commands", Style::default().fg(ACCENT_GREEN).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  /help       ", Style::default().fg(ACCENT_MAUVE)),
                Span::styled("Show this help", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  /clear      ", Style::default().fg(ACCENT_MAUVE)),
                Span::styled("Clear chat history", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  /quit       ", Style::default().fg(ACCENT_MAUVE)),
                Span::styled("Exit the application", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  /model      ", Style::default().fg(ACCENT_MAUVE)),
                Span::styled("Show current model info", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(vec![
                Span::styled("  /cost       ", Style::default().fg(ACCENT_MAUVE)),
                Span::styled("Show cost summary", Style::default().fg(FG_TEXT)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Press any key to close", Style::default().fg(FG_DIM).add_modifier(Modifier::ITALIC)),
            ]),
        ];

        Paragraph::new(help_text)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Estimate the display width of a character.
/// CJK characters take 2 columns, most others take 1.
fn char_width(c: char) -> usize {
    // CJK Unified Ideographs and common fullwidth ranges.
    if ('\u{1100}'..='\u{115F}').contains(&c)   // Hangul Jamo
        || ('\u{2E80}'..='\u{A4CF}').contains(&c)  // CJK radicals..Yi
        || ('\u{AC00}'..='\u{D7A3}').contains(&c)  // Hangul Syllables
        || ('\u{F900}'..='\u{FAFF}').contains(&c)  // CJK Compat Ideographs
        || ('\u{FE10}'..='\u{FE19}').contains(&c)  // Vertical forms
        || ('\u{FE30}'..='\u{FE6F}').contains(&c)  // CJK Compat Forms
        || ('\u{FF00}'..='\u{FF60}').contains(&c)  // Fullwidth Forms
        || ('\u{FFE0}'..='\u{FFE6}').contains(&c)  // Fullwidth Signs
        || ('\u{20000}'..='\u{2FA1F}').contains(&c) // CJK Ext B..Compat Supp
        || ('\u{30000}'..='\u{3134F}').contains(&c) // CJK Ext G..H
    {
        2
    } else {
        1
    }
}

/// Word-wrap a single line into chunks that fit within `max_width` display columns.
/// CJK-safe: measures by display width, not byte length.
fn wrap_line(line: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![line.to_string()];
    }

    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_width: usize = 0;

    for word in SplitKeepWhitespace::new(line) {
        let word_width: usize = word.chars().map(char_width).sum();

        // If the word alone exceeds max_width, force-break it character by character.
        if word_width > max_width {
            for ch in word.chars() {
                let cw = char_width(ch);
                if current_width + cw > max_width && !current.is_empty() {
                    result.push(current.trim_end().to_string());
                    current = String::new();
                    current_width = 0;
                }
                current.push(ch);
                current_width += cw;
            }
            continue;
        }

        if current_width + word_width > max_width {
            result.push(current.trim_end().to_string());
            current = word.trim_start().to_string();
            current_width = current.chars().map(char_width).sum();
        } else {
            current.push_str(word);
            current_width += word_width;
        }
    }

    if !current.is_empty() {
        result.push(current.trim_end().to_string());
    }

    if result.is_empty() {
        result.push(String::new());
    }

    result
}

/// Iterator that splits on whitespace but keeps the whitespace attached to each word.
struct SplitKeepWhitespace<'a> {
    remaining: &'a str,
}

impl<'a> SplitKeepWhitespace<'a> {
    fn new(s: &'a str) -> Self {
        Self { remaining: s }
    }
}

impl<'a> Iterator for SplitKeepWhitespace<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            return None;
        }
        // Find end of non-whitespace.
        let end_word = self.remaining
            .find(|c: char| c.is_whitespace())
            .unwrap_or(self.remaining.len());
        // Include trailing whitespace.
        let end_ws = self.remaining[end_word..]
            .find(|c: char| !c.is_whitespace())
            .map(|p| end_word + p)
            .unwrap_or(self.remaining.len());
        let (chunk, rest) = self.remaining.split_at(end_ws);
        self.remaining = rest;
        Some(chunk)
    }
}
