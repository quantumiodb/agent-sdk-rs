//! SSE (Server-Sent Events) stream parser for the Anthropic Messages API.
//!
//! Reads an HTTP response body as a stream of bytes, splits on SSE frame
//! boundaries, and yields parsed [`ApiStreamEvent`] items.

use bytes::Bytes;
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{debug, trace, warn};

use super::client::ApiError;
use super::types::ApiStreamEvent;

/// A stream of [`ApiStreamEvent`] parsed from an SSE byte stream.
///
/// This is the concrete implementation behind the [`ApiStream`](super::client::ApiStream) type alias.
pub struct SseStream {
    /// The underlying byte stream from the HTTP response body.
    body: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    /// Buffer for incomplete lines across chunk boundaries.
    buffer: String,
    /// The current SSE event type (from the most recent "event:" line).
    current_event_type: Option<String>,
    /// Accumulated data lines for the current event.
    current_data: String,
    /// Whether the stream has encountered a terminal error or finished.
    finished: bool,
}

impl SseStream {
    /// Create a new SSE stream from a reqwest response byte stream.
    pub fn new(body: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static) -> Self {
        Self {
            body: Box::pin(body),
            buffer: String::new(),
            current_event_type: None,
            current_data: String::new(),
            finished: false,
        }
    }

    /// Try to parse and yield the next event from buffered lines.
    ///
    /// Returns `Some(Ok(event))` if a complete event was parsed,
    /// `Some(Err(..))` on a parse error, or `None` if more data is needed.
    fn try_parse_event(&mut self) -> Option<Result<ApiStreamEvent, ApiError>> {
        // Process complete lines in the buffer.
        loop {
            // Find the next newline. SSE uses \n or \r\n as line terminators.
            let newline_pos = self.buffer.find('\n');
            let line = match newline_pos {
                Some(pos) => {
                    let raw_line = self.buffer[..pos].to_string();
                    self.buffer = self.buffer[pos + 1..].to_string();
                    raw_line.trim_end_matches('\r').to_string()
                }
                None => return None, // Need more data.
            };

            if line.is_empty() {
                // Empty line = event boundary. Dispatch if we have data.
                if !self.current_data.is_empty() {
                    let event_type = self.current_event_type.take();
                    let data = std::mem::take(&mut self.current_data);
                    return Some(self.dispatch_event(event_type.as_deref(), &data));
                }
                // Otherwise, just skip consecutive blank lines.
                continue;
            }

            if line.starts_with(':') {
                // Comment line — used for keep-alive. Skip.
                trace!(line = %line, "SSE comment (keep-alive)");
                continue;
            }

            if let Some(value) = line.strip_prefix("event:") {
                self.current_event_type = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                let value = value.trim_start();
                if !self.current_data.is_empty() {
                    self.current_data.push('\n');
                }
                self.current_data.push_str(value);
            } else if let Some(value) = line.strip_prefix("id:") {
                // SSE id field — not used by the Anthropic API, but handle gracefully.
                trace!(id = %value.trim(), "SSE id field");
            } else if let Some(value) = line.strip_prefix("retry:") {
                // SSE retry field — not used in our client, but handle gracefully.
                trace!(retry = %value.trim(), "SSE retry field");
            } else {
                // Unknown field — treat the entire line as a field name with empty value.
                // Per the SSE spec, this is valid but we just skip it.
                trace!(line = %line, "SSE unknown field");
            }
        }
    }

    /// Parse an SSE event from its type and data payload.
    fn dispatch_event(
        &self,
        event_type: Option<&str>,
        data: &str,
    ) -> Result<ApiStreamEvent, ApiError> {
        // The Anthropic API uses the "event:" field to specify event types.
        // The data is always JSON.
        let event_type = event_type.unwrap_or("message");

        // Special handling for [DONE] signal (used by some providers).
        if data == "[DONE]" {
            debug!("SSE stream received [DONE] signal");
            return Ok(ApiStreamEvent::MessageStop);
        }

        trace!(event_type = %event_type, data_len = data.len(), "Dispatching SSE event");

        // Parse the JSON data. The `type` field in the JSON should match the
        // event type from the SSE frame, but we use the event type to set it
        // explicitly in case it's missing from the JSON body.
        let mut json_value: serde_json::Value =
            serde_json::from_str(data).map_err(|e| ApiError::InvalidResponse {
                message: format!("Failed to parse SSE data as JSON: {e}"),
                body: data.to_string(),
            })?;

        // Ensure the JSON has a "type" field matching the SSE event type.
        if let Some(obj) = json_value.as_object_mut() {
            if !obj.contains_key("type") {
                obj.insert(
                    "type".to_string(),
                    serde_json::Value::String(event_type.to_string()),
                );
            }
        }

        serde_json::from_value(json_value).map_err(|e| ApiError::InvalidResponse {
            message: format!("Failed to deserialize SSE event '{event_type}': {e}"),
            body: data.to_string(),
        })
    }
}

impl Stream for SseStream {
    type Item = Result<ApiStreamEvent, ApiError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            // First, try to parse an event from data already in the buffer.
            if let Some(event) = self.try_parse_event() {
                match &event {
                    Ok(ApiStreamEvent::MessageStop) => {
                        self.finished = true;
                    }
                    Ok(ApiStreamEvent::Error { error }) => {
                        warn!(error_type = %error.error_type, message = %error.message, "SSE error event");
                    }
                    _ => {}
                }
                return Poll::Ready(Some(event));
            }

            // Need more data from the underlying stream.
            match self.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    match std::str::from_utf8(&chunk) {
                        Ok(text) => self.buffer.push_str(text),
                        Err(e) => {
                            self.finished = true;
                            return Poll::Ready(Some(Err(ApiError::InvalidResponse {
                                message: format!("SSE stream contained invalid UTF-8: {e}"),
                                body: String::new(),
                            })));
                        }
                    }
                    // Loop back to try parsing again with the new data.
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(ApiError::Network {
                        message: format!("Error reading SSE stream: {e}"),
                        source: e,
                    })));
                }
                Poll::Ready(None) => {
                    // Stream ended. Try to flush any remaining buffered event.
                    if !self.current_data.is_empty() {
                        let event_type = self.current_event_type.take();
                        let data = std::mem::take(&mut self.current_data);
                        self.finished = true;
                        return Poll::Ready(Some(self.dispatch_event(
                            event_type.as_deref(),
                            &data,
                        )));
                    }
                    self.finished = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    /// Helper: build an SSE byte stream from raw text.
    fn sse_bytes(text: &str) -> impl Stream<Item = Result<Bytes, reqwest::Error>> {
        let chunks = vec![Ok(Bytes::from(text.to_owned()))];
        stream::iter(chunks)
    }

    #[tokio::test]
    async fn parse_message_start() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );

        let mut stream = SseStream::new(sse_bytes(raw));
        let event = stream.next().await.unwrap().unwrap();
        match event {
            ApiStreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_01");
                assert_eq!(message.model, "claude-sonnet-4-6");
            }
            other => panic!("Expected MessageStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parse_content_block_delta() {
        let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );

        let mut stream = SseStream::new(sse_bytes(raw));
        let event = stream.next().await.unwrap().unwrap();
        match event {
            ApiStreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    crate::types::ContentDelta::TextDelta { text } => {
                        assert_eq!(text, "Hello");
                    }
                    other => panic!("Expected TextDelta, got {other:?}"),
                }
            }
            other => panic!("Expected ContentBlockDelta, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handles_chunked_data() {
        // Data split across two chunks.
        let chunk1 = "event: message_start\ndata: {\"type\":\"message_sta";
        let chunk2 = concat!(
            "rt\",\"message\":{\"id\":\"msg_02\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\"}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );

        let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from(chunk1.to_owned())),
            Ok(Bytes::from(chunk2.to_owned())),
        ];
        let mut stream = SseStream::new(stream::iter(chunks));

        let event = stream.next().await.unwrap().unwrap();
        match event {
            ApiStreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_02");
            }
            other => panic!("Expected MessageStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn skips_comments() {
        let raw = concat!(
            ": this is a comment\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );

        let mut stream = SseStream::new(sse_bytes(raw));
        let event = stream.next().await.unwrap().unwrap();
        assert!(matches!(event, ApiStreamEvent::MessageStop));
    }
}
