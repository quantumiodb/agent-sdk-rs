# Provider-Specific Quirks

Differences from the standard OpenAI Chat Completions streaming protocol,
discovered during integration testing.

---

## NVIDIA NIM — z-ai/glm4.7

**Endpoint:** `https://integrate.api.nvidia.com/v1/chat/completions`

### 1. `finish_reason` is always `null`

Standard OpenAI sets `finish_reason: "tool_calls"` in the final chunk when
the model issues a tool call, and `finish_reason: "stop"` for normal turns.

GLM4.7 **never** sets `finish_reason` — it is always `null` in every chunk,
including the final one before `[DONE]`.

```json
// Standard OpenAI (last chunk)
{"choices":[{"finish_reason":"tool_calls","delta":{}}]}

// GLM4.7 (last chunk)
{"choices":[{"finish_reason":null,"delta":{}}]}
{"choices":[],"usage":{...}}   // usage in a separate empty-choices chunk
[DONE]
```

**SDK fix:** On `[DONE]`, if `tool_block_map` has unclosed blocks, the SDK
synthesises `ContentBlockStop` for each block and emits
`MessageDelta { stop_reason: ToolUse }` to keep the agent loop correct.

---

### 2. Complete tool call in a single chunk

Standard OpenAI streams tool calls across multiple chunks: `name` first,
then `arguments` as incremental fragments.

GLM4.7 delivers the entire tool call — `id`, `name`, and the full
`arguments` JSON — in **one chunk**.

```json
// GLM4.7 (single chunk, complete)
{"tool_calls":[{
  "index": 0,
  "id": "019222a4-9b76-4b17-803a-9cb89cd585c5",
  "type": "function",
  "function": {
    "name": "Grep",
    "arguments": "{\"pattern\":\"PermissionDenied\",\"path\":\"src/tools/\",\"glob\":\"*.rs\",\"output_mode\":\"files_with_matches\"}"
  }
}]}
```

**SDK:** The existing incremental parser handles this correctly — it emits
`ContentBlockStart` (name) and immediately `ContentBlockDelta` (full
arguments) from the same chunk.

---

### 3. `reasoning_content` field (chain-of-thought)

GLM4.7 streams its chain-of-thought in a **`reasoning_content`** delta
field, not in `content`.  Standard OpenAI has no such field; Ollama models
(Qwen3, DeepSeek-R1) use **`reasoning`** instead.

```json
// GLM4.7 reasoning chunk
{"delta":{"content":null,"reasoning_content":"The user wants me to..."}}

// Ollama/Qwen3 reasoning chunk
{"delta":{"content":null,"reasoning":"The user wants me to..."}}
```

`content` is `null` while reasoning is streaming; the final answer arrives
in `content` after the tool call result is returned.

**SDK:** `OaiDelta` deserialises both fields; whichever is non-null is
emitted as `ThinkingDelta`. Priority: `reasoning_content` > `reasoning`.

---

### 4. `<tool_call>` tags echoed in `content` (multi-turn)

In some multi-turn responses (after a tool result is returned), GLM4.7 may
echo the tool invocation as plain text inside the `content` delta using
XML-like tags, **in addition to** the proper `tool_calls` JSON:

```
<tool_call>Bash<arg_key>command</arg_key><arg_value>ls src/</arg_value></tool_call>
```

The `tool_calls` JSON field is always authoritative and is what the SDK uses
to execute tools.  The text tags are cosmetic and appear only in `content`.

**SDK fix:** `OaiSseStream::clean_text()` strips `<tool_call>…</tool_call>`
blocks from text deltas before emitting `TextDelta` events, using a
stateful buffer to handle tags split across SSE chunks.

---

### 5. Usage in a separate chunk with empty `choices`

The token-usage object arrives in its own final chunk **after** the last
content chunk, with `choices: []`:

```json
{"choices":[],"usage":{"prompt_tokens":164,"completion_tokens":61,"total_tokens":225,...}}
```

**SDK:** The existing usage handler already checks `chunk.usage` before
iterating `choices`, so this is handled correctly.

---

### Summary table

| Behaviour | Standard OpenAI | GLM4.7 (NVIDIA NIM) |
|---|---|---|
| `finish_reason` | `"stop"` / `"tool_calls"` | always `null` |
| Tool call delivery | streamed fragments | single complete chunk |
| Chain-of-thought field | — | `reasoning_content` |
| Tool echo in `content` | no | sometimes (multi-turn) |
| Usage chunk | inline with last choice | separate empty-choices chunk |
