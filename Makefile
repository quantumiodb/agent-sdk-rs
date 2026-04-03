SHELL := /bin/bash
EXAMPLE ?= simple

# ── Unit tests (no network) ───────────────────────────────────────────────────
test-unit:
	cargo test --lib

# ── Ollama native (/api/chat) ─────────────────────────────────────────────────
run-ollama:
	@source .env.ollama && cargo run --example $(EXAMPLE)

test-ollama:
	@source .env.ollama && cargo test --test ollama_integration -- --test-threads=1

# ── Ollama via Anthropic Messages API format (/v1/messages) ───────────────────
run-anthropic-ollama:
	@source .env.anthropic-ollama && cargo run --example $(EXAMPLE)

test-anthropic-ollama:
	@source .env.anthropic-ollama && cargo test --test ollama_integration -- --test-threads=1

# ── NVIDIA NIM ────────────────────────────────────────────────────────────────
run-nvidia:
	cargo run --example nvidia_nim

# ── Run all provider smoke tests in sequence ──────────────────────────────────
test-all: test-unit test-ollama test-anthropic-ollama run-nvidia
	@echo "✓ All provider tests passed"

.PHONY: test-unit run-ollama test-ollama run-anthropic-ollama test-anthropic-ollama run-nvidia test-all
