# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the module uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `core.ReasoningDelta` and `Chunk.Reasoning`: streamed reasoning fragments carry a block
  index plus the provider's `Signature`/`Encrypted` tokens (Anthropic `signature_delta` and
  `redacted_thinking`, OpenAI reasoning item `id`/`encrypted_content`, Gemini
  `thoughtSignature`).
- `Usage.CacheWriteTokens` (Anthropic `cache_creation_input_tokens`), summed by `Usage.Add`.
- `core.CacheConfig` on `Request.Cache`: Anthropic places `cache_control` breakpoints on the
  tool list, system prompt and last N user turns (capped at four, 5m or 1h TTL); providers
  with automatic caching ignore it.
- `ReasoningConfig.Summary`: Anthropic `thinking.display`, OpenAI `reasoning.summary`, with
  `auto` and `summarized` translated between the two.
- `core.TokenCounter`, implemented by Anthropic via `POST /messages/count_tokens`;
  `llmkit.Open[llmkit.TokenCounter]` on other providers returns `ErrUnsupported`.
- `catalog` package: context windows, output limits, USD-per-Mtok prices (including cache
  read/write rates) and capabilities for Anthropic, OpenAI, Gemini, DeepSeek, xAI and Groq
  models, with `Lookup`, `Register`, `All`, `(Model).Cost(core.Usage)` and `DataAsOf`.
  Unknown models resolve to `Known == false` rather than a guess.
- Aliases `llmkit.TokenCounter`, `llmkit.CacheConfig`, `llmkit.ReasoningDelta`.

### Changed

- **Anthropic `Usage.InputTokens` now counts the whole prompt** (uncached + cache read +
  cache write), matching every other provider; `TotalTokens` follows. Code that added
  `CachedInputTokens` on top will double count.
- `Collect` emits one `ReasoningPart` per reasoning block, in block order, with signatures
  preserved; previously all reasoning text was merged into one unsigned part.
- Anthropic `max_tokens` default raised from 4096 to 16384, since thinking counts against it.
- Anthropic sends `system` as a block list whenever `Request.Cache` is set (a string otherwise).

### Fixed

- Streaming with extended thinking and tools on Anthropic no longer produces a `Collect`
  result the next turn rejects with HTTP 400: thinking signatures and redacted blocks are
  kept.

## [0.2.0] - 2026-09-18

### Added

- `Reranker` interface, implemented by Cohere (`/rerank`) and VoyageAI (`/rerank`).
- `MultimodalEmbedder` interface, implemented by VoyageAI (`/multimodalembeddings`) for
  text, image and video inputs.
- `cmd/llmkit` CLI with `chat`, `embed` and `resolve` commands.
- GitHub Pages site at https://richardwooding.github.io/llmkit/.

## [0.1.0] - 2026-09-18

### Added

- Core types and single-method interfaces: `Chatter`, `Streamer`, `Embedder`.
- Model-name factory: `llmkit.New`, `llmkit.Open[T]`, `llmkit.ParseModel`, registry with
  `<provider>/<model>` prefixes, bare-name matching and an Ollama fallback.
- Providers: OpenAI (Responses API), DeepSeek, Groq, x.ai, OpenRouter, Hugging Face,
  Ollama, Anthropic, Cohere, Vertex AI (REST), VoyageAI.
- Nested module `vertexgrpc` for Vertex AI over gRPC.
- Tool-calling loop `RunTools` and stream collector `Collect`.
- Public `openaicompat` transport for registering custom OpenAI-compatible endpoints.
