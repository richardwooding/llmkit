# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the module uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
