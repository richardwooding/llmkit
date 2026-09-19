# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llmkit` is a multi-vendor AI client library (`module github.com/richardwooding/llmkit`,
Go 1.27, pure Go, no cgo). The root module depends on the standard library plus
`golang.org/x/oauth2` only — do not add third-party imports. Vertex AI over gRPC is a
**nested module** in `vertexgrpc/` with its own `go.mod` so grpc/protobuf never enter the
root module graph.

## Commands

```sh
go build ./...
go test -race ./...                              # full suite, httptest only, no network
go test -race -run TestParseModelTable .         # single test
go test -tags integration -race ./...            # live smoke tests, skipped without API keys

gofumpt -w .                                     # required; gofmt is not enough
golangci-lint run ./...                          # strict v2 config, the bar is 0 issues

(cd vertexgrpc && go test -race ./... && golangci-lint run --config ../.golangci.yml ./...)
```

`gofumpt` and `golangci-lint` may not be on PATH; `go install mvdan.cc/gofumpt@latest`
and add `$(go env GOPATH)/bin`. CI runs build, `go test -race`, `go fix -diff`, vet and
golangci-lint v2.13.1 for both modules.

## Architecture

```
core/            LEAF: interfaces, Request/Message/Part/Response/Chunk/Embed*, Config, errors
internal/httpx/  one HTTP funnel (auth, timeout, error envelope → *core.APIError), SSE, NDJSON, MarshalWithExtra
openaicompat/    PUBLIC Chat Completions transport parameterised by Config{BaseURL, APIKeyEnv, Quirks}
openai/          Responses API (+ embeddings via openaicompat)
deepseek/ groq/ xai/ openrouter/ huggingface/   thin wrappers over openaicompat
anthropic/ cohere/ ollama/ vertex/ voyage/      native wire formats
catalog/         stdlib + core only: model metadata (limits, USD/Mtok prices, capabilities), Lookup/Register/Cost
(root)           alias.go re-exports core as llmkit.X; registry.go, factory.go (New/Open/As),
                 tools.go (RunTools), stream.go (Collect), default.go (provider order)
vertexgrpc/      nested module; users call llmkit.Register(vertexgrpc.Provider{})
cmd/llmkit/      CLI (chat / embed / resolve), stdlib flag, tests use a registered fake provider
docs/            GitHub Pages site (gloam design system; gloam.css/js vendored, synced weekly by gloam-sync.yml)
```

Import graph is acyclic by construction: `core ← httpx ← providers ← root ← vertexgrpc`.
Types live in `core` because providers need them and the root needs the providers.

Optional capabilities beyond chat/stream/embed: `core.Reranker` (Cohere, Voyage),
`core.MultimodalEmbedder` (Voyage) and `core.TokenCounter` (Anthropic,
`/messages/count_tokens` with a body stripped of generation parameters). Add new
ones as separate one-method interfaces.

Every provider has the same shape: `const ID`, `Provider struct{}` (`ID`, `Matches`,
`Open`), `Client` with `New(model, opts...)`, `Provider()`, `Model()`, and ONLY the
capability methods its API supports. **The method set is the capability matrix** —
`llmkit.Open[llmkit.Embedder]("deepseek-chat")` fails with `ErrUnsupported` because
`deepseek.Client` has no `Embed`.

### Things that are non-obvious and easy to break

- **No I/O at construction.** `New`/`Open` read env vars and build structs. Vertex ADC
  discovery is deferred to the first request (`sync.OnceValues`). Tests rely on this.
- **Unsupported parts fail before any HTTP call** with `core.Unsupported(ID, what)`. Each
  mapper's `switch p := part.(type)` is the source of truth; keep the README matrix in sync.
- **Stream contract** (every provider): the request is sent when iteration starts; setup
  errors are yielded as the first item; exactly one `ChunkFinish` is yielded last, carrying
  `FinishReason` and `Usage`; `break` in the consumer closes the body via the deferred
  `Close`. `Collect` depends on this.
- **`ToolCallDelta.Index`** keys argument reassembly in `Collect`. OpenAI-compat uses the
  wire `index`; Anthropic uses a tool-call ordinal derived from block index; Ollama/Gemini
  send complete calls, one chunk each.
- **`ReasoningDelta.Index`** is a separate reasoning-block ordinal (not the tool-call one and
  not the wire block index). `Collect` concatenates text per Index and keeps the last
  Signature/Encrypted, so signed thinking survives a round trip; blocks with nothing in them
  are dropped. Anthropic: ordinal per `thinking`/`redacted_thinking` block, `signature_delta`
  closes it. OpenAI: `output_index`, with id/encrypted_content on `response.output_item.done`.
  Gemini: one ordinal until a `thoughtSignature` closes it; a signature on a text or
  functionCall part is emitted as a bare reasoning chunk first, as the mapper does.
  `Chunk.Text` mirrors `Reasoning.Text` for ChunkReasoning so old consumers still print it.
- **Usage convention**: `InputTokens` is the whole prompt on every provider;
  `CachedInputTokens` and `CacheWriteTokens` are subsets of it; `TotalTokens = Input + Output`.
  Anthropic's wire `input_tokens` is only the uncached remainder, so `toUsage` adds
  `cache_read_input_tokens` and `cache_creation_input_tokens` back in (OpenAI, Gemini and
  DeepSeek already report the whole prompt). `message_delta` usage is merged field-wise.
- **Cache breakpoints** (`Request.Cache`, Anthropic only): `applyCache` marks the last tool,
  the system prompt (which becomes `[]wireBlock` whenever `Cache != nil`) and the last block of
  the last N user-role *wire* messages (tool results are user role), in that order, stopping at
  the API's limit of four so the stable prefix wins. TTL is `5m` (omitted on the wire) or `1h`;
  anything else fails before I/O. Other providers cache automatically and ignore the field.
- **Ollama and Gemini have no tool-call IDs.** IDs are synthesised as `call_1`, `call_2`…
  in order; Gemini `functionResponse` addresses tools by NAME, so `ToolResult.Name` is
  required for Vertex. `RunTools` always sets it.
- **Anthropic** forbids consecutive same-role messages and requires tool results in the next
  user turn; the mapper merges/hoists. `max_tokens` defaults to 16384 so thinking plus a long answer is not truncated. `error` events can
  arrive after HTTP 200 and are turned into `*core.APIError`.
- **OpenAI Responses API** tools are flat (`{type:"function", name, parameters}`), calls are
  addressed by `call_id`, and the body uses `max_output_tokens`. Chat Completions quirks
  live in `openaicompat.Quirks` (stream usage chunk with empty `choices`, Groq's `x_groq`
  usage, `reasoning_content` vs `reasoning`, `max_completion_tokens`).
- **DeepSeek** 400s if `reasoning_content` is echoed back, so ReasoningParts are never sent
  to Chat-Completions providers; `deepseek-reasoner` + tools is rejected up front.
- **`httpx.SSE` uses `bufio.Reader`, not `Scanner`**: Responses `response.completed` frames
  exceed 64 KB. It also joins multi-line `data:`, drops `:` comments (OpenRouter sends
  them) and treats `[DONE]` as end of stream.
- **Registry order is match priority.** `default.go` lists strict-prefix providers first,
  then Ollama (claims `:`), then Hugging Face (claims `org/model`). Groq and OpenRouter
  never match bare names. `hf` is an alias. Re-registering an ID replaces in place.
- **`Config`/`Quirks` are passed by value on purpose**; `hugeParam`/`rangeValCopy` are
  disabled in `.golangci.yml`. Optional numeric request fields are pointers — use Go 1.27's
  `new(0.5)` rather than a helper.
- **Nested module tagging**: tag root `vX.Y.Z` first, bump the `require github.com/richardwooding/llmkit`
  line in `vertexgrpc/go.mod` to it, then tag `vertexgrpc/vX.Y.Z`. Never add a `replace`; the
  committed `go.work` already makes local development use the sibling root module.

## Conventions

- Tests are black-box (`package x_test`), table-driven, `httptest.NewServer` with a captured
  request body, inline SSE/NDJSON fixtures as raw strings, `reflect.DeepEqual` on `[]core.Chunk`
  with `Raw` zeroed. No golden files. Live tests only behind `//go:build integration`.
- No comments explaining *what* code does; short *why* comments for non-obvious behaviour.
  Exported identifiers carry one-line doc comments (revive `exported`).
- Repeated wire strings become package constants (goconst 2/3). Keep functions under
  gocyclo 15 / gocognit 20 by splitting stream state machines per event.
- Commit prefixes: `feat:`, `fix:`, `docs:`, `ci:`, `test:`, `chore:`.

## Releasing

`v0.x` — breaking changes allowed but recorded under `[Unreleased]` in `CHANGELOG.md`.
Tag-push only; no GoReleaser (library). See the nested-module tagging order above.
