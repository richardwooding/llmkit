# llmkit

[![Go Reference](https://pkg.go.dev/badge/github.com/richardwooding/llmkit.svg)](https://pkg.go.dev/github.com/richardwooding/llmkit)
[![CI](https://github.com/richardwooding/llmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/richardwooding/llmkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

One Go client for twelve AI back-ends. Pick a model by name, code against a
one-method interface, and swap vendors without touching call sites.

```go
chat, err := llmkit.Open[llmkit.Chatter]("claude-sonnet-4-5")
resp, err := chat.Chat(ctx, &llmkit.Request{
	Messages: []llmkit.Message{llmkit.UserText("Explain iter.Seq2 in one paragraph.")},
})
fmt.Println(resp.Text())
```

Pure Go 1.27, no cgo. The core module depends only on the standard library and
`golang.org/x/oauth2` (Google credentials for Vertex AI). Vertex AI over gRPC
lives in a separate nested module so its dependency tree stays opt-in.

## Why

- **Small interfaces.** `Chatter`, `Streamer` and `Embedder` each have one
  method. Ask for exactly what you need with `llmkit.Open[T]`; a provider that
  lacks the capability fails at construction with `ErrUnsupported`, before any
  network call.
- **Model-name routing.** `"gpt-5"`, `"claude-sonnet-4-5"`, `"gemini-2.5-pro"`,
  `"deepseek-reasoner"`, `"grok-4"`, `"command-a-03-2025"`, `"voyage-3-large"`
  and `"llama3.2:3b"` resolve on their own. Anything ambiguous takes a prefix:
  `"groq/llama-3.3-70b-versatile"`, `"openrouter/openai/gpt-4o"`,
  `"hf/meta-llama/Llama-3.3-70B-Instruct"`, `"vertexgrpc/gemini-2.5-flash"`.
  Bare open-weight names fall back to a local Ollama daemon.
- **One message model.** Text, images, audio, documents, reasoning, tool calls
  and tool results are typed parts; each provider maps what it supports and
  rejects the rest up front.
- **Streaming as iterators.** `Stream` returns `iter.Seq2[Chunk, error]`; the
  request is sent when you start ranging and closed when you stop.
- **Escape hatches.** `Request.Extra` and `Request.ProviderOptions` merge raw
  fields into the wire body; every response keeps its raw JSON.

## Install

```sh
go get github.com/richardwooding/llmkit
go get github.com/richardwooding/llmkit/vertexgrpc   # optional, Vertex AI over gRPC
```

## Providers

| Provider | Names | Chat | Stream | Embed | Tools | Image | Audio | File/PDF | Auth |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| OpenAI (Responses API) | `gpt-*`, `o*`, `text-embedding-3-*` | ✅ | ✅ | ✅ | ✅ | ✅ | – | ✅ | `OPENAI_API_KEY` |
| DeepSeek | `deepseek-*` | ✅ | ✅ | – | ✅¹ | – | – | – | `DEEPSEEK_API_KEY` |
| Ollama | `name:tag`, bare fallback | ✅ | ✅ | ✅ | ✅ | ✅ | – | – | `OLLAMA_HOST` |
| Vertex AI (REST) | `gemini-*`, `text-embedding-*` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ADC / `GOOGLE_CLOUD_PROJECT` |
| Vertex AI (gRPC) | `vertexgrpc/…` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ADC |
| Anthropic | `claude-*` | ✅ | ✅ | – | ✅ | ✅ | – | ✅ | `ANTHROPIC_API_KEY` |
| Cohere | `command*`, `embed-*` | ✅ | ✅ | ✅ | ✅ | ✅ | – | – | `COHERE_API_KEY` |
| Groq | `groq/…` | ✅ | ✅ | – | ✅ | ✅ | – | – | `GROQ_API_KEY` |
| x.ai (Grok) | `grok-*` | ✅ | ✅ | – | ✅ | ✅ | – | – | `XAI_API_KEY` |
| Hugging Face | `org/model`, `hf/…` | ✅ | ✅ | ✅ | ✅ | ✅ | – | – | `HF_TOKEN` |
| OpenRouter | `openrouter/…` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `OPENROUTER_API_KEY` |
| VoyageAI | `voyage-*` | – | – | ✅ | – | – | – | – | `VOYAGE_API_KEY` |

¹ `deepseek-reasoner` rejects tool definitions; llmkit fails fast with `ErrUnsupported`.

Every provider reads its key from the environment variable shown, or from
`llmkit.WithAPIKey`. `llmkit.WithBaseURL`, `WithHTTPClient`, `WithHeader` and
`WithTimeout` apply to all of them.

## Usage

### Streaming

```go
stream, err := llmkit.Open[llmkit.Streamer]("llama3.2")
for chunk, err := range stream.Stream(ctx, req) {
	if err != nil {
		return err
	}
	switch chunk.Kind {
	case llmkit.ChunkText:
		fmt.Print(chunk.Text)
	case llmkit.ChunkFinish:
		fmt.Println("\n", chunk.Usage.OutputTokens, "tokens")
	}
}
```

`llmkit.Collect(stream.Stream(ctx, req))` turns a stream back into a `*Response`.

### Tool calling

```go
req := &llmkit.Request{
	Messages: []llmkit.Message{llmkit.UserText("Weather in Cape Town?")},
	Tools: []llmkit.Tool{{
		Name:        "weather",
		Description: "Current weather for a city",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}},
}
tools := map[string]llmkit.ToolFunc{
	"weather": func(ctx context.Context, args json.RawMessage) (string, error) {
		var in struct{ City string `json:"city"` }
		if err := json.Unmarshal(args, &in); err != nil {
			return "", err
		}
		return lookup(in.City), nil
	},
}
resp, err := llmkit.RunTools(ctx, chat, req, tools, 5)
```

`RunTools` appends assistant and tool messages to `req.Messages` until the
model stops calling tools, feeding tool errors back as error results.

### Multimodal input

```go
png, _ := os.ReadFile("chart.png")
pdf, _ := os.ReadFile("report.pdf")
req := &llmkit.Request{Messages: []llmkit.Message{llmkit.User(
	llmkit.Text("Summarise the chart and the report."),
	llmkit.Image(png, "image/png"),
	llmkit.File(pdf, "application/pdf", "report.pdf"),
)}}
```

Providers that cannot accept a part return `ErrUnsupported` before sending anything.

### Embeddings

```go
embed, err := llmkit.Open[llmkit.Embedder]("voyage-3-large")
out, err := embed.Embed(ctx, &llmkit.EmbedRequest{
	Inputs:    []string{"first document", "second document"},
	InputType: llmkit.EmbedDocument,
})
vectors := out.Embeddings // [][]float32, one per input
```

### Custom OpenAI-compatible endpoints

```go
llmkit.Register(openaicompat.NewProvider(openaicompat.Config{
	ID:          "vllm",
	BaseURL:     "http://gpu-box:8000/v1",
	KeyOptional: true,
	Quirks:      openaicompat.Quirks{Images: true, StreamUsage: true},
}))
chat, err := llmkit.Open[llmkit.Chatter]("vllm/my-finetune")
```

### Vertex AI over gRPC

```go
import "github.com/richardwooding/llmkit/vertexgrpc"

llmkit.Register(vertexgrpc.Provider{})
chat, err := llmkit.Open[llmkit.Chatter]("vertexgrpc/gemini-2.5-flash",
	vertexgrpc.WithProject("my-project"), vertexgrpc.WithLocation("europe-west1"))
```

### Errors, retries and rate limits

Provider failures are `*llmkit.APIError` values carrying the HTTP status, the
provider's error code and any `Retry-After` hint. `errors.Is(err,
llmkit.ErrRateLimited)` and `errors.Is(err, llmkit.ErrContextLength)` work
across vendors. llmkit does not retry; compose an `http.RoundTripper` such as
[hostrate](https://github.com/richardwooding/hostrate) via `WithHTTPClient`.

### Resolution rules

1. If the text before the first `/` is a registered provider ID or alias, that
   provider gets the rest (`openrouter/openai/gpt-4o`).
2. Otherwise the first provider whose `Matches` accepts the bare name wins:
   OpenAI, Anthropic, Vertex, DeepSeek, x.ai, Cohere, VoyageAI prefixes; Ollama
   for anything containing `:`; Hugging Face for `org/model`.
3. Otherwise the fallback: `llmkit.SetFallback`, then `LLMKIT_DEFAULT_PROVIDER`,
   then `ollama`.

## What this is not

- Not an agent framework. `RunTools` is a twenty-line loop; bring your own
  planning, memory and observability.
- Not a retry or caching layer. Both belong in the `http.Client` you pass in.
- Not a wrapper around vendor SDKs. Every REST provider is written against the
  wire format with `net/http`; only `vertexgrpc` pulls in Google's client.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT © 2026 Richard Wooding
