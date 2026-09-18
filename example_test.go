package llmkit_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"

	"github.com/richardwooding/llmkit"
	"github.com/richardwooding/llmkit/openaicompat"
)

func fakeServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream   bool `json:"stream"`
			Messages []struct {
				Role string `json:"role"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		switch {
		case req.Stream:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"content\":\", world\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		case len(req.Messages) == 1:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Cape Town\"}"}}]},"finish_reason":"tool_calls"}]}`)
		default:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"It is 24°C in Cape Town."},"finish_reason":"stop"}]}`)
		}
	}))
}

func Example() {
	srv := fakeServer()
	defer srv.Close()
	// Any OpenAI-compatible server can be registered under its own prefix.
	llmkit.Register(openaicompat.NewProvider(openaicompat.Config{ID: "demo", BaseURL: srv.URL, KeyOptional: true}))

	chat, err := llmkit.Open[llmkit.Chatter]("demo/my-model")
	if err != nil {
		panic(err)
	}
	req := &llmkit.Request{
		Messages: []llmkit.Message{llmkit.UserText("What's the weather in Cape Town?")},
		Tools: []llmkit.Tool{{
			Name:       "weather",
			Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		}},
	}
	tools := map[string]llmkit.ToolFunc{
		"weather": func(_ context.Context, args json.RawMessage) (string, error) {
			var in struct{ City string }
			_ = json.Unmarshal(args, &in)
			return "24°C in " + in.City, nil
		},
	}
	resp, err := llmkit.RunTools(context.Background(), chat, req, tools, 5)
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())

	stream, _ := llmkit.Open[llmkit.Streamer]("demo/my-model")
	for chunk, err := range stream.Stream(context.Background(), &llmkit.Request{Messages: []llmkit.Message{llmkit.UserText("Say hello")}}) {
		if err != nil {
			panic(err)
		}
		if chunk.Kind == llmkit.ChunkText {
			fmt.Print(chunk.Text)
		}
	}
	fmt.Println()
	// Output:
	// It is 24°C in Cape Town.
	// Hello, world
}

func ExampleParseModel() {
	for _, name := range []string{"gpt-5", "claude-sonnet-4-5", "llama3.2:3b", "openrouter/openai/gpt-4o", "meta-llama/Llama-3.3-70B-Instruct"} {
		p, model, err := llmkit.ParseModel(name)
		if err != nil {
			fmt.Println(name, "→", err)
			continue
		}
		fmt.Printf("%-36s → %s / %s\n", name, p.ID(), model)
	}
	// Output:
	// gpt-5                                → openai / gpt-5
	// claude-sonnet-4-5                    → anthropic / claude-sonnet-4-5
	// llama3.2:3b                          → ollama / llama3.2:3b
	// openrouter/openai/gpt-4o             → openrouter / openai/gpt-4o
	// meta-llama/Llama-3.3-70B-Instruct    → huggingface / meta-llama/Llama-3.3-70B-Instruct
}
