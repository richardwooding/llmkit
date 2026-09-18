//go:build integration

package llmkit_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/richardwooding/llmkit"
)

// Live smoke tests catch upstream wire-format drift. Each case runs only when
// its key is present; run with: go test -tags integration -race ./...
func TestLiveChatStreamTools(t *testing.T) {
	cases := []struct {
		model string
		env   string
	}{
		{"gpt-5-mini", "OPENAI_API_KEY"},
		{"claude-haiku-4-5", "ANTHROPIC_API_KEY"},
		{"deepseek-chat", "DEEPSEEK_API_KEY"},
		{"groq/llama-3.3-70b-versatile", "GROQ_API_KEY"},
		{"grok-4-fast", "XAI_API_KEY"},
		{"openrouter/openai/gpt-4o-mini", "OPENROUTER_API_KEY"},
		{"hf/meta-llama/Llama-3.3-70B-Instruct", "HF_TOKEN"},
		{"command-a-03-2025", "COHERE_API_KEY"},
		{"gemini-2.5-flash", "GOOGLE_CLOUD_PROJECT"},
		{"llama3.2", "OLLAMA_HOST"},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if os.Getenv(tc.env) == "" {
				t.Skipf("%s not set", tc.env)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			type chatStream interface {
				llmkit.Chatter
				llmkit.Streamer
			}
			c, err := llmkit.Open[chatStream](tc.model)
			if err != nil {
				t.Fatal(err)
			}
			req := &llmkit.Request{Messages: []llmkit.Message{llmkit.UserText("Reply with the single word: pong")}, MaxTokens: 512}
			resp, err := c.Chat(ctx, req)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
				t.Fatalf("chat text = %q", resp.Text())
			}
			streamed, err := llmkit.Collect(c.Stream(ctx, req))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(strings.ToLower(streamed.Text()), "pong") {
				t.Fatalf("stream text = %q", streamed.Text())
			}

			toolReq := &llmkit.Request{
				Messages: []llmkit.Message{llmkit.UserText("What is the weather in Cape Town? Use the weather tool.")},
				Tools: []llmkit.Tool{{
					Name: "weather", Description: "Current weather for a city",
					Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
				}},
				MaxTokens: 512,
			}
			tools := map[string]llmkit.ToolFunc{"weather": func(context.Context, json.RawMessage) (string, error) { return "sunny, 24C", nil }}
			final, err := llmkit.RunTools(ctx, c, toolReq, tools, 4)
			if err != nil {
				t.Fatal(err)
			}
			if len(toolReq.Messages) < 3 {
				t.Fatalf("model did not call the tool; messages=%d text=%q", len(toolReq.Messages), final.Text())
			}
		})
	}
}

func TestLiveEmbed(t *testing.T) {
	cases := []struct {
		model string
		env   string
	}{
		{"text-embedding-3-small", "OPENAI_API_KEY"},
		{"embed-v4.0", "COHERE_API_KEY"},
		{"voyage-3-lite", "VOYAGE_API_KEY"},
		{"text-embedding-005", "GOOGLE_CLOUD_PROJECT"},
		{"nomic-embed-text", "OLLAMA_HOST"},
		{"openrouter/openai/text-embedding-3-small", "OPENROUTER_API_KEY"},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if os.Getenv(tc.env) == "" {
				t.Skipf("%s not set", tc.env)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			e, err := llmkit.Open[llmkit.Embedder](tc.model)
			if err != nil {
				t.Fatal(err)
			}
			out, err := e.Embed(ctx, &llmkit.EmbedRequest{Inputs: []string{"hello", "world"}, InputType: llmkit.EmbedDocument})
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Embeddings) != 2 || len(out.Embeddings[0]) == 0 {
				t.Fatalf("embeddings = %d x %d", len(out.Embeddings), len(out.Embeddings[0]))
			}
		})
	}
}
