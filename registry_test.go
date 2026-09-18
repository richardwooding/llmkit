package llmkit_test

import (
	"context"
	"errors"
	"testing"

	"github.com/richardwooding/llmkit"
	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openaicompat"
)

func TestParseModelTable(t *testing.T) {
	t.Setenv(llmkit.DefaultProviderEnv, "")
	tests := []struct {
		in       string
		provider string
		model    string
	}{
		{"gpt-5", "openai", "gpt-5"},
		{"o3-mini", "openai", "o3-mini"},
		{"chatgpt-4o-latest", "openai", "chatgpt-4o-latest"},
		{"text-embedding-3-small", "openai", "text-embedding-3-small"},
		{"claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"},
		{"gemini-2.5-pro", "vertex", "gemini-2.5-pro"},
		{"text-embedding-005", "vertex", "text-embedding-005"},
		{"text-multilingual-embedding-002", "vertex", "text-multilingual-embedding-002"},
		{"deepseek-reasoner", "deepseek", "deepseek-reasoner"},
		{"grok-4", "xai", "grok-4"},
		{"command-a-03-2025", "cohere", "command-a-03-2025"},
		{"embed-v4.0", "cohere", "embed-v4.0"},
		{"voyage-3-large", "voyage", "voyage-3-large"},
		{"llama3.2:3b", "ollama", "llama3.2:3b"},
		{"hf.co/bartowski/x:Q4_K_M", "ollama", "hf.co/bartowski/x:Q4_K_M"},
		{"llama3.2", "ollama", "llama3.2"},
		{"qwen2.5", "ollama", "qwen2.5"},
		{"mistral", "ollama", "mistral"},
		{"meta-llama/Llama-3.3-70B-Instruct", "huggingface", "meta-llama/Llama-3.3-70B-Instruct"},
		{"openrouter/openai/gpt-4o", "openrouter", "openai/gpt-4o"},
		{"OpenAI/gpt-4o", "openai", "gpt-4o"},
		{"hf/meta-llama/Llama-3.3-70B-Instruct", "huggingface", "meta-llama/Llama-3.3-70B-Instruct"},
		{"groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b-versatile"},
		{"ollama/gpt-oss:20b", "ollama", "gpt-oss:20b"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			if _, ok := llmkit.Default.Lookup(tt.provider); !ok {
				t.Skipf("provider %q not registered yet", tt.provider)
			}
			p, model, err := llmkit.ParseModel(tt.in)
			if err != nil {
				t.Fatal(err)
			}
			if p.ID() != tt.provider || model != tt.model {
				t.Fatalf("got %s/%s want %s/%s", p.ID(), model, tt.provider, tt.model)
			}
		})
	}
}

func TestParseModelFallbacks(t *testing.T) {
	t.Setenv(llmkit.DefaultProviderEnv, "groq")
	p, _, err := llmkit.ParseModel("mistral")
	if err != nil || p.ID() != "groq" {
		t.Fatalf("env fallback: %v %v", p, err)
	}
	r := llmkit.NewRegistry(llmkit.Default.Providers()...)
	r.SetFallback("nope")
	if _, _, err := r.ParseModel("mistral"); !errors.Is(err, llmkit.ErrUnknownProvider) {
		t.Fatalf("err = %v", err)
	}
	if _, _, err := r.ParseModel(""); err == nil {
		t.Fatal("empty model must error")
	}
	custom := openaicompat.NewProvider(openaicompat.Config{ID: "vllm", BaseURL: "http://gpu:8000/v1", KeyOptional: true})
	r.Register(custom, "local")
	for _, in := range []string{"vllm/my-model", "local/my-model"} {
		p, model, err := r.ParseModel(in)
		if err != nil || p.ID() != "vllm" || model != "my-model" {
			t.Fatalf("%s: got %v %q %v", in, p, model, err)
		}
	}
	r.Register(openaicompat.NewProvider(openaicompat.Config{ID: "vllm", BaseURL: "http://other", KeyOptional: true}))
	if got := len(r.Providers()); got != len(llmkit.Default.Providers())+1 {
		t.Fatalf("re-register must replace, got %d providers", got)
	}
}

func TestOpenAndAs(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "k")
	chat, err := llmkit.Open[llmkit.Chatter]("deepseek-chat")
	if err != nil || chat == nil {
		t.Fatalf("Open Chatter: %v", err)
	}
	if _, err := llmkit.Open[llmkit.Embedder]("deepseek-chat"); !errors.Is(err, llmkit.ErrUnsupported) {
		t.Fatalf("deepseek embedder err = %v", err)
	}
	type chatStream interface {
		llmkit.Chatter
		llmkit.Streamer
	}
	if _, err := llmkit.Open[chatStream]("ollama/llama3.2"); err != nil {
		t.Fatal(err)
	}
	c, err := llmkit.New("llama3.2:3b")
	if err != nil || c.Provider() != "ollama" || c.Model() != "llama3.2:3b" {
		t.Fatalf("New = %v %v", c, err)
	}
	if _, err := llmkit.As[llmkit.Embedder](c); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GROQ_API_KEY", "")
	if _, err := llmkit.New("groq/llama-3.3-70b-versatile", llmkit.WithAPIKey("")); !errors.Is(err, llmkit.ErrMissingAPIKey) {
		t.Fatalf("missing key err = %v", err)
	}
}

type scripted struct {
	responses []*core.Response
	calls     int
	seen      [][]core.Message
}

func (s *scripted) Chat(_ context.Context, req *core.Request) (*core.Response, error) {
	s.seen = append(s.seen, append([]core.Message(nil), req.Messages...))
	if s.calls >= len(s.responses) {
		return nil, errors.New("no more responses")
	}
	r := s.responses[s.calls]
	s.calls++
	return r, nil
}
