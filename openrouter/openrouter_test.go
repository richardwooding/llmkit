package openrouter_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openrouter"
)

func TestReasoningAndHeaders(t *testing.T) {
	var body map[string]any
	var hdr http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hdr = r.Header.Clone()
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok","reasoning":"why"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	c, err := openrouter.New("openai/gpt-4o", core.WithAPIKey("k"), core.WithBaseURL(srv.URL), openrouter.WithReferer("https://app"), openrouter.WithTitle("App"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &core.Request{
		Messages:  []core.Message{core.User(core.Text("x"), core.File([]byte{1}, "application/pdf", "a.pdf"), core.Audio([]byte{2}, "audio/mp3"))},
		Reasoning: &core.ReasoningConfig{Effort: "low", BudgetTokens: 100},
	})
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Get("HTTP-Referer") != "https://app" || hdr.Get("X-Title") != "App" {
		t.Fatalf("headers = %v", hdr)
	}
	r := body["reasoning"].(map[string]any)
	if r["effort"] != "low" || r["max_tokens"] != float64(100) {
		t.Fatalf("reasoning = %v", r)
	}
	if resp.Message.Parts[0].(core.ReasoningPart).Text != "why" {
		t.Fatalf("resp = %+v", resp)
	}
	var cl core.Client = c
	if _, ok := cl.(core.Embedder); !ok {
		t.Fatal("openrouter must embed")
	}
	if (openrouter.Provider{}).Matches("openai/gpt-4o") {
		t.Fatal("openrouter must not claim bare names")
	}
}
