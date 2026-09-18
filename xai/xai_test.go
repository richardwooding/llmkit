package xai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/xai"
)

func TestReasoningEffortAndIdentity(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"hi","reasoning_content":"r"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	c, err := xai.New("grok-4", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Reasoning: &core.ReasoningConfig{Effort: "high"}, Seed: new(int64(1))})
	if err != nil {
		t.Fatal(err)
	}
	if body["reasoning_effort"] != "high" || body["seed"] != float64(1) || resp.Message.Parts[0].(core.ReasoningPart).Text != "r" {
		t.Fatalf("body=%v resp=%+v", body, resp)
	}
	var cl core.Client = c
	if _, ok := cl.(core.Embedder); ok {
		t.Fatal("xai has no embeddings")
	}
	if !(xai.Provider{}).Matches("grok-4") || (xai.Provider{}).Matches("gpt-4") {
		t.Fatal("Matches wrong")
	}
}
