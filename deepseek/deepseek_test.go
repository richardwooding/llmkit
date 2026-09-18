package deepseek_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/deepseek"
)

func TestReasonerRejectsTools(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hit = true }))
	defer srv.Close()
	c, err := deepseek.New("deepseek-reasoner", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Tools: []core.Tool{{Name: "f"}}})
	if !errors.Is(err, core.ErrUnsupported) || hit {
		t.Fatalf("err=%v hit=%v", err, hit)
	}
}

func TestReasoningContentAndNoEmbed(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"4","reasoning_content":"2+2"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	c, _ := deepseek.New("deepseek-chat", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("2+2")}, Reasoning: &core.ReasoningConfig{Effort: "high"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatal("deepseek must not send reasoning_effort")
	}
	if r, ok := resp.Message.Parts[0].(core.ReasoningPart); !ok || r.Text != "2+2" || resp.Text() != "4" {
		t.Fatalf("resp = %+v", resp)
	}
	var cl core.Client = c
	if _, ok := cl.(core.Embedder); ok {
		t.Fatal("deepseek client must not implement Embedder")
	}
	if !(deepseek.Provider{}).Matches("deepseek-chat") || (deepseek.Provider{}).Matches("gpt-4") {
		t.Fatal("Matches wrong")
	}
}
