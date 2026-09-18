package groq_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/groq"
)

func TestXGroqUsageAndIdentity(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"hi","reasoning":"r"},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":2,"completion_tokens":1}}}`)
	}))
	defer srv.Close()
	c, err := groq.New("llama-3.3-70b-versatile", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.User(core.Text("x"), core.Image([]byte{1}, "image/png"))}})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/chat/completions" || resp.Usage.TotalTokens != 3 || resp.Message.Parts[0].(core.ReasoningPart).Text != "r" {
		t.Fatalf("path=%s resp=%+v", path, resp)
	}
	var cl core.Client = c
	if _, ok := cl.(core.Embedder); ok {
		t.Fatal("groq has no embeddings")
	}
	if (groq.Provider{}).ID() != "groq" || (groq.Provider{}).Matches("llama-3.3-70b-versatile") {
		t.Fatal("identity wrong")
	}
}
