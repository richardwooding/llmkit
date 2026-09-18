package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/ollama"
)

type capture struct {
	path string
	body map[string]any
}

func newServer(t *testing.T, respond func(w http.ResponseWriter)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&cap.body)
		respond(w)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func TestHostNormalisation(t *testing.T) {
	tests := []struct{ env, want string }{
		{"", ollama.DefaultHost},
		{"localhost:11434", "http://localhost:11434"},
		{"http://gpu:11434/", "http://gpu:11434"},
		{"https://ollama.example.com", "https://ollama.example.com"},
	}
	for _, tt := range tests {
		t.Run(tt.env, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.env)
			srv, cap := newServer(t, func(w http.ResponseWriter) { _, _ = io.WriteString(w, `{"embeddings":[[1]]}`) })
			c, err := ollama.New("m", core.WithBaseURL(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}}); err != nil {
				t.Fatal(err)
			}
			if cap.path != "/api/embed" {
				t.Fatalf("path = %s", cap.path)
			}
		})
	}
	if _, err := ollama.New("m", core.WithBaseURL("http://a://b")); err == nil {
		t.Fatal("invalid host should error")
	}
	if !(ollama.Provider{}).Matches("llama3.2:3b") || !(ollama.Provider{}).Matches("hf.co/org/m") || (ollama.Provider{}).Matches("llama3.2") {
		t.Fatal("Matches wrong")
	}
}

func TestChatMapping(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter) {
		_, _ = io.WriteString(w, `{"model":"m","message":{"role":"assistant","content":"","thinking":"hmm","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":6}`)
	})
	c, err := ollama.New("m", core.WithBaseURL(srv.URL), ollama.WithKeepAlive("5m"))
	if err != nil {
		t.Fatal(err)
	}
	req := &core.Request{
		Messages: []core.Message{
			core.System("s"),
			core.User(core.Text("look"), core.Image([]byte{1}, "image/png")),
			core.Assistant(core.ReasoningPart{Text: "r"}, core.Text("ok"), core.ToolCall{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}),
			core.ToolResults(core.ToolResultText("call_1", "f", "42")),
		},
		Tools:       []core.Tool{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		MaxTokens:   9,
		Temperature: new(0.1),
		Seed:        new(int64(3)),
		Stop:        []string{"x"},
		Format:      &core.ResponseFormat{Type: core.FormatJSON},
		Reasoning:   &core.ReasoningConfig{Effort: "high"},
		Extra:       map[string]any{"keep_alive": "1m"},
	}
	resp, err := c.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	b := cap.body
	if cap.path != "/api/chat" || b["stream"] != false || b["think"] != true || b["format"] != "json" || b["keep_alive"] != "1m" {
		t.Fatalf("body = %v", b)
	}
	opts := b["options"].(map[string]any)
	if opts["num_predict"] != float64(9) || opts["temperature"] != 0.1 || opts["seed"] != float64(3) {
		t.Fatalf("options = %v", opts)
	}
	msgs := b["messages"].([]any)
	user := msgs[1].(map[string]any)
	if user["content"] != "look" || user["images"].([]any)[0] != "AQ==" {
		t.Fatalf("user = %v", user)
	}
	asst := msgs[2].(map[string]any)
	if asst["thinking"] != "r" || asst["content"] != "ok" || asst["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)["arguments"].(map[string]any)["a"] != float64(1) {
		t.Fatalf("assistant = %v", asst)
	}
	tool := msgs[3].(map[string]any)
	if tool["role"] != "tool" || tool["tool_name"] != "f" || tool["content"] != "42" {
		t.Fatalf("tool = %v", tool)
	}
	resp.Raw = nil
	want := &core.Response{
		Model: "m", FinishReason: core.FinishToolCalls,
		Message: core.Assistant(core.ReasoningPart{Text: "hmm"}, core.ToolCall{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}),
		Usage:   core.Usage{InputTokens: 4, OutputTokens: 6, TotalTokens: 10},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestUnsupportedBeforeIO(t *testing.T) {
	hit := false
	srv, _ := newServer(t, func(w http.ResponseWriter) { hit = true })
	c, _ := ollama.New("m", core.WithBaseURL(srv.URL))
	for _, m := range []core.Message{
		core.User(core.ImageURL("https://x/y.png")),
		core.User(core.Audio([]byte{1}, "audio/wav")),
		core.User(core.File([]byte{1}, "application/pdf", "f")),
	} {
		if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{m}}); !errors.Is(err, core.ErrUnsupported) {
			t.Fatalf("%T: %v", m.Parts[0], err)
		}
	}
	if hit {
		t.Fatal("server must not be hit")
	}
}

func TestStream(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter) {
		_, _ = io.WriteString(w, `{"model":"m","message":{"role":"assistant","content":"","thinking":"hm"},"done":false}
{"model":"m","message":{"role":"assistant","content":"Hel"},"done":false}
{"model":"m","message":{"role":"assistant","content":"lo"},"done":false}
{"model":"m","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},"done":false}
{"model":"m","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":4}
`)
	})
	c, _ := ollama.New("m", core.WithBaseURL(srv.URL))
	var got []core.Chunk
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	if cap.body["stream"] != true {
		t.Fatal("stream flag missing")
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "hm"},
		{Kind: core.ChunkText, Text: "Hel"},
		{Kind: core.ChunkText, Text: "lo"},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "call_1", Name: "f", Arguments: `{"a":1}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v\nwant %+v", got, want)
	}

	srv2, _ := newServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":"model 'm' not found"}`)
	})
	c2, _ := ollama.New("m", core.WithBaseURL(srv2.URL))
	var n int
	var gotErr error
	for _, err := range c2.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		gotErr = err
	}
	var apiErr *core.APIError
	if n != 1 || !errors.As(gotErr, &apiErr) || apiErr.Message != "model 'm' not found" {
		t.Fatalf("n=%d err=%v", n, gotErr)
	}
}

func TestEmbed(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter) {
		_, _ = io.WriteString(w, `{"model":"e","embeddings":[[0.1,0.2],[0.3]],"prompt_eval_count":5}`)
	})
	c, _ := ollama.New("e", core.WithBaseURL(srv.URL))
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a", "b"}, Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	if cap.body["dimensions"] != float64(2) || len(cap.body["input"].([]any)) != 2 {
		t.Fatalf("body = %v", cap.body)
	}
	if !reflect.DeepEqual(resp.Embeddings, [][]float32{{0.1, 0.2}, {0.3}}) || resp.Usage.InputTokens != 5 || resp.Model != "e" {
		t.Fatalf("resp = %+v", resp)
	}
	if _, err := c.Embed(context.Background(), &core.EmbedRequest{}); err == nil {
		t.Fatal("empty inputs must error")
	}
}
