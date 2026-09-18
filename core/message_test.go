package core_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/richardwooding/llmkit/core"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	msgs := []core.Message{
		core.System("be terse"),
		core.User(core.Text("hi"), core.Image([]byte{1, 2}, "image/png"), core.ImageURL("https://x/y.png"),
			core.Audio([]byte{3}, "audio/wav"), core.File([]byte{4}, "application/pdf", "a.pdf"), core.FileURL("https://x/a.pdf", "application/pdf")),
		core.Assistant(core.ReasoningPart{Text: "hmm", Signature: "sig"}, core.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}),
		core.ToolResults(core.ToolResultText("c1", "f", "ok"), core.ToolResult{CallID: "c2", Name: "g", IsError: true, Content: []core.Part{core.Image([]byte{9}, "image/png")}}),
	}
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	var back []core.Message
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(msgs, back) {
		t.Fatalf("round trip mismatch\n got %#v\nwant %#v", back, msgs)
	}
}

func TestMessageUnmarshalUnknownPart(t *testing.T) {
	var m core.Message
	err := json.Unmarshal([]byte(`{"role":"user","parts":[{"type":"video"}]}`), &m)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestMessageAccessors(t *testing.T) {
	m := core.Assistant(core.Text("a"), core.ToolCall{ID: "1", Name: "f"}, core.Text("b"))
	if got := m.Text(); got != "ab" {
		t.Fatalf("Text = %q", got)
	}
	if got := len(m.ToolCalls()); got != 1 {
		t.Fatalf("ToolCalls = %d", got)
	}
	var args struct{ X int }
	if err := (core.ToolCall{}).UnmarshalArgs(&args); err != nil {
		t.Fatal(err)
	}
	tc := core.ToolCall{Arguments: json.RawMessage(`{"X":3}`)}
	if err := tc.UnmarshalArgs(&args); err != nil || args.X != 3 {
		t.Fatalf("UnmarshalArgs = %v %v", args, err)
	}
}

func TestAPIErrorIs(t *testing.T) {
	tests := []struct {
		name   string
		err    *core.APIError
		target error
		want   bool
	}{
		{"429", &core.APIError{Status: http.StatusTooManyRequests}, core.ErrRateLimited, true},
		{"rate code", &core.APIError{Status: 400, Code: "rate_limit_exceeded"}, core.ErrRateLimited, true},
		{"400 not rate", &core.APIError{Status: 400}, core.ErrRateLimited, false},
		{"context code", &core.APIError{Code: "context_length_exceeded"}, core.ErrContextLength, true},
		{"anthropic msg", &core.APIError{Message: "prompt is too long: 250000 tokens"}, core.ErrContextLength, true},
		{"cohere msg", &core.APIError{Message: "too many tokens"}, core.ErrContextLength, true},
		{"other", &core.APIError{Message: "nope"}, core.ErrContextLength, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := errors.Is(tt.err, tt.target); got != tt.want {
				t.Fatalf("errors.Is = %v, want %v", got, tt.want)
			}
		})
	}
	if got := (&core.APIError{Provider: "p", Status: 500, Code: "x", Message: "boom"}).Error(); got != "p: HTTP 500 [x]: boom" {
		t.Fatalf("Error() = %q", got)
	}
	if !errors.Is(core.Unsupported("p", "audio"), core.ErrUnsupported) {
		t.Fatal("Unsupported should wrap ErrUnsupported")
	}
}

func TestConfig(t *testing.T) {
	type key struct{}
	hc := &http.Client{}
	cfg := core.NewConfig(core.WithAPIKey("k"), core.WithBaseURL("http://x"), core.WithHTTPClient(hc),
		core.WithHeader("X-A", "1"), core.WithHeader("X-A", "2"), core.WithValue(key{}, 42), nil)
	if cfg.APIKey != "k" || cfg.BaseURL != "http://x" || cfg.Client() != hc {
		t.Fatalf("cfg = %+v", cfg)
	}
	if got := cfg.Headers.Values("X-A"); len(got) != 2 {
		t.Fatalf("headers = %v", got)
	}
	if cfg.Value(key{}) != 42 || cfg.Value("missing") != nil {
		t.Fatal("Value lookup failed")
	}
	if (&core.Config{}).Client() == nil {
		t.Fatal("Client() must never be nil")
	}
}

func TestProviderExtra(t *testing.T) {
	r := &core.Request{Extra: map[string]any{"a": 1, "b": 1}, ProviderOptions: map[string]map[string]any{"p": {"b": 2}}}
	got := r.ProviderExtra("p")
	if got["a"] != 1 || got["b"] != 2 {
		t.Fatalf("ProviderExtra = %v", got)
	}
	if (&core.Request{}).ProviderExtra("p") != nil {
		t.Fatal("empty extra should be nil")
	}
	if u := (core.Usage{InputTokens: 1}).Add(core.Usage{InputTokens: 2, OutputTokens: 3}); u.InputTokens != 3 || u.OutputTokens != 3 {
		t.Fatalf("Add = %+v", u)
	}
}
