package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/richardwooding/llmkit"
	"github.com/richardwooding/llmkit/openaicompat"
)

func startFake(t *testing.T) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/embeddings"):
			_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[1,2]},{"index":1,"embedding":[3]}]}`)
		case strings.Contains(r.Header.Get("Accept"), "event-stream") && bytes.Contains(readBody(r), []byte(`"stream":true`)):
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		default:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`)
		}
	}))
	t.Cleanup(srv.Close)
	llmkit.Register(openaicompat.NewProvider(openaicompat.Config{ID: "fake", BaseURL: srv.URL, KeyOptional: true, Quirks: openaicompat.Quirks{EmbedPath: "/embeddings"}}))
}

func readBody(r *http.Request) []byte {
	b, _ := io.ReadAll(r.Body)
	return b
}

func TestCommands(t *testing.T) {
	startFake(t)
	tests := []struct {
		name  string
		args  []string
		stdin string
		want  string
	}{
		{"chat args", []string{"chat", "-m", "fake/m", "ping"}, "", "pong\n"},
		{"chat stdin stream", []string{"chat", "-m", "fake/m", "-stream", "-system", "s"}, "ping", "hi\n"},
		{"embed", []string{"embed", "-m", "fake/e", "a", "b"}, "", "[1,2]\n[3]\n"},
		{"embed stdin", []string{"embed", "-m", "fake/e"}, "a\nb\n", "[1,2]\n[3]\n"},
		{"resolve", []string{"resolve", "fake/m", "llama3.2:3b"}, "", "fake/m\tfake\tm\nllama3.2:3b\tollama\tllama3.2:3b\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			if err := run(tt.args, strings.NewReader(tt.stdin), &out); err != nil {
				t.Fatal(err)
			}
			if out.String() != tt.want {
				t.Fatalf("got %q want %q", out.String(), tt.want)
			}
		})
	}
	var out bytes.Buffer
	if err := run([]string{"chat", "-m", "fake/m", "-json", "ping"}, nil, &out); err != nil || !strings.Contains(out.String(), `"FinishReason": "stop"`) {
		t.Fatalf("json: %v %s", err, out.String())
	}
	for _, args := range [][]string{{}, {"bogus"}, {"chat"}, {"embed"}, {"resolve"}, {"chat", "-m", "fake/m"}} {
		if err := run(args, strings.NewReader(""), &out); err == nil {
			t.Fatalf("%v should fail", args)
		}
	}
}
