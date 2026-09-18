package httpx_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

func TestSSE(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []httpx.Event
	}{
		{"basic", "event: a\ndata: {\"x\":1}\n\ndata: two\n\n", []httpx.Event{{Name: "a", Data: `{"x":1}`}, {Data: "two"}}},
		{"multiline", "data: l1\ndata: l2\n\n", []httpx.Event{{Data: "l1\nl2"}}},
		{"comment and crlf", ": OPENROUTER PROCESSING\r\ndata:nospace\r\n\r\n", []httpx.Event{{Data: "nospace"}}},
		{"done", "data: a\n\ndata: [DONE]\n\ndata: after\n\n", []httpx.Event{{Data: "a"}}},
		{"eof without blank", "data: tail", []httpx.Event{{Data: "tail"}}},
		{"id", "id: 7\ndata: x\n\n", []httpx.Event{{ID: "7", Data: "x"}}},
		{"long line", "data: " + strings.Repeat("y", 200_000) + "\n\n", []httpx.Event{{Data: strings.Repeat("y", 200_000)}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []httpx.Event
			for ev, err := range httpx.SSE(strings.NewReader(tt.in)) {
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, ev)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %#v want %#v", got, tt.want)
			}
		})
	}
}

type failReader struct{ n int }

func (f *failReader) Read(p []byte) (int, error) {
	if f.n == 0 {
		f.n++
		return copy(p, "data: a\n\n"), nil
	}
	return 0, errors.New("boom")
}

func TestSSEReadError(t *testing.T) {
	var events int
	var gotErr error
	for _, err := range httpx.SSE(&failReader{}) {
		if err != nil {
			gotErr = err
			break
		}
		events++
	}
	if events != 1 || gotErr == nil {
		t.Fatalf("events=%d err=%v", events, gotErr)
	}
}

func TestNDJSON(t *testing.T) {
	var got []string
	for line, err := range httpx.NDJSON(strings.NewReader("{\"a\":1}\n\n{\"b\":2}\r\n{\"c\":3}")) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, string(line))
	}
	want := []string{`{"a":1}`, `{"b":2}`, `{"c":3}`}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v", got)
	}
}

func TestParseError(t *testing.T) {
	tests := []struct {
		name string
		body string
		hdr  http.Header
		want core.APIError
	}{
		{
			"openai", `{"error":{"message":"bad","type":"invalid_request_error","code":"context_length_exceeded"}}`, nil,
			core.APIError{Message: "bad", Type: "invalid_request_error", Code: "context_length_exceeded"},
		},
		{
			"anthropic", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, nil,
			core.APIError{Message: "Overloaded", Type: "overloaded_error"},
		},
		{
			"google", `{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`, nil,
			core.APIError{Message: "quota", Code: "429"},
		},
		{"cohere", `{"message":"too many tokens"}`, nil, core.APIError{Message: "too many tokens"}},
		{"ollama", `{"error":"model not found"}`, nil, core.APIError{Message: "model not found"}},
		{"voyage", `{"detail":"invalid key"}`, nil, core.APIError{Message: "invalid key"}},
		{"plain", `<html>gateway</html>`, nil, core.APIError{Message: "<html>gateway</html>"}},
		{"retry seconds", `{}`, http.Header{"Retry-After": {"2"}}, core.APIError{Message: "{}", RetryAfter: 2 * time.Second}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := httpx.ParseError("p", 400, tt.hdr, []byte(tt.body))
			tt.want.Provider, tt.want.Status, tt.want.Body = "p", 400, []byte(tt.body)
			if !reflect.DeepEqual(*got, tt.want) {
				t.Fatalf("got %+v want %+v", *got, tt.want)
			}
		})
	}
	got := httpx.ParseError("p", 503, http.Header{"Retry-After": {time.Now().Add(30 * time.Second).UTC().Format(http.TimeFormat)}}, nil)
	if got.RetryAfter < 20*time.Second || got.RetryAfter > 31*time.Second {
		t.Fatalf("RetryAfter = %v", got.RetryAfter)
	}
}

func TestClientPostJSON(t *testing.T) {
	var gotAuth, gotHdr, gotCT string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotHdr = r.Header.Get("X-Test")
		gotCT = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		if r.URL.Path == "/fail" {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"slow down"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()

	cfg := core.NewConfig(core.WithBaseURL(srv.URL+"/"), core.WithHeader("X-Test", "1"))
	c := httpx.New("p", cfg, "http://ignored")
	c.Auth = httpx.BearerAuth("k")
	body, _ := httpx.MarshalWithExtra(map[string]any{"a": 1}, map[string]any{"b": 2})
	var out struct{ OK bool }
	raw, err := c.PostJSON(context.Background(), "/x", body, &out)
	if err != nil || !out.OK || string(raw) != `{"ok":true}` {
		t.Fatalf("PostJSON = %s %v %v", raw, out, err)
	}
	if gotAuth != "Bearer k" || gotHdr != "1" || gotCT != "application/json" || gotBody["b"] != float64(2) {
		t.Fatalf("request: auth=%q hdr=%q ct=%q body=%v", gotAuth, gotHdr, gotCT, gotBody)
	}

	_, err = c.PostJSON(context.Background(), "/fail", body, nil)
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 429 || apiErr.Message != "slow down" || apiErr.RetryAfter != time.Second {
		t.Fatalf("err = %#v", err)
	}
	if !errors.Is(err, core.ErrRateLimited) {
		t.Fatal("expected ErrRateLimited")
	}
}

func TestClientPostStreamTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	c := httpx.New("p", core.NewConfig(core.WithTimeout(50*time.Millisecond)), srv.URL)
	rc, err := c.PostStream(context.Background(), "/s", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rc.Close() }()
	_, err = io.ReadAll(rc)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestDataURI(t *testing.T) {
	if got := httpx.DataURI("image/png", []byte{1}); got != "data:image/png;base64,AQ==" {
		t.Fatalf("DataURI = %q", got)
	}
	if got := httpx.MIMEOr("", []byte("\x89PNG\r\n\x1a\n")); got != "image/png" {
		t.Fatalf("MIMEOr sniff = %q", got)
	}
}
