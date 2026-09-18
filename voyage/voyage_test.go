package voyage_test

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
	"github.com/richardwooding/llmkit/voyage"
)

var _ core.Embedder = (*voyage.Client)(nil)

type capture struct {
	body map[string]any
	path string
	auth string
}

func newServer(t *testing.T, response string) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.auth = r.Header.Get("Authorization")
		cap.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&cap.body)
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func newClient(t *testing.T, url string) *voyage.Client {
	t.Helper()
	c, err := voyage.New("voyage-4", core.WithAPIKey("vk"), core.WithBaseURL(url))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestEmbed(t *testing.T) {
	srv, cap := newServer(t, `{"object":"list","data":[{"object":"embedding","index":1,"embedding":[0.3]},{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"voyage-4-large","usage":{"total_tokens":8}}`)
	c := newClient(t, srv.URL)
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{
		Inputs:          []string{"a", "b"},
		Dimensions:      512,
		InputType:       core.EmbedQuery,
		Extra:           map[string]any{"output_dtype": "int8"},
		ProviderOptions: map[string]map[string]any{"voyage": {"encoding_format": "base64"}, "other": {"z": 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if cap.path != "/embeddings" || cap.auth != "Bearer vk" {
		t.Fatalf("path=%s auth=%s", cap.path, cap.auth)
	}
	b := cap.body
	if b["model"] != "voyage-4" || b["input_type"] != "query" || b["output_dimension"] != float64(512) || b["truncation"] != true {
		t.Fatalf("body = %v", b)
	}
	if !reflect.DeepEqual(b["input"], []any{"a", "b"}) || b["output_dtype"] != "int8" || b["encoding_format"] != "base64" {
		t.Fatalf("body = %v", b)
	}
	if _, ok := b["z"]; ok {
		t.Fatal("other provider's options leaked")
	}
	resp.Raw = nil
	want := &core.EmbedResponse{
		Embeddings: [][]float32{{0.1, 0.2}, {0.3}},
		Model:      "voyage-4-large",
		Usage:      core.Usage{InputTokens: 8, TotalTokens: 8},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestEmbedDefaults(t *testing.T) {
	srv, cap := newServer(t, `{"data":[{"index":0,"embedding":[1]}],"usage":{"total_tokens":1}}`)
	c := newClient(t, srv.URL)
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}, InputType: core.EmbedDocument})
	if err != nil {
		t.Fatal(err)
	}
	if cap.body["input_type"] != "document" {
		t.Fatalf("input_type = %v", cap.body["input_type"])
	}
	if _, ok := cap.body["output_dimension"]; ok {
		t.Fatal("output_dimension must be omitted when Dimensions is zero")
	}
	if resp.Model != "voyage-4" {
		t.Fatalf("model = %q", resp.Model)
	}

	srv2, cap2 := newServer(t, `{"data":[],"usage":{}}`)
	if _, err := newClient(t, srv2.URL).Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}}); err != nil {
		t.Fatal(err)
	}
	if _, ok := cap2.body["input_type"]; ok {
		t.Fatal("input_type must be omitted when unset")
	}
}

func TestEmbedEmptyInputsAndAPIError(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hit = true
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"detail":"Provided API key is invalid."}`)
	}))
	t.Cleanup(srv.Close)
	c := newClient(t, srv.URL)
	if _, err := c.Embed(context.Background(), &core.EmbedRequest{}); err == nil || hit {
		t.Fatalf("empty inputs: err=%v hit=%v", err, hit)
	}
	if _, err := c.Embed(context.Background(), nil); err == nil || hit {
		t.Fatalf("nil request: err=%v hit=%v", err, hit)
	}
	_, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}})
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 401 || apiErr.Message != "Provided API key is invalid." {
		t.Fatalf("err = %v", err)
	}
}

func TestMissingKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	_, err := voyage.New("voyage-4")
	if !errors.Is(err, core.ErrMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("VOYAGE_API_KEY", "from-env")
	c, err := voyage.New("voyage-4")
	if err != nil || c.Provider() != "voyage" || c.Model() != "voyage-4" {
		t.Fatalf("client = %v %v", c, err)
	}
}

func TestProviderMatches(t *testing.T) {
	p := voyage.Provider{}
	if p.ID() != "voyage" {
		t.Fatalf("id = %s", p.ID())
	}
	tests := map[string]bool{
		"voyage-4":         true,
		"voyage-code-3":    true,
		"Voyage-4-Large":   true,
		"voyager":          false,
		"embed-v4.0":       false,
		"text-embedding-3": false,
		"":                 false,
	}
	for model, want := range tests {
		if got := p.Matches(model); got != want {
			t.Errorf("Matches(%q) = %v, want %v", model, got, want)
		}
	}
	t.Setenv("VOYAGE_API_KEY", "k")
	c, err := p.Open("voyage-4", nil)
	if err != nil || c.Model() != "voyage-4" {
		t.Fatalf("Open = %v %v", c, err)
	}
}

func TestClientIsEmbedderOnly(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "k")
	c, err := voyage.Provider{}.Open("voyage-4", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.(core.Embedder); !ok {
		t.Fatal("client must implement Embedder")
	}
	if _, ok := c.(core.Chatter); ok {
		t.Fatal("client must not implement Chatter")
	}
	if _, ok := c.(core.Streamer); ok {
		t.Fatal("client must not implement Streamer")
	}
}
