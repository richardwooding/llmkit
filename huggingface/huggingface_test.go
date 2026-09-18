package huggingface_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/huggingface"
)

func TestEmbed(t *testing.T) {
	var path, auth string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, auth = r.URL.Path, r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `[[0.1,0.2],[0.3,0.4]]`)
	}))
	defer srv.Close()
	c, err := huggingface.New("org/model", core.WithAPIKey("hf_x"), core.WithBaseURL(srv.URL+"/v1"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/hf-inference/models/org/model/pipeline/feature-extraction" || auth != "Bearer hf_x" {
		t.Fatalf("path=%s auth=%s", path, auth)
	}
	if len(body["inputs"].([]any)) != 2 || !reflect.DeepEqual(resp.Embeddings, [][]float32{{0.1, 0.2}, {0.3, 0.4}}) {
		t.Fatalf("body=%v resp=%+v", body, resp)
	}
}

func TestEmbedTokenLevelPooled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `[[[1,2],[3,4]]]`)
	}))
	defer srv.Close()
	c, _ := huggingface.New("org/model", core.WithAPIKey("k"), core.WithBaseURL(srv.URL+"/v1"))
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}})
	if err != nil || !reflect.DeepEqual(resp.Embeddings, [][]float32{{2, 3}}) {
		t.Fatalf("resp=%+v err=%v", resp, err)
	}
}

func TestIdentity(t *testing.T) {
	p := huggingface.Provider{}
	if p.ID() != "huggingface" || !p.Matches("org/model") || p.Matches("llama3:8b") || p.Matches("gpt-4") {
		t.Fatal("identity/matches wrong")
	}
	c, err := p.Open("org/model", core.NewConfig(core.WithAPIKey("k")))
	if err != nil || c.Provider() != "huggingface" || c.Model() != "org/model" {
		t.Fatalf("Open = %v %v", c, err)
	}
	if _, ok := c.(core.Embedder); !ok {
		t.Fatal("should embed")
	}
}
