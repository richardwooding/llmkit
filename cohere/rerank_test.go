package cohere_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/richardwooding/llmkit/cohere"
	"github.com/richardwooding/llmkit/core"
)

func TestRerank(t *testing.T) {
	var path string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"id":"r1","results":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.4}],"meta":{"billed_units":{"search_units":1}}}`)
	}))
	defer srv.Close()
	c, err := cohere.New("rerank-v3.5", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	var _ core.Reranker = c
	resp, err := c.Rerank(context.Background(), &core.RerankRequest{Query: "q", Documents: []string{"a", "b", "c"}, TopN: 2, Extra: map[string]any{"max_tokens_per_doc": 512}})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/rerank" || body["model"] != "rerank-v3.5" || body["query"] != "q" || body["top_n"] != float64(2) || body["max_tokens_per_doc"] != float64(512) || len(body["documents"].([]any)) != 3 {
		t.Fatalf("path=%s body=%v", path, body)
	}
	want := []core.RerankResult{{Index: 2, Score: 0.9}, {Index: 0, Score: 0.4}}
	if !reflect.DeepEqual(resp.Results, want) || resp.Model != "rerank-v3.5" {
		t.Fatalf("resp = %+v", resp)
	}
	if _, err := c.Rerank(context.Background(), &core.RerankRequest{Query: "q"}); err == nil {
		t.Fatal("empty documents must error")
	}
}
