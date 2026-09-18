package voyage_test

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

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/voyage"
)

func TestEmbedMultimodal(t *testing.T) {
	var path string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"data":[{"index":1,"embedding":[0.3]},{"index":0,"embedding":[0.1,0.2]}],"model":"voyage-multimodal-3","usage":{"text_tokens":3,"image_pixels":100,"total_tokens":10}}`)
	}))
	defer srv.Close()
	c, err := voyage.New("voyage-multimodal-3", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	var _ core.MultimodalEmbedder = c
	resp, err := c.EmbedMultimodal(context.Background(), &core.MultimodalEmbedRequest{
		Inputs: [][]core.Part{
			{core.Text("a cat"), core.Image([]byte{1}, "image/png")},
			{core.ImageURL("https://x/y.jpg"), core.FileURL("https://x/v.mp4", "video/mp4")},
		},
		InputType: core.EmbedQuery,
	})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/multimodalembeddings" || body["input_type"] != "query" || body["truncation"] != true {
		t.Fatalf("path=%s body=%v", path, body)
	}
	inputs := body["inputs"].([]any)
	first := inputs[0].(map[string]any)["content"].([]any)
	if first[0].(map[string]any)["text"] != "a cat" || !strings.HasPrefix(first[1].(map[string]any)["image_base64"].(string), "data:image/png;base64,") {
		t.Fatalf("first input = %v", first)
	}
	second := inputs[1].(map[string]any)["content"].([]any)
	if second[0].(map[string]any)["type"] != "image_url" || second[1].(map[string]any)["type"] != "video_url" {
		t.Fatalf("second input = %v", second)
	}
	if !reflect.DeepEqual(resp.Embeddings, [][]float32{{0.1, 0.2}, {0.3}}) || resp.Usage.TotalTokens != 10 {
		t.Fatalf("resp = %+v", resp)
	}
	_, err = c.EmbedMultimodal(context.Background(), &core.MultimodalEmbedRequest{Inputs: [][]core.Part{{core.Audio([]byte{1}, "audio/wav")}}})
	if !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("audio err = %v", err)
	}
	_, err = c.EmbedMultimodal(context.Background(), &core.MultimodalEmbedRequest{Inputs: [][]core.Part{{core.File([]byte{1}, "application/pdf", "x")}}})
	if !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("pdf err = %v", err)
	}
}

func TestRerank(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"data":[{"index":1,"relevance_score":0.8},{"index":0,"relevance_score":0.2}],"model":"rerank-2","usage":{"total_tokens":12}}`)
	}))
	defer srv.Close()
	c, _ := voyage.New("rerank-2", core.WithAPIKey("k"), core.WithBaseURL(srv.URL))
	var _ core.Reranker = c
	resp, err := c.Rerank(context.Background(), &core.RerankRequest{Query: "q", Documents: []string{"a", "b"}, TopN: 1})
	if err != nil {
		t.Fatal(err)
	}
	if body["top_k"] != float64(1) || body["query"] != "q" {
		t.Fatalf("body = %v", body)
	}
	if !reflect.DeepEqual(resp.Results, []core.RerankResult{{Index: 1, Score: 0.8}, {Index: 0, Score: 0.2}}) || resp.Usage.TotalTokens != 12 || resp.Model != "rerank-2" {
		t.Fatalf("resp = %+v", resp)
	}
}
