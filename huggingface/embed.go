package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
	"github.com/richardwooding/llmkit/openaicompat"
)

var quirks = openaicompat.Quirks{
	Images:      true,
	StreamUsage: true,
	JSONSchema:  true,
}

func matches(model string) bool {
	return strings.Contains(model, "/") && !strings.Contains(model, ":")
}

type embedClient struct {
	model string
	http  *httpx.Client
}

// Embed calls the feature-extraction pipeline for the model and returns one
// vector per input. Token-level outputs are mean-pooled.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	body, err := httpx.MarshalWithExtra(map[string]any{"inputs": req.Inputs}, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var raw json.RawMessage
	raw, err = c.embed.http.PostJSON(ctx, c.embedURL(), body, &raw)
	if err != nil {
		return nil, err
	}
	vectors, err := decodeVectors(raw, len(req.Inputs))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", ID, err)
	}
	return &core.EmbedResponse{Embeddings: vectors, Model: c.Model(), Raw: raw}, nil
}

func (c *Client) embedURL() string {
	base := strings.TrimSuffix(c.embed.http.BaseURL, "/v1")
	return base + "/hf-inference/models/" + c.Model() + "/pipeline/feature-extraction"
}

func decodeVectors(raw json.RawMessage, n int) ([][]float32, error) {
	var flat [][]float32
	if err := json.Unmarshal(raw, &flat); err == nil {
		if n == 1 && len(flat) > 0 && len(flat[0]) > 0 {
			return [][]float32{meanPool(flat)}, nil
		}
		return flat, nil
	}
	var tokens [][][]float32
	if err := json.Unmarshal(raw, &tokens); err != nil {
		return nil, fmt.Errorf("unexpected embeddings shape: %w", err)
	}
	out := make([][]float32, 0, len(tokens))
	for _, t := range tokens {
		out = append(out, meanPool(t))
	}
	return out, nil
}

func meanPool(rows [][]float32) []float32 {
	if len(rows) == 0 {
		return nil
	}
	if len(rows) == 1 {
		return rows[0]
	}
	out := make([]float32, len(rows[0]))
	for _, r := range rows {
		for i := range out {
			if i < len(r) {
				out[i] += r[i]
			}
		}
	}
	for i := range out {
		out[i] /= float32(len(rows))
	}
	return out
}
