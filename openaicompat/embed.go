package openaicompat

import (
	"context"
	"fmt"
	"sort"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Embed calls the endpoint's embeddings API. It fails with ErrUnsupported when
// Quirks.EmbedPath is empty.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if c.cfg.Quirks.EmbedPath == "" {
		return nil, core.Unsupported(c.cfg.ID, "embeddings")
	}
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", c.cfg.ID)
	}
	body, err := httpx.MarshalWithExtra(embedRequest{
		Model: c.model, Input: req.Inputs, Dimensions: req.Dimensions, EncodingFormat: "float",
	}, req.ProviderExtra(c.cfg.ID))
	if err != nil {
		return nil, err
	}
	var out embedResponse
	raw, err := c.http.PostJSON(ctx, c.cfg.Quirks.EmbedPath, body, &out)
	if err != nil {
		return nil, err
	}
	sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].Index < out.Data[j].Index })
	resp := &core.EmbedResponse{Model: out.Model, Raw: raw, Embeddings: make([][]float32, 0, len(out.Data))}
	for _, d := range out.Data {
		resp.Embeddings = append(resp.Embeddings, d.Embedding)
	}
	if out.Usage != nil {
		resp.Usage = usage(out.Usage)
	}
	if resp.Model == "" {
		resp.Model = c.model
	}
	return resp, nil
}
