package vertex

import (
	"context"
	"fmt"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Embed calls the model's :predict endpoint and returns one vector per input.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	w := embedRequest{
		Instances:  make([]embedInstance, 0, len(req.Inputs)),
		Parameters: embedParameters{OutputDimensionality: req.Dimensions, AutoTruncate: true},
	}
	task := taskType(req.InputType)
	for _, in := range req.Inputs {
		w.Instances = append(w.Instances, embedInstance{Content: in, TaskType: task})
	}
	body, err := httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out embedResponse
	raw, err := c.http.PostJSON(ctx, c.modelPath("predict"), body, &out)
	if err != nil {
		return nil, googleError(err)
	}
	resp := &core.EmbedResponse{Model: c.model, Raw: raw, Embeddings: make([][]float32, 0, len(out.Predictions))}
	for i := range out.Predictions {
		e := &out.Predictions[i].Embeddings
		resp.Embeddings = append(resp.Embeddings, e.Values)
		resp.Usage.InputTokens += e.Statistics.TokenCount
	}
	resp.Usage.TotalTokens = resp.Usage.InputTokens
	return resp, nil
}

func taskType(t core.EmbedInputType) string {
	switch t {
	case core.EmbedQuery:
		return "RETRIEVAL_QUERY"
	case core.EmbedDocument:
		return "RETRIEVAL_DOCUMENT"
	default:
		return ""
	}
}
