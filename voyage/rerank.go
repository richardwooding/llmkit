package voyage

import (
	"context"
	"fmt"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopK      int      `json:"top_k,omitempty"`
}

type rerankResponse struct {
	Data []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Rerank calls /rerank and returns documents ordered by relevance.
func (c *Client) Rerank(ctx context.Context, req *core.RerankRequest) (*core.RerankResponse, error) {
	if req == nil || req.Query == "" || len(req.Documents) == 0 {
		return nil, fmt.Errorf("%s: rerank needs a query and documents", ID)
	}
	body, err := httpx.MarshalWithExtra(rerankRequest{Model: c.model, Query: req.Query, Documents: req.Documents, TopK: req.TopN}, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out rerankResponse
	raw, err := c.http.PostJSON(ctx, "/rerank", body, &out)
	if err != nil {
		return nil, err
	}
	resp := &core.RerankResponse{
		Model: out.Model, Raw: raw, Results: make([]core.RerankResult, 0, len(out.Data)),
		Usage: core.Usage{InputTokens: out.Usage.TotalTokens, TotalTokens: out.Usage.TotalTokens},
	}
	for _, r := range out.Data {
		resp.Results = append(resp.Results, core.RerankResult{Index: r.Index, Score: r.RelevanceScore})
	}
	if resp.Model == "" {
		resp.Model = c.model
	}
	return resp, nil
}
