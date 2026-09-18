package cohere

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
	TopN      int      `json:"top_n,omitempty"`
}

type rerankResponse struct {
	ID      string `json:"id"`
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
	Meta struct {
		BilledUnits struct {
			SearchUnits float64 `json:"search_units"`
		} `json:"billed_units"`
	} `json:"meta"`
}

// Rerank calls /rerank and returns documents ordered by relevance.
func (c *Client) Rerank(ctx context.Context, req *core.RerankRequest) (*core.RerankResponse, error) {
	if req == nil || req.Query == "" || len(req.Documents) == 0 {
		return nil, fmt.Errorf("%s: rerank needs a query and documents", ID)
	}
	body, err := httpx.MarshalWithExtra(rerankRequest{Model: c.model, Query: req.Query, Documents: req.Documents, TopN: req.TopN}, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out rerankResponse
	raw, err := c.http.PostJSON(ctx, "/rerank", body, &out)
	if err != nil {
		return nil, err
	}
	resp := &core.RerankResponse{Model: c.model, Raw: raw, Results: make([]core.RerankResult, 0, len(out.Results))}
	for _, r := range out.Results {
		resp.Results = append(resp.Results, core.RerankResult{Index: r.Index, Score: r.RelevanceScore})
	}
	return resp, nil
}
