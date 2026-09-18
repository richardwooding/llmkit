package core

import (
	"context"
	"encoding/json"
)

// Reranker scores documents against a query.
type Reranker interface {
	Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error)
}

// RerankRequest asks for documents ordered by relevance to Query. TopN limits
// the results when positive.
type RerankRequest struct {
	Query           string
	Documents       []string
	TopN            int
	Extra           map[string]any
	ProviderOptions map[string]map[string]any
}

// ProviderExtra returns the merged provider-specific overrides for id.
func (r *RerankRequest) ProviderExtra(id string) map[string]any {
	if r == nil {
		return nil
	}
	return mergeExtra(r.Extra, r.ProviderOptions[id])
}

// RerankResult is one scored document; Index refers to RerankRequest.Documents.
type RerankResult struct {
	Index int
	Score float64
}

// RerankResponse lists results best first.
type RerankResponse struct {
	Results []RerankResult
	Model   string
	Usage   Usage
	Raw     json.RawMessage
}

// MultimodalEmbedder embeds inputs that mix text with images or other media.
type MultimodalEmbedder interface {
	EmbedMultimodal(ctx context.Context, req *MultimodalEmbedRequest) (*EmbedResponse, error)
}

// MultimodalEmbedRequest carries one vector's worth of parts per input.
type MultimodalEmbedRequest struct {
	Inputs          [][]Part
	Dimensions      int
	InputType       EmbedInputType
	Extra           map[string]any
	ProviderOptions map[string]map[string]any
}

// ProviderExtra returns the merged provider-specific overrides for id.
func (r *MultimodalEmbedRequest) ProviderExtra(id string) map[string]any {
	if r == nil {
		return nil
	}
	return mergeExtra(r.Extra, r.ProviderOptions[id])
}
