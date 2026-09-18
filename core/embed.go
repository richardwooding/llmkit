package core

import "encoding/json"

// EmbedInputType hints whether inputs are search queries or documents.
type EmbedInputType string

// Embedding input types.
const (
	EmbedQuery    EmbedInputType = "query"
	EmbedDocument EmbedInputType = "document"
)

// EmbedRequest asks for one vector per input.
type EmbedRequest struct {
	Inputs          []string
	Dimensions      int
	InputType       EmbedInputType
	Extra           map[string]any
	ProviderOptions map[string]map[string]any
}

// ProviderExtra returns the merged provider-specific overrides for id.
func (r *EmbedRequest) ProviderExtra(id string) map[string]any {
	if r == nil {
		return nil
	}
	return mergeExtra(r.Extra, r.ProviderOptions[id])
}

// EmbedResponse holds one vector per input, in input order.
type EmbedResponse struct {
	Embeddings [][]float32
	Model      string
	Usage      Usage
	Raw        json.RawMessage
}
