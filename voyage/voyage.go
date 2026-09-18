// Package voyage is the Voyage AI embeddings provider. Its Client implements
// only core.Embedder; use "voyage/<model>" names.
package voyage

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// ID is the provider identifier used in "voyage/<model>" names.
const ID = "voyage"

// DefaultBaseURL is the Voyage AI API root.
const DefaultBaseURL = "https://api.voyageai.com/v1"

const keyEnv = "VOYAGE_API_KEY"

// Provider registers Voyage AI with a llmkit Registry.
type Provider struct{}

// ID returns "voyage".
func (Provider) ID() string { return ID }

// Matches claims model names starting with "voyage-".
func (Provider) Matches(model string) bool {
	return strings.HasPrefix(strings.ToLower(model), "voyage-")
}

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

// Client embeds text with one Voyage model.
type Client struct {
	model string
	http  *httpx.Client
}

// New builds a Client. The API key comes from VOYAGE_API_KEY unless WithAPIKey is given.
func New(model string, opts ...core.Option) (*Client, error) {
	return open(model, core.NewConfig(opts...))
}

func open(model string, cfg *core.Config) (*Client, error) {
	if cfg == nil {
		cfg = core.NewConfig()
	}
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv(keyEnv)
	}
	if key == "" {
		return nil, fmt.Errorf("%s: set %s: %w", ID, keyEnv, core.ErrMissingAPIKey)
	}
	hc := httpx.New(ID, cfg, DefaultBaseURL)
	hc.Auth = httpx.BearerAuth(key)
	return &Client{model: model, http: hc}, nil
}

// Provider returns "voyage".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

type embedRequest struct {
	Model           string   `json:"model"`
	Input           []string `json:"input"`
	InputType       string   `json:"input_type,omitempty"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	Truncation      bool     `json:"truncation"`
}

type embedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed calls /embeddings and returns one vector per input, in input order.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	w := embedRequest{Model: c.model, Input: req.Inputs, InputType: string(req.InputType), OutputDimension: req.Dimensions, Truncation: true}
	body, err := httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out embedResponse
	raw, err := c.http.PostJSON(ctx, "/embeddings", body, &out)
	if err != nil {
		return nil, err
	}
	sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].Index < out.Data[j].Index })
	resp := &core.EmbedResponse{
		Embeddings: make([][]float32, 0, len(out.Data)),
		Model:      out.Model,
		Usage:      core.Usage{InputTokens: out.Usage.TotalTokens, TotalTokens: out.Usage.TotalTokens},
		Raw:        raw,
	}
	for _, d := range out.Data {
		resp.Embeddings = append(resp.Embeddings, d.Embedding)
	}
	if resp.Model == "" {
		resp.Model = c.model
	}
	return resp, nil
}
