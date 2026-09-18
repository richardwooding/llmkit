// Package cohere is the Cohere v2 provider: Command chat models with tools,
// streaming and thinking, plus Embed models. Use "cohere/<model>" names.
package cohere

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// ID is the provider identifier used in "cohere/<model>" names.
const ID = "cohere"

// DefaultBaseURL is the Cohere v2 API root.
const DefaultBaseURL = "https://api.cohere.com/v2"

const keyEnv = "COHERE_API_KEY"

var modelPrefixes = []string{"command", "embed-", "rerank-", "c4ai-", "aya-"}

// Provider registers Cohere with a llmkit Registry.
type Provider struct{}

// ID returns "cohere".
func (Provider) ID() string { return ID }

// Matches claims Command, Embed, Rerank, C4AI and Aya model names.
func (Provider) Matches(model string) bool {
	m := strings.ToLower(model)
	for _, p := range modelPrefixes {
		if strings.HasPrefix(m, p) {
			return true
		}
	}
	return false
}

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

// Client talks to Cohere for one model.
type Client struct {
	model string
	http  *httpx.Client
}

// New builds a Client. The API key comes from COHERE_API_KEY unless WithAPIKey is given.
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

// Provider returns "cohere".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

// Chat performs a non-streaming completion.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	body, err := c.body(req, false)
	if err != nil {
		return nil, err
	}
	var out chatResponse
	raw, err := c.http.PostJSON(ctx, "/chat", body, &out)
	if err != nil {
		return nil, err
	}
	resp := out.toResponse(c.model)
	resp.Raw = raw
	return resp, nil
}

// Embed calls /embed and returns one float vector per input.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	w := embedRequest{
		Model:           c.model,
		Texts:           req.Inputs,
		InputType:       embedInputType(req.InputType),
		EmbeddingTypes:  []string{"float"},
		OutputDimension: req.Dimensions,
		Truncate:        "END",
	}
	body, err := httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out embedResponse
	raw, err := c.http.PostJSON(ctx, "/embed", body, &out)
	if err != nil {
		return nil, err
	}
	tokens := out.Meta.BilledUnits
	if tokens == nil {
		tokens = out.Meta.Tokens
	}
	resp := &core.EmbedResponse{Embeddings: out.Embeddings.Float, Model: c.model, Raw: raw}
	if tokens != nil {
		n := int(tokens.InputTokens)
		resp.Usage = core.Usage{InputTokens: n, TotalTokens: n}
	}
	return resp, nil
}

func embedInputType(t core.EmbedInputType) string {
	if t == core.EmbedQuery {
		return "search_query"
	}
	return "search_document"
}
