// Package ollama talks to a local or remote Ollama daemon over its native
// /api/chat and /api/embed endpoints.
package ollama

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// ID is the provider identifier used in "ollama/<model>" names.
const ID = "ollama"

// DefaultHost is used when OLLAMA_HOST is unset.
const DefaultHost = "http://localhost:11434"

// Provider registers Ollama with a llmkit Registry.
type Provider struct{}

// ID returns "ollama".
func (Provider) ID() string { return ID }

// Matches claims names with an Ollama tag ("llama3.2:3b") or an hf.co path.
func (Provider) Matches(model string) bool {
	return strings.Contains(model, ":") || strings.HasPrefix(model, "hf.co/")
}

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

type keepAliveKey struct{}

// WithKeepAlive controls how long the model stays loaded after a request
// (Ollama duration syntax such as "5m", or "-1" for forever).
func WithKeepAlive(d string) core.Option { return core.WithValue(keepAliveKey{}, d) }

// Client talks to one Ollama model.
type Client struct {
	model     string
	http      *httpx.Client
	keepAlive string
}

// New builds a Client. The host comes from OLLAMA_HOST unless WithBaseURL is given.
func New(model string, opts ...core.Option) (*Client, error) {
	return open(model, core.NewConfig(opts...))
}

func open(model string, cfg *core.Config) (*Client, error) {
	if cfg == nil {
		cfg = core.NewConfig()
	}
	base := cfg.BaseURL
	if base == "" {
		base = os.Getenv("OLLAMA_HOST")
	}
	base, err := normalizeHost(base)
	if err != nil {
		return nil, err
	}
	hc := httpx.New(ID, cfg, base)
	hc.BaseURL = base
	if cfg.APIKey != "" {
		hc.Auth = httpx.BearerAuth(cfg.APIKey)
	}
	c := &Client{model: model, http: hc}
	if ka, ok := cfg.Value(keepAliveKey{}).(string); ok {
		c.keepAlive = ka
	}
	return c, nil
}

func normalizeHost(h string) (string, error) {
	h = strings.TrimSpace(h)
	if h == "" {
		return DefaultHost, nil
	}
	if !strings.Contains(h, "://") {
		h = "http://" + h
	}
	if strings.Count(h, "://") != 1 {
		return "", fmt.Errorf("%s: invalid host %q", ID, h)
	}
	return strings.TrimRight(h, "/"), nil
}

// Provider returns "ollama".
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
	raw, err := c.http.PostJSON(ctx, "/api/chat", body, &out)
	if err != nil {
		return nil, err
	}
	resp := out.toResponse(c.model)
	resp.Raw = raw
	return resp, nil
}

// Embed calls /api/embed and returns one vector per input.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	w := embedRequest{Model: c.model, Input: req.Inputs, Dimensions: req.Dimensions, KeepAlive: c.keepAlive}
	body, err := httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out embedResponse
	raw, err := c.http.PostJSON(ctx, "/api/embed", body, &out)
	if err != nil {
		return nil, err
	}
	model := out.Model
	if model == "" {
		model = c.model
	}
	return &core.EmbedResponse{
		Embeddings: out.Embeddings,
		Model:      model,
		Usage:      core.Usage{InputTokens: out.PromptEvalCount, TotalTokens: out.PromptEvalCount},
		Raw:        raw,
	}, nil
}
