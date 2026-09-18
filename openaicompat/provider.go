package openaicompat

import (
	"context"
	"fmt"
	"os"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Provider is a core.Provider for one compatible endpoint.
type Provider struct {
	cfg Config
}

// NewProvider wraps cfg as a core.Provider. Register the result with
// llmkit.Register to make "<id>/<model>" resolvable.
func NewProvider(cfg Config) *Provider {
	return &Provider{cfg: cfg}
}

// ID returns the provider identifier.
func (p *Provider) ID() string { return p.cfg.ID }

// Matches reports whether a bare model name belongs to this endpoint.
func (p *Provider) Matches(model string) bool {
	return p.cfg.Match != nil && p.cfg.Match(model)
}

// Open returns a *Client for model.
func (p *Provider) Open(model string, cfg *core.Config) (core.Client, error) {
	return NewClient(model, p.cfg, cfg)
}

// Client speaks Chat Completions to one endpoint for one model.
type Client struct {
	model string
	cfg   Config
	http  *httpx.Client
}

// NewClient builds a Client; no network I/O happens here.
func NewClient(model string, ecfg Config, cfg *core.Config) (*Client, error) {
	if cfg == nil {
		cfg = core.NewConfig()
	}
	key := cfg.APIKey
	if key == "" && ecfg.APIKeyEnv != "" {
		key = os.Getenv(ecfg.APIKeyEnv)
	}
	if key == "" && !ecfg.KeyOptional {
		return nil, fmt.Errorf("%s: set %s: %w", ecfg.ID, ecfg.APIKeyEnv, core.ErrMissingAPIKey)
	}
	hc := httpx.New(ecfg.ID, cfg, ecfg.BaseURL)
	if key != "" {
		hc.Auth = httpx.BearerAuth(key)
	}
	for k, v := range ecfg.Headers {
		if hc.Headers == nil {
			hc.Headers = map[string][]string{}
		}
		if hc.Headers.Get(k) == "" {
			hc.Headers.Set(k, v)
		}
	}
	return &Client{model: model, cfg: ecfg, http: hc}, nil
}

// Provider returns the endpoint ID.
func (c *Client) Provider() string { return c.cfg.ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

// Chat performs a non-streaming completion.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	body, err := c.body(req, false)
	if err != nil {
		return nil, err
	}
	var out chatResponse
	raw, err := c.http.PostJSON(ctx, "/chat/completions", body, &out)
	if err != nil {
		return nil, err
	}
	resp, err := c.toResponse(&out)
	if err != nil {
		return nil, err
	}
	resp.Raw = raw
	return resp, nil
}
