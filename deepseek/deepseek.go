// Package deepseek is the DeepSeek chat provider (OpenAI-compatible, no embeddings).
package deepseek

import (
	"context"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openaicompat"
)

// ID is the provider identifier used in "deepseek/<model>" names.
const ID = "deepseek"

// Config is the endpoint definition shared with openaicompat.
var Config = openaicompat.Config{ //nolint:gosec // APIKeyEnv names an environment variable, not a credential
	ID:        ID,
	BaseURL:   "https://api.deepseek.com",
	APIKeyEnv: "DEEPSEEK_API_KEY",
	Match:     openaicompat.PrefixMatcher("deepseek-"),
	Quirks:    quirks,
}

// Provider registers deepseek with a llmkit Registry.
type Provider struct{}

// ID returns "deepseek".
func (Provider) ID() string { return ID }

// Matches reports whether a bare model name belongs to deepseek.
func (Provider) Matches(model string) bool { return Config.Match != nil && Config.Match(model) }

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

// Client talks to deepseek for one model.
type Client struct {
	inner *openaicompat.Client
}

// New builds a Client. The API key comes from DEEPSEEK_API_KEY unless WithAPIKey is given.
func New(model string, opts ...core.Option) (*Client, error) {
	return open(model, core.NewConfig(opts...))
}

func open(model string, cfg *core.Config) (*Client, error) {
	inner, err := openaicompat.NewClient(model, Config, cfg)
	if err != nil {
		return nil, err
	}
	return &Client{inner: inner}, nil
}

// Provider returns "deepseek".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.inner.Model() }

// Chat performs a completion.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	return c.inner.Chat(ctx, req)
}

// Stream performs a streaming completion.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return c.inner.Stream(ctx, req)
}
