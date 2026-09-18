// Package groq is the Groq chat provider (OpenAI-compatible, no embeddings). Use "groq/<model>" names.
package groq

import (
	"context"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openaicompat"
)

// ID is the provider identifier used in "groq/<model>" names.
const ID = "groq"

// Config is the endpoint definition shared with openaicompat.
var Config = openaicompat.Config{ //nolint:gosec // APIKeyEnv names an environment variable, not a credential
	ID:        ID,
	BaseURL:   "https://api.groq.com/openai/v1",
	APIKeyEnv: "GROQ_API_KEY",
	Match:     nil,
	Quirks:    quirks,
}

// Provider registers groq with a llmkit Registry.
type Provider struct{}

// ID returns "groq".
func (Provider) ID() string { return ID }

// Matches reports whether a bare model name belongs to groq.
func (Provider) Matches(model string) bool { return Config.Match != nil && Config.Match(model) }

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

// Client talks to groq for one model.
type Client struct {
	inner *openaicompat.Client
}

// New builds a Client. The API key comes from GROQ_API_KEY unless WithAPIKey is given.
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

// Provider returns "groq".
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
