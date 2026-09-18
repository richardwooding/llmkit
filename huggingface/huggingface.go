// Package huggingface is the Hugging Face Inference Providers router (OpenAI-compatible chat plus feature-extraction embeddings).
package huggingface

import (
	"context"
	"iter"
	"os"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
	"github.com/richardwooding/llmkit/openaicompat"
)

// ID is the provider identifier used in "huggingface/<model>" names.
const ID = "huggingface"

// Config is the endpoint definition shared with openaicompat.
var Config = openaicompat.Config{
	ID:        ID,
	BaseURL:   "https://router.huggingface.co/v1",
	APIKeyEnv: "HF_TOKEN",
	Match:     matches,
	Quirks:    quirks,
}

// Provider registers huggingface with a llmkit Registry.
type Provider struct{}

// ID returns "huggingface".
func (Provider) ID() string { return ID }

// Matches reports whether a bare model name belongs to huggingface.
func (Provider) Matches(model string) bool { return Config.Match != nil && Config.Match(model) }

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

// Client talks to huggingface for one model.
type Client struct {
	inner *openaicompat.Client
	embed embedClient
}

// New builds a Client. The API key comes from HF_TOKEN unless WithAPIKey is given.
func New(model string, opts ...core.Option) (*Client, error) {
	return open(model, core.NewConfig(opts...))
}

func open(model string, cfg *core.Config) (*Client, error) {
	inner, err := openaicompat.NewClient(model, Config, cfg)
	if err != nil {
		return nil, err
	}
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv(Config.APIKeyEnv)
	}
	hc := httpx.New(ID, cfg, Config.BaseURL)
	hc.Auth = httpx.BearerAuth(key)
	return &Client{inner: inner, embed: embedClient{model: model, http: hc}}, nil
}

// Provider returns "huggingface".
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
