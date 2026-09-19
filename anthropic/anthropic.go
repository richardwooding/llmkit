// Package anthropic talks to the Claude Messages API at api.anthropic.com.
package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// ID is the provider identifier used in "anthropic/<model>" names.
const ID = "anthropic"

// DefaultBaseURL is used unless core.WithBaseURL is given.
const DefaultBaseURL = "https://api.anthropic.com/v1"

// DefaultVersion is the anthropic-version header sent unless WithVersion is given.
const DefaultVersion = "2023-06-01"

const (
	keyEnv           = "ANTHROPIC_API_KEY"
	headerAPIKey     = "x-api-key"
	headerVersion    = "anthropic-version"
	headerBeta       = "anthropic-beta"
	messagesPath     = "/messages"
	countTokensPath  = "/messages/count_tokens"
	defaultMaxTokens = 4096
)

// Provider registers Anthropic with a llmkit Registry.
type Provider struct{}

// ID returns "anthropic".
func (Provider) ID() string { return ID }

// Matches claims model names starting with "claude".
func (Provider) Matches(model string) bool {
	return strings.HasPrefix(strings.ToLower(model), "claude")
}

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

type versionKey struct{}

// WithBeta sends the given beta feature flags in the anthropic-beta header.
func WithBeta(features ...string) core.Option {
	return core.WithHeader(headerBeta, strings.Join(features, ","))
}

// WithVersion overrides the anthropic-version header.
func WithVersion(v string) core.Option { return core.WithValue(versionKey{}, v) }

// Client talks to one Claude model. It implements core.Chatter, core.Streamer
// and core.TokenCounter; Anthropic offers no embeddings endpoint.
type Client struct {
	model string
	http  *httpx.Client
}

// New builds a Client. The key comes from ANTHROPIC_API_KEY unless
// core.WithAPIKey is given. No network I/O happens here.
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
	hc.Auth = httpx.HeaderAuth(headerAPIKey, key)
	if hc.Headers == nil {
		hc.Headers = http.Header{}
	}
	if v, ok := cfg.Value(versionKey{}).(string); ok && v != "" {
		hc.Headers.Set(headerVersion, v)
	} else if hc.Headers.Get(headerVersion) == "" {
		hc.Headers.Set(headerVersion, DefaultVersion)
	}
	return &Client{model: model, http: hc}, nil
}

// Provider returns "anthropic".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

// Chat performs a non-streaming completion.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	body, err := c.body(req, false)
	if err != nil {
		return nil, err
	}
	var out wireResponse
	raw, err := c.http.PostJSON(ctx, messagesPath, body, &out)
	if err != nil {
		return nil, err
	}
	resp := out.toResponse()
	resp.Raw = raw
	return resp, nil
}

// CountTokens asks /messages/count_tokens how many input tokens req would
// consume. Request.Extra and ProviderOptions are not sent.
func (c *Client) CountTokens(ctx context.Context, req *core.Request) (int, error) {
	body, err := c.countBody(req)
	if err != nil {
		return 0, err
	}
	var out wireCountResponse
	if _, err := c.http.PostJSON(ctx, countTokensPath, body, &out); err != nil {
		return 0, err
	}
	return out.InputTokens, nil
}
