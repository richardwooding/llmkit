// Package openai talks to the OpenAI platform: chat over the Responses API
// (POST /responses) and vectors over POST /embeddings.
//
// Request.Stop and Request.Seed have no Responses API equivalent and are
// ignored. ReasoningPart.Signature carries the reasoning item id and
// ReasoningPart.Encrypted its encrypted_content; both are echoed back on later
// turns so multi-turn tool use works with reasoning models.
package openai

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
	"github.com/richardwooding/llmkit/openaicompat"
)

// ID is the provider identifier used in "openai/<model>" names.
const ID = "openai"

// DefaultBaseURL is used when WithBaseURL is not given.
const DefaultBaseURL = "https://api.openai.com/v1"

const (
	keyEnv        = "OPENAI_API_KEY"
	pathResponses = "/responses"
	pathEmbed     = "/embeddings"
)

var modelPrefixes = []string{"gpt", "chatgpt-", "o1", "o3", "o4", "text-embedding-3", "text-embedding-ada"}

// Provider registers OpenAI with a llmkit Registry.
type Provider struct{}

// ID returns "openai".
func (Provider) ID() string { return ID }

// Matches claims gpt*, chatgpt-*, o1/o3/o4* and text-embedding-* names.
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

// WithOrganization sets the OpenAI-Organization header on every request.
func WithOrganization(org string) core.Option { return core.WithHeader("OpenAI-Organization", org) }

// WithProject sets the OpenAI-Project header on every request.
func WithProject(project string) core.Option { return core.WithHeader("OpenAI-Project", project) }

// Client talks to one OpenAI model.
type Client struct {
	model string
	http  *httpx.Client
	embed *openaicompat.Client
}

// New builds a Client. The API key comes from OPENAI_API_KEY unless WithAPIKey is given.
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
	embed, err := openaicompat.NewClient(model, openaicompat.Config{
		ID: ID, BaseURL: DefaultBaseURL, APIKeyEnv: keyEnv, Quirks: openaicompat.Quirks{EmbedPath: pathEmbed},
	}, cfg)
	if err != nil {
		return nil, err
	}
	return &Client{model: model, http: hc, embed: embed}, nil
}

// Provider returns "openai".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

// Chat performs a non-streaming Responses API call.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	body, err := c.body(req, false)
	if err != nil {
		return nil, err
	}
	var out wireResponse
	raw, err := c.http.PostJSON(ctx, pathResponses, body, &out)
	if err != nil {
		return nil, err
	}
	resp, err := toResponse(&out, raw)
	if err != nil {
		return nil, err
	}
	resp.Raw = raw
	return resp, nil
}

// Embed calls POST /embeddings and returns one vector per input.
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	return c.embed.Embed(ctx, req)
}
