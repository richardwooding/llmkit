// Package vertex talks to Gemini and Google embedding models on Vertex AI over
// REST, authenticating with OAuth2 access tokens (Application Default
// Credentials by default).
package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// ID is the provider identifier used in "vertex/<model>" names.
const ID = "vertex"

// DefaultLocation is used when neither WithLocation nor GOOGLE_CLOUD_LOCATION is set.
const DefaultLocation = "us-central1"

const (
	scopeCloudPlatform = "https://www.googleapis.com/auth/cloud-platform"
	locationGlobal     = "global"
)

var modelPrefixes = []string{"gemini-", "imagen-", "text-embedding-0", "text-multilingual-embedding-", "gemini-embedding-"}

// Provider registers Vertex AI with a llmkit Registry.
type Provider struct{}

// ID returns "vertex".
func (Provider) ID() string { return ID }

// Matches claims Google publisher model names (gemini-*, imagen-*, text-embedding-*).
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

type (
	projectKey     struct{}
	locationKey    struct{}
	tokenSourceKey struct{}
	accessTokenKey struct{}
)

// WithProject sets the Google Cloud project, overriding GOOGLE_CLOUD_PROJECT.
func WithProject(project string) core.Option { return core.WithValue(projectKey{}, project) }

// WithLocation sets the Vertex AI region ("us-central1", "europe-west4", "global"), overriding GOOGLE_CLOUD_LOCATION.
func WithLocation(location string) core.Option { return core.WithValue(locationKey{}, location) }

// WithTokenSource supplies OAuth2 credentials instead of Application Default Credentials.
func WithTokenSource(ts oauth2.TokenSource) core.Option { return core.WithValue(tokenSourceKey{}, ts) }

// WithAccessToken uses a fixed bearer token, handy for CI and tests.
func WithAccessToken(token string) core.Option { return core.WithValue(accessTokenKey{}, token) }

// Client talks to one Vertex AI publisher model.
type Client struct {
	model    string
	project  string
	location string
	http     *httpx.Client
	tokens   func() (oauth2.TokenSource, error)
}

// New builds a Client. Credentials are resolved lazily on the first request,
// so construction never touches the network. A missing project fails with
// core.ErrMissingAPIKey.
func New(model string, opts ...core.Option) (*Client, error) {
	return open(model, core.NewConfig(opts...))
}

func open(model string, cfg *core.Config) (*Client, error) {
	if cfg == nil {
		cfg = core.NewConfig()
	}
	project := stringValue(cfg, projectKey{})
	if project == "" {
		project = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}
	if project == "" {
		project = os.Getenv("GCLOUD_PROJECT")
	}
	if project == "" {
		return nil, fmt.Errorf("%s: set GOOGLE_CLOUD_PROJECT or use vertex.WithProject: %w", ID, core.ErrMissingAPIKey)
	}
	location := stringValue(cfg, locationKey{})
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
	}
	if location == "" {
		location = DefaultLocation
	}
	c := &Client{model: model, project: project, location: location, tokens: tokenSource(cfg)}
	c.http = httpx.New(ID, cfg, hostFor(location))
	c.http.Auth = c.authorize
	return c, nil
}

func stringValue(cfg *core.Config, key any) string {
	s, _ := cfg.Value(key).(string)
	return s
}

func hostFor(location string) string {
	if location == locationGlobal {
		return "https://aiplatform.googleapis.com"
	}
	return "https://" + location + "-aiplatform.googleapis.com"
}

func tokenSource(cfg *core.Config) func() (oauth2.TokenSource, error) {
	if ts, ok := cfg.Value(tokenSourceKey{}).(oauth2.TokenSource); ok && ts != nil {
		return fixed(ts)
	}
	if tok := stringValue(cfg, accessTokenKey{}); tok != "" {
		return fixed(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: tok}))
	}
	if cfg.APIKey != "" {
		return fixed(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: cfg.APIKey}))
	}
	return sync.OnceValues(func() (oauth2.TokenSource, error) {
		// Background rather than the first request's ctx: the source outlives that request.
		ts, err := google.DefaultTokenSource(context.Background(), scopeCloudPlatform)
		if err != nil {
			return nil, fmt.Errorf("%s: application default credentials: %w", ID, err)
		}
		return oauth2.ReuseTokenSource(nil, ts), nil
	})
}

func fixed(ts oauth2.TokenSource) func() (oauth2.TokenSource, error) {
	return func() (oauth2.TokenSource, error) { return ts, nil }
}

func (c *Client) authorize(_ context.Context, req *http.Request) error {
	ts, err := c.tokens()
	if err != nil {
		return err
	}
	tok, err := ts.Token()
	if err != nil {
		return fmt.Errorf("access token: %w", err)
	}
	tok.SetAuthHeader(req)
	return nil
}

// Provider returns "vertex".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

func (c *Client) modelPath(verb string) string {
	return fmt.Sprintf("/v1/projects/%s/locations/%s/publishers/google/models/%s:%s", c.project, c.location, c.model, verb)
}

// Chat performs a non-streaming generateContent call.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	body, err := c.body(req)
	if err != nil {
		return nil, err
	}
	var out wireResponse
	raw, err := c.http.PostJSON(ctx, c.modelPath("generateContent"), body, &out)
	if err != nil {
		return nil, googleError(err)
	}
	resp := c.toResponse(&out)
	resp.Raw = raw
	return resp, nil
}

// googleError swaps the numeric code httpx picks out of Google's error
// envelope for the more useful status string (RESOURCE_EXHAUSTED, ...).
func googleError(err error) error {
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	var env struct {
		Error struct {
			Status string `json:"status"`
		} `json:"error"`
	}
	if json.Unmarshal(apiErr.Body, &env) == nil && env.Error.Status != "" {
		apiErr.Code = env.Error.Status
	}
	return err
}
