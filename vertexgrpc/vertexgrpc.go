package vertexgrpc

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"google.golang.org/api/option"
	"google.golang.org/grpc"

	"github.com/richardwooding/llmkit/core"
)

// ID is the provider identifier used in "vertexgrpc/<model>" names.
const ID = "vertexgrpc"

// DefaultLocation is used when neither WithLocation nor GOOGLE_CLOUD_LOCATION is set.
const DefaultLocation = "us-central1"

// Provider registers Vertex AI gRPC with a llmkit Registry.
type Provider struct{}

// ID returns "vertexgrpc".
func (Provider) ID() string { return ID }

// Matches never claims bare names; use the "vertexgrpc/" prefix.
func (Provider) Matches(string) bool { return false }

// Open builds a Client from a core.Config.
func (Provider) Open(model string, cfg *core.Config) (core.Client, error) { return open(model, cfg) }

type (
	projectKey    struct{}
	locationKey   struct{}
	clientOptsKey struct{}
	grpcConnKey   struct{}
)

// WithProject sets the Google Cloud project, overriding GOOGLE_CLOUD_PROJECT.
func WithProject(id string) core.Option { return core.WithValue(projectKey{}, id) }

// WithLocation sets the Vertex AI region ("us-central1", "europe-west4", "global").
func WithLocation(loc string) core.Option { return core.WithValue(locationKey{}, loc) }

// WithClientOptions appends google.golang.org/api options to the PredictionClient.
func WithClientOptions(opts ...option.ClientOption) core.Option {
	return func(c *core.Config) {
		existing, _ := c.Value(clientOptsKey{}).([]option.ClientOption)
		core.WithValue(clientOptsKey{}, append(existing, opts...))(c)
	}
}

// WithGRPCConn uses an existing connection instead of dialing the regional endpoint.
func WithGRPCConn(conn *grpc.ClientConn) core.Option { return core.WithValue(grpcConnKey{}, conn) }

// Client talks to one Vertex AI model over gRPC.
type Client struct {
	model    string
	project  string
	location string
	timeout  time.Duration
	started  atomic.Bool
	pc       func() (*aiplatform.PredictionClient, error)
}

// New builds a Client; the gRPC connection is opened on first use.
func New(model string, opts ...core.Option) (*Client, error) {
	return open(model, core.NewConfig(opts...))
}

func open(model string, cfg *core.Config) (*Client, error) {
	if cfg == nil {
		cfg = core.NewConfig()
	}
	project, _ := cfg.Value(projectKey{}).(string)
	if project == "" {
		project = firstEnv("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT")
	}
	if project == "" {
		return nil, fmt.Errorf("%s: set GOOGLE_CLOUD_PROJECT or use vertexgrpc.WithProject: %w", ID, core.ErrMissingAPIKey)
	}
	location, _ := cfg.Value(locationKey{}).(string)
	if location == "" {
		location = firstEnv("GOOGLE_CLOUD_LOCATION")
	}
	if location == "" {
		location = DefaultLocation
	}
	c := &Client{model: model, project: project, location: location, timeout: cfg.Timeout}
	clientOpts := c.clientOptions(cfg)
	c.pc = sync.OnceValues(func() (*aiplatform.PredictionClient, error) {
		return aiplatform.NewPredictionClient(context.Background(), clientOpts...)
	})
	return c, nil
}

// clientOptions puts the regional endpoint first so any WithEndpoint or
// WithGRPCConn supplied by the caller takes precedence.
func (c *Client) clientOptions(cfg *core.Config) []option.ClientOption {
	opts := []option.ClientOption{option.WithEndpoint(endpoint(c.location))}
	if conn, ok := cfg.Value(grpcConnKey{}).(*grpc.ClientConn); ok && conn != nil {
		opts = append(opts, option.WithGRPCConn(conn))
	}
	extra, _ := cfg.Value(clientOptsKey{}).([]option.ClientOption)
	return append(opts, extra...)
}

func endpoint(location string) string {
	if location == "global" {
		return "aiplatform.googleapis.com:443"
	}
	return location + "-aiplatform.googleapis.com:443"
}

func firstEnv(names ...string) string {
	for _, n := range names {
		if v := strings.TrimSpace(os.Getenv(n)); v != "" {
			return v
		}
	}
	return ""
}

// Provider returns "vertexgrpc".
func (c *Client) Provider() string { return ID }

// Model returns the model name.
func (c *Client) Model() string { return c.model }

// Close releases the underlying gRPC connection if one was opened.
func (c *Client) Close() error {
	if !c.started.Load() {
		return nil
	}
	pc, err := c.pc()
	if err != nil {
		return nil
	}
	return pc.Close()
}

func (c *Client) client() (*aiplatform.PredictionClient, error) {
	c.started.Store(true)
	pc, err := c.pc()
	if err != nil {
		return nil, fmt.Errorf("%s: create client: %w", ID, err)
	}
	return pc, nil
}

func (c *Client) context(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.timeout > 0 {
		return context.WithTimeout(ctx, c.timeout)
	}
	return context.WithCancel(ctx)
}

func (c *Client) modelName() string {
	if strings.HasPrefix(c.model, "projects/") {
		return c.model
	}
	return fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", c.project, c.location, c.model)
}

// Chat performs a non-streaming GenerateContent call.
func (c *Client) Chat(ctx context.Context, req *core.Request) (*core.Response, error) {
	greq, err := c.request(req)
	if err != nil {
		return nil, err
	}
	pc, err := c.client()
	if err != nil {
		return nil, err
	}
	ctx, cancel := c.context(ctx)
	defer cancel()
	resp, err := pc.GenerateContent(ctx, greq)
	if err != nil {
		return nil, apiError(err)
	}
	return c.response(resp)
}
